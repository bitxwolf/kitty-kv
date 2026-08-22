"""
kitty/pipeline.py
-----------------
Phase 4 — Runtime Coordination Layer.

Implements the token-by-token state machine that manages all four memory
regions of the Kitty KV cache:

  1. Attention Sink Buffer (S=32 tokens, always FP16)
  2. Key Q-Buffer (G=128 token accumulator; quantisation triggered when full)
  3. Value Local Buffer (R=128 token sliding window, FP16)
  4. Value Q-Buffer (tokens evicted from Local Buffer, quantised when G tokens)

Per reference spec §5 (Runtime Coordination Layer):

    [ New K, V token generated (FP16) ]
                   │
                   ▼
      Is Sink Buffer full?
      ├── NO  → Write K,V to Sink (FP16, permanent)
      └── YES → Write K to Key Q-Buffer
                Write V to Value Local Buffer (sliding window)
                If Local Buffer overflows:
                  Evict oldest V token → Value Q-Buffer
                If Key Q-Buffer full (size == G):
                  → Quantise Key Q-Buffer → Page in layout
                  → Quantise Value Q-Buffer → Page in layout
                  → Clear both Q-Buffers

The pipeline handles num_layers independently.  Each layer maintains its own
per-layer sink/qbuffer state because the Key/Value activations differ per layer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from .config import KittyConfig
from .layout import PageCentricKVLayoutManager

# Import quantisation kernels (fall back to PyTorch on CPU)
from .kernels.quantize_key import quantize_key_page
from .kernels.quantize_value import quantize_value_page


class KittyInferencePipeline:
    """Runtime KV cache buffer state machine for Kitty.

    One instance manages the entire model (all layers, all heads).

    Parameters
    ----------
    config:
        KittyConfig for this model.
    layout:
        Pre-allocated :class:`~kitty.layout.PageCentricKVLayoutManager`.
    boost_idx_uint8:
        Tensor of shape ``(num_layers, num_kv_heads, head_dim)`` uint8.
        From :class:`~kitty.sensitivity.KittySensitivityProfiler`.
    """

    def __init__(
        self,
        config: KittyConfig,
        layout: PageCentricKVLayoutManager,
        boost_idx_uint8: torch.Tensor,
    ) -> None:
        self.config = config
        self.layout = layout
        self.boost_idx_uint8 = boost_idx_uint8  # (L, H, D) uint8

        L = config.num_layers
        H = config.num_kv_heads
        D = config.head_dim
        S = config.sink_size
        G = config.q_buffer_size
        R = config.local_value_window
        dev = torch.device(config.device)

        if config.dtype == "float16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.bfloat16

        # -------------------------------------------------------- Sink buffers
        # Shape: (L, H, S, D)  — static FP16 sinks, never quantised
        self.sink_k = torch.zeros(L, H, S, D, dtype=self._dtype, device=dev)
        self.sink_v = torch.zeros(L, H, S, D, dtype=self._dtype, device=dev)
        self.sink_fill = 0  # number of sink tokens filled (shared across layers)

        # ------------------------------------------- Key Q-Buffer (per-layer)
        # Shape: (L, H, G, D)  — accumulates up to G FP16 keys per layer
        self.key_qbuf = torch.zeros(L, H, G, D, dtype=self._dtype, device=dev)
        self.key_qbuf_len = 0  # tokens currently in qbuffer (same across layers)

        # ------------------------------------- Value Local Buffer (per-layer)
        # Shape: (L, H, R, D)  — sliding window of R most-recent FP16 values
        self.val_local = torch.zeros(L, H, R, D, dtype=self._dtype, device=dev)
        self.val_local_len = 0  # tokens currently in local buf

        # ------------------------------------- Value Q-Buffer (per-layer)
        # Shape: (L, H, G, D)  — evicted tokens waiting to be quantised
        self.val_qbuf = torch.zeros(L, H, G, D, dtype=self._dtype, device=dev)
        self.val_qbuf_len = 0  # evicted tokens ready for quantisation

        # -------------------------------------------------- Page counters
        # page_counter[(seq_id, layer_id)] = next page number to allocate
        self._page_counter: Dict[Tuple[int, int], int] = defaultdict(int)

    # ------------------------------------------------------------------ push

    def push_token(
        self,
        layer_id: int,
        key: torch.Tensor,      # (num_kv_heads, head_dim) float16
        value: torch.Tensor,    # (num_kv_heads, head_dim) float16
        seq_id: int = 0,
    ) -> None:
        """Push one new FP16 KV token into the pipeline for a given layer.

        Parameters
        ----------
        layer_id:
            Index of the transformer layer (0-indexed).
        key, value:
            New key and value tensors of shape ``(num_kv_heads, head_dim)``.
        seq_id:
            Sequence identifier (batch slot).
        """
        cfg = self.config
        S, G, R = cfg.sink_size, cfg.q_buffer_size, cfg.local_value_window

        key = key.to(self._dtype).to(self.sink_k.device)
        value = value.to(self._dtype).to(self.sink_v.device)

        # ---------------------------------------- Sink Buffer
        if self.sink_fill < S:
            pos = self.sink_fill
            self.sink_k[layer_id, :, pos, :] = key    # (H, D)
            self.sink_v[layer_id, :, pos, :] = value
            # Increment sink_fill only on the first layer to avoid double-counting
            # (all layers share the same "token count" for sinks)
            if layer_id == 0:
                self.sink_fill += 1
            return

        # ---------------------------------------- Key Q-Buffer
        kqlen = self.key_qbuf_len
        self.key_qbuf[layer_id, :, kqlen, :] = key    # (H, D)

        # ---------------------------------------- Value Local Buffer (sliding window)
        if self.val_local_len < R:
            # Local buffer not yet full — just insert
            self.val_local[layer_id, :, self.val_local_len, :] = value
        else:
            # Local buffer full — evict oldest token into Value Q-Buffer
            oldest = self.val_local[layer_id, :, 0, :].clone()  # (H, D)
            # Shift local buffer left by 1
            self.val_local[layer_id, :, :R - 1, :] = self.val_local[layer_id, :, 1:R, :]
            # Insert new value at end
            self.val_local[layer_id, :, R - 1, :] = value

            # Push evicted oldest token into Value Q-Buffer
            vqlen = self.val_qbuf_len
            self.val_qbuf[layer_id, :, vqlen, :] = oldest

        # Increment lens only on first layer to stay coherent
        if layer_id == 0:
            if self.val_local_len < R:
                self.val_local_len += 1
            else:
                self.val_qbuf_len += 1

            self.key_qbuf_len += 1

            # ---------------------------------------- Trigger quantisation
            if self.key_qbuf_len == G:
                self._trigger_quantization(layer_id, seq_id)

    def _trigger_quantization(self, triggering_layer_id: int, seq_id: int) -> None:
        """Quantise all layers' Q-Buffers and write pages to the layout.

        Called when the Key Q-Buffer fills to G tokens.  Both Key and Value
        Q-Buffers are quantised together across all layers.

        Parameters
        ----------
        triggering_layer_id:
            The layer that triggered this event (informational).
        seq_id:
            Sequence being processed.
        """
        cfg = self.config
        G = cfg.q_buffer_size
        H = cfg.num_kv_heads
        D = cfg.head_dim
        K = cfg.d_boost

        for l in range(cfg.num_layers):
            page_num = self._page_counter[(seq_id, l)]
            page = self.layout.allocate_page(seq_id=seq_id, layer_id=l, page_number=page_num)
            phys = page.physical_idx
            self._page_counter[(seq_id, l)] += 1

            # Get pool views for this page
            out_baseline = self.layout.key_baseline_page(phys)  # (H, D, G//4)
            out_boost    = self.layout.key_boost_page(phys)     # (H, K, G//4)
            out_meta     = self.layout.key_meta_page(phys)      # (H, 2, D)

            # Quantise each head's key block independently
            for h in range(H):
                b_idx_h = self.boost_idx_uint8[l, h]  # (D,) uint8
                sentinel = cfg.sentinel

                # key block: (G, D) → transpose to (D, G) for per-channel quant
                key_block = self.key_qbuf[l, h, :G, :].T.contiguous()  # (D, G)

                quantize_key_page(
                    key=key_block,
                    boost_idx=b_idx_h,
                    D_boost=K,
                    sentinel=sentinel,
                    out_baseline=out_baseline[h],   # (D, G//4)
                    out_boost=out_boost[h],          # (K, G//4)
                    out_scale=out_meta[h, 0],        # (D,) scale row
                    out_zero=out_meta[h, 1],         # (D,) zero-point row
                )

            # Quantise value Q-Buffer for this layer
            out_vpacked = self.layout.value_page(phys)   # (H, G, D//4)
            out_vmeta   = self.layout.value_meta_page(phys)  # (H, G, 2)

            vqlen = self.val_qbuf_len
            for h in range(H):
                val_block = self.val_qbuf[l, h, :vqlen, :]  # (vqlen, D)
                # Pad if vqlen < G (shouldn't happen when G_key == G_val, but be safe)
                if vqlen < G:
                    pad = torch.zeros(G - vqlen, cfg.head_dim, dtype=self._dtype,
                                      device=val_block.device)
                    val_block = torch.cat([val_block, pad], dim=0)  # (G, D)

                quantize_value_page(
                    value=val_block.contiguous(),
                    out_packed=out_vpacked[h],   # (G, D//4)
                    out_scale=out_vmeta[h, :, 0],  # (G,)
                    out_zero=out_vmeta[h, :, 1],   # (G,)
                )

        # Reset Q-Buffers
        self.key_qbuf.zero_()
        self.key_qbuf_len = 0
        self.val_qbuf.zero_()
        self.val_qbuf_len = 0

    # ---------------------------------------------------------------- query

    def get_attention_context(self, layer_id: int, seq_id: int = 0) -> dict:
        """Return all KV data needed for the attention computation at decode step.

        Returns
        -------
        dict with keys:
          - ``sink_k``:    (H, sink_fill, D) float16 — attention sinks
          - ``sink_v``:    (H, sink_fill, D) float16
          - ``key_qbuf``:  (H, qbuf_len, D) float16 — unquantised keys
          - ``val_local``: (H, local_len, D) float16 — FP16 value local window
          - ``page_list``: List[int] — physical page indices, ordered
        """
        s = self.sink_fill
        q = self.key_qbuf_len
        r = self.val_local_len

        return {
            "sink_k":    self.sink_k[layer_id, :, :s, :],
            "sink_v":    self.sink_v[layer_id, :, :s, :],
            "key_qbuf":  self.key_qbuf[layer_id, :, :q, :],
            "val_local": self.val_local[layer_id, :, :r, :],
            "page_list": self.layout.get_page_list(seq_id=seq_id, layer_id=layer_id),
        }

    # ---------------------------------------------------------------- reset

    def reset_sequence(self, seq_id: int = 0) -> None:
        """Clear all buffer state and free all pages for a finished sequence.

        Parameters
        ----------
        seq_id:
            The sequence to clear.
        """
        # Free all pages for this sequence
        self.layout.free_sequence(seq_id)

        # Clear all per-sequence page counters
        keys_to_remove = [k for k in self._page_counter if k[0] == seq_id]
        for k in keys_to_remove:
            del self._page_counter[k]

        # Reset shared buffer counters
        self.sink_fill = 0
        self.key_qbuf_len = 0
        self.val_local_len = 0
        self.val_qbuf_len = 0

        # Zero out buffers (optional but clean for the next sequence)
        self.sink_k.zero_()
        self.sink_v.zero_()
        self.key_qbuf.zero_()
        self.val_local.zero_()
        self.val_qbuf.zero_()

    def stats(self) -> dict:
        """Return a human-readable summary of the current pipeline state."""
        return {
            "sink_fill": self.sink_fill,
            "sink_capacity": self.config.sink_size,
            "key_qbuf_len": self.key_qbuf_len,
            "key_qbuf_capacity": self.config.q_buffer_size,
            "val_local_len": self.val_local_len,
            "val_local_capacity": self.config.local_value_window,
            "val_qbuf_len": self.val_qbuf_len,
            **self.layout.memory_stats(),
        }
