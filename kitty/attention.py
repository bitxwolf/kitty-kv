"""
kitty/attention.py
------------------
KittyPagedAttention — drop-in nn.Module replacement for standard transformer
attention, wired into the Kitty runtime pipeline.

Also provides KittyAttentionFactory for convenient creation of all per-layer
attention modules from a single KittyConfig.

Supports Grouped Query Attention (GQA) where num_q_heads >= num_kv_heads.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KittyConfig
from .layout import PageCentricKVLayoutManager
from .pipeline import KittyInferencePipeline
from .kernels.attention import paged_attention


class KittyPagedAttention(nn.Module):
    """Drop-in replacement for standard multi-head (or GQA) attention.

    During **decode** (seq_len == 1):
      1. Pushes the new K/V token into the pipeline for this layer.
      2. Retrieves the full attention context (sinks + quantised pages + qbuf).
      3. Calls the paged attention kernel to produce the output.

    During **prefill** (seq_len > 1):
      1. Pushes each token sequentially into the pipeline.
      2. Runs standard scaled dot-product attention (torch.nn.functional) over
         all accumulated tokens for speed — quantisation happens in background.

    Parameters
    ----------
    config:
        Global KittyConfig.
    num_q_heads:
        Number of query attention heads (may be > num_kv_heads for GQA).
    layout:
        Shared PageCentricKVLayoutManager.
    pipeline:
        Shared KittyInferencePipeline.
    layer_id:
        Zero-indexed transformer layer this module handles.
    """

    def __init__(
        self,
        config: KittyConfig,
        num_q_heads: int,
        layout: PageCentricKVLayoutManager,
        pipeline: KittyInferencePipeline,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_q_heads = num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_per_kv = num_q_heads // config.num_kv_heads
        self.layer_id = layer_id
        self.layout = layout
        self.pipeline = pipeline
        self.scale = 1.0 / math.sqrt(config.head_dim)

    def forward(
        self,
        q: torch.Tensor,                       # (B, num_q_heads, T, D)
        k: torch.Tensor,                       # (B, num_kv_heads, T, D)
        v: torch.Tensor,                       # (B, num_kv_heads, T, D)
        seq_id: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Kitty paged attention.

        Parameters
        ----------
        q, k, v:
            Query / Key / Value tensors with batch on dim 0.
        seq_id:
            Sequence slot identifier (for multi-batch support).
        attention_mask:
            Optional causal mask (ignored for decode — causality is inherent
            in the streaming pipeline).

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(B, num_q_heads, T, D)``.
        """
        B, nqh, T, D = q.shape
        assert B == 1, "KittyPagedAttention currently supports batch_size=1"
        assert nqh == self.num_q_heads
        assert D == self.head_dim

        if T == 1:
            return self._decode_step(q, k, v, seq_id)
        else:
            return self._prefill(q, k, v, seq_id)

    def _decode_step(
        self,
        q: torch.Tensor,   # (1, num_q_heads, 1, D)
        k: torch.Tensor,   # (1, num_kv_heads, 1, D)
        v: torch.Tensor,   # (1, num_kv_heads, 1, D)
        seq_id: int,
    ) -> torch.Tensor:
        """Single-token decode step."""
        # Push new K/V into the pipeline
        k_token = k[0, :, 0, :]   # (H, D)
        v_token = v[0, :, 0, :]   # (H, D)
        self.pipeline.push_token(
            layer_id=self.layer_id,
            key=k_token,
            value=v_token,
            seq_id=seq_id,
        )

        # Get full attention context
        ctx = self.pipeline.get_attention_context(self.layer_id, seq_id)

        # Dispatch paged attention kernel
        q_vec = q[0, :, 0, :]   # (num_q_heads, D)
        out = paged_attention(
            q=q_vec,
            page_list=ctx["page_list"],
            key_baseline=self.layout.key_baseline,
            key_boost=self.layout.key_boost,
            key_meta=self.layout.key_metadata,
            value_pool=self.layout.value_pool,
            value_meta=self.layout.value_metadata,
            boost_idx=self.pipeline.boost_idx_uint8[self.layer_id],  # (H, D)
            D_boost=self.config.d_boost,
            sink_k=ctx["sink_k"],
            sink_v=ctx["sink_v"],
            qbuf_k=ctx["key_qbuf"],
            qbuf_v=ctx["val_local"],
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        # out: (num_q_heads, D) → (1, num_q_heads, 1, D)
        return out.unsqueeze(0).unsqueeze(2)

    def _prefill(
        self,
        q: torch.Tensor,   # (1, num_q_heads, T, D)
        k: torch.Tensor,   # (1, num_kv_heads, T, D)
        v: torch.Tensor,   # (1, num_kv_heads, T, D)
        seq_id: int,
    ) -> torch.Tensor:
        """Prefill pass: process T tokens and push them all to the pipeline.

        Uses standard scaled dot-product attention over the entire sequence
        for the prefill output, while also streaming tokens into the pipeline
        (which may trigger page quantisation for long prefills).
        """
        T = q.shape[2]

        # Stream each token into the pipeline
        for t in range(T):
            k_t = k[0, :, t, :]   # (H, D)
            v_t = v[0, :, t, :]   # (H, D)
            self.pipeline.push_token(
                layer_id=self.layer_id,
                key=k_t,
                value=v_t,
                seq_id=seq_id,
            )

        # Standard causal SDPA for the prefill output
        # GQA: expand KV heads to match Q heads
        k_exp = k.repeat_interleave(self.q_per_kv, dim=1)  # (1, nqh, T, D)
        v_exp = v.repeat_interleave(self.q_per_kv, dim=1)

        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
        )  # (1, num_q_heads, T, D)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class KittyAttentionFactory:
    """Convenience factory that creates the full Kitty attention stack.

    Creates:
      - One :class:`~kitty.layout.PageCentricKVLayoutManager`
      - One :class:`~kitty.pipeline.KittyInferencePipeline`
      - ``num_layers`` :class:`KittyPagedAttention` modules

    Example
    -------
    >>> from kitty import KittyConfig
    >>> from kitty.sensitivity import KittySensitivityProfiler
    >>> profiler = KittySensitivityProfiler.load("kitty_profiles", config)
    >>> layout, pipeline, attn_layers = KittyAttentionFactory.from_config(
    ...     config=config,
    ...     boost_idx_uint8=profiler.boost_idx_uint8,
    ...     num_q_heads=32,
    ... )
    """

    @classmethod
    def from_config(
        cls,
        config: KittyConfig,
        boost_idx_uint8: torch.Tensor,  # (L, H, D) uint8
        num_q_heads: int,
    ) -> Tuple[PageCentricKVLayoutManager, KittyInferencePipeline, List[KittyPagedAttention]]:
        """Build the complete Kitty attention infrastructure.

        Parameters
        ----------
        config:
            KittyConfig instance.
        boost_idx_uint8:
            Boost index array from the sensitivity profiler.
        num_q_heads:
            Number of query heads (>= num_kv_heads for GQA).

        Returns
        -------
        layout:
            Shared layout manager with pre-allocated pools.
        pipeline:
            Shared runtime pipeline.
        attn_layers:
            List of ``num_layers`` KittyPagedAttention modules.
        """
        layout = PageCentricKVLayoutManager(config, boost_idx_uint8)
        pipeline = KittyInferencePipeline(config, layout, boost_idx_uint8)
        attn_layers = [
            KittyPagedAttention(
                config=config,
                num_q_heads=num_q_heads,
                layout=layout,
                pipeline=pipeline,
                layer_id=i,
            )
            for i in range(config.num_layers)
        ]
        return layout, pipeline, attn_layers
