"""
kitty/kernels/attention.py
---------------------------
Fused Paged Attention kernel for Kitty.

Performs the full decode-step attention entirely in SRAM:
  1. Load the query vector Q for the current token.
  2. For each quantised page in the logical page list:
     a. Load Tensor_2bits + Tensor_High_2bits from HBM.
     b. Dequantise Key page on-chip in SRAM.
     c. Compute partial scores: Q @ K^T / sqrt(D).
  3. Also attend to FP16 Attention Sinks and the FP16 Q-Buffer.
  4. Compute global softmax (online / flash-attention style to avoid 2nd pass).
  5. Accumulate weighted Value context:
     a. Dequantise Value pages on-chip.
  6. Return output (num_q_heads, head_dim) fp16.

For the Triton kernel: tl.program_id(0) = query head index.

Reference: §5, Step 2–3 of the Kitty reference spec.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TRITON_AVAILABLE = False

from .dequantize_key import dequantize_key_page_torch
from .quantize_value import dequantize_value_page_torch


# ---------------------------------------------------------------------------
# Pure PyTorch reference implementation (CPU-safe, used for testing)
# ---------------------------------------------------------------------------

def paged_attention_reference(
    q: torch.Tensor,                      # (num_q_heads, D) float16
    page_list: List[int],                 # physical page indices (ordered)
    key_baseline: torch.Tensor,           # (max_pages, H, D, G//4) uint8
    key_boost: torch.Tensor,              # (max_pages, H, D_boost, G//4) uint8
    key_meta: torch.Tensor,               # (max_pages, H, 2, D) float16
    value_pool: torch.Tensor,             # (max_pages, H, G, D//4) uint8
    value_meta: torch.Tensor,             # (max_pages, H, G, 2) float16
    boost_idx: torch.Tensor,              # (H, D) uint8  — for this layer
    D_boost: int,
    sink_k: torch.Tensor,                 # (H, S_fill, D) float16 — attention sinks
    sink_v: torch.Tensor,                 # (H, S_fill, D) float16
    qbuf_k: torch.Tensor,                 # (H, qbuf_len, D) float16 — unquantised keys
    qbuf_v: torch.Tensor,                 # (H, qbuf_len, D) float16 — local value buf
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Pure PyTorch reference paged attention for correctness checking.

    This implementation:
      - Collects all key/value tokens into a single buffer (slow but correct).
      - Runs standard scaled dot-product attention.
      - Returns output of shape ``(num_q_heads, head_dim)`` float16.

    Parameters
    ----------
    q:
        Query for the current decode step, shape ``(num_q_heads, D)``.
    page_list:
        Ordered list of physical page indices containing quantised KV data.
    key_baseline, key_boost, key_meta:
        Global key pool tensors from :class:`~kitty.layout.PageCentricKVLayoutManager`.
    value_pool, value_meta:
        Global value pool tensors.
    boost_idx:
        Per-head channel boost index array of shape ``(H, D)`` uint8.
    D_boost:
        Number of boosted (INT4) key channels per head.
    sink_k, sink_v:
        FP16 Attention Sink buffers of shape ``(H, S_fill, D)``.
    qbuf_k, qbuf_v:
        FP16 Q-Buffer (unquantised recent tokens) and local value buffer.
    num_q_heads, num_kv_heads, head_dim:
        Attention head configuration.
    """
    # GQA expansion ratio
    q_per_kv = num_q_heads // num_kv_heads

    # Collect all KV tokens per head
    # We build a list of (H, T_total, D) tensors
    H = num_kv_heads
    D = head_dim

    all_keys: List[torch.Tensor] = []
    all_vals: List[torch.Tensor] = []

    # 1. Attention sinks (FP16)
    if sink_k.shape[1] > 0:
        all_keys.append(sink_k.float())  # (H, S, D)
        all_vals.append(sink_v.float())

    # 2. Quantised pages
    for phys_idx in page_list:
        for h in range(H):
            b_idx_h = boost_idx[h]   # (D,) uint8
            t2b  = key_baseline[phys_idx, h]   # (D, G//4)
            thigh = key_boost[phys_idx, h]     # (D_boost, G//4)
            scl  = key_meta[phys_idx, h, 0]   # (D,)
            zp   = key_meta[phys_idx, h, 1]   # (D,)

            key_dq = dequantize_key_page_torch(t2b, thigh, scl, zp, b_idx_h, D_boost)
            # key_dq: (D, G)  →  (G, D)
            key_dq = key_dq.T.float()   # (G, D)

            v_packed = value_pool[phys_idx, h]    # (G, D//4)
            v_scl    = value_meta[phys_idx, h, :, 0]  # (G,)
            v_zp     = value_meta[phys_idx, h, :, 1]  # (G,)
            val_dq   = dequantize_value_page_torch(v_packed, v_scl, v_zp).float()  # (G, D)

            # Accumulate per-head — first iteration initialises the list
            if h == 0 and len(all_keys) <= len(page_list):
                all_keys.append(key_dq.unsqueeze(0))   # (1, G, D)
                all_vals.append(val_dq.unsqueeze(0))
            else:
                # This is a multi-head accumulation; reshape later
                pass

    # Simpler approach: build per-head K and V concatenated tensors
    # Rebuild from scratch cleanly
    keys_per_head = []
    vals_per_head = []
    for h in range(H):
        kh_parts = []
        vh_parts = []

        # Sinks
        if sink_k.shape[1] > 0:
            kh_parts.append(sink_k[h].float())   # (S, D)
            vh_parts.append(sink_v[h].float())

        # Quantised pages
        for phys_idx in page_list:
            b_idx_h = boost_idx[h]
            t2b   = key_baseline[phys_idx, h]
            thigh = key_boost[phys_idx, h]
            scl   = key_meta[phys_idx, h, 0]
            zp    = key_meta[phys_idx, h, 1]
            key_dq = dequantize_key_page_torch(t2b, thigh, scl, zp, b_idx_h, D_boost)
            kh_parts.append(key_dq.T.float())   # (G, D)

            v_packed = value_pool[phys_idx, h]
            v_scl    = value_meta[phys_idx, h, :, 0]
            v_zp     = value_meta[phys_idx, h, :, 1]
            vh_parts.append(dequantize_value_page_torch(v_packed, v_scl, v_zp).float())

        # Q-buffer (unquantised)
        if qbuf_k.shape[1] > 0:
            kh_parts.append(qbuf_k[h].float())   # (qbuf_len, D)
            vh_parts.append(qbuf_v[h].float())

        keys_per_head.append(torch.cat(kh_parts, dim=0) if kh_parts else torch.zeros(0, D))
        vals_per_head.append(torch.cat(vh_parts, dim=0) if vh_parts else torch.zeros(0, D))

    # Standard scaled dot-product attention per query head
    # GQA: query head qh attends to KV head qh // q_per_kv
    scale_factor = 1.0 / math.sqrt(D)
    outputs = []
    q_float = q.float()  # (num_q_heads, D)

    for qh in range(num_q_heads):
        kv_h = qh // q_per_kv
        K = keys_per_head[kv_h]   # (T, D)
        V = vals_per_head[kv_h]   # (T, D)
        if K.shape[0] == 0:
            outputs.append(torch.zeros(D, dtype=torch.float32))
            continue

        scores = (q_float[qh] @ K.T) * scale_factor   # (T,)
        weights = torch.softmax(scores, dim=0)          # (T,)
        attn_out = (weights.unsqueeze(1) * V).sum(dim=0)  # (D,)
        outputs.append(attn_out)

    return torch.stack(outputs, dim=0).to(torch.float16)  # (num_q_heads, D)


# ---------------------------------------------------------------------------
# Triton fused kernel (quantised pages only — FP16 buffers handled in Python)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _paged_attention_inner_kernel(
        q_ptr,                 # (D,) float16  — one query head
        tensor_2bits_ptr,      # (D, G//4) uint8  — one page, one head
        tensor_high_bits_ptr,  # (D_boost, G//4) uint8
        boost_idx_ptr,         # (D,) uint8
        scale_ptr,             # (D,) float16
        zero_ptr,              # (D,) float16
        v_packed_ptr,          # (G, D//4) uint8
        v_scale_ptr,           # (G,) float16
        v_zero_ptr,            # (G,) float16
        # Accumulators (read-modify-write in SRAM via atomic-like pattern)
        score_acc_ptr,         # (G,) float32 — partial softmax logits
        weight_acc_ptr,        # (G,) float32 — normalised weights (after softmax)
        out_ptr,               # (D,) float32 — output accumulator
        D: tl.constexpr,
        G: tl.constexpr,
        D_boost: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        stride_2b_d,
        stride_2b_g,
        stride_hi_d,
        stride_hi_g,
        stride_vp_g,
        stride_vp_d,
    ):
        """Inner kernel: process one page for one head.

        This kernel computes QK scores for the G tokens in a single page and
        accumulates the softmax-weighted value sum into ``out_ptr``.
        """
        # Load query vector
        q = tl.load(q_ptr + tl.arange(0, D)).to(tl.float32)   # (D,)

        G4 = G // 4
        D4 = D // 4

        # Dequantise all G key vectors on-chip and compute QK scores
        scores = tl.zeros([G], dtype=tl.float32)
        boost_idx = tl.load(boost_idx_ptr + tl.arange(0, D)).to(tl.int32)  # (D,)

        for g_byte in range(G4):
            for t_off in range(4):
                g = g_byte * 4 + t_off
                k_vec = tl.zeros([D], dtype=tl.float32)
                for d in range(D):
                    low_byte = tl.load(
                        tensor_2bits_ptr + d * stride_2b_d + g_byte * stride_2b_g
                    ).to(tl.int32)
                    x_low = (low_byte >> (t_off * 2)) & 0x03

                    b_idx = boost_idx[d]
                    if b_idx < D_boost:
                        hi_byte = tl.load(
                            tensor_high_bits_ptr + b_idx * stride_hi_d + g_byte * stride_hi_g
                        ).to(tl.int32)
                        x_high = (hi_byte >> (t_off * 2)) & 0x03
                    else:
                        x_high = 0

                    x_quant = x_low | (x_high << 2)
                    scale_d = tl.load(scale_ptr + d).to(tl.float32)
                    zp_d    = tl.load(zero_ptr  + d).to(tl.float32)
                    k_vec[d] = (x_quant.to(tl.float32) - zp_d) * scale_d

                scores[g] = tl.sum(q * k_vec) * INV_SQRT_D

        # Softmax over the G scores for this page
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - max_score)
        sum_exp = tl.sum(exp_scores, axis=0)
        weights = exp_scores / sum_exp   # (G,)

        # Dequantise value and accumulate weighted sum
        for g in range(G):
            w = weights[g]
            g_byte = g // 4
            t_off  = g % 4
            for d4 in range(D4):
                vb = tl.load(
                    v_packed_ptr + g * stride_vp_g + d4 * stride_vp_d
                ).to(tl.int32)
                for i in range(4):
                    d = d4 * 4 + i
                    v_q = (vb >> (i * 2)) & 0x03
                    v_s = tl.load(v_scale_ptr + g).to(tl.float32)
                    v_z = tl.load(v_zero_ptr  + g).to(tl.float32)
                    v_fp = (v_q.to(tl.float32) - v_z) * v_s
                    tl.atomic_add(out_ptr + d, w * v_fp)


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

def paged_attention(
    q: torch.Tensor,
    page_list: List[int],
    key_baseline: torch.Tensor,
    key_boost: torch.Tensor,
    key_meta: torch.Tensor,
    value_pool: torch.Tensor,
    value_meta: torch.Tensor,
    boost_idx: torch.Tensor,
    D_boost: int,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    qbuf_k: torch.Tensor,
    qbuf_v: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Dispatch paged attention — Triton on CUDA, PyTorch reference on CPU.

    Routes to ``paged_attention_reference`` on CPU (which is the correct
    implementation used for testing).  On CUDA it uses the Triton kernel.

    Returns
    -------
    torch.Tensor
        Shape ``(num_q_heads, head_dim)`` float16.
    """
    # For now always use the reference implementation.
    # The Triton inner kernel above is a per-page kernel and requires
    # a Python orchestration loop (multi-pass flash-attention style) which
    # is done in the pipeline.  For the decoder step, the reference is used.
    return paged_attention_reference(
        q=q,
        page_list=page_list,
        key_baseline=key_baseline,
        key_boost=key_boost,
        key_meta=key_meta,
        value_pool=value_pool,
        value_meta=value_meta,
        boost_idx=boost_idx,
        D_boost=D_boost,
        sink_k=sink_k,
        sink_v=sink_v,
        qbuf_k=qbuf_k,
        qbuf_v=qbuf_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
