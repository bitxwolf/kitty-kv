"""
kitty/kernels/quantize_key.py
------------------------------
Key Cache quantisation kernel.

Implements per-channel grouped quantisation with the Dense-Sparse Decomposition
layout:

  - All channels (boosted and non-boosted) have their LOW 2 bits written to
    ``Tensor_2bits`` of shape ``(D, G//4)`` packed uint8.
  - Boosted channels additionally have their HIGH 2 bits written to
    ``Tensor_High_2bits`` of shape ``(D_boost, G//4)`` packed uint8.

Per reference spec §2.2 (Key Cache Quantization):
  S_c = (max(X) - min(X)) / (2^b - 1)        b=2 (INT2) or b=4 (INT4)
  Z_c = round(-min(X) / S_c)
  X̂_c = clip(round(X / S_c) + Z_c, 0, 2^b - 1)
"""

from __future__ import annotations

import torch
from typing import Tuple

# Try to import Triton; fall back gracefully on CPU-only systems
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure PyTorch fallback (CPU + GPU, used for testing and Triton fallback)
# ---------------------------------------------------------------------------

def quantize_key_page_torch(
    key: torch.Tensor,        # (D, G) float16
    boost_idx: torch.Tensor,  # (D,)   uint8  — sentinel for non-boosted
    D_boost: int,
    sentinel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch key page quantisation.

    Parameters
    ----------
    key:
        FP16 key tensor of shape ``(D, G)`` where D = head_dim, G = page size.
    boost_idx:
        Per-channel boost index array of shape ``(D,)``, dtype uint8.
        Non-boosted channels carry ``sentinel`` (e.g. D_boost + 1 or 255).
    D_boost:
        Number of boosted (INT4) channels.
    sentinel:
        Sentinel value for non-boosted channels in ``boost_idx``.

    Returns
    -------
    tensor_2bits : (D, G//4) uint8
        Packed low 2-bit values for all channels.
    tensor_high_2bits : (D_boost, G//4) uint8
        Packed high 2-bit values for boosted channels.
    scale : (D,) float16
        Per-channel quantisation scale.
    zero_point : (D,) float16
        Per-channel quantisation zero-point (as float16, applied as integer).
    """
    key = key.float()  # work in float32 for precision
    D, G = key.shape
    assert G % 4 == 0, "G must be divisible by 4 for uint8 packing"

    is_boosted = boost_idx < sentinel  # (D,) bool

    # ---------------------------------------------------------------- Per-channel scale/ZP
    x_min = key.min(dim=1).values   # (D,)
    x_max = key.max(dim=1).values   # (D,)
    x_range = (x_max - x_min).clamp(min=1e-8)

    # Bit depth per channel: 4 for boosted, 2 for non-boosted  (float for vectorised ops)
    b = torch.where(is_boosted, torch.tensor(4.0), torch.tensor(2.0))
    max_quant = (2.0 ** b - 1.0)   # (D,) float32: 15.0 for INT4, 3.0 for INT2

    scale = x_range / max_quant    # (D,)
    # Element-wise clamp — avoids type mismatch between int scalar and tensor
    zp_raw = (-x_min / scale).round()
    zero_point = torch.minimum(torch.maximum(zp_raw, torch.zeros_like(zp_raw)), max_quant)

    # ----------------------------------------------------------- Quantise all channels
    # X_hat = clip(round(X / S) + Z, 0, 2^b-1)
    scale_bc = scale.unsqueeze(1)      # (D, 1)
    zp_bc    = zero_point.unsqueeze(1) # (D, 1)
    mq_bc    = max_quant.unsqueeze(1)  # (D, 1)

    x_quant_f = (key / scale_bc).round() + zp_bc
    # Element-wise clamp using tensors (avoids scalar/tensor mismatch in PyTorch 2.1+)
    x_quant = torch.minimum(
        torch.maximum(x_quant_f, torch.zeros_like(x_quant_f)), mq_bc
    ).to(torch.uint8)  # (D, G)

    # ---------------------------------------- Split into low-2-bit and high-2-bit
    x_low  = x_quant & 0x03          # (D, G)  — always stored in Tensor_2bits
    x_high = (x_quant >> 2) & 0x03   # (D, G)  — only meaningful for boosted

    # --------------------------------------------------- Pack 4 values per byte
    G4 = G // 4
    # Tensor_2bits: (D, G//4) uint8
    tensor_2bits = torch.zeros(D, G4, dtype=torch.uint8)
    for i in range(4):
        tensor_2bits |= (x_low[:, i::4] << (i * 2)).to(torch.uint8)

    # Tensor_High_2bits: (D_boost, G//4) uint8
    tensor_high_2bits = torch.zeros(D_boost, G4, dtype=torch.uint8)
    boosted_channels = torch.where(is_boosted)[0]  # indices of boosted channels
    for ch_d in boosted_channels:
        b_idx = int(boost_idx[ch_d].item())
        if b_idx < D_boost:
            for i in range(4):
                tensor_high_2bits[b_idx] |= (x_high[ch_d, i::4] << (i * 2)).to(torch.uint8)

    scale_fp16 = scale.to(torch.float16)
    zp_fp16 = zero_point.to(torch.float16)

    return tensor_2bits, tensor_high_2bits, scale_fp16, zp_fp16


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _quantize_key_page_kernel(
        key_ptr,                  # (D, G) float16
        tensor_2bits_ptr,         # (D, G//4) uint8 output
        tensor_high_2bits_ptr,    # (D_boost, G//4) uint8 output
        scale_ptr,                # (D,) float16 output
        zero_ptr,                 # (D,) float16 output
        boost_idx_ptr,            # (D,) uint8
        D: tl.constexpr,
        G: tl.constexpr,
        D_boost: tl.constexpr,
        sentinel: tl.constexpr,
        stride_key_d,
        stride_key_g,
        stride_2b_d,
        stride_2b_g,
        stride_hi_d,
        stride_hi_g,
    ):
        """One Triton program handles one channel (pid_d)."""
        pid_d = tl.program_id(0)
        if pid_d >= D:
            return

        G4 = G // 4

        # Load boost index for this channel
        b_idx = tl.load(boost_idx_ptr + pid_d).to(tl.int32)
        is_boosted = b_idx < D_boost

        # Bit depth
        b_bits: tl.constexpr = 4 if is_boosted else 2
        max_q = (1 << b_bits) - 1  # 15 or 3

        # Load all G values for this channel and compute min/max
        x_vals = tl.load(
            key_ptr + pid_d * stride_key_d + tl.arange(0, G) * stride_key_g
        ).to(tl.float32)

        x_min = tl.min(x_vals, axis=0)
        x_max = tl.max(x_vals, axis=0)
        x_range = tl.maximum(x_max - x_min, 1e-8)

        scale = x_range / max_q
        zero_pt = tl.math.round(-x_min / scale)
        zero_pt = tl.minimum(tl.maximum(zero_pt, 0.0), float(max_q))

        # Store scale and zero-point
        tl.store(scale_ptr + pid_d, scale.to(tl.float16))
        tl.store(zero_ptr + pid_d, zero_pt.to(tl.float16))

        # Quantise all G values
        x_quant = tl.math.round(x_vals / scale) + zero_pt
        x_quant = tl.minimum(tl.maximum(x_quant, 0.0), float(max_q)).to(tl.int32)

        x_low  = x_quant & 0x03
        x_high = (x_quant >> 2) & 0x03

        # Pack 4 tokens per byte and store to Tensor_2bits
        for byte_i in range(0, G4):
            packed_low = (
                (x_low[byte_i * 4 + 0] << 0) |
                (x_low[byte_i * 4 + 1] << 2) |
                (x_low[byte_i * 4 + 2] << 4) |
                (x_low[byte_i * 4 + 3] << 6)
            )
            out_addr = tensor_2bits_ptr + pid_d * stride_2b_d + byte_i * stride_2b_g
            tl.store(out_addr, packed_low.to(tl.uint8))

            # Boosted: also pack high bits into Tensor_High_2bits
            if is_boosted:
                packed_high = (
                    (x_high[byte_i * 4 + 0] << 0) |
                    (x_high[byte_i * 4 + 1] << 2) |
                    (x_high[byte_i * 4 + 2] << 4) |
                    (x_high[byte_i * 4 + 3] << 6)
                )
                hi_addr = tensor_high_2bits_ptr + b_idx * stride_hi_d + byte_i * stride_hi_g
                tl.store(hi_addr, packed_high.to(tl.uint8))


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

def quantize_key_page(
    key: torch.Tensor,          # (D, G) float16, CUDA
    boost_idx: torch.Tensor,    # (D,) uint8, CUDA
    D_boost: int,
    sentinel: int,
    out_baseline: torch.Tensor,  # (D, G//4) uint8, CUDA — pre-allocated
    out_boost: torch.Tensor,     # (D_boost, G//4) uint8, CUDA — pre-allocated
    out_scale: torch.Tensor,     # (D,) float16, CUDA — pre-allocated
    out_zero: torch.Tensor,      # (D,) float16, CUDA — pre-allocated
) -> None:
    """Quantise one key cache page (G tokens) in-place into pre-allocated outputs.

    Routes to the Triton kernel when CUDA is available; otherwise falls back to
    the pure PyTorch implementation and copies results into the output tensors.

    Parameters
    ----------
    key:
        FP16 key block of shape ``(D, G)``.
    boost_idx:
        Per-channel boost index array of shape ``(D,)``.
    D_boost:
        Number of boosted channels.
    sentinel:
        Sentinel value for non-boosted channels.
    out_baseline, out_boost, out_scale, out_zero:
        Pre-allocated output tensors (views into the global pool).
    """
    if _TRITON_AVAILABLE and key.is_cuda:
        D, G = key.shape
        G4 = G // 4
        grid = (D,)
        _quantize_key_page_kernel[grid](
            key,
            out_baseline,
            out_boost,
            out_scale,
            out_zero,
            boost_idx,
            D=D, G=G, D_boost=D_boost, sentinel=sentinel,
            stride_key_d=key.stride(0), stride_key_g=key.stride(1),
            stride_2b_d=out_baseline.stride(0), stride_2b_g=out_baseline.stride(1),
            stride_hi_d=out_boost.stride(0), stride_hi_g=out_boost.stride(1),
        )
    else:
        # PyTorch fallback — compute and copy into pre-allocated outputs
        t2b, thigh, scale, zero = quantize_key_page_torch(
            key.cpu(), boost_idx.cpu(), D_boost, sentinel
        )
        out_baseline.copy_(t2b.to(out_baseline.device))
        out_boost.copy_(thigh.to(out_boost.device))
        out_scale.copy_(scale.to(out_scale.device))
        out_zero.copy_(zero.to(out_zero.device))
