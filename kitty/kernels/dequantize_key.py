"""
kitty/kernels/dequantize_key.py
--------------------------------
Key Cache dequantisation kernel.

Reverses the Dense-Sparse decomposition packing:

  1. Load the packed low-2-bit byte from ``Tensor_2bits``.
  2. Extract 4 values using bitwise shifts:  val = (byte >> shift) & 0x03
  3. Look up ``Boost_IDX_uint8[d]``; if < D_boost, load corresponding high-2-bit
     byte from ``Tensor_High_2bits``.
  4. Reconstitute INT4 value:  x_quant = x_low | (x_high << 2)
  5. Dequantise to FP16:       x_fp16  = (x_quant - ZP) * Scale

Reference: §4.1 & §4.2 of the Kitty reference spec.
"""

from __future__ import annotations

import torch
from typing import Tuple

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure PyTorch fallback
# ---------------------------------------------------------------------------

def dequantize_key_page_torch(
    tensor_2bits: torch.Tensor,        # (D, G//4) uint8
    tensor_high_2bits: torch.Tensor,   # (D_boost, G//4) uint8
    scale: torch.Tensor,               # (D,) float16
    zero_point: torch.Tensor,          # (D,) float16
    boost_idx: torch.Tensor,           # (D,) uint8
    D_boost: int,
) -> torch.Tensor:
    """Pure PyTorch key page dequantisation.

    Parameters
    ----------
    tensor_2bits:
        Packed low-2-bit values for all D channels, shape ``(D, G//4)`` uint8.
    tensor_high_2bits:
        Packed high-2-bit values for boosted channels, shape ``(D_boost, G//4)`` uint8.
    scale:
        Per-channel scale, shape ``(D,)`` float16.
    zero_point:
        Per-channel zero-point, shape ``(D,)`` float16.
    boost_idx:
        Per-channel offset into ``tensor_high_2bits``. Non-boosted channels
        carry ``sentinel`` (>= D_boost).
    D_boost:
        Number of boosted channels.

    Returns
    -------
    torch.Tensor
        Dequantised key block of shape ``(D, G)`` float32.
    """
    D, G4 = tensor_2bits.shape
    G = G4 * 4

    # Unpack all low-2-bit values  →  (D, G)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)   # 4 shifts per byte
    # tensor_2bits: (D, G4)  →  broadcast over shifts →  (D, G4, 4)
    t2b_expanded = tensor_2bits.unsqueeze(-1)  # (D, G4, 1)
    x_low = ((t2b_expanded >> shifts) & 0x03).to(torch.int32)  # (D, G4, 4)
    x_low = x_low.reshape(D, G)                                # (D, G)

    # Unpack high-2-bit values for ALL channels (sentinel channels will be 0)
    x_high = torch.zeros(D, G, dtype=torch.int32)
    boost_idx_int = boost_idx.to(torch.int32)
    is_boosted = boost_idx_int < D_boost  # (D,) bool

    for d_idx in range(D):
        if is_boosted[d_idx]:
            b_idx = int(boost_idx_int[d_idx].item())
            hi_bytes = tensor_high_2bits[b_idx]          # (G4,) uint8
            hi_expanded = hi_bytes.unsqueeze(-1)          # (G4, 1)
            hi_vals = ((hi_expanded >> shifts) & 0x03).to(torch.int32)  # (G4, 4)
            x_high[d_idx] = hi_vals.reshape(G)

    # Reconstitute: x_quant = x_low | (x_high << 2)
    x_quant = (x_low | (x_high << 2)).float()  # (D, G)

    # Dequantise: x_fp16 = (x_quant - ZP) * Scale
    scale_f = scale.float().unsqueeze(1)     # (D, 1)
    zp_f    = zero_point.float().unsqueeze(1) # (D, 1)
    x_fp32  = (x_quant - zp_f) * scale_f    # (D, G)

    return x_fp32


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _dequantize_key_page_kernel(
        tensor_2bits_ptr,          # (D, G//4) uint8
        tensor_high_2bits_ptr,     # (D_boost, G//4) uint8
        boost_idx_ptr,             # (D,) uint8
        scale_ptr,                 # (D,) float16
        zero_ptr,                  # (D,) float16
        out_ptr,                   # (D, G) float16 output
        D: tl.constexpr,
        G: tl.constexpr,
        D_boost: tl.constexpr,
        stride_2b_d,
        stride_2b_g,
        stride_hi_d,
        stride_hi_g,
        stride_out_d,
        stride_out_g,
    ):
        """One Triton program handles one channel (pid_d)."""
        pid_d = tl.program_id(0)
        if pid_d >= D:
            return

        G4 = G // 4

        # Load per-channel metadata
        scale = tl.load(scale_ptr + pid_d).to(tl.float32)
        zero_pt = tl.load(zero_ptr + pid_d).to(tl.float32)
        b_idx = tl.load(boost_idx_ptr + pid_d).to(tl.int32)
        is_boosted = b_idx < D_boost

        # Process G//4 packed bytes, each yielding 4 values
        for byte_i in range(0, G4):
            # Load the packed low-2-bit byte
            low_byte = tl.load(
                tensor_2bits_ptr + pid_d * stride_2b_d + byte_i * stride_2b_g
            ).to(tl.int32)

            # Optionally load high-2-bit byte
            if is_boosted:
                high_byte = tl.load(
                    tensor_high_2bits_ptr + b_idx * stride_hi_d + byte_i * stride_hi_g
                ).to(tl.int32)
            else:
                high_byte = 0

            # Extract and dequantise 4 tokens
            for t in range(4):
                shift = t * 2
                x_low  = (low_byte  >> shift) & 0x03
                x_high = (high_byte >> shift) & 0x03

                x_quant = x_low | (x_high << 2)
                x_fp32  = (x_quant.to(tl.float32) - zero_pt) * scale

                token_idx = byte_i * 4 + t
                out_addr  = out_ptr + pid_d * stride_out_d + token_idx * stride_out_g
                tl.store(out_addr, x_fp32.to(tl.float16))


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------

def dequantize_key_page(
    tensor_2bits: torch.Tensor,        # (D, G//4) uint8
    tensor_high_2bits: torch.Tensor,   # (D_boost, G//4) uint8
    scale: torch.Tensor,               # (D,) float16
    zero_point: torch.Tensor,          # (D,) float16
    boost_idx: torch.Tensor,           # (D,) uint8
    D_boost: int,
    out: torch.Tensor,                 # (D, G) float16 — pre-allocated output
) -> None:
    """Dequantise one key cache page into a pre-allocated FP16 output tensor.

    Routes to the Triton kernel when CUDA is available, else uses the PyTorch
    fallback and copies results into ``out``.

    Parameters
    ----------
    tensor_2bits:
        Packed low-2-bit tensor, shape ``(D, G//4)`` uint8.
    tensor_high_2bits:
        Packed high-2-bit tensor for boosted channels, shape ``(D_boost, G//4)`` uint8.
    scale:
        Per-channel quantisation scale, shape ``(D,)`` float16.
    zero_point:
        Per-channel zero-point, shape ``(D,)`` float16.
    boost_idx:
        Channel-to-boosted-offset mapping, shape ``(D,)`` uint8.
    D_boost:
        Number of boosted channels.
    out:
        Pre-allocated output tensor of shape ``(D, G)`` float16.
    """
    if _TRITON_AVAILABLE and tensor_2bits.is_cuda:
        D, G4 = tensor_2bits.shape
        G = G4 * 4
        grid = (D,)
        _dequantize_key_page_kernel[grid](
            tensor_2bits,
            tensor_high_2bits,
            boost_idx,
            scale,
            zero_point,
            out,
            D=D, G=G, D_boost=D_boost,
            stride_2b_d=tensor_2bits.stride(0),
            stride_2b_g=tensor_2bits.stride(1),
            stride_hi_d=tensor_high_2bits.stride(0),
            stride_hi_g=tensor_high_2bits.stride(1),
            stride_out_d=out.stride(0),
            stride_out_g=out.stride(1),
        )
    else:
        result = dequantize_key_page_torch(
            tensor_2bits.cpu(),
            tensor_high_2bits.cpu(),
            scale.cpu(),
            zero_point.cpu(),
            boost_idx.cpu(),
            D_boost,
        )
        out.copy_(result.to(out.dtype).to(out.device))
