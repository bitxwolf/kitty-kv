"""
kitty/kernels/quantize_value.py
--------------------------------
Value Cache quantisation and dequantisation kernels.

The Value Cache uses uniform **per-token** INT2 quantisation across the head
dimension D (unlike the Key Cache which is per-channel).

Per reference spec §2.2 (Value Cache Quantization):
  S_t = (max(V_t) - min(V_t)) / 3        (since 2^2 - 1 = 3)
  Z_t = round(-min(V_t) / S_t)
  V̂_t = clip(round(V_t / S_t) + Z_t, 0, 3)

Values are packed 4 per byte along the D dimension:
  packed_byte = (v0 & 0x3) | ((v1 & 0x3) << 2) | ((v2 & 0x3) << 4) | ((v3 & 0x3) << 6)

Input:  (G, D) float16 — G tokens, D head dim
Output: packed (G, D//4) uint8, scale (G,) float16, zero_point (G,) float16
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
# Pure PyTorch fallback — quantise
# ---------------------------------------------------------------------------

def quantize_value_page_torch(
    value: torch.Tensor,   # (G, D) float16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token INT2 quantisation of a value cache page.

    Parameters
    ----------
    value:
        FP16 value tensor of shape ``(G, D)`` where G = page size (tokens),
        D = head dimension.

    Returns
    -------
    packed : (G, D//4) uint8
        Packed INT2 values, 4 per byte along the D dimension.
    scale : (G,) float16
        Per-token quantisation scale.
    zero_point : (G,) float16
        Per-token quantisation zero-point.
    """
    value_f = value.float()  # (G, D)
    G, D = value_f.shape
    assert D % 4 == 0, "head_dim D must be divisible by 4 for packing"

    # Per-token min/max
    v_min = value_f.min(dim=1).values   # (G,)
    v_max = value_f.max(dim=1).values   # (G,)
    v_range = (v_max - v_min).clamp(min=1e-8)  # guard against zero-range tokens

    scale = v_range / 3.0               # (G,)  — 2^2 - 1 = 3
    # Element-wise clamp (avoids scalar/tensor type mismatch in newer PyTorch)
    zp_raw = (-v_min / scale).round()
    zero_point = torch.minimum(
        torch.maximum(zp_raw, torch.zeros_like(zp_raw)),
        torch.full_like(zp_raw, 3.0)
    )

    # Quantise: V_hat = clip(round(V / S) + Z, 0, 3)
    s_bc = scale.unsqueeze(1)      # (G, 1)
    z_bc = zero_point.unsqueeze(1) # (G, 1)
    v_quant_f = (value_f / s_bc).round() + z_bc
    v_quant = torch.minimum(
        torch.maximum(v_quant_f, torch.zeros_like(v_quant_f)),
        torch.full_like(v_quant_f, 3.0)
    ).to(torch.uint8)  # (G, D)

    # Pack 4 values per byte along D
    D4 = D // 4
    packed = torch.zeros(G, D4, dtype=torch.uint8)
    for i in range(4):
        packed |= (v_quant[:, i::4] << (i * 2)).to(torch.uint8)

    return packed, scale.to(torch.float16), zero_point.to(torch.float16)


# ---------------------------------------------------------------------------
# Pure PyTorch fallback — dequantise
# ---------------------------------------------------------------------------

def dequantize_value_page_torch(
    packed: torch.Tensor,       # (G, D//4) uint8
    scale: torch.Tensor,        # (G,) float16
    zero_point: torch.Tensor,   # (G,) float16
) -> torch.Tensor:
    """Dequantise a packed value cache page back to float32.

    Parameters
    ----------
    packed:
        Packed INT2 values of shape ``(G, D//4)`` uint8.
    scale:
        Per-token scale, shape ``(G,)`` float16.
    zero_point:
        Per-token zero-point, shape ``(G,)`` float16.

    Returns
    -------
    torch.Tensor
        Dequantised values of shape ``(G, D)`` float32.
    """
    G, D4 = packed.shape
    D = D4 * 4
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)

    # Unpack: (G, D4, 1) >> shifts → (G, D4, 4) → (G, D)
    packed_exp = packed.unsqueeze(-1)                        # (G, D4, 1)
    v_quant = ((packed_exp >> shifts) & 0x03).to(torch.int32)  # (G, D4, 4)
    # Interleave correctly: column i in D4 gives positions i, i+D4, i+2*D4, ...
    # We packed as v_quant[:, i::4] << (i*2), so to unpack we read the i-th
    # nibble from each group of 4 bytes
    v_quant_flat = v_quant.reshape(G, D)  # (G, D)

    # Dequantise: (v_quant - ZP) * Scale
    s_bc = scale.float().unsqueeze(1)      # (G, 1)
    z_bc = zero_point.float().unsqueeze(1)  # (G, 1)
    return (v_quant_flat.float() - z_bc) * s_bc  # (G, D)


# ---------------------------------------------------------------------------
# Triton kernel — quantise
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _quantize_value_page_kernel(
        value_ptr,     # (G, D) float16 input
        packed_ptr,    # (G, D//4) uint8 output
        scale_ptr,     # (G,) float16 output
        zero_ptr,      # (G,) float16 output
        G: tl.constexpr,
        D: tl.constexpr,
        stride_v_g,
        stride_v_d,
        stride_p_g,
        stride_p_d,
    ):
        """One Triton program handles one token (pid_g)."""
        pid_g = tl.program_id(0)
        if pid_g >= G:
            return

        D4 = D // 4

        # Load all D values for this token
        vals = tl.load(
            value_ptr + pid_g * stride_v_g + tl.arange(0, D) * stride_v_d
        ).to(tl.float32)

        v_min = tl.min(vals, axis=0)
        v_max = tl.max(vals, axis=0)
        v_range = tl.maximum(v_max - v_min, 1e-8)

        scale = v_range / 3.0
        zero_pt = tl.math.round(-v_min / scale)
        zero_pt = tl.minimum(tl.maximum(zero_pt, 0.0), 3.0)

        tl.store(scale_ptr + pid_g, scale.to(tl.float16))
        tl.store(zero_ptr  + pid_g, zero_pt.to(tl.float16))

        # Quantise all D values
        v_quant = tl.math.round(vals / scale) + zero_pt
        v_quant = tl.minimum(tl.maximum(v_quant, 0.0), 3.0).to(tl.int32)

        # Pack groups of 4 and store
        for byte_i in range(0, D4):
            packed_val = (
                (v_quant[byte_i * 4 + 0] << 0) |
                (v_quant[byte_i * 4 + 1] << 2) |
                (v_quant[byte_i * 4 + 2] << 4) |
                (v_quant[byte_i * 4 + 3] << 6)
            )
            out_addr = packed_ptr + pid_g * stride_p_g + byte_i * stride_p_d
            tl.store(out_addr, packed_val.to(tl.uint8))


# ---------------------------------------------------------------------------
# Public launchers
# ---------------------------------------------------------------------------

def quantize_value_page(
    value: torch.Tensor,        # (G, D) float16, CUDA
    out_packed: torch.Tensor,   # (G, D//4) uint8, CUDA — pre-allocated
    out_scale: torch.Tensor,    # (G,) float16, CUDA — pre-allocated
    out_zero: torch.Tensor,     # (G,) float16, CUDA — pre-allocated
) -> None:
    """Quantise a value cache page in-place into pre-allocated output tensors.

    Routes to Triton kernel when CUDA is available, else falls back to PyTorch.

    Parameters
    ----------
    value:
        FP16 value block of shape ``(G, D)``.
    out_packed, out_scale, out_zero:
        Pre-allocated output views into the global pool.
    """
    if _TRITON_AVAILABLE and value.is_cuda:
        G, D = value.shape
        grid = (G,)
        _quantize_value_page_kernel[grid](
            value, out_packed, out_scale, out_zero,
            G=G, D=D,
            stride_v_g=value.stride(0), stride_v_d=value.stride(1),
            stride_p_g=out_packed.stride(0), stride_p_d=out_packed.stride(1),
        )
    else:
        packed, scale, zero = quantize_value_page_torch(value.cpu())
        out_packed.copy_(packed.to(out_packed.device))
        out_scale.copy_(scale.to(out_scale.device))
        out_zero.copy_(zero.to(out_zero.device))
