"""
kitty/kernels/__init__.py
--------------------------
Public interface for the Kitty quantisation/dequantisation kernels.

Triton JIT kernels are loaded lazily — the import works cleanly on CPU-only
machines, which fall back to pure PyTorch implementations automatically.
"""

from .quantize_key import quantize_key_page, quantize_key_page_torch
from .dequantize_key import dequantize_key_page, dequantize_key_page_torch
from .quantize_value import (
    quantize_value_page,
    quantize_value_page_torch,
    dequantize_value_page_torch,
)
from .attention import paged_attention, paged_attention_reference

__all__ = [
    "quantize_key_page",
    "quantize_key_page_torch",
    "dequantize_key_page",
    "dequantize_key_page_torch",
    "quantize_value_page",
    "quantize_value_page_torch",
    "dequantize_value_page_torch",
    "paged_attention",
    "paged_attention_reference",
]
