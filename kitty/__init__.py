"""
kitty/__init__.py
-----------------
Public API for the Kitty 2-bit KV Cache Quantization framework.
"""

from .config import KittyConfig
from .sensitivity import KittySensitivityProfiler
from .layout import PageCentricKVLayoutManager
from .pipeline import KittyInferencePipeline
from .attention import KittyPagedAttention, KittyAttentionFactory

__version__ = "0.1.0"
__author__ = "Kitty Framework"
__description__ = (
    "2-bit KV Cache Quantization with per-channel boosted precision "
    "and Dense-Sparse memory decomposition for near-lossless LLM inference."
)

__all__ = [
    "KittyConfig",
    "KittySensitivityProfiler",
    "PageCentricKVLayoutManager",
    "KittyInferencePipeline",
    "KittyPagedAttention",
    "KittyAttentionFactory",
]
