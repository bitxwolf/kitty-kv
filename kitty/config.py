"""
kitty/config.py
---------------
Central configuration dataclass for the Kitty 2-bit KV Cache Quantization
framework. Every tuneable hyper-parameter and hardware constant lives here so
that the rest of the codebase can import a single source of truth.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class KittyConfig:
    """Full configuration for the Kitty KV Cache Quantization framework.

    Attributes
    ----------
    num_layers:
        Number of transformer decoder layers in the target model.
    num_kv_heads:
        Number of KV attention heads (after GQA grouping).
    head_dim:
        Dimension of each KV attention head (typically 128).
    sink_size:
        Number of initial *Attention Sink* tokens retained in FP16.  These
        receive consistently high attention weights and must not be quantized.
        Paper default: S = 32.
    q_buffer_size:
        Size G of the FP16 Quantization Buffer.  Quantization is triggered once
        exactly G tokens have accumulated.  Paper default: G = 128.
    local_value_window:
        Number of most-recent Value Cache tokens retained in FP16 (the sliding
        window / Local Buffer).  Paper default: R = 128.
    boost_rate:
        Fraction of Key Cache channels promoted from INT2 to INT4 precision.
        - Kitty standard: 0.125  (12.5 %, 16/128 channels for D=128)
        - Kitty-Pro:      0.250  (25.0 %, 32/128 channels for D=128)
    max_seq_len:
        Maximum supported sequence length (used to pre-compute pool sizes).
    max_batch_size:
        Maximum batch size supported at inference time.
    dtype:
        Full-precision floating-point type used for FP16 buffers. Either
        ``"float16"`` or ``"bfloat16"``.
    device:
        Target CUDA device string (e.g. ``"cuda:0"``).
    profile_save_dir:
        Directory where the offline sensitivity profiler saves/loads its masks
        and index arrays.
    variant:
        Human-readable variant label — ``"kitty"`` or ``"kitty-pro"``.
    """

    # ------------------------------------------------------------------ model
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    # --------------------------------------------------------- memory regions
    sink_size: int = 32           # S  — attention sinks, always FP16
    q_buffer_size: int = 128      # G  — quantisation trigger threshold
    local_value_window: int = 128  # R  — value cache FP16 sliding window

    # ------------------------------------------------------ precision control
    boost_rate: float = 0.125     # 12.5 % for Kitty; 25.0 % for Kitty-Pro

    # ------------------------------------------------------- runtime / layout
    max_seq_len: int = 32768
    max_batch_size: int = 1
    dtype: Literal["float16", "bfloat16"] = "float16"
    device: str = "cuda:0"

    # --------------------------------------------------------- profiler paths
    profile_save_dir: str = "kitty_profiles"

    # ----------------------------------------------------- convenience labels
    variant: Literal["kitty", "kitty-pro"] = "kitty"

    # ---------------------------------------------------------------- derived
    @property
    def d_boost(self) -> int:
        """Number of boosted (INT4) channels per head: K = round(r × D)."""
        return round(self.boost_rate * self.head_dim)

    @property
    def max_pages(self) -> int:
        """Maximum number of quantised pages that can ever be allocated.

        A page holds exactly G = ``q_buffer_size`` tokens.  We need one page
        per G tokens per layer per head, across the full maximum sequence.
        We add a small safety margin of 2 extra pages per sequence.
        """
        tokens_per_page = self.q_buffer_size
        pages_per_seq = (
            (self.max_seq_len - self.sink_size) // tokens_per_page + 2
        )
        return pages_per_seq * self.max_batch_size

    @property
    def sentinel(self) -> int:
        """Sentinel value written into ``Boost_IDX_uint8`` for non-boosted
        channels.  Must be distinguishable from any valid offset in
        [0, D_boost-1] and fit in a uint8 byte (max 255)."""
        return min(self.d_boost + 1, 255)

    # -------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        """Serialise the config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "KittyConfig":
        """Deserialise a config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    # ------------------------------------------------------------ class factories
    @classmethod
    def kitty_pro(cls, **kwargs) -> "KittyConfig":
        """Return a Kitty-Pro config (25 % boost rate)."""
        return cls(boost_rate=0.25, variant="kitty-pro", **kwargs)

    def __post_init__(self) -> None:
        # Validate
        assert 0 < self.boost_rate < 1, "boost_rate must be in (0, 1)"
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 (packing)"
        assert self.q_buffer_size % 4 == 0, (
            "q_buffer_size (G) must be divisible by 4 (uint8 packing)"
        )
        assert self.sink_size >= 0
        assert self.local_value_window >= 0
        assert self.dtype in ("float16", "bfloat16")

    def __repr__(self) -> str:
        return (
            f"KittyConfig(variant={self.variant!r}, "
            f"num_layers={self.num_layers}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"boost_rate={self.boost_rate} -> d_boost={self.d_boost}, "
            f"G={self.q_buffer_size}, S={self.sink_size}, R={self.local_value_window})"
        )
