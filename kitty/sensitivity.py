"""
kitty/sensitivity.py
--------------------
Phase 1 — Offline Channel-Sensitivity Profiling.

This module implements the KittySensitivityProfiler, which hooks into the Key
Projection linear layers of a target transformer model, feeds a calibration
dataset through, and computes the per-head, per-channel importance score

    s_i = (1/T) * Σ_{t=1}^{T} |x_{i,t}|

from which the top-K boosted (INT4) channels are selected and the
``Boost_IDX_uint8`` index mapping is built.

Reference: §3.2, Algorithm 2 of the Kitty paper; §2.1 of the reference spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import KittyConfig


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _ChannelAccumulator:
    """Running sum of |activation| per channel for a single (layer, head)."""

    def __init__(self, num_kv_heads: int, head_dim: int, device: str) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # sum_abs[h, d] = Σ |x_{h,d,t}| over all calibration tokens
        self.sum_abs = torch.zeros(
            num_kv_heads, head_dim, dtype=torch.float32, device=device
        )
        self.token_count = 0  # total tokens seen

    @torch.no_grad()
    def update(self, key: torch.Tensor) -> None:
        """Accumulate absolute values from a key tensor.

        Parameters
        ----------
        key:
            Shape ``(batch, num_kv_heads, seq_len, head_dim)`` — the raw FP16
            key projections *before* any quantisation.
        """
        # Flatten batch and seq dims → (N, num_kv_heads, head_dim)
        n = key.shape[0] * key.shape[2]
        flat = key.reshape(-1, self.num_kv_heads, self.head_dim)
        # |x|.sum(over N tokens) → (num_kv_heads, head_dim)
        self.sum_abs += flat.abs().float().sum(dim=0)
        self.token_count += n

    def scores(self) -> torch.Tensor:
        """Return per-channel average magnitude, shape (num_kv_heads, head_dim)."""
        if self.token_count == 0:
            raise RuntimeError("No tokens have been accumulated yet.")
        return self.sum_abs / self.token_count


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class KittySensitivityProfiler:
    """Offline sensitivity profiler that determines which Key Cache channels
    should be promoted to INT4 precision.

    Usage
    -----
    >>> profiler = KittySensitivityProfiler(config, model)
    >>> profiler.register_hooks(layer_indices=[0, 1, 2, ...])
    >>> for batch in calibration_dataloader:
    ...     model(**batch)
    >>> profiler.remove_hooks()
    >>> profiler.save(config.profile_save_dir)
    # Or query directly:
    >>> masks = profiler.boost_masks        # (num_layers, num_kv_heads, D)
    >>> idx   = profiler.boost_idx_uint8   # (num_layers, num_kv_heads, D)
    """

    def __init__(
        self,
        config: KittyConfig,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        self.model = model
        self._accumulators: Dict[int, _ChannelAccumulator] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Populated after compute()
        self.scores_: Optional[torch.Tensor] = None          # (L, H, D) float32
        self.boost_masks_: Optional[torch.Tensor] = None     # (L, H, D) bool
        self.boost_idx_uint8_: Optional[torch.Tensor] = None  # (L, H, D) uint8

    # ------------------------------------------------------------------ hooks

    def register_hooks(
        self,
        layer_indices: Sequence[int],
        key_proj_getter: Optional[Callable[[nn.Module, int], nn.Module]] = None,
    ) -> None:
        """Attach forward hooks to the Key Projection layers of the model.

        Parameters
        ----------
        layer_indices:
            Indices of the transformer layers to profile.
        key_proj_getter:
            Optional callable ``(model, layer_idx) → nn.Module`` that returns
            the Key Projection linear layer for a given index.  If not
            provided, a best-effort heuristic is used to find ``k_proj`` or
            ``key`` sub-modules in common model families (LLaMA, Qwen, GPT-2).
        """
        if self.model is None:
            raise RuntimeError(
                "A model must be provided to KittySensitivityProfiler to "
                "register hooks.  Pass model= at construction time."
            )

        for layer_idx in layer_indices:
            acc = _ChannelAccumulator(
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.device,
            )
            self._accumulators[layer_idx] = acc

            k_proj = self._find_key_proj(layer_idx, key_proj_getter)

            def _make_hook(accumulator: _ChannelAccumulator) -> Callable:
                def _hook(module: nn.Module, inp, out: torch.Tensor) -> None:
                    # out shape: (batch, seq_len, num_kv_heads * head_dim)
                    # Reshape to (batch, seq_len, num_kv_heads, head_dim) then
                    # permute to (batch, num_kv_heads, seq_len, head_dim)
                    B, T, _ = out.shape
                    H = accumulator.num_kv_heads
                    D = accumulator.head_dim
                    key = out.detach().reshape(B, T, H, D).permute(0, 2, 1, 3)
                    accumulator.update(key)
                return _hook

            handle = k_proj.register_forward_hook(_make_hook(acc))
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ----------------------------------------------------------------- compute

    def compute(self) -> None:
        """Convert accumulated magnitudes into masks and index arrays.

        Must be called after all calibration batches have been forwarded.
        """
        cfg = self.config
        L = cfg.num_layers
        H = cfg.num_kv_heads
        D = cfg.head_dim
        K = cfg.d_boost          # number of boosted channels
        sentinel = cfg.sentinel

        if not self._accumulators:
            raise RuntimeError(
                "No accumulators found.  Did you call register_hooks() and "
                "forward the model over the calibration data?"
            )

        scores = torch.zeros(L, H, D, dtype=torch.float32)
        for layer_idx, acc in self._accumulators.items():
            scores[layer_idx] = acc.scores()

        # Top-K boosted channels per (layer, head)
        # boost_masks: (L, H, D) bool
        topk_indices = scores.topk(K, dim=-1).indices  # (L, H, K)
        boost_masks = torch.zeros(L, H, D, dtype=torch.bool)
        boost_masks.scatter_(-1, topk_indices, True)

        # Build Boost_IDX_uint8: valid offset into the boosted subset, or sentinel
        # For each (l, h), the K boosted channels are given compact offsets 0..K-1
        # in the order they appear in the sorted topk.
        boost_idx = torch.full((L, H, D), sentinel, dtype=torch.uint8)
        # Use argsort to get per-row sorted orders
        sorted_scores = scores.argsort(dim=-1, descending=True)  # (L, H, D)
        for rank in range(K):
            channel = sorted_scores[..., rank]  # (L, H)
            # Expand channel to index boost_idx: set boost_idx[l,h,channel[l,h]] = rank
            l_idx = torch.arange(L).unsqueeze(1).expand(L, H)
            h_idx = torch.arange(H).unsqueeze(0).expand(L, H)
            boost_idx[l_idx, h_idx, channel] = rank

        self.scores_ = scores
        self.boost_masks_ = boost_masks
        self.boost_idx_uint8_ = boost_idx

    # ----------------------------------------------------------------- I/O

    def save(self, directory: str | Path) -> None:
        """Save profiling results to disk.

        Saves:
        - ``scores.pt``         — raw per-channel magnitude scores
        - ``boost_masks.pt``    — boolean precision-boost masks
        - ``boost_idx.pt``      — uint8 physical-offset index array
        - ``meta.json``         — config snapshot
        """
        if self.scores_ is None:
            self.compute()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.scores_, directory / "scores.pt")
        torch.save(self.boost_masks_, directory / "boost_masks.pt")
        torch.save(self.boost_idx_uint8_, directory / "boost_idx.pt")

        meta = {
            "num_layers": self.config.num_layers,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
            "d_boost": self.config.d_boost,
            "boost_rate": self.config.boost_rate,
            "token_count": {
                k: v.token_count for k, v in self._accumulators.items()
            },
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[KittySensitivityProfiler] Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path, config: KittyConfig) -> "KittySensitivityProfiler":
        """Load a previously saved profiler result.

        Parameters
        ----------
        directory:
            Path to the directory saved by :meth:`save`.
        config:
            KittyConfig matching the saved profile.
        """
        directory = Path(directory)
        profiler = cls(config)
        profiler.scores_ = torch.load(directory / "scores.pt", weights_only=True)
        profiler.boost_masks_ = torch.load(directory / "boost_masks.pt", weights_only=True)
        profiler.boost_idx_uint8_ = torch.load(directory / "boost_idx.pt", weights_only=True)
        print(f"[KittySensitivityProfiler] Loaded from {directory}")
        return profiler

    # ---------------------------------------------------------------- helpers

    @property
    def boost_masks(self) -> torch.Tensor:
        """Boolean mask of shape (num_layers, num_kv_heads, head_dim)."""
        if self.boost_masks_ is None:
            self.compute()
        return self.boost_masks_  # type: ignore[return-value]

    @property
    def boost_idx_uint8(self) -> torch.Tensor:
        """Physical-offset index array of shape (num_layers, num_kv_heads, head_dim),
        dtype uint8.  Non-boosted channels hold the sentinel value."""
        if self.boost_idx_uint8_ is None:
            self.compute()
        return self.boost_idx_uint8_  # type: ignore[return-value]

    def _find_key_proj(
        self,
        layer_idx: int,
        getter: Optional[Callable[[nn.Module, int], nn.Module]],
    ) -> nn.Module:
        """Return the Key Projection linear layer for a given transformer layer."""
        if getter is not None:
            return getter(self.model, layer_idx)

        # Best-effort heuristics for common model families
        # Tries: model.layers[i].self_attn.k_proj  (LLaMA / Qwen)
        #        model.transformer.h[i].attn.k_proj  (GPT-2 style)
        #        model.model.layers[i].self_attn.k_proj  (wrapped variants)
        candidates = [
            lambda m, i: m.layers[i].self_attn.k_proj,
            lambda m, i: m.model.layers[i].self_attn.k_proj,
            lambda m, i: m.transformer.h[i].attn.k_proj,
            lambda m, i: m.transformer.h[i].attn.c_attn,  # GPT-2 combined QKV
        ]
        for fn in candidates:
            try:
                module = fn(self.model, layer_idx)
                if isinstance(module, nn.Module):
                    return module
            except (AttributeError, IndexError, TypeError):
                continue

        raise RuntimeError(
            f"Could not automatically find the Key Projection layer for "
            f"layer {layer_idx}.  Please pass a key_proj_getter callable."
        )

    def summary(self) -> str:
        """Return a human-readable summary of the profiling results."""
        if self.scores_ is None:
            return "KittySensitivityProfiler — not yet computed."
        cfg = self.config
        lines = [
            f"KittySensitivityProfiler Summary",
            f"  Layers: {cfg.num_layers}, KV Heads: {cfg.num_kv_heads}, "
            f"Head Dim: {cfg.head_dim}",
            f"  Boost rate: {cfg.boost_rate:.1%} → {cfg.d_boost} INT4 channels/head",
            f"  Sentinel value: {cfg.sentinel}",
        ]
        if self.scores_ is not None:
            max_score = self.scores_.max().item()
            mean_score = self.scores_.mean().item()
            lines += [
                f"  Max channel score: {max_score:.4f}",
                f"  Mean channel score: {mean_score:.4f}",
            ]
        return "\n".join(lines)
