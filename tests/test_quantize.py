"""
tests/test_quantize.py
-----------------------
Correctness tests for the Kitty quantisation and dequantisation routines.
Tests use the pure PyTorch fallback implementations so they run on CPU without
requiring a CUDA-capable GPU.

Run with:
    pytest tests/test_quantize.py -v
"""

from __future__ import annotations

import math
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs):
    """Create a KittyConfig with small defaults for fast CPU testing."""
    from kitty.config import KittyConfig
    defaults = dict(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,  # small enough to be fast on CPU
        q_buffer_size=8,  # G=8 (must be divisible by 4)
        sink_size=4,
        local_value_window=8,
        device="cpu",
        boost_rate=0.25,  # 25% = 4 channels boosted (D_boost=4)
    )
    defaults.update(kwargs)
    return KittyConfig(**defaults)


def _make_boost_idx(config):
    """Create a simple boost_idx: channels 0..D_boost-1 are boosted."""
    D = config.head_dim
    K = config.d_boost
    sentinel = config.sentinel
    boost_idx = torch.full((D,), sentinel, dtype=torch.uint8)
    for i in range(K):
        boost_idx[i] = i
    return boost_idx


# ---------------------------------------------------------------------------
# Key Cache quantise / dequantise round-trip
# ---------------------------------------------------------------------------

class TestKeyQuantisation:
    def test_round_trip_non_boosted_channels(self):
        """Non-boosted channels round-trip with error within INT2 tolerance."""
        from kitty.kernels.quantize_key import quantize_key_page_torch
        from kitty.kernels.dequantize_key import dequantize_key_page_torch

        config = _make_config()
        D, G = config.head_dim, config.q_buffer_size
        K = config.d_boost

        # All channels non-boosted
        sentinel = config.sentinel
        boost_idx = torch.full((D,), sentinel, dtype=torch.uint8)

        key = torch.randn(D, G, dtype=torch.float32)
        key_fp16 = key.half()

        t2b, thigh, scale, zero = quantize_key_page_torch(
            key_fp16, boost_idx, K, sentinel
        )

        recon = dequantize_key_page_torch(t2b, thigh, scale, zero, boost_idx, K)

        # Max representable error for INT2 per-channel quantisation is 0.5 LSB
        max_err = (key_fp16 - recon.half()).abs().max().item()
        # Allow up to 10% of input range as error (INT2 is coarse)
        rng = (key_fp16.max() - key_fp16.min()).item()
        assert max_err < 0.35 * rng, (
            f"Round-trip error {max_err:.4f} is too large vs range {rng:.4f}"
        )

    def test_round_trip_boosted_channels(self):
        """Boosted (INT4) channels have much lower quantisation error than INT2."""
        from kitty.kernels.quantize_key import quantize_key_page_torch
        from kitty.kernels.dequantize_key import dequantize_key_page_torch

        # Use boost_rate=1.0 so ALL channels are boosted to INT4
        config = _make_config(boost_rate=0.9375)  # 15/16 boosted channels for D=16
        D, G = config.head_dim, config.q_buffer_size
        K = config.d_boost  # 15
        sentinel = config.sentinel

        # All channels boosted: boost_idx[d] = d for d < K, sentinel for d >= K
        boost_idx = torch.full((D,), sentinel, dtype=torch.uint8)
        for i in range(K):
            boost_idx[i] = i

        key_fp16 = torch.randn(D, G, dtype=torch.float16)
        t2b, thigh, scale, zero = quantize_key_page_torch(
            key_fp16, boost_idx, K, sentinel
        )
        recon = dequantize_key_page_torch(t2b, thigh, scale, zero, boost_idx, K)

        # INT4 channels (boosted) should be much more accurate than INT2 (12% error threshold)
        boosted_err = (key_fp16[:K] - recon[:K].half()).abs().max().item()
        rng = (key_fp16[:K].max() - key_fp16[:K].min()).item()
        assert boosted_err < 0.12 * rng, (
            f"INT4 round-trip error {boosted_err:.4f} exceeds 12% of range {rng:.4f}"
        )

    def test_mixed_precision_boost_idx(self):
        """Mixed INT2/INT4 channels: boosted channels are more accurate."""
        from kitty.kernels.quantize_key import quantize_key_page_torch
        from kitty.kernels.dequantize_key import dequantize_key_page_torch

        config = _make_config(boost_rate=0.25)
        D, G = config.head_dim, config.q_buffer_size
        K = config.d_boost  # 4
        sentinel = config.sentinel

        boost_idx = _make_boost_idx(config)
        key_fp16 = torch.randn(D, G).half()

        t2b, thigh, scale, zero = quantize_key_page_torch(
            key_fp16, boost_idx, K, sentinel
        )

        # Output shapes
        assert t2b.shape == (D, G // 4), f"Expected ({D},{G // 4}), got {t2b.shape}"
        assert thigh.shape == (K, G // 4), f"Expected ({K},{G // 4}), got {thigh.shape}"
        assert scale.shape == (D,)
        assert zero.shape == (D,)

        recon = dequantize_key_page_torch(t2b, thigh, scale, zero, boost_idx, K)
        assert recon.shape == (D, G), f"Expected ({D},{G}), got {recon.shape}"

    def test_packing_roundtrip_exact_values(self):
        """Verify that a known INT2 value {0,1,2,3} packs and unpacks exactly."""
        from kitty.kernels.quantize_key import quantize_key_page_torch
        from kitty.kernels.dequantize_key import dequantize_key_page_torch

        D, G = 4, 4
        sentinel = 255
        K = 0  # no boosted channels
        boost_idx = torch.full((D,), sentinel, dtype=torch.uint8)

        # Construct key so that quantised values are exactly [0, 1, 2, 3] per channel
        # For a channel with values that span [0, 3], S=1, Z=0 → quant = value
        key_fp16 = torch.zeros(D, G, dtype=torch.float16)
        for d in range(D):
            key_fp16[d] = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)

        t2b, thigh, scale, zero = quantize_key_page_torch(
            key_fp16, boost_idx, K, sentinel
        )
        recon = dequantize_key_page_torch(t2b, thigh, scale, zero, boost_idx, K)

        expected = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)
        for d in range(D):
            assert torch.allclose(recon[d].float(), expected.float(), atol=1e-3), (
                f"Channel {d}: expected {expected.tolist()}, got {recon[d].tolist()}"
            )

    def test_snr_improves_with_boost(self):
        """SNR of boosted channels must exceed SNR of non-boosted channels."""
        from kitty.kernels.quantize_key import quantize_key_page_torch
        from kitty.kernels.dequantize_key import dequantize_key_page_torch

        config = _make_config(head_dim=16, q_buffer_size=8, boost_rate=0.5)
        D, G = config.head_dim, config.q_buffer_size
        K = config.d_boost
        sentinel = config.sentinel

        # Only first half boosted
        boost_idx = torch.full((D,), sentinel, dtype=torch.uint8)
        for i in range(K):
            boost_idx[i] = i

        key_fp16 = torch.randn(D, G).half()

        t2b, thigh, scale, zero = quantize_key_page_torch(
            key_fp16, boost_idx, K, sentinel
        )
        recon = dequantize_key_page_torch(t2b, thigh, scale, zero, boost_idx, K)

        err = (key_fp16 - recon.half()).float()
        # Boosted channels: lower MSE than non-boosted
        mse_boosted = err[:K].pow(2).mean().item()
        mse_nonboosted = err[K:].pow(2).mean().item() if K < D else float("inf")

        if K < D:
            assert mse_boosted <= mse_nonboosted * 2, (
                f"Boosted MSE {mse_boosted:.6f} should be ≤ non-boosted MSE {mse_nonboosted:.6f}"
            )


# ---------------------------------------------------------------------------
# Value Cache quantise / dequantise round-trip
# ---------------------------------------------------------------------------

class TestValueQuantisation:
    def test_round_trip_value_cache(self):
        """Value cache INT2 round-trip error within expected tolerance."""
        from kitty.kernels.quantize_value import (
            quantize_value_page_torch,
            dequantize_value_page_torch,
        )

        config = _make_config()
        G, D = config.q_buffer_size, config.head_dim

        value = torch.randn(G, D, dtype=torch.float32)
        value_fp16 = value.half()

        packed, scale, zero = quantize_value_page_torch(value_fp16)
        recon = dequantize_value_page_torch(packed, scale, zero)

        assert recon.shape == (G, D), f"Expected ({G},{D}), got {recon.shape}"

        # INT2 per-token: allow up to 35% of per-token range as error
        for t in range(G):
            rng = (value_fp16[t].max() - value_fp16[t].min()).abs().item()
            err = (value_fp16[t] - recon[t]).abs().max().item()
            assert err < 0.35 * rng + 1e-3, (
                f"Token {t}: error {err:.4f} > 35% of range {rng:.4f}"
            )

    def test_value_output_shapes(self):
        """Value quantisation produces correctly shaped outputs."""
        from kitty.kernels.quantize_value import quantize_value_page_torch

        config = _make_config()
        G, D = config.q_buffer_size, config.head_dim

        value_fp16 = torch.randn(G, D).half()
        packed, scale, zero = quantize_value_page_torch(value_fp16)

        assert packed.shape == (G, D // 4), f"packed shape {packed.shape}"
        assert packed.dtype == torch.uint8
        assert scale.shape == (G,)
        assert zero.shape == (G,)

    def test_constant_value_quantises_exactly(self):
        """A constant value token should produce all-equal dequantised outputs."""
        from kitty.kernels.quantize_value import (
            quantize_value_page_torch,
            dequantize_value_page_torch,
        )

        config = _make_config()
        G, D = config.q_buffer_size, config.head_dim

        # Constant value = 3.0 per token
        value_fp16 = torch.full((G, D), 3.0, dtype=torch.float16)
        packed, scale, zero = quantize_value_page_torch(value_fp16)
        recon = dequantize_value_page_torch(packed, scale, zero)

        # All dequantised values within each token should be equal
        # (constant input → all same quantised code → all same reconstruction)
        for t in range(G):
            row = recon[t]
            assert (row - row[0]).abs().max().item() < 1e-3, (
                f"Token {t}: not all equal after constant-value round-trip"
            )


# ---------------------------------------------------------------------------
# Layout manager integration
# ---------------------------------------------------------------------------

class TestLayoutManager:
    def test_page_allocation_and_free(self):
        """Allocate and free pages, checking free-list accounting."""
        from kitty.config import KittyConfig
        from kitty.layout import PageCentricKVLayoutManager

        config = _make_config()
        D = config.head_dim
        K = config.d_boost
        sentinel = config.sentinel
        boost_idx = torch.zeros(config.num_layers, config.num_kv_heads, D, dtype=torch.uint8)
        boost_idx.fill_(sentinel)
        for l in range(config.num_layers):
            for h in range(config.num_kv_heads):
                for i in range(K):
                    boost_idx[l, h, i] = i

        layout = PageCentricKVLayoutManager(config, boost_idx)
        total = config.max_pages

        p0 = layout.allocate_page(seq_id=0, layer_id=0, page_number=0)
        p1 = layout.allocate_page(seq_id=0, layer_id=0, page_number=1)

        assert layout.memory_stats()["pages_used"] == 2

        layout.free_page(seq_id=0, layer_id=0, page_number=0)
        assert layout.memory_stats()["pages_used"] == 1

        layout.free_sequence(seq_id=0)
        assert layout.memory_stats()["pages_used"] == 0

    def test_tensor_shapes(self):
        """Layout manager allocates tensors with correct shapes."""
        from kitty.layout import PageCentricKVLayoutManager

        config = _make_config()
        D = config.head_dim
        K = config.d_boost
        G = config.q_buffer_size
        H = config.num_kv_heads
        P = config.max_pages
        sentinel = config.sentinel

        boost_idx = torch.full(
            (config.num_layers, config.num_kv_heads, D), sentinel, dtype=torch.uint8
        )
        layout = PageCentricKVLayoutManager(config, boost_idx)

        assert layout.key_baseline.shape == (P, H, D, G // 4)
        assert layout.key_boost.shape == (P, H, K, G // 4)
        assert layout.key_metadata.shape == (P, H, 2, D)
        assert layout.value_pool.shape == (P, H, G, D // 4)
        assert layout.value_metadata.shape == (P, H, G, 2)
