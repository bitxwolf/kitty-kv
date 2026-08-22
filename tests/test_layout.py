"""
tests/test_layout.py
---------------------
Tests for PageCentricKVLayoutManager page allocation, pool shapes, and accessors.
"""

from __future__ import annotations

import pytest
import torch

from kitty.config import KittyConfig
from kitty.layout import PageCentricKVLayoutManager


def _make_config(**kw):
    defaults = dict(
        num_layers=2, num_kv_heads=2, head_dim=16,
        q_buffer_size=8, sink_size=4, local_value_window=8,
        boost_rate=0.25, device="cpu",
    )
    defaults.update(kw)
    return KittyConfig(**defaults)


def _make_layout(config: KittyConfig) -> PageCentricKVLayoutManager:
    D, K, sentinel = config.head_dim, config.d_boost, config.sentinel
    boost_idx = torch.full(
        (config.num_layers, config.num_kv_heads, D), sentinel, dtype=torch.uint8
    )
    for i in range(K):
        boost_idx[:, :, i] = i
    return PageCentricKVLayoutManager(config, boost_idx)


class TestPageAllocation:
    def test_allocate_returns_unique_phys_indices(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        pages = [layout.allocate_page(seq_id=0, layer_id=0, page_number=i)
                 for i in range(5)]
        phys = [p.physical_idx for p in pages]
        assert len(set(phys)) == 5, "All physical indices must be unique"

    def test_free_returns_page_to_pool(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        p0 = layout.allocate_page(seq_id=0, layer_id=0, page_number=0)
        phys0 = p0.physical_idx
        layout.free_page(seq_id=0, layer_id=0, page_number=0)
        # The freed page should be reusable
        p1 = layout.allocate_page(seq_id=1, layer_id=0, page_number=0)
        assert p1.physical_idx == phys0  # same slot reused

    def test_pool_exhaustion_raises(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        total = cfg.max_pages
        for i in range(total):
            layout.allocate_page(seq_id=0, layer_id=0, page_number=i)
        with pytest.raises(RuntimeError, match="exhausted"):
            layout.allocate_page(seq_id=0, layer_id=0, page_number=total)

    def test_free_sequence_releases_all(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        for i in range(4):
            layout.allocate_page(seq_id=7, layer_id=0, page_number=i)
        assert layout.memory_stats()["pages_used"] == 4
        layout.free_sequence(seq_id=7)
        assert layout.memory_stats()["pages_used"] == 0


class TestPoolShapes:
    def test_key_baseline_shape(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        P, H, D, G = cfg.max_pages, cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size
        assert layout.key_baseline.shape == (P, H, D, G // 4)
        assert layout.key_baseline.dtype == torch.uint8

    def test_key_boost_shape(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        P, H, K, G = cfg.max_pages, cfg.num_kv_heads, cfg.d_boost, cfg.q_buffer_size
        assert layout.key_boost.shape == (P, H, K, G // 4)

    def test_key_metadata_shape(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        P, H, D = cfg.max_pages, cfg.num_kv_heads, cfg.head_dim
        assert layout.key_metadata.shape == (P, H, 2, D)

    def test_value_pool_shape(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        P, H, G, D = cfg.max_pages, cfg.num_kv_heads, cfg.q_buffer_size, cfg.head_dim
        assert layout.value_pool.shape == (P, H, G, D // 4)

    def test_value_metadata_shape(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        P, H, G = cfg.max_pages, cfg.num_kv_heads, cfg.q_buffer_size
        assert layout.value_metadata.shape == (P, H, G, 2)


class TestAccessors:
    def test_page_accessor_returns_correct_slice(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        page = layout.allocate_page(seq_id=0, layer_id=0, page_number=0)
        phys = page.physical_idx

        baseline_view = layout.key_baseline_page(phys)
        H, D, G4 = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size // 4
        assert baseline_view.shape == (H, D, G4)

        boost_view = layout.key_boost_page(phys)
        assert boost_view.shape == (H, cfg.d_boost, G4)

    def test_get_page_list_ordered(self):
        cfg = _make_config()
        layout = _make_layout(cfg)
        phys_list = []
        for i in range(3):
            p = layout.allocate_page(seq_id=0, layer_id=0, page_number=i)
            phys_list.append(p.physical_idx)
        retrieved = layout.get_page_list(seq_id=0, layer_id=0)
        assert retrieved == phys_list
