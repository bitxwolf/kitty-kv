"""
tests/test_attention.py
------------------------
Integration tests for the Kitty paged attention pipeline.

Tests the end-to-end flow:
  push_token → Q-Buffer fills → quantise → page written → paged_attention_reference

All tests run on CPU using PyTorch fallback implementations.
"""

from __future__ import annotations

import math
import pytest
import torch

from kitty.config import KittyConfig
from kitty.layout import PageCentricKVLayoutManager
from kitty.pipeline import KittyInferencePipeline
from kitty.kernels.attention import paged_attention_reference


def _make_config(**kw):
    defaults = dict(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        q_buffer_size=8,    # G=8 so we can fill quickly in tests
        sink_size=4,
        local_value_window=8,
        boost_rate=0.25,
        device="cpu",
    )
    defaults.update(kw)
    return KittyConfig(**defaults)


def _make_boost_idx(config: KittyConfig) -> torch.Tensor:
    D, K, sentinel = config.head_dim, config.d_boost, config.sentinel
    boost_idx = torch.full(
        (config.num_layers, config.num_kv_heads, D), sentinel, dtype=torch.uint8
    )
    for i in range(K):
        boost_idx[:, :, i] = i
    return boost_idx


def _make_pipeline(config):
    boost_idx = _make_boost_idx(config)
    layout = PageCentricKVLayoutManager(config, boost_idx)
    pipeline = KittyInferencePipeline(config, layout, boost_idx)
    return pipeline, layout, boost_idx


class TestPipelineState:
    def test_sink_fills_correctly(self):
        cfg = _make_config()
        pipeline, _, _ = _make_pipeline(cfg)
        H, D = cfg.num_kv_heads, cfg.head_dim

        for t in range(cfg.sink_size):
            k = torch.randn(H, D, dtype=torch.float16)
            v = torch.randn(H, D, dtype=torch.float16)
            pipeline.push_token(layer_id=0, key=k, value=v)

        assert pipeline.sink_fill == cfg.sink_size
        assert pipeline.key_qbuf_len == 0

    def test_qbuf_accumulates_after_sink(self):
        cfg = _make_config()
        pipeline, _, _ = _make_pipeline(cfg)
        H, D = cfg.num_kv_heads, cfg.head_dim

        # Fill sinks
        for _ in range(cfg.sink_size):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        # Push 3 more (below G=8 threshold)
        for _ in range(3):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        assert pipeline.key_qbuf_len == 3
        assert pipeline.val_local_len == 3

    def test_quantisation_triggered_at_G(self):
        cfg = _make_config()
        pipeline, layout, _ = _make_pipeline(cfg)
        H, D, G = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size

        # Fill sinks + exactly G more tokens (should trigger quantisation)
        for _ in range(cfg.sink_size + G):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        # After quantisation, qbuf should be reset
        assert pipeline.key_qbuf_len == 0
        # One page should be allocated
        pages = layout.get_page_list(seq_id=0, layer_id=0)
        assert len(pages) == 1

    def test_reset_sequence_clears_pages(self):
        cfg = _make_config()
        pipeline, layout, _ = _make_pipeline(cfg)
        H, D, G = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size

        # Trigger one quantisation
        for _ in range(cfg.sink_size + G):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        pipeline.reset_sequence(seq_id=0)
        assert pipeline.sink_fill == 0
        assert pipeline.key_qbuf_len == 0
        assert layout.memory_stats()["pages_used"] == 0

    def test_multiple_pages_accumulate(self):
        cfg = _make_config()
        pipeline, layout, _ = _make_pipeline(cfg)
        H, D, G = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size

        # Trigger 3 quantisation cycles
        n_tokens = cfg.sink_size + 3 * G
        for _ in range(n_tokens):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        pages = layout.get_page_list(seq_id=0, layer_id=0)
        assert len(pages) == 3


class TestPagedAttentionReference:
    def test_output_shape(self):
        cfg = _make_config()
        pipeline, layout, boost_idx = _make_pipeline(cfg)
        H, D, G = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size
        num_q_heads = H  # no GQA for simplicity

        # Fill and trigger one page
        for _ in range(cfg.sink_size + G):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        ctx = pipeline.get_attention_context(layer_id=0, seq_id=0)
        q = torch.randn(num_q_heads, D, dtype=torch.float16)

        out = paged_attention_reference(
            q=q,
            page_list=ctx["page_list"],
            key_baseline=layout.key_baseline,
            key_boost=layout.key_boost,
            key_meta=layout.key_metadata,
            value_pool=layout.value_pool,
            value_meta=layout.value_metadata,
            boost_idx=boost_idx[0],   # (H, D)
            D_boost=cfg.d_boost,
            sink_k=ctx["sink_k"],
            sink_v=ctx["sink_v"],
            qbuf_k=ctx["key_qbuf"],
            qbuf_v=ctx["val_local"],
            num_q_heads=num_q_heads,
            num_kv_heads=H,
            head_dim=D,
        )

        assert out.shape == (num_q_heads, D)
        assert out.dtype == torch.float16

    def test_attention_output_finite(self):
        """Attention output must not contain NaN or Inf."""
        cfg = _make_config()
        pipeline, layout, boost_idx = _make_pipeline(cfg)
        H, D, G = cfg.num_kv_heads, cfg.head_dim, cfg.q_buffer_size
        num_q_heads = H

        for _ in range(cfg.sink_size + G):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        ctx = pipeline.get_attention_context(layer_id=0, seq_id=0)
        q = torch.randn(num_q_heads, D, dtype=torch.float16)

        out = paged_attention_reference(
            q=q, page_list=ctx["page_list"],
            key_baseline=layout.key_baseline, key_boost=layout.key_boost,
            key_meta=layout.key_metadata, value_pool=layout.value_pool,
            value_meta=layout.value_metadata, boost_idx=boost_idx[0],
            D_boost=cfg.d_boost, sink_k=ctx["sink_k"], sink_v=ctx["sink_v"],
            qbuf_k=ctx["key_qbuf"], qbuf_v=ctx["val_local"],
            num_q_heads=num_q_heads, num_kv_heads=H, head_dim=D,
        )

        assert torch.isfinite(out).all(), "Attention output contains NaN or Inf"

    def test_attention_sink_only(self):
        """When only sinks are present (no quantised pages), attention still works."""
        cfg = _make_config()
        pipeline, layout, boost_idx = _make_pipeline(cfg)
        H, D = cfg.num_kv_heads, cfg.head_dim
        num_q_heads = H

        # Only fill sinks (no quantisation)
        for _ in range(cfg.sink_size):
            pipeline.push_token(0, torch.randn(H, D).half(), torch.randn(H, D).half())

        ctx = pipeline.get_attention_context(layer_id=0, seq_id=0)
        assert len(ctx["page_list"]) == 0  # no quantised pages

        q = torch.randn(num_q_heads, D, dtype=torch.float16)
        out = paged_attention_reference(
            q=q, page_list=[], key_baseline=layout.key_baseline,
            key_boost=layout.key_boost, key_meta=layout.key_metadata,
            value_pool=layout.value_pool, value_meta=layout.value_metadata,
            boost_idx=boost_idx[0], D_boost=cfg.d_boost,
            sink_k=ctx["sink_k"], sink_v=ctx["sink_v"],
            qbuf_k=ctx["key_qbuf"], qbuf_v=ctx["val_local"],
            num_q_heads=num_q_heads, num_kv_heads=H, head_dim=D,
        )
        assert out.shape == (num_q_heads, D)
        assert torch.isfinite(out).all()
