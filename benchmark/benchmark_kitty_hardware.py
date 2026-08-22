# -*- coding: utf-8 -*-
"""
benchmark/benchmark_kitty_hardware.py
--------------------------------------
Hardware validation and benchmarking suite for the Kitty KV Cache framework.

Measures:
  1. Memory compression factor  — KV cache VRAM vs FP16 baseline
  2. Quantisation kernel latency — time per page for key/value quantisation
  3. Dequantisation kernel latency — time per page for key dequantisation
  4. Attention decode latency — time per decode step vs FP16 baseline
  5. Throughput — tokens per second for simulated decode-only workloads

Designed to run on NVIDIA RTX 5050 (or any CUDA GPU).  Falls back to CPU
timing if CUDA is unavailable (useful for CI / sanity checks).

Usage
-----
    python benchmark/benchmark_kitty_hardware.py --seq_len 4096 --num_runs 50
    python benchmark/benchmark_kitty_hardware.py --profile  # uses torch.profiler
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 output so emoji and Unicode chars print cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kitty.config import KittyConfig
from kitty.layout import PageCentricKVLayoutManager
from kitty.pipeline import KittyInferencePipeline
from kitty.kernels.quantize_key import quantize_key_page, quantize_key_page_torch
from kitty.kernels.dequantize_key import dequantize_key_page, dequantize_key_page_torch
from kitty.kernels.quantize_value import quantize_value_page, quantize_value_page_torch


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cuda_or_cpu() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_runs(fn, num_runs: int) -> float:
    """Return median wall-clock time in milliseconds for fn() over num_runs."""
    times = []
    for _ in range(num_runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


def _make_boost_idx(config: KittyConfig, device: str) -> torch.Tensor:
    """Build a simple boost_idx where the first d_boost channels are boosted."""
    D = config.head_dim
    K = config.d_boost
    sentinel = config.sentinel
    boost_idx = torch.full((config.num_layers, config.num_kv_heads, D),
                           sentinel, dtype=torch.uint8, device=device)
    for i in range(K):
        boost_idx[:, :, i] = i
    return boost_idx


# ---------------------------------------------------------------------------
# Benchmark 1: Memory compression
# ---------------------------------------------------------------------------

def benchmark_memory_compression(config: KittyConfig, device: str) -> Dict[str, float]:
    """Compare VRAM footprint of FP16 vs Kitty for a given max_seq_len."""
    L = config.num_layers
    H = config.num_kv_heads
    D = config.head_dim
    T = config.max_seq_len
    dtype_bytes = 2  # float16

    fp16_bytes = L * 2 * H * T * D * dtype_bytes  # 2 for K+V

    # Kitty layout footprint
    boost_idx = _make_boost_idx(config, device)
    layout = PageCentricKVLayoutManager(config, boost_idx)
    stats = layout.memory_stats()
    kitty_bytes = stats["total_MB"] * 1024 * 1024

    # Sink + qbuffer + local FP16 buffers
    S, G, R = config.sink_size, config.q_buffer_size, config.local_value_window
    fp16_buf_bytes = L * 2 * H * (S + G + R) * D * dtype_bytes  # 2 for K+V

    kitty_total = kitty_bytes + fp16_buf_bytes

    return {
        "fp16_cache_MB": fp16_bytes / (1024 ** 2),
        "kitty_quant_MB": kitty_bytes / (1024 ** 2),
        "kitty_fp16_buf_MB": fp16_buf_bytes / (1024 ** 2),
        "kitty_total_MB": kitty_total / (1024 ** 2),
        "compression_ratio": fp16_bytes / kitty_total,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Quantisation kernel latency
# ---------------------------------------------------------------------------

def benchmark_quantize_key(
    config: KittyConfig, device: str, num_runs: int
) -> Dict[str, float]:
    """Time the key cache quantisation for one page."""
    D, G = config.head_dim, config.q_buffer_size
    K = config.d_boost
    sentinel = config.sentinel

    boost_idx = torch.full((D,), sentinel, dtype=torch.uint8, device=device)
    for i in range(K):
        boost_idx[i] = i

    key = torch.randn(D, G, dtype=torch.float16, device=device)
    out_baseline = torch.zeros(D, G // 4, dtype=torch.uint8, device=device)
    out_boost    = torch.zeros(K, G // 4, dtype=torch.uint8, device=device)
    out_scale    = torch.zeros(D, dtype=torch.float16, device=device)
    out_zero     = torch.zeros(D, dtype=torch.float16, device=device)

    def _run():
        quantize_key_page(key, boost_idx, K, sentinel, out_baseline, out_boost, out_scale, out_zero)

    # Warmup
    for _ in range(5):
        _run()

    latency_ms = _timed_runs(_run, num_runs)
    bytes_processed = D * G * 2  # float16

    return {
        "latency_ms": latency_ms,
        "bandwidth_GBps": bytes_processed / (latency_ms * 1e-3) / 1e9,
    }


def benchmark_dequantize_key(
    config: KittyConfig, device: str, num_runs: int
) -> Dict[str, float]:
    """Time the key cache dequantisation for one page."""
    D, G = config.head_dim, config.q_buffer_size
    K = config.d_boost
    sentinel = config.sentinel

    boost_idx = torch.full((D,), sentinel, dtype=torch.uint8, device=device)
    for i in range(K):
        boost_idx[i] = i

    key = torch.randn(D, G, dtype=torch.float16, device=device)
    out_baseline = torch.zeros(D, G // 4, dtype=torch.uint8, device=device)
    out_boost    = torch.zeros(K, G // 4, dtype=torch.uint8, device=device)
    out_scale    = torch.zeros(D, dtype=torch.float16, device=device)
    out_zero     = torch.zeros(D, dtype=torch.float16, device=device)
    quantize_key_page(key, boost_idx, K, sentinel, out_baseline, out_boost, out_scale, out_zero)

    out_fp16 = torch.zeros(D, G, dtype=torch.float16, device=device)

    def _run():
        dequantize_key_page(out_baseline, out_boost, out_scale, out_zero,
                            boost_idx, K, out_fp16)

    for _ in range(5):
        _run()

    latency_ms = _timed_runs(_run, num_runs)
    bytes_written = D * G * 2

    return {
        "latency_ms": latency_ms,
        "bandwidth_GBps": bytes_written / (latency_ms * 1e-3) / 1e9,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Decode throughput
# ---------------------------------------------------------------------------

def benchmark_decode_throughput(
    config: KittyConfig, device: str, num_decode_steps: int
) -> Dict[str, float]:
    """Simulate a decode-only workload and measure tokens per second."""
    from kitty.sensitivity import KittySensitivityProfiler

    boost_idx = _make_boost_idx(config, device)
    layout = PageCentricKVLayoutManager(config, boost_idx)
    pipeline = KittyInferencePipeline(config, layout, boost_idx)

    H = config.num_kv_heads
    D = config.head_dim
    layer_id = 0
    seq_id = 0

    def _generate_token():
        k = torch.randn(H, D, dtype=torch.float16, device=device)
        v = torch.randn(H, D, dtype=torch.float16, device=device)
        pipeline.push_token(layer_id=layer_id, key=k, value=v, seq_id=seq_id)

    # Warmup: fill sinks
    for _ in range(config.sink_size):
        _generate_token()

    _sync()
    t_start = time.perf_counter()
    for _ in range(num_decode_steps):
        _generate_token()
    _sync()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    tokens_per_sec = num_decode_steps / elapsed

    return {
        "num_steps": num_decode_steps,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": elapsed * 1000 / num_decode_steps,
    }


# ---------------------------------------------------------------------------
# Torch profiler trace (optional)
# ---------------------------------------------------------------------------

def run_torch_profiler(config: KittyConfig, device: str) -> None:
    """Run quantisation under torch.profiler and print the top ops."""
    if not torch.cuda.is_available():
        print("[profiler] CUDA not available; skipping torch.profiler trace.")
        return

    D, G = config.head_dim, config.q_buffer_size
    K = config.d_boost
    sentinel = config.sentinel

    boost_idx = torch.full((D,), sentinel, dtype=torch.uint8, device=device)
    for i in range(K):
        boost_idx[i] = i

    key = torch.randn(D, G, dtype=torch.float16, device=device)
    out_baseline = torch.zeros(D, G // 4, dtype=torch.uint8, device=device)
    out_boost    = torch.zeros(K, G // 4, dtype=torch.uint8, device=device)
    out_scale    = torch.zeros(D, dtype=torch.float16, device=device)
    out_zero     = torch.zeros(D, dtype=torch.float16, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(20):
            quantize_key_page(key, boost_idx, K, sentinel,
                              out_baseline, out_boost, out_scale, out_zero)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(config: KittyConfig, device: str, num_runs: int) -> None:
    """Print a full benchmark report."""
    print("=" * 65)
    print(f"  Kitty Benchmark Report - {config}")
    print(f"  Device: {device}  |  Runs: {num_runs}")
    print("=" * 65)

    print("\n📦 Memory Compression")
    mem = benchmark_memory_compression(config, device)
    print(f"  FP16 KV cache:    {mem['fp16_cache_MB']:.1f} MB")
    print(f"  Kitty quant pool: {mem['kitty_quant_MB']:.1f} MB")
    print(f"  Kitty FP16 bufs:  {mem['kitty_fp16_buf_MB']:.1f} MB")
    print(f"  Kitty total:      {mem['kitty_total_MB']:.1f} MB")
    print(f"  Compression:      {mem['compression_ratio']:.2f}×")

    print("\n⚡ Key Quantisation Latency")
    kq = benchmark_quantize_key(config, device, num_runs)
    print(f"  Latency:   {kq['latency_ms']:.4f} ms/page")
    print(f"  Bandwidth: {kq['bandwidth_GBps']:.2f} GB/s")

    print("\n⚡ Key Dequantisation Latency")
    kdq = benchmark_dequantize_key(config, device, num_runs)
    print(f"  Latency:   {kdq['latency_ms']:.4f} ms/page")
    print(f"  Bandwidth: {kdq['bandwidth_GBps']:.2f} GB/s")

    num_steps = min(500, config.max_seq_len - config.sink_size - 1)
    print(f"\n🚀 Decode Throughput ({num_steps} steps, 1 layer)")
    tp = benchmark_decode_throughput(config, device, num_steps)
    print(f"  Tokens/sec:    {tp['tokens_per_sec']:.1f}")
    print(f"  ms/token:      {tp['ms_per_token']:.3f}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kitty Hardware Benchmark")
    parser.add_argument("--seq_len",   type=int, default=8192, help="Max sequence length")
    parser.add_argument("--num_runs",  type=int, default=100,  help="Timing repetitions")
    parser.add_argument("--num_layers",type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim",  type=int, default=128)
    parser.add_argument("--boost_rate", type=float, default=0.125,
                        help="Kitty=0.125, Kitty-Pro=0.25")
    parser.add_argument("--profile",   action="store_true",
                        help="Run torch.profiler trace")
    parser.add_argument("--device",    type=str, default=_cuda_or_cpu())
    args = parser.parse_args()

    config = KittyConfig(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        boost_rate=args.boost_rate,
        max_seq_len=args.seq_len,
        device=args.device,
    )

    print_report(config, args.device, args.num_runs)

    if args.profile:
        run_torch_profiler(config, args.device)
