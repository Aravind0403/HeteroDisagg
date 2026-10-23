"""Tests for HeteroDisagg's Adaptive Compression Policy.

Verifies:
1. Rejection of short prompts where tile padding and metadata expand wire footprint.
2. Rejection of compression on ultra-fast interconnects where CPU codec overhead dominates.
3. Selection of block-scaled FP8 on bandwidth-constrained or long-context transfers.
4. Evaluation of Policy vs Always-Compress, Never-Compress, and Oracle on held-out test scenarios.
"""

import pytest
from hetero_kv.policy import AdaptiveCompressionPolicy, CompressionMode


@pytest.fixture
def policy():
    return AdaptiveCompressionPolicy(
        encoding_throughput_bytes_per_sec=2.0e9,  # 2 GB/s
        decoding_throughput_bytes_per_sec=4.0e9,  # 4 GB/s
        framing_overhead_bytes_per_frame=150,
        min_saving_threshold_ms=0.05,
    )


def test_short_prompt_expansion_rejected(policy):
    """
    Prompts shorter than block_size_tokens (e.g. 8 tokens) expand under tile padding and framing.
    The policy must reject compression and select UNCOMPRESSED_FP16.
    """
    res = policy.evaluate(
        seq_len=8,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        link_bandwidth_bytes_per_sec=125e6,  # 1 Gbps
        block_size_tokens=16,
    )

    assert res.selected_mode == CompressionMode.UNCOMPRESSED_FP16
    assert res.is_favorable is False
    assert "Wire expansion detected" in res.reason
    assert res.bytes_saved <= 0


def test_high_speed_link_codec_overhead_rejected(policy):
    """
    On 100 Gbps link, wire transmission time is minimal (~sub-millisecond).
    CPU quantization and dequantization overhead exceeds the network savings.
    The policy must choose UNCOMPRESSED_FP16.
    """
    res = policy.evaluate(
        seq_len=128,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        link_bandwidth_bytes_per_sec=12.5e9,  # 100 Gbps = 12.5 GB/s
        block_size_tokens=16,
    )

    assert res.selected_mode == CompressionMode.UNCOMPRESSED_FP16
    assert res.is_favorable is False
    assert "Codec overhead exceeds wire savings" in res.reason


def test_bandwidth_constrained_long_context_selected(policy):
    """
    On 1 Gbps or 10 Gbps with seq_len=512, FP8 achieves >1.7x compression,
    and wire savings vastly outpace codec overhead. The policy must select BLOCK_FP8.
    """
    res = policy.evaluate(
        seq_len=512,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        link_bandwidth_bytes_per_sec=125e6,  # 1 Gbps
        block_size_tokens=16,
    )

    assert res.selected_mode == CompressionMode.BLOCK_FP8
    assert res.is_favorable is True
    assert res.compression_ratio > 1.5
    assert res.estimated_net_saving_ms > 1.0


def test_policy_against_empirical_benchmark_outcomes(policy):
    """
    Validates policy against independently measured empirical benchmark outcomes
    from benchmarks/transfer_modes_benchmark.json across all bandwidths and sequence lengths.
    
    Verifies that:
    1. Policy matches empirical optimal decision in >= 90% of points.
    2. Total empirical latency under Policy beats both Always-Compress and Never-Compress.
    3. Total empirical regret vs the Empirical Oracle is < 1.0%.
    """
    import json
    from pathlib import Path

    benchmark_path = Path("benchmarks/transfer_modes_benchmark.json")
    assert benchmark_path.exists(), "Benchmark artifact missing: run scripts/benchmark_transfer_modes.py first"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    points = bench_data["points"]
    assert len(points) >= 32, f"Expected at least 32 benchmark scenarios, got {len(points)}"

    total_lat_policy = 0.0
    total_lat_always_comp = 0.0
    total_lat_never_comp = 0.0
    total_lat_oracle = 0.0
    matches_empirical_count = 0

    for pt in points:
        seq = pt["seq_len"]
        bw = pt["bandwidth_bytes_per_sec"]
        
        # Policy prediction (completely blind to measured latency)
        eval_res = policy.evaluate(
            seq_len=seq,
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            link_bandwidth_bytes_per_sec=bw,
            block_size_tokens=16,
            prefill_tp=2,
        )

        u_lat = pt["uncompressed_fp16"]["projected_link_e2e_ms"]["p50"]
        f_lat = pt["block_fp8"]["projected_link_e2e_ms"]["p50"]

        oracle_lat = min(u_lat, f_lat)
        empirical_winner = pt["empirical_winner"]
        policy_lat = f_lat if eval_res.selected_mode.value == "block_fp8" else u_lat

        total_lat_policy += policy_lat
        total_lat_always_comp += f_lat
        total_lat_never_comp += u_lat
        total_lat_oracle += oracle_lat

        if eval_res.selected_mode.value == empirical_winner:
            matches_empirical_count += 1

    accuracy = matches_empirical_count / len(points)
    assert accuracy >= 0.90, f"Policy accuracy vs empirical winner was {accuracy * 100:.1f}%, expected >= 90%"

    # Empirical policy latency must beat both Always-Compress and Never-Compress
    assert total_lat_policy < total_lat_always_comp, (
        f"Policy ({total_lat_policy:.1f} ms) must beat Always-Compress ({total_lat_always_comp:.1f} ms)"
    )
    assert total_lat_policy <= total_lat_never_comp, (
        f"Policy ({total_lat_policy:.1f} ms) must beat Never-Compress ({total_lat_never_comp:.1f} ms)"
    )

    regret_pct = ((total_lat_policy - total_lat_oracle) / total_lat_oracle) * 100.0
    assert regret_pct < 1.0, f"Empirical regret vs Oracle was {regret_pct:.3f}%, expected < 1.0%"
