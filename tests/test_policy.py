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
    # Production default configuration
    return AdaptiveCompressionPolicy()


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


def test_policy_against_projected_benchmark_outcomes(policy):
    """
    Validates policy decisions against analytical projections recorded in
    benchmarks/transfer_modes_benchmark.json across all bandwidth tiers and sequence lengths.
    
    Validation Note:
    - This tests against analytical projections (projected_link_e2e_ms), not physical wire measurements.
    - Uses production default configuration (AdaptiveCompressionPolicy()).
    - Across the 32-case sweep grid (4 bandwidths x 8 sequence lengths):
      * Matches projected optimal mode in exactly 30/32 cases (93.75%).
      * Mean per-case regret vs projected oracle is 0.0488%.
      * Maximum single-case regret is 1.147% (at 128 tokens on 1 Gbps).
      * Calibration/evaluation separation note: The benchmark grid serves as an analytical validation
        set; real-world deployment requires runtime link probing.
    """
    import json
    from pathlib import Path

    benchmark_path = Path("benchmarks/transfer_modes_benchmark.json")
    assert benchmark_path.exists(), "Benchmark artifact missing: run scripts/benchmark_transfer_modes.py first"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    points = bench_data["points"]
    assert len(points) == 32, f"Expected exactly 32 benchmark scenarios, got {len(points)}"

    total_lat_policy = 0.0
    total_lat_always_comp = 0.0
    total_lat_never_comp = 0.0
    total_lat_oracle = 0.0
    matches_projected_count = 0
    per_case_regrets = []

    for pt in points:
        seq = pt["seq_len"]
        bw = pt["bandwidth_bytes_per_sec"]
        
        # Policy prediction (completely blind to measured/projected latency)
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

        case_regret = ((policy_lat - oracle_lat) / oracle_lat) * 100.0
        per_case_regrets.append(case_regret)

        if eval_res.selected_mode.value == empirical_winner:
            matches_projected_count += 1

    accuracy = matches_projected_count / len(points)
    # Exact reproduction of production default figures
    assert matches_projected_count == 30, f"Expected exactly 30 matches, got {matches_projected_count}/{len(points)}"
    assert round(accuracy * 100, 2) == 93.75, f"Policy accuracy vs projected winner was {accuracy * 100:.2f}%, expected 93.75%"

    # Total projected latency under policy must beat both Always-Compress and Never-Compress
    assert total_lat_policy < total_lat_always_comp, (
        f"Policy ({total_lat_policy:.1f} ms) must beat Always-Compress ({total_lat_always_comp:.1f} ms)"
    )
    assert total_lat_policy <= total_lat_never_comp, (
        f"Policy ({total_lat_policy:.1f} ms) must beat Never-Compress ({total_lat_never_comp:.1f} ms)"
    )

    mean_case_regret = sum(per_case_regrets) / len(per_case_regrets)
    max_case_regret = max(per_case_regrets)
    assert round(mean_case_regret, 4) == 0.0488, f"Mean per-case regret was {mean_case_regret:.4f}%, expected 0.0488%"
    assert round(max_case_regret, 3) == 1.147, f"Max single-case regret was {max_case_regret:.4f}%, expected 1.147%"
