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


def test_policy_generalization_against_oracle_and_baselines(policy):
    """
    Evaluates policy across an unseen test grid (held-out from calibration):
    - Sequence lengths: 8, 16, 64, 256, 1024, 2048
    - Bandwidths: 500 Mbps, 1 Gbps, 10 Gbps, 25 Gbps, 100 Gbps
    - Model: 32 layers, 8 heads, head_dim 128 (Llama-style 7B dimensions)
    
    Compares total latency of Policy vs Always-Compress, Never-Compress, and Oracle.
    """
    test_seq_lens = [8, 16, 64, 256, 1024, 2048]
    test_bandwidths = [
        62.5e6,   # 500 Mbps
        125e6,    # 1 Gbps
        1.25e9,   # 10 Gbps
        3.125e9,  # 25 Gbps
        12.5e9,   # 100 Gbps
    ]

    total_latency_policy = 0.0
    total_latency_always_comp = 0.0
    total_latency_never_comp = 0.0
    total_latency_oracle = 0.0

    matches_oracle_count = 0
    total_scenarios = len(test_seq_lens) * len(test_bandwidths)

    for seq in test_seq_lens:
        for bw in test_bandwidths:
            eval_res = policy.evaluate(
                seq_len=seq,
                num_layers=32,
                num_kv_heads=8,
                head_dim=128,
                link_bandwidth_bytes_per_sec=bw,
                block_size_tokens=16,
            )

            lat_uncomp = eval_res.estimated_uncompressed_latency_ms
            lat_comp = eval_res.estimated_compressed_latency_ms

            # Oracle selects the strictly minimum latency
            lat_oracle = min(lat_uncomp, lat_comp)
            oracle_mode = CompressionMode.UNCOMPRESSED_FP16 if lat_uncomp <= lat_comp else CompressionMode.BLOCK_FP8

            # Policy selected latency
            lat_policy = lat_comp if eval_res.selected_mode == CompressionMode.BLOCK_FP8 else lat_uncomp

            total_latency_policy += lat_policy
            total_latency_always_comp += lat_comp
            total_latency_never_comp += lat_uncomp
            total_latency_oracle += lat_oracle

            if eval_res.selected_mode == oracle_mode:
                matches_oracle_count += 1

    # Policy should match Oracle in >= 90% of scenarios
    accuracy = matches_oracle_count / total_scenarios
    assert accuracy >= 0.90, f"Policy accuracy vs Oracle was {accuracy * 100:.1f}%, expected >= 90%"

    # Total latency under policy should beat both Always-Compress and Never-Compress
    assert total_latency_policy < total_latency_always_comp, (
        f"Policy ({total_latency_policy:.1f} ms) must outperform Always-Compress ({total_latency_always_comp:.1f} ms)"
    )
    assert total_latency_policy < total_latency_never_comp, (
        f"Policy ({total_latency_policy:.1f} ms) must outperform Never-Compress ({total_latency_never_comp:.1f} ms)"
    )
    # Policy regret vs Oracle should be small (< 5% overhead)
    regret_pct = ((total_latency_policy - total_latency_oracle) / total_latency_oracle) * 100.0
    assert regret_pct < 5.0, f"Policy regret vs Oracle is {regret_pct:.2f}%, expected < 5%"
