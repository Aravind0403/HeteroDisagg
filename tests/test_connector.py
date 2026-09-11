"""Tests for HeteroKVConnector pipeline transfer and benchmarking."""

import pytest
import torch
from hetero_kv.connector import HeteroKVConnector, TransferMetrics


def test_connector_single_layer():
    torch.manual_seed(42)
    # Llama 3 8B setup: 8 KV heads, prefill_tp=4, decode_tp=1
    connector = HeteroKVConnector(
        num_kv_heads=8,
        prefill_tp=4,
        decode_tp=1,
        enable_fp8=True,
    )

    # 128 tokens, batch size 1
    ground_truth = torch.randn(1, 128, 8, 128, dtype=torch.float32)

    # Slice into prefill shards
    prefill_shards = {}
    for p in range(4):
        prefill_shards[p] = ground_truth[..., p * 2 : (p + 1) * 2, :]

    decode_shards, payloads, quant_ms, dequant_ms = connector.transfer_layer_kv(
        layer_idx=0, prefill_rank_shards=prefill_shards
    )

    assert 0 in decode_shards
    assert decode_shards[0].shape == ground_truth.shape
    assert len(payloads) == 4
    assert quant_ms >= 0
    assert dequant_ms >= 0


def test_connector_benchmark_full_transfer():
    torch.manual_seed(42)
    connector_fp8 = HeteroKVConnector(
        num_kv_heads=8,
        prefill_tp=4,
        decode_tp=2,
        enable_fp8=True,
        interconnect_bandwidth_gb_s=32.0,  # PCIe Gen4
    )

    metrics = connector_fp8.benchmark_full_transfer(
        num_layers=4,
        batch_size=1,
        seq_len=256,
        head_dim=128,
        dtype=torch.float16,
    )

    assert isinstance(metrics, TransferMetrics)
    assert metrics.num_layers == 4
    assert metrics.compression_ratio >= 1.9
    assert metrics.average_cosine_similarity > 0.999
    assert metrics.simulated_wire_latency_ms > 0
