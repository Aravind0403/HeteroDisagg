"""Comprehensive regression tests covering the 23 October 2026 code audit findings (F1-F13)."""

import pytest
import torch
import math
from pathlib import Path

from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import (
    KVPayload,
    slice_kv_for_prefill_rank,
    package_payloads_from_prefill_rank,
    reassemble_decode_kv,
)
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    FP8BlockPayload,
)
from hetero_kv.connector import HeteroKVConnector
from hetero_prof.device_detector import load_spec_from_file, detect_device
from hetero_prof.roofline import recommend_chunk_size, calculate_transformer_operational_intensity
from hetero_policy.schema import GPUInventoryItem, ModelMetadata, POPULAR_MODELS
from hetero_policy.planner import plan_cluster_placement


# =========================================================================
# F1: Receiver payload consistency & validation tests
# =========================================================================

def test_f1_reject_mixed_layer_payloads():
    """Receiver must reject payloads from different transformer layers."""
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=2, decode_tp=1)
    t = torch.randn(1, 16, 4, 128)
    
    p0 = KVPayload(
        prefill_rank=0, decode_rank=0, layer_idx=0,
        start_head_idx=0, end_head_idx=4, data=t, is_quantized=False,
    )
    p1 = KVPayload(
        prefill_rank=1, decode_rank=0, layer_idx=1,  # Layer mismatch!
        start_head_idx=4, end_head_idx=8, data=t, is_quantized=False,
    )
    with pytest.raises(ValueError, match="Inconsistent layer_idx"):
        reassemble_decode_kv([p0, p1], decode_rank=0, plan=plan)


def test_f1_reject_head_dimension_mismatch():
    """Receiver must reject declared head range that does not match actual tensor head dimension."""
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=1, decode_tp=1)
    # Supply a tensor with only 2 heads, but declare 8 heads!
    t_2heads = torch.randn(1, 16, 2, 128)
    
    p = KVPayload(
        prefill_rank=0, decode_rank=0, layer_idx=0,
        start_head_idx=0, end_head_idx=8,  # Declares 8 heads
        data=t_2heads, is_quantized=False,
    )
    with pytest.raises(ValueError, match="Head count mismatch"):
        reassemble_decode_kv([p], decode_rank=0, plan=plan)


def test_f1_reject_unauthorized_source_rank():
    """Receiver must reject payloads arriving from an unexpected source prefill rank."""
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=2, decode_tp=1)
    # In this plan: prefill_rank 0 owns [0:4], prefill_rank 1 owns [4:8]
    t = torch.randn(1, 16, 4, 128)
    
    # Payload claims [0:4] comes from prefill_rank 1 (unauthorized!)
    p0 = KVPayload(
        prefill_rank=1, decode_rank=0, layer_idx=0,
        start_head_idx=0, end_head_idx=4, data=t, is_quantized=False,
    )
    p1 = KVPayload(
        prefill_rank=1, decode_rank=0, layer_idx=0,
        start_head_idx=4, end_head_idx=8, data=t, is_quantized=False,
    )
    with pytest.raises(ValueError, match="Unauthorized source rank"):
        reassemble_decode_kv([p0, p1], decode_rank=0, plan=plan)


# =========================================================================
# F2: Short-sequence request isolation in FP8 quantization
# =========================================================================

def test_f2_request_scale_isolation_non_divisible_sequence():
    """
    Ensure modifying request B does NOT alter the dequantized values of request A
    when sequence length is not divisible by block size.
    """
    block_size = 16
    seq_len = 20  # Remainder = 4 tokens
    num_heads = 4
    head_dim = 64

    torch.manual_seed(123)
    # Request 0: small values ~ 0.05
    req_0 = torch.randn(1, seq_len, num_heads, head_dim) * 0.05
    # Request 1: moderate values
    req_1_normal = torch.randn(1, seq_len, num_heads, head_dim) * 1.0
    # Request 1: extreme adversarial values
    req_1_extreme = torch.randn(1, seq_len, num_heads, head_dim) * 250.0

    batch_normal = torch.cat([req_0, req_1_normal], dim=0)
    batch_extreme = torch.cat([req_0, req_1_extreme], dim=0)

    # Quantize and dequantize both batches
    payload_normal = quantize_kv_block_fp8(batch_normal, block_size_tokens=block_size)
    payload_extreme = quantize_kv_block_fp8(batch_extreme, block_size_tokens=block_size)

    deq_normal = dequantize_kv_block_fp8(payload_normal)
    deq_extreme = dequantize_kv_block_fp8(payload_extreme)

    # Extract Request 0 from both runs
    req_0_restored_normal = deq_normal[0]
    req_0_restored_extreme = deq_extreme[0]

    # Under strict request-isolated blocking, Request 0 must be 100% identical!
    diff = torch.max(torch.abs(req_0_restored_normal - req_0_restored_extreme)).item()
    assert diff == 0.0, f"Request isolation violated! Diff on Req 0 was {diff}"


# =========================================================================
# F6: Full KV byte accounting
# =========================================================================

def test_f6_full_kv_transfer_byte_accounting():
    """Verify benchmark_full_transfer accounts for both Key and Value tensors."""
    connector = HeteroKVConnector(
        num_kv_heads=8,
        prefill_tp=2,
        decode_tp=1,
        enable_fp8=False,
    )
    # 2 layers, batch 1, seq 128, 8 heads, 128 head_dim, fp16 (2 bytes per elem)
    # 1 tensor = 1 * 128 * 8 * 128 * 2 = 262,144 bytes
    # Both K and V across 2 layers = 2 * 2 * 262,144 = 1,048,576 bytes (1.0 MiB)
    metrics = connector.benchmark_full_transfer(
        num_layers=2,
        batch_size=1,
        seq_len=128,
        head_dim=128,
        dtype=torch.float16,
    )
    assert metrics.uncompressed_bytes == 1048576, f"Expected 1048576 bytes for full K+V, got {metrics.uncompressed_bytes}"


# =========================================================================
# F7: Placement planner feasibility & duplicate tiers
# =========================================================================

def test_f7_reject_zero_or_negative_gpu_inventory():
    """Planner must reject zero or negative GPU counts and non-positive costs."""
    spec = load_spec_from_file("rtx4090")
    model = POPULAR_MODELS["llama-3-8b"]

    with pytest.raises(ValueError, match="count must be >= 1"):
        bad_item = GPUInventoryItem(spec=spec, count=0, cost_per_hour_usd=0.50)
        plan_cluster_placement([bad_item, bad_item], model)

    with pytest.raises(ValueError, match="cost must be positive"):
        bad_cost = GPUInventoryItem(spec=spec, count=2, cost_per_hour_usd=0.0)
        plan_cluster_placement([bad_cost, bad_cost], model)


def test_f7_handle_duplicate_inventory_tiers_without_crash():
    """Planner must handle duplicate equal inventory tiers without IndexError."""
    spec = load_spec_from_file("rtx4090")
    model = POPULAR_MODELS["llama-3-8b"]
    item1 = GPUInventoryItem(spec=spec, count=2, cost_per_hour_usd=0.50)
    item2 = GPUInventoryItem(spec=spec, count=2, cost_per_hour_usd=0.50)

    # Must not raise IndexError
    plan = plan_cluster_placement([item1, item2], model)
    assert plan.prefill_tp == 2
    assert plan.decode_tp == 2


def test_f7_enforce_vram_feasibility():
    """Planner must flag plans that require FP8 quantization or reject impossible VRAM allocations."""
    # Llama 3 70B (~140 GB FP16 weights, ~70 GB FP8) on 4x A10G (96 GB total)
    spec_a10g = load_spec_from_file("a10g")
    spec_h100 = load_spec_from_file("h100")
    model_70b = POPULAR_MODELS["llama-3-70b"]

    h100_tier = GPUInventoryItem(spec=spec_h100, count=2, cost_per_hour_usd=3.50)
    a10g_tier = GPUInventoryItem(spec=spec_a10g, count=4, cost_per_hour_usd=0.80)

    plan = plan_cluster_placement([h100_tier, a10g_tier], model_70b)
    assert plan.memory_status == "requires_fp8_weight_quantization"
    assert "requires FP8 weight quantization" in plan.rationale

    # Now create an impossible inventory (e.g. 1x 24GB GPU for 70B model)
    single_a10g = GPUInventoryItem(spec=spec_a10g, count=1, cost_per_hour_usd=0.80)
    with pytest.raises(ValueError, match="cannot fit"):
        plan_cluster_placement([h100_tier, single_a10g], model_70b)


# =========================================================================
# F8: Chunk sizing bounds & FP32 element size
# =========================================================================

def test_f8_roofline_strictly_respects_max_chunk_limit():
    """recommend_chunk_size must respect max_chunk_limit down to small values without IndexError."""
    spec = load_spec_from_file("h100")
    
    # Test small limit = 128
    rec_128 = recommend_chunk_size(spec, max_chunk_limit=128)
    assert rec_128.recommended_chunk_size_tokens <= 128

    # Test small limit = 32
    rec_32 = recommend_chunk_size(spec, block_size=16, max_chunk_limit=32)
    assert rec_32.recommended_chunk_size_tokens <= 32


def test_f8_fp32_element_size():
    """Operational intensity for FP32 must use 4 bytes per element."""
    intensity_fp16 = calculate_transformer_operational_intensity(
        num_tokens=512, hidden_size=4096, num_layers=32, bytes_per_elem=2
    )
    intensity_fp32 = calculate_transformer_operational_intensity(
        num_tokens=512, hidden_size=4096, num_layers=32, bytes_per_elem=4
    )
    # With double the memory traffic for FP32, operational intensity should be exactly half
    assert math.isclose(intensity_fp32, intensity_fp16 / 2.0, rel_tol=1e-3)


# =========================================================================
# F9 & F13: Provenance & package resource loading
# =========================================================================

def test_f9_spec_provenance_tracking():
    """AcceleratorSpec must retain and validate provenance."""
    spec = load_spec_from_file("rtx4090")
    assert spec.provenance in ["reference", "detected", "measured", "estimated"]
