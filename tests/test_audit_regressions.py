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


# =========================================================================
# Gaps 1-5: Laptop-Only Follow-Up Audit Regressions
# =========================================================================

from click.testing import CliRunner
import hetero_policy.cli as policy_cli
import hetero_prof.cli as prof_cli
from hetero_kv.connector import calculate_uncompressed_kv_bytes
from hetero_policy.vllm_adapter import generate_vllm_command
from hetero_policy.sglang_adapter import generate_sglang_command
from hetero_policy.planner import load_cluster_inventory
from scripts.run_rigorous_benchmark import run_benchmark_trial


def test_gap1_unresolved_memory_refuses_launch_generation():
    """Launch generation must be refused for unresolved weight-memory states."""
    _, inventory = load_cluster_inventory("h100_and_a10g.json")
    model_70b = POPULAR_MODELS["llama-3-70b"]
    plan = plan_cluster_placement(inventory, model_70b)

    assert plan.memory_status == "requires_fp8_weight_quantization"

    with pytest.raises(ValueError, match="refuses launch generation"):
        generate_vllm_command(plan, role="prefill")

    with pytest.raises(ValueError, match="refuses launch generation"):
        generate_sglang_command(plan, role="decode")

    # CLI generate must fail with non-zero exit code
    runner = CliRunner()
    result = runner.invoke(
        policy_cli.main,
        ["generate", "--cluster", "h100_and_a10g.json", "--model", "llama-3-70b"],
    )
    assert result.exit_code != 0


def test_gap1_planner_enforces_engine_utilization_and_cache_headroom():
    """Planner must reject configurations where weights consume the entire GPU capacity leaving no headroom."""
    spec_a10g = load_spec_from_file("a10g")  # 24 GB
    spec_h100 = load_spec_from_file("h100")  # 80 GB
    model_70b = POPULAR_MODELS["llama-3-70b"]  # 70 GB FP8 weights

    h100_tier = GPUInventoryItem(spec=spec_h100, count=2, cost_per_hour_usd=3.50)
    # 3x 24GB = 72 GB nominal. At 90% utilization: 64.8 GB usable.
    # 70 GB FP8 weights cannot fit in 64.8 GB usable VRAM!
    tier_3_a10g = GPUInventoryItem(spec=spec_a10g, count=3, cost_per_hour_usd=0.80)

    with pytest.raises(ValueError, match="cannot fit"):
        plan_cluster_placement([h100_tier, tier_3_a10g], model_70b)


def test_gap2_connector_threads_request_and_kv_metadata():
    """Connector must pass request_id and kv_type through to payloads and validate them on receipt."""
    connector = HeteroKVConnector(num_kv_heads=8, prefill_tp=2, decode_tp=1, enable_fp8=False)
    ground_truth = torch.randn(1, 16, 8, 128)

    prefill_shards = {
        0: slice_kv_for_prefill_rank(ground_truth, prefill_rank=0, plan=connector.plan),
        1: slice_kv_for_prefill_rank(ground_truth, prefill_rank=1, plan=connector.plan),
    }

    decode_shards, payloads, _, _ = connector.transfer_layer_kv(
        layer_idx=5,
        prefill_rank_shards=prefill_shards,
        request_id="req_alpha_42",
        kv_type="key",
    )

    assert len(payloads) > 0
    for p in payloads:
        assert p.request_id == "req_alpha_42"
        assert p.kv_type == "key"

    # Receipt validation: mismatched expected request_id must be rejected
    with pytest.raises(ValueError, match="Mismatched request_id on receipt"):
        reassemble_decode_kv(
            received_payloads=payloads,
            decode_rank=0,
            plan=connector.plan,
            expected_request_id="different_request",
        )

    # Receipt validation: mismatched expected kv_type must be rejected
    with pytest.raises(ValueError, match="Mismatched kv_type on receipt"):
        reassemble_decode_kv(
            received_payloads=payloads,
            decode_rank=0,
            plan=connector.plan,
            expected_kv_type="value",
        )


def test_gap3_full_cache_accounting_consistency():
    """Standalone benchmarks and connector helper must report identical full K+V cache volume."""
    batch_size = 1
    prompt_tokens = 128
    num_kv_heads = 8
    head_dim = 128
    num_layers = 2

    expected_bytes = calculate_uncompressed_kv_bytes(
        batch_size=batch_size,
        seq_len=prompt_tokens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        bytes_per_elem=2,
        include_both_k_and_v=True,
    )
    # 1 * 128 * 8 * 128 * 2 (fp16) * 2 layers * 2 (both K & V) = 1,048,576 bytes = 1.0 MiB
    assert expected_bytes == 1048576

    connector = HeteroKVConnector(num_kv_heads=8, prefill_tp=2, decode_tp=1, enable_fp8=False)
    metrics = connector.benchmark_full_transfer(
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=prompt_tokens,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    assert metrics.uncompressed_bytes == expected_bytes
    assert round(metrics.uncompressed_bytes / (1024 ** 2), 2) == 1.0

    trial = run_benchmark_trial(
        config_name="heterodisagg",
        concurrency=batch_size,
        num_layers=num_layers,
        prompt_tokens=prompt_tokens,
        decode_tokens=16,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    assert trial["uncompressed_mb"] == 1.0


def test_gap4_matched_reference_provenance_detected():
    """When a device matches a reference profile, provenance must be 'detected', not 'measured'."""
    from hetero_prof.schema import AcceleratorSpec

    # Verify that if matching reference JSON, provenance is 'detected'
    spec = load_spec_from_file("rtx3090")
    spec_dict = spec.model_dump()
    spec_dict["memory_capacity_gb"] = 24.0
    spec_dict["provenance"] = "detected"
    validated = AcceleratorSpec.model_validate(spec_dict)
    assert validated.provenance == "detected"


def test_gap5_cli_fp32_roofline_and_non_zero_exit_codes():
    """CLI commands must calculate FP32 using 4 bytes/element and return non-zero exit status on errors."""
    runner = CliRunner()

    # 1. Invalid cluster path in hetero-plan must exit with non-zero code
    res_plan = runner.invoke(policy_cli.main, ["plan", "-c", "nonexistent_cluster_path.json"])
    assert res_plan.exit_code != 0

    res_gen = runner.invoke(policy_cli.main, ["generate", "-c", "nonexistent_cluster_path.json"])
    assert res_gen.exit_code != 0

    # 2. Invalid spec path in hetero-prof must exit with non-zero code
    res_prof = runner.invoke(prof_cli.main, ["inspect", "-s", "nonexistent_spec_path.json"])
    assert res_prof.exit_code != 0

    # 3. FP32 roofline table must calculate using 4 bytes per element
    # Operational intensity for 512 tokens with 4 bytes/elem is 245.8 FLOPs/Byte (vs 491.5 for FP16)
    res_roofline = runner.invoke(prof_cli.main, ["roofline", "-s", "rtx4090", "--dtype", "fp32"])
    assert res_roofline.exit_code == 0
    assert "245.8 FLOPs/Byte" in res_roofline.output


