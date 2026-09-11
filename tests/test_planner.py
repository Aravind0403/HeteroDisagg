"""Tests for heuristic cluster placement planner."""

import pytest
from hetero_policy.planner import load_cluster_inventory, plan_cluster_placement
from hetero_policy.schema import POPULAR_MODELS, ClusterPlacementPlan


def test_load_cluster_inventory():
    cluster_name, inventory = load_cluster_inventory("l40s_and_3090.json")
    assert len(inventory) == 2
    assert any("L40S" in item.spec.device_name for item in inventory)
    assert any("3090" in item.spec.device_name for item in inventory)


def test_plan_l40s_and_3090_placement():
    cluster_name, inventory = load_cluster_inventory("l40s_and_3090.json")
    model = POPULAR_MODELS["llama-3-8b"]

    plan = plan_cluster_placement(inventory, model, cluster_name=cluster_name)

    assert isinstance(plan, ClusterPlacementPlan)
    # L40S has higher Compute Efficiency (TFLOPS/$) -> Assigned to Prefill
    assert "L40S" in plan.prefill_assignment.device_name
    assert plan.prefill_tp == 2

    # RTX 3090 has higher Bandwidth Efficiency (GB/s/$) -> Assigned to Decode
    assert "3090" in plan.decode_assignment.device_name
    assert plan.decode_tp in [1, 2]

    # Total cost = 2 * 1.20 + 2 * 0.30 = $3.00
    assert plan.total_cluster_cost_per_hour_usd == 3.00


def test_plan_h100_and_a10g_placement():
    cluster_name, inventory = load_cluster_inventory("h100_and_a10g.json")
    model = POPULAR_MODELS["llama-3-70b"]

    plan = plan_cluster_placement(inventory, model, cluster_name=cluster_name)

    # H100 SXM5 assigned to Prefill
    assert "H100" in plan.prefill_assignment.device_name
    assert plan.prefill_tp == 2

    # A10G assigned to Decode
    assert "A10G" in plan.decode_assignment.device_name
    # 4 A10Gs, 8 KV heads -> decode_tp=4
    assert plan.decode_tp == 4


def test_qwen_gqa_divisibility():
    # Qwen-2.5-7B has only 4 KV heads
    cluster_name, inventory = load_cluster_inventory("l40s_and_3090.json")
    model = POPULAR_MODELS["qwen-2.5-7b"]

    plan = plan_cluster_placement(inventory, model, cluster_name=cluster_name)
    assert plan.num_kv_heads == 4
    assert 4 % plan.prefill_tp == 0
    assert 4 % plan.decode_tp == 0
