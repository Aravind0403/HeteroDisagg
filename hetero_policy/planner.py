"""Heuristic cluster placement planner for heterogeneous disaggregated serving."""

import json
from pathlib import Path
from typing import List, Tuple
from hetero_prof.device_detector import load_spec_from_file
from hetero_prof.roofline import recommend_chunk_size
from hetero_policy.schema import (
    GPUInventoryItem,
    ModelMetadata,
    RoleAssignment,
    ClusterPlacementPlan,
    POPULAR_MODELS,
)


def load_cluster_inventory(cluster_path: str | Path) -> Tuple[str, List[GPUInventoryItem]]:
    """Load cluster inventory from JSON file."""
    path = Path(cluster_path)
    if not path.exists():
        # Check configs/clusters/
        candidate = Path(__file__).resolve().parent.parent / "configs" / "clusters" / cluster_path
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Cluster config not found: {cluster_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cluster_name = data.get("cluster_name", "HeteroCluster")
    items: List[GPUInventoryItem] = []
    for raw in data.get("inventory", []):
        spec = load_spec_from_file(raw["device_spec"])
        items.append(
            GPUInventoryItem(
                spec=spec,
                count=int(raw["count"]),
                cost_per_hour_usd=float(raw["cost_per_hour_usd"]),
            )
        )
    return cluster_name, items


def _find_best_tp_degree(available_count: int, num_kv_heads: int) -> int:
    """
    Find the largest valid Tensor Parallelism degree <= available_count
    that evenly divides num_kv_heads.
    """
    for tp in [8, 4, 2, 1]:
        if tp <= available_count and num_kv_heads % tp == 0:
            return tp
    return 1


def plan_cluster_placement(
    inventory: List[GPUInventoryItem],
    model: ModelMetadata,
    cluster_name: str = "HeteroCluster",
) -> ClusterPlacementPlan:
    """
    Given a cluster inventory and a target model, deterministically assign
    accelerator tiers to Prefill vs. Decode roles and determine TP degrees.
    """
    if not inventory:
        raise ValueError("Cannot plan placement for an empty GPU inventory.")

    if len(inventory) == 1:
        # Single tier cluster: split GPUs between Prefill and Decode
        item = inventory[0]
        if item.count < 2:
            raise ValueError(f"Need at least 2 GPUs to form disaggregated prefill/decode pools, got {item.count}.")
        prefill_count = item.count // 2
        decode_count = item.count - prefill_count
        prefill_item = GPUInventoryItem(spec=item.spec, count=prefill_count, cost_per_hour_usd=item.cost_per_hour_usd)
        decode_item = GPUInventoryItem(spec=item.spec, count=decode_count, cost_per_hour_usd=item.cost_per_hour_usd)
    else:
        # Multi-tier cluster: sort by compute efficiency (TFLOPS/$) and bandwidth efficiency (GB/s/$)
        # Tier with highest Compute Efficiency gets assigned to Prefill
        sorted_by_compute = sorted(
            inventory, key=lambda x: x.compute_efficiency_tflops_per_dollar, reverse=True
        )
        prefill_item = sorted_by_compute[0]

        # Remaining tiers evaluated for Bandwidth Efficiency
        remaining_tiers = [x for x in inventory if x != prefill_item]
        sorted_by_bandwidth = sorted(
            remaining_tiers, key=lambda x: x.bandwidth_efficiency_gb_s_per_dollar, reverse=True
        )
        decode_item = sorted_by_bandwidth[0]

    # Calculate valid TP degrees respecting GQA head divisibility
    prefill_tp = _find_best_tp_degree(prefill_item.count, model.num_kv_heads)
    decode_tp = _find_best_tp_degree(decode_item.count, model.num_kv_heads)

    # Derive optimal chunk size for the prefill accelerator
    chunk_rec = recommend_chunk_size(
        prefill_item.spec,
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        dtype="fp16",
    )

    prefill_hourly_cost = prefill_item.count * prefill_item.cost_per_hour_usd
    decode_hourly_cost = decode_item.count * decode_item.cost_per_hour_usd
    total_cost = prefill_hourly_cost + decode_hourly_cost

    prefill_assignment = RoleAssignment(
        role="prefill",
        device_name=prefill_item.spec.device_name,
        num_gpus=prefill_item.count,
        tp_degree=prefill_tp,
        optimal_chunk_size_tokens=chunk_rec.recommended_chunk_size_tokens,
        hourly_cost_usd=round(prefill_hourly_cost, 2),
    )

    decode_assignment = RoleAssignment(
        role="decode",
        device_name=decode_item.spec.device_name,
        num_gpus=decode_item.count,
        tp_degree=decode_tp,
        optimal_chunk_size_tokens=chunk_rec.recommended_chunk_size_tokens,
        hourly_cost_usd=round(decode_hourly_cost, 2),
    )

    rationale = (
        f"Assigned {prefill_item.count}x {prefill_item.spec.device_name} to Prefill (prefill_tp={prefill_tp}) "
        f"yielding {prefill_item.compute_efficiency_tflops_per_dollar:.1f} TFLOPS/$ for compute-heavy prompt encoding. "
        f"Assigned {decode_item.count}x {decode_item.spec.device_name} to Decode (decode_tp={decode_tp}) "
        f"yielding {decode_item.bandwidth_efficiency_gb_s_per_dollar:.1f} GB/s/$ for memory-bound token generation. "
        f"Hardware-derived chunk size set to {chunk_rec.recommended_chunk_size_tokens} tokens."
    )

    return ClusterPlacementPlan(
        cluster_name=cluster_name,
        model_name=model.model_name,
        prefill_assignment=prefill_assignment,
        decode_assignment=decode_assignment,
        total_cluster_cost_per_hour_usd=round(total_cost, 2),
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        num_kv_heads=model.num_kv_heads,
        rationale=rationale,
    )
