"""Tests for KV head layout and bipartite partition planner."""

import pytest
from hetero_kv.layout import KVPartitionPlan, KVHeadSlice


def test_valid_partition_plan_4_to_1():
    # 8 KV heads, prefill_tp=4, decode_tp=1
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=4, decode_tp=1)
    assert plan.heads_per_prefill_rank == 2
    assert plan.heads_per_decode_rank == 8

    # Prefill slices: rank 0 has [0, 2), rank 1 has [2, 4), rank 2 has [4, 6), rank 3 has [6, 8)
    assert plan.prefill_head_slices[0] == KVHeadSlice(start_head_idx=0, end_head_idx=2)
    assert plan.prefill_head_slices[1] == KVHeadSlice(start_head_idx=2, end_head_idx=4)
    assert plan.prefill_head_slices[2] == KVHeadSlice(start_head_idx=4, end_head_idx=6)
    assert plan.prefill_head_slices[3] == KVHeadSlice(start_head_idx=6, end_head_idx=8)

    # Decode slice: rank 0 has all [0, 8)
    assert plan.decode_head_slices[0] == KVHeadSlice(start_head_idx=0, end_head_idx=8)

    # 4 routes, all going to decode_rank 0
    assert len(plan.transfer_routes) == 4
    for route in plan.transfer_routes:
        assert route.decode_rank == 0


def test_valid_partition_plan_4_to_2():
    # 8 KV heads, prefill_tp=4, decode_tp=2
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=4, decode_tp=2)
    assert plan.heads_per_prefill_rank == 2
    assert plan.heads_per_decode_rank == 4

    # Decode 0 needs heads [0, 4) -> supplied by prefill 0 and 1
    senders_to_0 = plan.get_sender_ranks_for_decode_rank(0)
    assert senders_to_0 == [0, 1]

    # Decode 1 needs heads [4, 8) -> supplied by prefill 2 and 3
    senders_to_1 = plan.get_sender_ranks_for_decode_rank(1)
    assert senders_to_1 == [2, 3]

    # Exactly 4 targeted point-to-point routes (not 4x2=8 full broadcast)
    assert len(plan.transfer_routes) == 4


def test_invalid_gqa_divisibility():
    # 8 KV heads cannot be divided across 3 ranks
    with pytest.raises(ValueError, match="must be evenly divisible"):
        KVPartitionPlan.create(num_kv_heads=8, prefill_tp=3, decode_tp=1)

    with pytest.raises(ValueError, match="must be evenly divisible"):
        KVPartitionPlan.create(num_kv_heads=8, prefill_tp=4, decode_tp=3)


def test_head_slice_intersection():
    slice_a = KVHeadSlice(start_head_idx=0, end_head_idx=4)
    slice_b = KVHeadSlice(start_head_idx=2, end_head_idx=6)
    slice_c = KVHeadSlice(start_head_idx=6, end_head_idx=8)

    assert slice_a.overlaps_with(slice_b)
    inter = slice_a.intersection(slice_b)
    assert inter is not None
    assert inter.start_head_idx == 2
    assert inter.end_head_idx == 4

    assert not slice_a.overlaps_with(slice_c)
    assert slice_a.intersection(slice_c) is None
