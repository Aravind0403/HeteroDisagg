"""Tests for asymmetric slicing and reassembly across ranks."""

import pytest
import torch
from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import (
    slice_kv_for_prefill_rank,
    package_payloads_from_prefill_rank,
    reassemble_decode_kv,
)
from hetero_kv.quant.fp8_block import compute_cosine_similarity


def test_reshard_exact_match_fp16():
    torch.manual_seed(42)
    # Llama 3 8B dimensions: 8 KV heads, head_dim 128
    # 512 tokens, batch size 2
    full_kv = torch.randn(2, 512, 8, 128, dtype=torch.float32)

    # Setup: prefill_tp=4 -> decode_tp=2
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=4, decode_tp=2)

    # Prefill side: carve full tensor into 4 rank slices, then package without quantization
    all_payloads = []
    for p_rank in range(4):
        p_slice = slice_kv_for_prefill_rank(full_kv, prefill_rank=p_rank, plan=plan)
        payloads = package_payloads_from_prefill_rank(
            p_slice, prefill_rank=p_rank, layer_idx=0, plan=plan, enable_fp8=False
        )
        all_payloads.extend(payloads)

    # Decode side: reassemble for decode_rank 0 and decode_rank 1
    decode_0 = reassemble_decode_kv(all_payloads, decode_rank=0, plan=plan)
    decode_1 = reassemble_decode_kv(all_payloads, decode_rank=1, plan=plan)

    # In FP16 without quantization, reassembly must match ground truth bit-for-bit
    assert torch.equal(decode_0, full_kv[..., 0:4, :])
    assert torch.equal(decode_1, full_kv[..., 4:8, :])


def test_reshard_fp8_accuracy_4_to_1():
    torch.manual_seed(42)
    full_kv = torch.randn(1, 256, 8, 128, dtype=torch.float32)

    # Setup: prefill_tp=4 -> decode_tp=1
    plan = KVPartitionPlan.create(num_kv_heads=8, prefill_tp=4, decode_tp=1)

    all_payloads = []
    for p_rank in range(4):
        p_slice = slice_kv_for_prefill_rank(full_kv, prefill_rank=p_rank, plan=plan)
        payloads = package_payloads_from_prefill_rank(
            p_slice, prefill_rank=p_rank, layer_idx=0, plan=plan, enable_fp8=True
        )
        all_payloads.extend(payloads)

    decode_0 = reassemble_decode_kv(all_payloads, decode_rank=0, plan=plan)

    # Decode rank 0 should have all 8 heads
    assert decode_0.shape == full_kv.shape
    sim = compute_cosine_similarity(full_kv, decode_0)
    assert sim > 0.999
