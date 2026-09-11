"""Tests for Zero-Fork vLLM and SGLang engine adapters."""

import pytest
from hetero_policy.planner import load_cluster_inventory, plan_cluster_placement
from hetero_policy.schema import POPULAR_MODELS
from hetero_policy.vllm_adapter import (
    generate_vllm_args,
    generate_vllm_command,
    generate_vllm_config_dict,
)
from hetero_policy.sglang_adapter import (
    generate_sglang_args,
    generate_sglang_command,
    generate_sglang_config_dict,
)


def test_vllm_adapter_prefill_and_decode():
    _, inventory = load_cluster_inventory("l40s_and_3090.json")
    model = POPULAR_MODELS["llama-3-8b"]
    plan = plan_cluster_placement(inventory, model)

    prefill_args = generate_vllm_args(plan, role="prefill", port=8000)
    assert "vllm" in prefill_args
    assert "--tensor-parallel-size" in prefill_args
    tp_idx = prefill_args.index("--tensor-parallel-size")
    assert prefill_args[tp_idx + 1] == str(plan.prefill_tp)

    assert "--max-num-batched-tokens" in prefill_args
    chunk_idx = prefill_args.index("--max-num-batched-tokens")
    assert prefill_args[chunk_idx + 1] == str(plan.prefill_assignment.optimal_chunk_size_tokens)

    # Decode args
    decode_cmd = generate_vllm_command(plan, role="decode", port=8001)
    assert "--port 8001" in decode_cmd

    # Config dict
    config = generate_vllm_config_dict(plan, role="prefill")
    assert config["role"] == "prefill"
    assert config["block_size"] == 16


def test_sglang_adapter():
    _, inventory = load_cluster_inventory("l40s_and_3090.json")
    model = POPULAR_MODELS["llama-3-8b"]
    plan = plan_cluster_placement(inventory, model)

    sglang_args = generate_sglang_args(plan, role="prefill", port=30000)
    assert "sglang.launch_server" in " ".join(sglang_args)
    assert "--chunked-prefill-size" in sglang_args
    assert "--tp" in sglang_args

    sglang_cmd = generate_sglang_command(plan, role="decode", port=30001)
    assert "--port 30001" in sglang_cmd
