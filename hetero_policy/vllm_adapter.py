"""Zero-Fork configuration generator and validator for vLLM."""

from typing import Dict, Any, List
from hetero_policy.schema import ClusterPlacementPlan


def generate_vllm_args(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> List[str]:
    """
    Generate command-line arguments for launching a native vLLM instance
    customized for the target node's hardware role.
    """
    role_lower = role.lower()
    if role_lower == "prefill":
        assignment = plan.prefill_assignment
        tp = plan.prefill_tp
    elif role_lower == "decode":
        assignment = plan.decode_assignment
        tp = plan.decode_tp
    else:
        raise ValueError(f"Unknown role '{role}', choose 'prefill' or 'decode'.")

    args = [
        "vllm", "serve", plan.model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--max-num-batched-tokens", str(assignment.optimal_chunk_size_tokens),
        "--enable-chunked-prefill", "True",
        "--block-size", "16",
        "--gpu-memory-utilization", "0.90",
    ]

    # If prefill_tp != decode_tp, configure FP8 wire KV cache
    if plan.prefill_tp != plan.decode_tp:
        args.extend(["--kv-cache-dtype", "fp8_e4m3"])

    return args


def generate_vllm_command(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> str:
    """Format vLLM launch arguments as a runnable bash command string."""
    args = generate_vllm_args(plan, role=role, host=host, port=port)
    return " ".join(args)


def generate_vllm_config_dict(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
) -> Dict[str, Any]:
    """Generate a structured JSON configuration dict for vLLM."""
    role_lower = role.lower()
    assignment = plan.prefill_assignment if role_lower == "prefill" else plan.decode_assignment
    tp = plan.prefill_tp if role_lower == "prefill" else plan.decode_tp

    return {
        "model": plan.model_name,
        "tensor_parallel_size": tp,
        "max_num_batched_tokens": assignment.optimal_chunk_size_tokens,
        "enable_chunked_prefill": True,
        "block_size": 16,
        "gpu_memory_utilization": 0.90,
        "kv_cache_dtype": "fp8_e4m3" if plan.prefill_tp != plan.decode_tp else "auto",
        "role": role_lower,
        "device_name": assignment.device_name,
    }
