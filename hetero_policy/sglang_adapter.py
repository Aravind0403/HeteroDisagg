"""Zero-Fork configuration generator and validator for SGLang."""

from typing import Dict, Any, List
from hetero_policy.schema import ClusterPlacementPlan


def generate_sglang_args(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
    host: str = "0.0.0.0",
    port: int = 30000,
) -> List[str]:
    """
    Generate command-line arguments for launching a native SGLang server instance
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
        "python", "-m", "sglang.launch_server",
        "--model-path", plan.model_name,
        "--host", host,
        "--port", str(port),
        "--tp", str(tp),
        "--chunked-prefill-size", str(assignment.optimal_chunk_size_tokens),
        "--mem-fraction-static", "0.88",
    ]

    return args


def generate_sglang_command(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
    host: str = "0.0.0.0",
    port: int = 30000,
) -> str:
    """Format SGLang launch arguments as a runnable bash command string."""
    args = generate_sglang_args(plan, role=role, host=host, port=port)
    return " ".join(args)


def generate_sglang_config_dict(
    plan: ClusterPlacementPlan,
    role: str = "prefill",
) -> Dict[str, Any]:
    """Generate a structured JSON configuration dict for SGLang."""
    role_lower = role.lower()
    assignment = plan.prefill_assignment if role_lower == "prefill" else plan.decode_assignment
    tp = plan.prefill_tp if role_lower == "prefill" else plan.decode_tp

    return {
        "model_path": plan.model_name,
        "tp": tp,
        "chunked_prefill_size": assignment.optimal_chunk_size_tokens,
        "mem_fraction_static": 0.88,
        "role": role_lower,
        "device_name": assignment.device_name,
    }
