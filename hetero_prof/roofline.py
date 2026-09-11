"""Roofline modeling and hardware-adaptive chunked prefill sizing."""

import math
from typing import Optional, List
from hetero_prof.schema import AcceleratorSpec, RooflinePoint, ChunkSizingRecommendation


def calculate_inflection_point(peak_tflops: float, memory_bandwidth_gb_s: float) -> float:
    """
    Calculate the arithmetic intensity inflection point in FLOPs per Byte:
    inflection_point = (peak_tflops * 1000) / memory_bandwidth_gb_s
    """
    if memory_bandwidth_gb_s <= 0:
        raise ValueError("Memory bandwidth must be greater than zero.")
    return (peak_tflops * 1000.0) / memory_bandwidth_gb_s


def evaluate_roofline(
    spec: AcceleratorSpec,
    precision: str = "fp16",
    workload_intensity: Optional[float] = None,
) -> RooflinePoint:
    """
    Evaluate an accelerator's roofline ceiling and classify a workload's execution regime.
    """
    inflection = spec.get_inflection_point_flops_per_byte(precision)
    precision_lower = precision.lower()
    if precision_lower == "fp32":
        peak_tflops = spec.compute_tflops.fp32
    elif precision_lower == "fp16":
        peak_tflops = spec.compute_tflops.fp16
    elif precision_lower == "bf16":
        peak_tflops = spec.compute_tflops.bf16
    elif precision_lower == "fp8":
        if spec.compute_tflops.fp8 is None:
            raise ValueError(f"FP8 compute is not supported on {spec.device_name}.")
        peak_tflops = spec.compute_tflops.fp8
    else:
        raise ValueError(f"Unknown precision format: {precision}")

    regime = None
    if workload_intensity is not None:
        if workload_intensity < inflection:
            regime = "memory_bandwidth_bound"
        else:
            regime = "compute_bound"

    return RooflinePoint(
        precision=precision,
        peak_tflops=peak_tflops,
        memory_bandwidth_gb_s=spec.memory_bandwidth_gb_s,
        inflection_point_flops_per_byte=round(inflection, 2),
        workload_operational_intensity=round(workload_intensity, 2) if workload_intensity else None,
        execution_regime=regime,
    )


def calculate_transformer_operational_intensity(
    num_tokens: int,
    hidden_size: int = 4096,
    num_layers: int = 32,
    bytes_per_elem: int = 2,
) -> float:
    """
    Calculate the operational intensity (FLOPs / Byte) of prefilling a chunk of tokens
    through a Transformer model.

    In a standard Transformer block:
    - Matrix multiplications (Q, K, V, Output projections + FFN Gate, Up, Down projections):
      Total FLOPs per token per layer ~ 24 * (hidden_size ^ 2)
    - Total FLOPs for chunk = num_tokens * num_layers * 24 * (hidden_size ^ 2)

    Memory traffic:
    - Weight bytes loaded (amortized once per forward pass across all tokens in the chunk):
      weights_bytes = 12 * (hidden_size ^ 2) * num_layers * bytes_per_elem
    - Activation bytes read/written per token:
      activations_bytes = num_tokens * num_layers * hidden_size * bytes_per_elem * 4
    - Total bytes = weights_bytes + activations_bytes

    As num_tokens increases, the fixed weight streaming cost is amortized, driving
    operational intensity up until it asymptotes toward the activation ceiling.
    """
    if num_tokens <= 0:
        return 0.0

    total_flops = float(num_tokens) * float(num_layers) * 24.0 * (float(hidden_size) ** 2)
    weight_bytes = 12.0 * (float(hidden_size) ** 2) * float(num_layers) * float(bytes_per_elem)
    activation_bytes = float(num_tokens) * float(num_layers) * float(hidden_size) * float(bytes_per_elem) * 4.0
    total_bytes = weight_bytes + activation_bytes

    return total_flops / total_bytes


def recommend_chunk_size(
    spec: AcceleratorSpec,
    hidden_size: int = 4096,
    num_layers: int = 32,
    block_size: int = 16,
    dtype: str = "fp16",
    max_chunk_limit: int = 8192,
) -> ChunkSizingRecommendation:
    """
    Calculate the hardware-derived optimal prefill chunk size (in tokens).

    Finds the smallest token chunk size where the model's operational intensity
    reaches or exceeds the accelerator's roofline inflection point, ensuring the
    GPU operates in its high-efficiency compute-bound regime.
    """
    bytes_per_elem = 1 if dtype.lower() == "fp8" else 2
    inflection = spec.get_inflection_point_flops_per_byte(dtype)

    candidate_sizes: List[int] = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    candidate_sizes = [c for c in candidate_sizes if c <= max_chunk_limit]

    # Find the smallest chunk size that reaches at least 85% of inflection point
    saturation_threshold = inflection * 0.85
    optimal_tokens = candidate_sizes[-1]
    min_saturation_tokens = candidate_sizes[-1]

    found_min = False
    for tokens in range(block_size, max_chunk_limit + 1, block_size):
        intensity = calculate_transformer_operational_intensity(
            tokens, hidden_size=hidden_size, num_layers=num_layers, bytes_per_elem=bytes_per_elem
        )
        if not found_min and intensity >= saturation_threshold:
            min_saturation_tokens = tokens
            found_min = True

        if intensity >= inflection:
            # Found full compute saturation
            optimal_tokens = tokens
            break

    # Snap to nearest power-of-two or multiple of block_size
    # Ensure minimum reasonable chunk (e.g. 512 tokens) for modern GPUs
    final_recommended = max(512, min_saturation_tokens)
    # Align to block_size
    final_recommended = ((final_recommended + block_size - 1) // block_size) * block_size

    rationale = (
        f"For {spec.device_name} running {dtype.upper()}, the inflection point is "
        f"{inflection:.1f} FLOPs/Byte. A prefill chunk size of {final_recommended} tokens "
        f"achieves sufficient arithmetic intensity to saturate the {spec.num_sms} SMs "
        f"without causing excessive activation memory overhead."
    )

    return ChunkSizingRecommendation(
        accelerator_name=spec.device_name,
        precision=dtype,
        inflection_point_flops_per_byte=round(inflection, 1),
        recommended_chunk_size_tokens=final_recommended,
        minimum_saturation_chunk_tokens=min_saturation_tokens,
        rationale=rationale,
    )
