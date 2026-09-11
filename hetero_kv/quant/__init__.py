"""In-flight FP8 wire quantization kernels and utilities."""

from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    compute_cosine_similarity,
    compute_mean_relative_error,
    FP8BlockPayload,
)

__all__ = [
    "quantize_kv_block_fp8",
    "dequantize_kv_block_fp8",
    "compute_cosine_similarity",
    "compute_mean_relative_error",
    "FP8BlockPayload",
]
