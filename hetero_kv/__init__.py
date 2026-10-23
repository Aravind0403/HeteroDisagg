"""HeteroKVConnector: Asymmetric KV cache re-sharding and wire quantization."""

from hetero_kv.layout import KVHeadSlice, BipartiteTransferRoute, KVPartitionPlan
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    FP8BlockPayload,
    compute_cosine_similarity,
    compute_mean_relative_error,
)
from hetero_kv.reshard import (
    KVPayload,
    slice_kv_for_prefill_rank,
    package_payloads_from_prefill_rank,
    reassemble_decode_kv,
)
from hetero_kv.connector import HeteroKVConnector, TransferMetrics, calculate_uncompressed_kv_bytes
from hetero_kv.policy import AdaptiveCompressionPolicy, CompressionMode, PolicyEvaluationResult

from hetero_kv.lifecycle import safe_cleanup_processes, verify_pid_dead

__all__ = [
    "KVHeadSlice",
    "BipartiteTransferRoute",
    "KVPartitionPlan",
    "quantize_kv_block_fp8",
    "dequantize_kv_block_fp8",
    "FP8BlockPayload",
    "compute_cosine_similarity",
    "compute_mean_relative_error",
    "KVPayload",
    "slice_kv_for_prefill_rank",
    "package_payloads_from_prefill_rank",
    "reassemble_decode_kv",
    "HeteroKVConnector",
    "TransferMetrics",
    "calculate_uncompressed_kv_bytes",
    "AdaptiveCompressionPolicy",
    "CompressionMode",
    "PolicyEvaluationResult",
    "safe_cleanup_processes",
    "verify_pid_dead",
]
