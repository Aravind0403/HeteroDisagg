"""Asymmetric KV cache slicing, serialization, and reassembly across mismatched ranks."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import torch

from hetero_kv.layout import KVPartitionPlan, KVHeadSlice
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    FP8BlockPayload,
)


@dataclass
class KVPayload:
    """Wire packet holding a KV cache slice and its routing metadata."""
    prefill_rank: int
    decode_rank: int
    layer_idx: int
    start_head_idx: int
    end_head_idx: int
    data: Union[torch.Tensor, FP8BlockPayload]
    is_quantized: bool
    request_id: Optional[str] = None
    kv_type: Optional[str] = None  # e.g., 'key' or 'value'

    @property
    def num_heads(self) -> int:
        return self.end_head_idx - self.start_head_idx

    @property
    def payload_bytes(self) -> int:
        if self.is_quantized and isinstance(self.data, FP8BlockPayload):
            return self.data.payload_bytes
        elif isinstance(self.data, torch.Tensor):
            return self.data.nbytes
        return 0


def slice_kv_for_prefill_rank(
    full_kv_tensor: torch.Tensor,
    prefill_rank: int,
    plan: KVPartitionPlan,
) -> torch.Tensor:
    """
    Extract the local KV head slice belonging to a given prefill rank.
    Assumes head dimension is at index -2 (standard in [..., num_heads, head_dim]).
    """
    if prefill_rank not in plan.prefill_head_slices:
        raise ValueError(f"prefill_rank {prefill_rank} is not in partition plan (prefill_tp={plan.prefill_tp})")

    head_slice = plan.prefill_head_slices[prefill_rank]
    return full_kv_tensor[..., head_slice.start_head_idx : head_slice.end_head_idx, :].contiguous()


def package_payloads_from_prefill_rank(
    prefill_kv_slice: torch.Tensor,
    prefill_rank: int,
    layer_idx: int,
    plan: KVPartitionPlan,
    enable_fp8: bool = True,
    block_size_tokens: int = 16,
    request_id: Optional[str] = None,
    kv_type: Optional[str] = None,
) -> List[KVPayload]:
    """
    Given the KV cache slice held by a prefill rank, carve it into targeted
    payloads for each designated decode rank according to the bipartite partition plan.
    """
    routes = plan.get_routes_from_prefill_rank(prefill_rank)
    rank_start_head = plan.prefill_head_slices[prefill_rank].start_head_idx
    payloads: List[KVPayload] = []

    for route in routes:
        # Calculate local offset inside this prefill rank's slice
        rel_start = route.head_slice.start_head_idx - rank_start_head
        rel_end = route.head_slice.end_head_idx - rank_start_head

        # Slice the tensor along the head dimension
        sub_slice = prefill_kv_slice[..., rel_start:rel_end, :].contiguous()

        if enable_fp8:
            compressed = quantize_kv_block_fp8(sub_slice, block_size_tokens=block_size_tokens)
            payload_data = compressed
            is_quantized = True
        else:
            payload_data = sub_slice
            is_quantized = False

        payloads.append(
            KVPayload(
                prefill_rank=prefill_rank,
                decode_rank=route.decode_rank,
                layer_idx=layer_idx,
                start_head_idx=route.head_slice.start_head_idx,
                end_head_idx=route.head_slice.end_head_idx,
                data=payload_data,
                is_quantized=is_quantized,
                request_id=request_id,
                kv_type=kv_type,
            )
        )

    return payloads


def reassemble_decode_kv(
    received_payloads: List[KVPayload],
    decode_rank: int,
    plan: KVPartitionPlan,
) -> torch.Tensor:
    """
    Reassemble incoming payloads from designated prefill ranks into the contiguous
    KV cache layout expected by the target decode rank.
    Validates layer consistency, request identity, source-rank ownership, and tensor shapes.
    """
    if decode_rank not in plan.decode_head_slices:
        raise ValueError(f"decode_rank {decode_rank} is not in partition plan (decode_tp={plan.decode_tp})")

    expected_slice = plan.decode_head_slices[decode_rank]
    filtered_payloads = [p for p in received_payloads if p.decode_rank == decode_rank]

    if not filtered_payloads:
        raise ValueError(f"No payloads found for decode_rank {decode_rank}")

    # Validate layer, request, and kv_type consistency across all payloads
    expected_layer = filtered_payloads[0].layer_idx
    expected_request = filtered_payloads[0].request_id
    expected_kv_type = filtered_payloads[0].kv_type

    for p in filtered_payloads:
        if p.layer_idx != expected_layer:
            raise ValueError(
                f"Inconsistent layer_idx in payloads for decode_rank {decode_rank}: "
                f"expected {expected_layer}, got {p.layer_idx}"
            )
        if p.request_id != expected_request:
            raise ValueError(
                f"Inconsistent request_id in payloads for decode_rank {decode_rank}: "
                f"expected {expected_request}, got {p.request_id}"
            )
        if p.kv_type != expected_kv_type:
            raise ValueError(
                f"Inconsistent kv_type in payloads for decode_rank {decode_rank}: "
                f"expected {expected_kv_type}, got {p.kv_type}"
            )

    # Validate source-rank ownership against partition plan routes
    decode_routes = {
        (r.head_slice.start_head_idx, r.head_slice.end_head_idx): r.prefill_rank
        for r in plan.transfer_routes
        if r.decode_rank == decode_rank
    }

    for p in filtered_payloads:
        head_key = (p.start_head_idx, p.end_head_idx)
        if head_key not in decode_routes:
            raise ValueError(
                f"Unexpected head slice [{p.start_head_idx}:{p.end_head_idx}] for decode_rank {decode_rank}"
            )
        expected_prefill = decode_routes[head_key]
        if p.prefill_rank != expected_prefill:
            raise ValueError(
                f"Unauthorized source rank for head slice [{p.start_head_idx}:{p.end_head_idx}]: "
                f"expected prefill_rank {expected_prefill}, got {p.prefill_rank}"
            )

    # Sort payloads by their starting head index to guarantee correct ordering
    sorted_payloads = sorted(filtered_payloads, key=lambda p: p.start_head_idx)

    uncompressed_slices: List[torch.Tensor] = []
    current_head = expected_slice.start_head_idx

    for p in sorted_payloads:
        if p.start_head_idx != current_head:
            raise ValueError(
                f"Head index gap detected for decode_rank {decode_rank}: "
                f"expected head {current_head}, got {p.start_head_idx}"
            )

        if p.is_quantized and isinstance(p.data, FP8BlockPayload):
            restored = dequantize_kv_block_fp8(p.data)
        elif isinstance(p.data, torch.Tensor):
            restored = p.data
        else:
            raise TypeError(f"Unrecognized payload data type: {type(p.data)}")

        if restored.dim() < 2:
            raise ValueError(f"Restored tensor must have >=2 dimensions, got shape {restored.shape}")

        declared_heads = p.end_head_idx - p.start_head_idx
        actual_heads = restored.shape[-2]
        if actual_heads != declared_heads:
            raise ValueError(
                f"Head count mismatch in payload: declared {declared_heads} heads "
                f"({p.start_head_idx}:{p.end_head_idx}), but actual tensor has {actual_heads} heads (shape={restored.shape})"
            )

        if uncompressed_slices:
            if (
                restored.shape[:-2] != uncompressed_slices[0].shape[:-2]
                or restored.shape[-1] != uncompressed_slices[0].shape[-1]
            ):
                raise ValueError(
                    f"Shape mismatch across payload slices: expected non-head shape {uncompressed_slices[0].shape[:-2]} "
                    f"and head_dim {uncompressed_slices[0].shape[-1]}, got {restored.shape[:-2]} and {restored.shape[-1]}"
                )

        uncompressed_slices.append(restored)
        current_head = p.end_head_idx

    if current_head != expected_slice.end_head_idx:
        raise ValueError(
            f"Incomplete KV heads for decode_rank {decode_rank}: "
            f"ended at {current_head}, expected {expected_slice.end_head_idx}"
        )

    # Concatenate along the KV head dimension (dimension -2)
    return torch.cat(uncompressed_slices, dim=-2)
