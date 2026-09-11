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
    """
    if decode_rank not in plan.decode_head_slices:
        raise ValueError(f"decode_rank {decode_rank} is not in partition plan (decode_tp={plan.decode_tp})")

    expected_slice = plan.decode_head_slices[decode_rank]
    filtered_payloads = [p for p in received_payloads if p.decode_rank == decode_rank]

    if not filtered_payloads:
        raise ValueError(f"No payloads found for decode_rank {decode_rank}")

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

        uncompressed_slices.append(restored)
        current_head = p.end_head_idx

    if current_head != expected_slice.end_head_idx:
        raise ValueError(
            f"Incomplete KV heads for decode_rank {decode_rank}: "
            f"ended at {current_head}, expected {expected_slice.end_head_idx}"
        )

    # Concatenate along the KV head dimension (dimension -2)
    return torch.cat(uncompressed_slices, dim=-2)
