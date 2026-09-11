"""KV Head Layout and Bipartite Partition Planner for Asymmetric Parallelism."""

from typing import List, Dict, Tuple, Set
from pydantic import BaseModel, Field


class KVHeadSlice(BaseModel):
    """Represents a contiguous range of KV heads owned or needed by a GPU rank."""
    start_head_idx: int = Field(..., description="Inclusive starting KV head index")
    end_head_idx: int = Field(..., description="Exclusive ending KV head index")

    @property
    def num_heads(self) -> int:
        return self.end_head_idx - self.start_head_idx

    def overlaps_with(self, other: "KVHeadSlice") -> bool:
        """Check if this head slice overlaps with another head slice."""
        return max(self.start_head_idx, other.start_head_idx) < min(self.end_head_idx, other.end_head_idx)

    def intersection(self, other: "KVHeadSlice") -> "KVHeadSlice | None":
        """Compute the intersecting sub-slice between two head slices."""
        start = max(self.start_head_idx, other.start_head_idx)
        end = min(self.end_head_idx, other.end_head_idx)
        if start < end:
            return KVHeadSlice(start_head_idx=start, end_head_idx=end)
        return None


class BipartiteTransferRoute(BaseModel):
    """A direct point-to-point transfer route from a prefill rank to a decode rank."""
    prefill_rank: int
    decode_rank: int
    head_slice: KVHeadSlice


class KVPartitionPlan(BaseModel):
    """
    Bipartite partition and routing plan for asymmetric tensor parallelism:
    prefill_tp (number of prefill GPU ranks) transferring to decode_tp (number of decode GPU ranks).
    """
    num_kv_heads: int
    prefill_tp: int
    decode_tp: int
    heads_per_prefill_rank: int
    heads_per_decode_rank: int
    prefill_head_slices: Dict[int, KVHeadSlice]
    decode_head_slices: Dict[int, KVHeadSlice]
    transfer_routes: List[BipartiteTransferRoute]

    @classmethod
    def create(cls, num_kv_heads: int, prefill_tp: int, decode_tp: int) -> "KVPartitionPlan":
        """
        Validate GQA head divisibility and construct the bipartite routing plan.
        """
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if prefill_tp <= 0 or decode_tp <= 0:
            raise ValueError(f"Tensor parallelism degrees must be positive, got prefill_tp={prefill_tp}, decode_tp={decode_tp}")

        # Grouped Query Attention (GQA) divisibility check
        if num_kv_heads % prefill_tp != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be evenly divisible by prefill_tp ({prefill_tp}). "
                f"Cannot partition {num_kv_heads} heads across {prefill_tp} ranks."
            )
        if num_kv_heads % decode_tp != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be evenly divisible by decode_tp ({decode_tp}). "
                f"Cannot partition {num_kv_heads} heads across {decode_tp} ranks."
            )

        heads_per_prefill = num_kv_heads // prefill_tp
        heads_per_decode = num_kv_heads // decode_tp

        prefill_slices: Dict[int, KVHeadSlice] = {}
        for rank in range(prefill_tp):
            start = rank * heads_per_prefill
            end = start + heads_per_prefill
            prefill_slices[rank] = KVHeadSlice(start_head_idx=start, end_head_idx=end)

        decode_slices: Dict[int, KVHeadSlice] = {}
        for rank in range(decode_tp):
            start = rank * heads_per_decode
            end = start + heads_per_decode
            decode_slices[rank] = KVHeadSlice(start_head_idx=start, end_head_idx=end)

        # Build targeted bipartite transfer routes
        routes: List[BipartiteTransferRoute] = []
        for p_rank, p_slice in prefill_slices.items():
            for d_rank, d_slice in decode_slices.items():
                overlap = p_slice.intersection(d_slice)
                if overlap is not None:
                    routes.append(
                        BipartiteTransferRoute(
                            prefill_rank=p_rank,
                            decode_rank=d_rank,
                            head_slice=overlap,
                        )
                    )

        return cls(
            num_kv_heads=num_kv_heads,
            prefill_tp=prefill_tp,
            decode_tp=decode_tp,
            heads_per_prefill_rank=heads_per_prefill,
            heads_per_decode_rank=heads_per_decode,
            prefill_head_slices=prefill_slices,
            decode_head_slices=decode_slices,
            transfer_routes=routes,
        )

    def get_routes_from_prefill_rank(self, prefill_rank: int) -> List[BipartiteTransferRoute]:
        """Get all transfer routes where the specified prefill rank is the sender."""
        return [r for r in self.transfer_routes if r.prefill_rank == prefill_rank]

    def get_routes_to_decode_rank(self, decode_rank: int) -> List[BipartiteTransferRoute]:
        """Get all transfer routes where the specified decode rank is the receiver."""
        return [r for r in self.transfer_routes if r.decode_rank == decode_rank]

    def get_sender_ranks_for_decode_rank(self, decode_rank: int) -> List[int]:
        """Get the unique list of prefill ranks that send KV cache to this decode rank."""
        return sorted(list(set(r.prefill_rank for r in self.get_routes_to_decode_rank(decode_rank))))
