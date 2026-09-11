"""HeteroKVConnector: Orchestrates asymmetric KV cache re-sharding and pipelined transfers."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch

from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import (
    KVPayload,
    slice_kv_for_prefill_rank,
    package_payloads_from_prefill_rank,
    reassemble_decode_kv,
)
from hetero_kv.quant.fp8_block import compute_cosine_similarity


@dataclass
class TransferMetrics:
    """Performance telemetry for KV cache transfer across mismatched ranks."""
    num_layers: int
    uncompressed_bytes: int
    wire_bytes_transferred: int
    compression_ratio: float
    quantization_time_ms: float
    simulated_wire_latency_ms: float
    dequantization_time_ms: float
    total_transfer_time_ms: float
    effective_bandwidth_gb_s: float
    average_cosine_similarity: float


class HeteroKVConnector:
    """
    High-level connector managing asymmetric KV cache transfer between
    a prefill worker pool and a decode worker pool.
    """

    def __init__(
        self,
        num_kv_heads: int,
        prefill_tp: int,
        decode_tp: int,
        enable_fp8: bool = True,
        block_size_tokens: int = 16,
        interconnect_bandwidth_gb_s: float = 32.0,  # Default: PCIe Gen4 x16
    ):
        self.num_kv_heads = num_kv_heads
        self.prefill_tp = prefill_tp
        self.decode_tp = decode_tp
        self.enable_fp8 = enable_fp8
        self.block_size_tokens = block_size_tokens
        self.interconnect_bandwidth_gb_s = interconnect_bandwidth_gb_s

        # Initialize and validate the bipartite partition plan
        self.plan = KVPartitionPlan.create(
            num_kv_heads=num_kv_heads,
            prefill_tp=prefill_tp,
            decode_tp=decode_tp,
        )

    def transfer_layer_kv(
        self,
        layer_idx: int,
        prefill_rank_shards: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], List[KVPayload], float, float]:
        """
        Execute asymmetric transfer for a single layer:
        1. Package and optionally quantize payloads on the prefill side.
        2. Route and reassemble on the decode side.
        
        Returns:
            decode_rank_tensors: Dict mapping decode_rank to its reassembled KV tensor.
            payloads: List of transferred KVPayload packets.
            quant_ms: Time spent packaging and quantizing.
            dequant_ms: Time spent reassembling and dequantizing.
        """
        t0 = time.perf_counter()
        all_payloads: List[KVPayload] = []

        for p_rank in range(self.prefill_tp):
            prefill_slice = prefill_rank_shards[p_rank]
            rank_payloads = package_payloads_from_prefill_rank(
                prefill_kv_slice=prefill_slice,
                prefill_rank=p_rank,
                layer_idx=layer_idx,
                plan=self.plan,
                enable_fp8=self.enable_fp8,
                block_size_tokens=self.block_size_tokens,
            )
            all_payloads.extend(rank_payloads)

        t1 = time.perf_counter()
        quant_ms = (t1 - t0) * 1000.0

        # Decode side: reassemble tensors
        t2 = time.perf_counter()
        decode_shards: Dict[int, torch.Tensor] = {}
        for d_rank in range(self.decode_tp):
            reassembled = reassemble_decode_kv(
                received_payloads=all_payloads,
                decode_rank=d_rank,
                plan=self.plan,
            )
            decode_shards[d_rank] = reassembled

        t3 = time.perf_counter()
        dequant_ms = (t3 - t2) * 1000.0

        return decode_shards, all_payloads, quant_ms, dequant_ms

    def benchmark_full_transfer(
        self,
        num_layers: int,
        batch_size: int,
        seq_len: int,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
    ) -> TransferMetrics:
        """
        Benchmark simulated end-to-end multi-layer KV transfer pipeline.
        Generates synthetic KV tensors, partitions across prefill ranks, transfers,
        and verifies cosine similarity against un-partitioned ground truth.
        """
        total_quant_ms = 0.0
        total_dequant_ms = 0.0
        total_wire_bytes = 0
        total_uncompressed_bytes = 0
        similarity_scores: List[float] = []

        for layer in range(num_layers):
            # Generate ground-truth full KV tensor: (batch_size, seq_len, num_kv_heads, head_dim)
            ground_truth = torch.randn(
                batch_size, seq_len, self.num_kv_heads, head_dim, dtype=dtype
            )
            uncompressed_layer_bytes = ground_truth.nbytes
            total_uncompressed_bytes += uncompressed_layer_bytes

            # Partition into prefill rank shards
            prefill_shards: Dict[int, torch.Tensor] = {}
            for p_rank in range(self.prefill_tp):
                prefill_shards[p_rank] = slice_kv_for_prefill_rank(
                    ground_truth, prefill_rank=p_rank, plan=self.plan
                )

            # Transfer layer
            decode_shards, payloads, q_ms, dq_ms = self.transfer_layer_kv(
                layer_idx=layer,
                prefill_rank_shards=prefill_shards,
            )
            total_quant_ms += q_ms
            total_dequant_ms += dq_ms

            for p in payloads:
                total_wire_bytes += p.payload_bytes

            # Glue decode shards together to check cosine similarity against ground truth
            reconstructed_full = torch.cat(
                [decode_shards[d] for d in range(self.decode_tp)], dim=-2
            )
            sim = compute_cosine_similarity(ground_truth, reconstructed_full)
            similarity_scores.append(sim)

        # Calculate simulated wire transfer time based on interconnect bandwidth
        # (Wire bytes / (Bandwidth in bytes/sec)) * 1000 for ms
        wire_bw_bytes_per_sec = self.interconnect_bandwidth_gb_s * (1024 ** 3)
        wire_latency_ms = (total_wire_bytes / wire_bw_bytes_per_sec) * 1000.0

        total_time_ms = total_quant_ms + wire_latency_ms + total_dequant_ms
        compression_ratio = total_uncompressed_bytes / max(1, total_wire_bytes)
        effective_bw_gb_s = (total_uncompressed_bytes / (1024 ** 3)) / (total_time_ms / 1000.0)

        return TransferMetrics(
            num_layers=num_layers,
            uncompressed_bytes=total_uncompressed_bytes,
            wire_bytes_transferred=total_wire_bytes,
            compression_ratio=round(compression_ratio, 2),
            quantization_time_ms=round(total_quant_ms, 2),
            simulated_wire_latency_ms=round(wire_latency_ms, 2),
            dequantization_time_ms=round(total_dequant_ms, 2),
            total_transfer_time_ms=round(total_time_ms, 2),
            effective_bandwidth_gb_s=round(effective_bw_gb_s, 2),
            average_cosine_similarity=round(sum(similarity_scores) / len(similarity_scores), 5),
        )
