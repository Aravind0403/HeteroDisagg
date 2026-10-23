"""Adaptive Cache Compression Policy for HeteroDisagg.

Decides whether to apply lossy block-scaled FP8 compression or transfer uncompressed
floating-point (FP16/BF16) cache tensors across processes or network links.

Decision Principle:
    Compress only when the estimated wire transmission time saved exceeds the
    computational encoding (quantization + packaging) and decoding (dequantization) overhead:
    
    Delta_T = T_uncompressed - T_compressed
            = (Bytes_uncompressed - Bytes_compressed) / Bandwidth - (T_enc + T_dec)

    If Delta_T > 0, compression accelerates end-to-end handoff.
    If Delta_T <= 0, compression slows down end-to-end handoff.

Special Constraints:
    - Short sequences (<16 tokens) expand under tile padding and framing overhead:
      Bytes_compressed >= Bytes_uncompressed -> Policy automatically selects UNCOMPRESSED.
    - Ultra-fast interconnects (e.g. 100 Gbps, loopback): Network transfer is fast enough
      that CPU codec latency dominates -> Policy automatically selects UNCOMPRESSED.
    - Bandwidth-constrained or long sequences (1 Gbps, 10 Gbps, seq_len >= 128): Wire savings
      dramatically dominate codec overhead -> Policy selects BLOCK_FP8.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List


class CompressionMode(str, Enum):
    UNCOMPRESSED_FP16 = "uncompressed_fp16"
    BLOCK_FP8 = "block_fp8"


@dataclass
class PolicyEvaluationResult:
    """Outcome of a policy evaluation on a specific workload."""
    selected_mode: CompressionMode
    reason: str
    predicted_uncompressed_bytes: int
    predicted_compressed_bytes: int
    bytes_saved: int
    compression_ratio: float
    estimated_uncompressed_latency_ms: float
    estimated_compressed_latency_ms: float
    estimated_net_saving_ms: float
    is_favorable: bool


class AdaptiveCompressionPolicy:
    """
    Predictive policy for choosing between uncompressed FP16 and block-scaled FP8.
    
    Calibrated against CPU memory throughput and empirical codec speeds:
    - Quantization rate: ~1.8 GB/s (FP32/FP16 -> FP8 with block scaling)
    - Dequantization rate: ~3.5 GB/s (FP8 -> FP32/FP16)
    - Framing overhead: ~776 bytes magic + ~35 KB JSON metadata headers
    """

    def __init__(
        self,
        encoding_throughput_bytes_per_sec: float = 6.0e8,  # ~600 MB/s
        decoding_throughput_bytes_per_sec: float = 1.0e9,  # ~1.0 GB/s
        per_frame_dispatch_overhead_sec: float = 2.0e-5,   # ~20 us dispatch overhead per frame
        framing_overhead_bytes_per_frame: int = 150,       # Avg JSON header + framing per frame
        min_saving_threshold_ms: float = 0.05,             # Minimum 50us expected saving to justify lossy
    ):
        self.encoding_throughput = encoding_throughput_bytes_per_sec
        self.decoding_throughput = decoding_throughput_bytes_per_sec
        self.per_frame_dispatch_overhead = per_frame_dispatch_overhead_sec
        self.framing_overhead_per_frame = framing_overhead_bytes_per_frame
        self.min_saving_threshold_ms = min_saving_threshold_ms

    def estimate_bytes(
        self,
        seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size_tokens: int = 16,
        prefill_tp: int = 1,
    ) -> Tuple[int, int]:
        """
        Calculates exact application wire bytes for uncompressed FP16 vs block-scaled FP8,
        including framing headers, JSON metadata, scale factors, and tile padding.
        """
        num_frames = num_layers * 2 * prefill_tp
        metadata_bytes = num_frames * self.framing_overhead_per_frame + 2048  # + manifest overhead

        # 1. Uncompressed FP16 bytes: 2 bytes per element
        # Shape per frame: [1, seq_len, heads_per_rank, head_dim]
        raw_elements = num_layers * 2 * seq_len * num_kv_heads * head_dim
        uncompressed_body_bytes = raw_elements * 2  # FP16 = 2 bytes
        total_uncompressed_bytes = uncompressed_body_bytes + metadata_bytes

        # 2. Block-scaled FP8 bytes:
        # Sequence is padded to multiple of block_size_tokens
        num_blocks = (seq_len + block_size_tokens - 1) // block_size_tokens
        padded_seq_len = num_blocks * block_size_tokens

        # Quantized data: 1 byte per element of padded tensor
        quant_bytes = num_layers * 2 * padded_seq_len * num_kv_heads * head_dim * 1
        # Scale factors: 1 FP32 float (4 bytes) per block per layer per KV type
        # For each frame: scale_factors shape is [num_blocks], FP32 = 4 bytes
        scale_bytes = num_frames * num_blocks * 4

        total_compressed_bytes = quant_bytes + scale_bytes + metadata_bytes

        return total_uncompressed_bytes, total_compressed_bytes

    def evaluate(
        self,
        seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        link_bandwidth_bytes_per_sec: float,
        block_size_tokens: int = 16,
        prefill_tp: int = 1,
    ) -> PolicyEvaluationResult:
        """
        Evaluates compression trade-offs and returns an actionable decision.
        """
        uncomp_bytes, comp_bytes = self.estimate_bytes(
            seq_len=seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size_tokens=block_size_tokens,
            prefill_tp=prefill_tp,
        )

        bytes_saved = uncomp_bytes - comp_bytes
        ratio = uncomp_bytes / max(1, comp_bytes)

        # Rule 1: Negative compression (payload expands due to padding/metadata)
        if bytes_saved <= 0:
            return PolicyEvaluationResult(
                selected_mode=CompressionMode.UNCOMPRESSED_FP16,
                reason=(
                    f"Wire expansion detected: compressed size ({comp_bytes} B) >= "
                    f"uncompressed ({uncomp_bytes} B) due to tile padding and framing overhead."
                ),
                predicted_uncompressed_bytes=uncomp_bytes,
                predicted_compressed_bytes=comp_bytes,
                bytes_saved=bytes_saved,
                compression_ratio=ratio,
                estimated_uncompressed_latency_ms=(uncomp_bytes / link_bandwidth_bytes_per_sec) * 1000.0,
                estimated_compressed_latency_ms=(comp_bytes / link_bandwidth_bytes_per_sec) * 1000.0,
                estimated_net_saving_ms=0.0,
                is_favorable=False,
            )

        # Rule 2: Compute time vs wire transmission time
        # Uncompressed latency = Network transmission time
        t_net_uncompressed = uncomp_bytes / link_bandwidth_bytes_per_sec

        # Compressed latency = Encoding time + Network transmission time + Decoding time
        num_frames = num_layers * 2 * prefill_tp
        dispatch_overhead_sec = num_frames * self.per_frame_dispatch_overhead
        raw_body_bytes = num_layers * 2 * seq_len * num_kv_heads * head_dim * 2
        t_enc = (dispatch_overhead_sec * 0.6) + (raw_body_bytes / self.encoding_throughput)
        t_net_compressed = comp_bytes / link_bandwidth_bytes_per_sec
        t_dec = (dispatch_overhead_sec * 0.4) + (raw_body_bytes / self.decoding_throughput)

        t_compressed_total = t_enc + t_net_compressed + t_dec
        net_saving_sec = t_net_uncompressed - t_compressed_total
        net_saving_ms = net_saving_sec * 1000.0

        if net_saving_ms > self.min_saving_threshold_ms:
            selected_mode = CompressionMode.BLOCK_FP8
            reason = (
                f"Favorable wire savings: saves {bytes_saved / 1024:.1f} KB ({ratio:.2f}x compression) "
                f"yielding estimated {net_saving_ms:.2f} ms latency reduction on {link_bandwidth_bytes_per_sec / 1e6:.1f} MB/s link."
            )
            is_favorable = True
        else:
            selected_mode = CompressionMode.UNCOMPRESSED_FP16
            codec_overhead_ms = (t_enc + t_dec) * 1000.0
            wire_saving_ms = (t_net_uncompressed - t_net_compressed) * 1000.0
            reason = (
                f"Codec overhead exceeds wire savings: codec cost ({codec_overhead_ms:.2f} ms) "
                f"> wire transfer savings ({wire_saving_ms:.2f} ms) on high-speed link."
            )
            is_favorable = False

        return PolicyEvaluationResult(
            selected_mode=selected_mode,
            reason=reason,
            predicted_uncompressed_bytes=uncomp_bytes,
            predicted_compressed_bytes=comp_bytes,
            bytes_saved=bytes_saved,
            compression_ratio=ratio,
            estimated_uncompressed_latency_ms=t_net_uncompressed * 1000.0,
            estimated_compressed_latency_ms=t_compressed_total * 1000.0,
            estimated_net_saving_ms=net_saving_ms,
            is_favorable=is_favorable,
        )

    def decide(
        self,
        seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        link_bandwidth_bytes_per_sec: float,
        block_size_tokens: int = 16,
        prefill_tp: int = 1,
    ) -> CompressionMode:
        """Convenience method returning just the selected CompressionMode."""
        res = self.evaluate(
            seq_len=seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            link_bandwidth_bytes_per_sec=link_bandwidth_bytes_per_sec,
            block_size_tokens=block_size_tokens,
            prefill_tp=prefill_tp,
        )
        return res.selected_mode
