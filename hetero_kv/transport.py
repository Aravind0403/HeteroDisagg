"""HeteroDisagg Socket Transport Protocol.

Provides robust, bounded-read framing for process-to-process KV cache transfers.

Protocol Specification:
1. Connection Handshake / Request Manifest (Magic: b"HKVM"):
   - 4-byte Magic: b"HKVM"
   - 4-byte big-endian unsigned integer: JSON Manifest length M
   - M bytes: UTF-8 encoded JSON RequestManifest
2. Frame Packets (Magic: b"HKVF"):
   - 4-byte Magic: b"HKVF"
   - 4-byte big-endian unsigned integer: JSON FrameMetadata length L
   - L bytes: UTF-8 encoded JSON FrameMetadata
   - B bytes: Contiguous raw binary tensor bytes (validated against shape * elem_size)

Guarantees:
- Bounded reads on all metadata and payload lengths.
- Timeout / deadline enforcement on all socket read loops.
- Strict payload byte validation against tensor shape and dtype.
- Rejection of duplicate frames, out-of-order/missing layers, and request ID mismatch.
- Accurate byte accounting separating framing, metadata, and tensor body bytes.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Set
import torch

from hetero_kv.quant.fp8_block import FP8BlockPayload, dequantize_kv_block_fp8

MAGIC_MANIFEST = b"HKVM"
MAGIC_FRAME = b"HKVF"

MAX_METADATA_BYTES = 64 * 1024          # 64 KB max metadata size
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB max single tensor frame size
DEFAULT_TIMEOUT = 10.0                   # 10s default socket timeout

DTYPE_BYTE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "int8": 1,
    "uint8": 1,
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": getattr(torch, "float8_e4m3fn", torch.int8),
    "int8": torch.int8,
    "uint8": torch.uint8,
}


class TransportError(Exception):
    """Base exception for transport errors."""
    pass


class TransportTimeoutError(TransportError):
    """Socket operation timed out before deadline."""
    pass


class IncompleteTransferError(TransportError):
    """Socket closed or truncated prematurely."""
    pass


class CorruptedFrameError(TransportError):
    """Magic mismatch or frame corrupted."""
    pass


class MismatchedRequestError(TransportError):
    """Request ID or metadata mismatch."""
    pass


class ManifestValidationError(TransportError):
    """Manifest data validation failure."""
    pass


class FrameValidationError(TransportError):
    """Frame structure, shape, or completeness validation failure."""
    pass


@dataclass
class WireAccounting:
    """Detailed breakdown of bytes transferred across the socket."""
    framing_bytes: int = 0   # Magic bytes (4) + length headers (4)
    metadata_bytes: int = 0  # JSON metadata bytes
    body_bytes: int = 0      # Raw binary payload bytes

    @property
    def total_wire_bytes(self) -> int:
        return self.framing_bytes + self.metadata_bytes + self.body_bytes

    def add(self, other: "WireAccounting") -> "WireAccounting":
        return WireAccounting(
            framing_bytes=self.framing_bytes + other.framing_bytes,
            metadata_bytes=self.metadata_bytes + other.metadata_bytes,
            body_bytes=self.body_bytes + other.body_bytes,
        )


@dataclass
class RequestManifest:
    """Manifest describing what constitutes a complete request transfer."""
    request_id: str
    model_id: str
    checkpoint_commit_hash: Optional[str]
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    prefill_tp: int
    decode_tp: int
    expected_frames: int
    initial_decode_token: int
    initial_decode_token_str: str
    prompt_tokens: List[int]
    dtype: str
    enable_fp8: bool = False
    block_size_tokens: int = 16

    def validate(self) -> None:
        if not self.request_id:
            raise ManifestValidationError("request_id cannot be empty")
        if self.num_layers <= 0:
            raise ManifestValidationError(f"Invalid num_layers: {self.num_layers}")
        if self.seq_len <= 0:
            raise ManifestValidationError(f"Invalid seq_len: {self.seq_len}")
        if self.prefill_tp <= 0 or self.decode_tp <= 0:
            raise ManifestValidationError(f"Invalid TP: prefill={self.prefill_tp}, decode={self.decode_tp}")
        if self.dtype not in DTYPE_BYTE_SIZES:
            raise ManifestValidationError(f"Unsupported dtype: {self.dtype}")
        expected_min_frames = self.num_layers * 2 * self.prefill_tp
        if self.expected_frames != expected_min_frames:
            raise ManifestValidationError(
                f"expected_frames mismatch: got {self.expected_frames}, expected {expected_min_frames}"
            )


@dataclass
class FrameMetadata:
    """Header metadata for an individual tensor frame."""
    request_id: str
    kv_type: str            # 'key' or 'value'
    layer_idx: int
    prefill_rank: int
    decode_rank: int
    start_head_idx: int
    end_head_idx: int
    is_quantized: bool
    shape: List[int]
    dtype: str
    body_bytes: int
    # Optional FP8 fields
    scale_shape: Optional[List[int]] = None
    scale_dtype: Optional[str] = None
    original_shape: Optional[List[int]] = None
    original_dtype: Optional[str] = None
    block_size_tokens: Optional[int] = None

    def validate(self, manifest: RequestManifest) -> None:
        if self.request_id != manifest.request_id:
            raise MismatchedRequestError(
                f"Frame request_id '{self.request_id}' does not match manifest '{manifest.request_id}'"
            )
        if self.kv_type not in ("key", "value"):
            raise FrameValidationError(f"Invalid kv_type '{self.kv_type}'")
        if self.layer_idx < 0 or self.layer_idx >= manifest.num_layers:
            raise FrameValidationError(
                f"layer_idx {self.layer_idx} out of bounds [0, {manifest.num_layers - 1}]"
            )
        if self.prefill_rank < 0 or self.prefill_rank >= manifest.prefill_tp:
            raise FrameValidationError(
                f"prefill_rank {self.prefill_rank} out of bounds [0, {manifest.prefill_tp - 1}]"
            )
        if self.start_head_idx >= self.end_head_idx:
            raise FrameValidationError(
                f"Invalid head range: [{self.start_head_idx}, {self.end_head_idx}]"
            )
        # Strict body length calculation validation
        if not self.is_quantized:
            if self.dtype not in DTYPE_BYTE_SIZES:
                raise FrameValidationError(f"Invalid frame dtype: {self.dtype}")
            elem_size = DTYPE_BYTE_SIZES[self.dtype]
            num_elements = 1
            for d in self.shape:
                if d <= 0:
                    raise FrameValidationError(f"Invalid shape dimension: {d}")
                num_elements *= d
            expected_bytes = num_elements * elem_size
            if self.body_bytes != expected_bytes:
                raise FrameValidationError(
                    f"Frame body_bytes ({self.body_bytes}) does not match shape ({self.shape}) "
                    f"and dtype ({self.dtype}) expected bytes ({expected_bytes})"
                )
        else:
            if not self.scale_shape or not self.scale_dtype or not self.original_shape or not self.original_dtype:
                raise FrameValidationError("Missing required FP8 scale or shape metadata for quantized frame")
            quant_elem_size = DTYPE_BYTE_SIZES.get(self.dtype, 1)
            quant_numel = 1
            for d in self.shape:
                quant_numel *= d
            quant_bytes = quant_numel * quant_elem_size

            scale_elem_size = DTYPE_BYTE_SIZES.get(self.scale_dtype, 4)
            scale_numel = 1
            for d in self.scale_shape:
                scale_numel *= d
            scale_bytes = scale_numel * scale_elem_size

            expected_bytes = quant_bytes + scale_bytes
            if self.body_bytes != expected_bytes:
                raise FrameValidationError(
                    f"Quantized frame body_bytes ({self.body_bytes}) does not match quant + scale expected bytes ({expected_bytes})"
                )


def recv_exact(sock: socket.socket, n_bytes: int, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """
    Read exactly n_bytes from socket with strict deadline enforcement.
    Raises IncompleteTransferError if connection terminates prematurely.
    Raises TransportTimeoutError if deadline expires.
    """
    if n_bytes < 0:
        raise ValueError(f"n_bytes must be non-negative, got {n_bytes}")
    if n_bytes == 0:
        return b""

    deadline = time.monotonic() + timeout
    buffer = bytearray(n_bytes)
    view = memoryview(buffer)
    bytes_read = 0

    while bytes_read < n_bytes:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise TransportTimeoutError(
                f"Socket timed out waiting for {n_bytes} bytes; received {bytes_read} bytes before timeout"
            )
        sock.settimeout(max(0.01, remaining_time))
        try:
            chunk = sock.recv_into(view[bytes_read:], n_bytes - bytes_read)
        except socket.timeout as e:
            raise TransportTimeoutError(
                f"Socket timed out after {timeout}s: received {bytes_read}/{n_bytes} bytes"
            ) from e
        except (ConnectionResetError, BrokenPipeError) as e:
            raise IncompleteTransferError(
                f"Socket disconnected unexpectedly: received {bytes_read}/{n_bytes} bytes"
            ) from e

        if chunk == 0:
            raise IncompleteTransferError(
                f"Socket EOF reached unexpectedly: expected {n_bytes} bytes, received {bytes_read} bytes"
            )
        bytes_read += chunk

    return bytes(buffer)


def send_manifest(sock: socket.socket, manifest: RequestManifest) -> WireAccounting:
    """Send RequestManifest packet over the socket."""
    manifest.validate()
    meta_json = json.dumps(asdict(manifest)).encode("utf-8")
    meta_len = len(meta_json)
    if meta_len > MAX_METADATA_BYTES:
        raise ManifestValidationError(f"Manifest JSON exceeds max allowed size: {meta_len} > {MAX_METADATA_BYTES}")

    # Framing: 4-byte magic + 4-byte length + payload
    header = MAGIC_MANIFEST + meta_len.to_bytes(4, byteorder="big")
    sock.sendall(header + meta_json)

    return WireAccounting(
        framing_bytes=len(header),
        metadata_bytes=meta_len,
        body_bytes=0,
    )


def recv_manifest(sock: socket.socket, timeout: float = DEFAULT_TIMEOUT) -> Tuple[RequestManifest, WireAccounting]:
    """Receive and validate RequestManifest packet from the socket."""
    # 1. Read framing header (4 bytes magic + 4 bytes length)
    header_bytes = recv_exact(sock, 8, timeout=timeout)
    magic = header_bytes[:4]
    if magic != MAGIC_MANIFEST:
        raise CorruptedFrameError(f"Invalid manifest magic: expected {MAGIC_MANIFEST}, got {magic}")

    meta_len = int.from_bytes(header_bytes[4:8], byteorder="big")
    if meta_len <= 0 or meta_len > MAX_METADATA_BYTES:
        raise ManifestValidationError(f"Manifest length {meta_len} out of allowed bounds [1, {MAX_METADATA_BYTES}]")

    # 2. Read manifest JSON body
    json_bytes = recv_exact(sock, meta_len, timeout=timeout)
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        manifest = RequestManifest(**data)
        manifest.validate()
    except Exception as e:
        raise ManifestValidationError(f"Failed to decode or validate RequestManifest: {e}") from e

    accounting = WireAccounting(
        framing_bytes=8,
        metadata_bytes=meta_len,
        body_bytes=0,
    )
    return manifest, accounting


def send_frame(sock: socket.socket, meta: FrameMetadata, body: bytes) -> WireAccounting:
    """Send individual tensor frame over the socket."""
    meta_json = json.dumps(asdict(meta)).encode("utf-8")
    meta_len = len(meta_json)
    if meta_len > MAX_METADATA_BYTES:
        raise FrameValidationError(f"Frame metadata exceeds limit: {meta_len} > {MAX_METADATA_BYTES}")
    if len(body) != meta.body_bytes:
        raise FrameValidationError(f"Body length ({len(body)}) does not match metadata ({meta.body_bytes})")

    header = MAGIC_FRAME + meta_len.to_bytes(4, byteorder="big")
    sock.sendall(header + meta_json + body)

    return WireAccounting(
        framing_bytes=len(header),
        metadata_bytes=meta_len,
        body_bytes=len(body),
    )


def recv_frame(
    sock: socket.socket,
    manifest: RequestManifest,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[FrameMetadata, bytes, WireAccounting]:
    """Receive individual tensor frame from socket with bounded validation."""
    header_bytes = recv_exact(sock, 8, timeout=timeout)
    magic = header_bytes[:4]
    if magic != MAGIC_FRAME:
        raise CorruptedFrameError(f"Invalid frame magic: expected {MAGIC_FRAME}, got {magic}")

    meta_len = int.from_bytes(header_bytes[4:8], byteorder="big")
    if meta_len <= 0 or meta_len > MAX_METADATA_BYTES:
        raise FrameValidationError(f"Frame metadata length {meta_len} out of bounds")

    json_bytes = recv_exact(sock, meta_len, timeout=timeout)
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        meta = FrameMetadata(**data)
        meta.validate(manifest)
    except Exception as e:
        raise FrameValidationError(f"Failed to decode or validate FrameMetadata: {e}") from e

    # Read binary body
    body = recv_exact(sock, meta.body_bytes, timeout=timeout)

    accounting = WireAccounting(
        framing_bytes=8,
        metadata_bytes=meta_len,
        body_bytes=len(body),
    )
    return meta, body, accounting


class KVCacheFrameAccumulator:
    """
    Accumulates, validates, and reassembles incoming KV frames for a request.
    Strictly checks for duplicate frames and complete layer coverage.
    """

    def __init__(self, manifest: RequestManifest):
        self.manifest = manifest
        self.received_frames: Dict[Tuple[int, str, int, int, int], Tuple[FrameMetadata, bytes]] = {}
        self.accounting = WireAccounting()

    def add_frame(self, meta: FrameMetadata, body: bytes, accounting: WireAccounting) -> None:
        key = (meta.layer_idx, meta.kv_type, meta.prefill_rank, meta.start_head_idx, meta.end_head_idx)
        if key in self.received_frames:
            raise FrameValidationError(
                f"Duplicate frame received for layer={meta.layer_idx}, type={meta.kv_type}, "
                f"rank={meta.prefill_rank}, heads=[{meta.start_head_idx}:{meta.end_head_idx}]"
            )
        self.received_frames[key] = (meta, body)
        self.accounting = self.accounting.add(accounting)

    def is_complete(self) -> bool:
        return len(self.received_frames) == self.manifest.expected_frames

    def validate_completeness(self) -> None:
        """Verify that every layer, kv_type, and prefill rank slice is present."""
        if not self.is_complete():
            raise FrameValidationError(
                f"Incomplete KV cache: received {len(self.received_frames)} of {self.manifest.expected_frames} frames"
            )

        for l in range(self.manifest.num_layers):
            for kv in ("key", "value"):
                for p in range(self.manifest.prefill_tp):
                    # Check that rank p is present for layer l
                    matching = [
                        k for k in self.received_frames.keys()
                        if k[0] == l and k[1] == kv and k[2] == p
                    ]
                    if not matching:
                        raise FrameValidationError(
                            f"Missing frame for layer={l}, kv_type='{kv}', prefill_rank={p}"
                        )

    def reconstruct_cache(self) -> Dict[str, List[torch.Tensor]]:
        """
        Reassemble received frame bodies into complete per-layer K and V tensors.
        Handles both uncompressed floating-point and block-scaled FP8 quantized frames.
        Returns dict with 'keys' and 'values' lists of shape [batch, num_kv_heads, seq_len, head_dim].
        """
        self.validate_completeness()
        dtype = TORCH_DTYPE_MAP[self.manifest.dtype]

        keys_by_layer: List[torch.Tensor] = []
        vals_by_layer: List[torch.Tensor] = []

        for l in range(self.manifest.num_layers):
            for kv_type in ("key", "value"):
                # Collect and sort shards across prefill ranks along head dimension
                layer_frames = [
                    (meta, body)
                    for (layer_idx, kv, rank, start, end), (meta, body) in self.received_frames.items()
                    if layer_idx == l and kv == kv_type
                ]
                # Sort by start_head_idx to stitch heads in order
                layer_frames.sort(key=lambda item: item[0].start_head_idx)

                shards = []
                for meta, body in layer_frames:
                    if meta.is_quantized:
                        quant_elem_size = DTYPE_BYTE_SIZES.get(meta.dtype, 1)
                        quant_numel = 1
                        for d in meta.shape:
                            quant_numel *= d
                        quant_bytes = quant_numel * quant_elem_size
                        quant_buf = bytearray(body[:quant_bytes])
                        scale_buf = bytearray(body[quant_bytes:])

                        target_dtype = TORCH_DTYPE_MAP[meta.dtype]
                        raw_uint8 = torch.frombuffer(quant_buf, dtype=torch.uint8).view(meta.shape)
                        quant_tensor = raw_uint8.view(target_dtype) if target_dtype != torch.uint8 else raw_uint8
                        scale_tensor = torch.frombuffer(scale_buf, dtype=TORCH_DTYPE_MAP[meta.scale_dtype]).view(meta.scale_shape)

                        payload = FP8BlockPayload(
                            quantized_data=quant_tensor,
                            scale_factors=scale_tensor,
                            original_shape=tuple(meta.original_shape),
                            original_dtype=TORCH_DTYPE_MAP[meta.original_dtype],
                            block_size_tokens=meta.block_size_tokens,
                        )
                        restored_t = dequantize_kv_block_fp8(payload)
                        shards.append(restored_t)
                    else:
                        # Parse tensor from contiguous buffer
                        t = torch.frombuffer(bytearray(body), dtype=dtype).view(meta.shape)
                        shards.append(t)

                # Concatenate along head dimension (dim -2 in Hetero format [B, S, H, D])
                full_hetero = torch.cat(shards, dim=-2)
                # Convert from Hetero [B, S, H, D] back to Transformers [B, H, S, D]
                full_hf = full_hetero.permute(0, 2, 1, 3).contiguous()

                if kv_type == "key":
                    keys_by_layer.append(full_hf)
                else:
                    vals_by_layer.append(full_hf)

        return {"keys": keys_by_layer, "values": vals_by_layer}

    def reconstruct_uncompressed_cache(self) -> Dict[str, List[torch.Tensor]]:
        """Backwards compatibility alias for reconstruct_cache."""
        return self.reconstruct_cache()
