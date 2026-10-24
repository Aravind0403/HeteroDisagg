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
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import math
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Set
import torch

from hetero_kv.layout import KVPartitionPlan, KVHeadSlice
from hetero_kv.quant.fp8_block import FP8BlockPayload, dequantize_kv_block_fp8

MAGIC_MANIFEST = b"HKVM"
MAGIC_FRAME = b"HKVF"
MAGIC_ACK_MANIFEST = b"ACKM"
MAGIC_REJ_MANIFEST = b"REJM"
MAGIC_ACK_TRANSFER = b"ACKT"
MAGIC_VAL_PASSED = b"VALP"
MAGIC_VAL_FAILED = b"VALF"

MAX_METADATA_BYTES = 64 * 1024                # 64 KB max metadata size
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024        # 1 GB max single tensor frame size
MAX_TENSOR_BYTES = 1024 * 1024 * 1024         # 1 GB max decoded tensor size
MAX_REQUEST_BYTES = 16 * 1024 * 1024 * 1024   # 16 GB max cumulative bytes for a single request
DEFAULT_TIMEOUT = 10.0                         # 10s default socket timeout

SUPPORTED_UNCOMPRESSED_DTYPES = {"float32", "float16", "bfloat16"}
SUPPORTED_QUANTIZED_DTYPES = {"float8_e4m3fn", "int8"}
SUPPORTED_SCALE_DTYPES = {"float32", "float16", "bfloat16"}

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
    """Request ID, model identity, checkpoint revision, or destination rank mismatch."""
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
    decode_rank: int = 0

    def validate(self) -> None:
        if not self.request_id:
            raise ManifestValidationError("request_id cannot be empty")
        if not self.model_id:
            raise ManifestValidationError("model_id cannot be empty")
        if self.num_layers <= 0:
            raise ManifestValidationError(f"Invalid num_layers: {self.num_layers}")
        if self.num_kv_heads <= 0:
            raise ManifestValidationError(f"Invalid num_kv_heads: {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ManifestValidationError(f"Invalid head_dim: {self.head_dim}")
        if self.seq_len <= 0:
            raise ManifestValidationError(f"Invalid seq_len: {self.seq_len}")
        if self.prefill_tp <= 0 or self.decode_tp <= 0:
            raise ManifestValidationError(f"Invalid TP: prefill={self.prefill_tp}, decode={self.decode_tp}")
        if self.decode_rank < 0 or self.decode_rank >= self.decode_tp:
            raise ManifestValidationError(
                f"decode_rank {self.decode_rank} out of bounds [0, {self.decode_tp - 1}]"
            )
        if self.dtype not in SUPPORTED_UNCOMPRESSED_DTYPES:
            raise ManifestValidationError(
                f"Unsupported dtype '{self.dtype}', allowed: {sorted(SUPPORTED_UNCOMPRESSED_DTYPES)}"
            )

        # Construct and validate bipartite partition plan
        try:
            plan = KVPartitionPlan.create(self.num_kv_heads, self.prefill_tp, self.decode_tp)
        except ValueError as e:
            raise ManifestValidationError(f"Invalid partition geometry: {e}") from e

        target_routes = [r for r in plan.transfer_routes if r.decode_rank == self.decode_rank]
        expected_frames_for_rank = len(target_routes) * 2 * self.num_layers
        if self.expected_frames != expected_frames_for_rank:
            raise ManifestValidationError(
                f"expected_frames mismatch for decode_rank {self.decode_rank}: got {self.expected_frames}, "
                f"expected {expected_frames_for_rank} based on partition plan"
            )


def validate_manifest_compatibility(
    manifest: RequestManifest,
    model_config: Any,
    expected_model_id: Optional[str] = None,
    expected_decode_rank: Optional[int] = None,
    expected_checkpoint_commit_hash: Optional[str] = None,
    expected_dtype: Optional[str] = None,
) -> None:
    """
    Validates that an incoming RequestManifest matches the receiving consumer's
    expected model architecture, identity, checkpoint revision, target decode rank,
    and tensor dtype before accepting cache frames.
    """
    manifest.validate()

    if expected_model_id is not None and manifest.model_id != expected_model_id:
        raise MismatchedRequestError(
            f"Manifest model_id '{manifest.model_id}' does not match consumer expected '{expected_model_id}'"
        )

    # Validate resolved checkpoint revision / commit hash
    resolved_commit_hash = expected_checkpoint_commit_hash or getattr(model_config, "_commit_hash", None)
    if resolved_commit_hash is not None:
        if manifest.checkpoint_commit_hash is None:
            raise MismatchedRequestError(
                f"Manifest lacks checkpoint revision, but consumer expects revision '{resolved_commit_hash}'"
            )
        if manifest.checkpoint_commit_hash != resolved_commit_hash:
            raise MismatchedRequestError(
                f"Manifest checkpoint revision '{manifest.checkpoint_commit_hash}' "
                f"does not match consumer resolved revision '{resolved_commit_hash}'"
            )

    if expected_decode_rank is not None and manifest.decode_rank != expected_decode_rank:
        raise ManifestValidationError(
            f"Manifest decode_rank {manifest.decode_rank} does not match consumer expected {expected_decode_rank}"
        )

    if expected_dtype is not None and manifest.dtype != expected_dtype:
        raise ManifestValidationError(
            f"Manifest dtype '{manifest.dtype}' does not match consumer expected dtype '{expected_dtype}'"
        )

    config_num_layers = getattr(model_config, "num_hidden_layers", getattr(model_config, "num_layers", None))
    if config_num_layers is not None and manifest.num_layers != config_num_layers:
        raise ManifestValidationError(
            f"Manifest num_layers ({manifest.num_layers}) does not match model config ({config_num_layers})"
        )

    config_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", None))
    if config_kv_heads is not None and manifest.num_kv_heads != config_kv_heads:
        raise ManifestValidationError(
            f"Manifest num_kv_heads ({manifest.num_kv_heads}) does not match model config ({config_kv_heads})"
        )

    config_head_dim = getattr(model_config, "head_dim", None)
    if config_head_dim is None and hasattr(model_config, "hidden_size") and hasattr(model_config, "num_attention_heads"):
        config_head_dim = model_config.hidden_size // model_config.num_attention_heads
    if config_head_dim is not None and manifest.head_dim != config_head_dim:
        raise ManifestValidationError(
            f"Manifest head_dim ({manifest.head_dim}) does not match model config ({config_head_dim})"
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

    def validate(
        self,
        manifest: RequestManifest,
        plan: Optional[KVPartitionPlan] = None,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_tensor_bytes: int = MAX_TENSOR_BYTES,
    ) -> None:
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
        if self.decode_rank < 0 or self.decode_rank >= manifest.decode_tp:
            raise FrameValidationError(
                f"decode_rank {self.decode_rank} out of bounds [0, {manifest.decode_tp - 1}]"
            )
        if self.decode_rank != manifest.decode_rank:
            raise FrameValidationError(
                f"Frame decode_rank {self.decode_rank} does not match manifest decode_rank {manifest.decode_rank}"
            )
        if self.start_head_idx >= self.end_head_idx:
            raise FrameValidationError(
                f"Invalid head range: [{self.start_head_idx}, {self.end_head_idx}]"
            )

        # Validate transfer route and head ownership against KVPartitionPlan
        if plan is None:
            plan = KVPartitionPlan.create(manifest.num_kv_heads, manifest.prefill_tp, manifest.decode_tp)

        matching_routes = [
            r for r in plan.transfer_routes
            if r.prefill_rank == self.prefill_rank
            and r.decode_rank == self.decode_rank
            and r.head_slice.start_head_idx == self.start_head_idx
            and r.head_slice.end_head_idx == self.end_head_idx
        ]
        if not matching_routes:
            raise FrameValidationError(
                f"Unauthorized route: no transfer route in partition plan from prefill_rank {self.prefill_rank} "
                f"to decode_rank {self.decode_rank} for head slice [{self.start_head_idx}:{self.end_head_idx}]"
            )

        # Enforce encoded-body payload limit before allocation
        if self.body_bytes <= 0:
            raise FrameValidationError("Frame body_bytes must be strictly positive")
        if self.body_bytes > max_payload_bytes:
            raise FrameValidationError(
                f"Frame body size {self.body_bytes} exceeds maximum allowed payload limit {max_payload_bytes}"
            )

        # Strict validation of shapes, sequence lengths, dtypes, and scale geometry
        declared_num_heads = self.end_head_idx - self.start_head_idx

        if not self.is_quantized:
            if self.dtype not in SUPPORTED_UNCOMPRESSED_DTYPES:
                raise FrameValidationError(
                    f"Unsupported uncompressed frame dtype '{self.dtype}', allowed: {sorted(SUPPORTED_UNCOMPRESSED_DTYPES)}"
                )
            if self.dtype != manifest.dtype:
                raise FrameValidationError(
                    f"Frame dtype '{self.dtype}' does not match manifest dtype '{manifest.dtype}'"
                )
            if len(self.shape) != 4:
                raise FrameValidationError(
                    f"Uncompressed frame shape must be 4D [batch, seq_len, num_heads, head_dim], got {self.shape}"
                )
            if self.shape[0] != 1:
                raise FrameValidationError(
                    f"Batch dimension {self.shape[0]} not supported; codec requires batch size 1"
                )
            if self.shape[1] != manifest.seq_len:
                raise FrameValidationError(
                    f"Frame seq_len ({self.shape[1]}) does not match manifest seq_len ({manifest.seq_len})"
                )
            if self.shape[2] != declared_num_heads:
                raise FrameValidationError(
                    f"Frame head count dimension ({self.shape[2]}) does not match declared heads ({declared_num_heads})"
                )
            if self.shape[3] != manifest.head_dim:
                raise FrameValidationError(
                    f"Frame head_dim ({self.shape[3]}) does not match manifest head_dim ({manifest.head_dim})"
                )

            elem_size = DTYPE_BYTE_SIZES[self.dtype]
            num_elements = math.prod(self.shape)
            expected_bytes = num_elements * elem_size
            if self.body_bytes != expected_bytes:
                raise FrameValidationError(
                    f"Frame body_bytes ({self.body_bytes}) does not match shape ({self.shape}) "
                    f"and dtype ({self.dtype}) expected bytes ({expected_bytes})"
                )
            if expected_bytes > max_tensor_bytes:
                raise FrameValidationError(
                    f"Decoded tensor size {expected_bytes} exceeds limit {max_tensor_bytes}"
                )
        else:
            if not self.scale_shape or not self.scale_dtype or not self.original_shape or not self.original_dtype:
                raise FrameValidationError("Missing required FP8 scale or shape metadata for quantized frame")
            if self.dtype not in SUPPORTED_QUANTIZED_DTYPES:
                raise FrameValidationError(
                    f"Unsupported FP8 quantized dtype '{self.dtype}', allowed: {sorted(SUPPORTED_QUANTIZED_DTYPES)}"
                )
            if self.scale_dtype not in SUPPORTED_SCALE_DTYPES:
                raise FrameValidationError(
                    f"Unsupported FP8 scale dtype '{self.scale_dtype}', allowed: {sorted(SUPPORTED_SCALE_DTYPES)}"
                )
            if self.original_dtype not in SUPPORTED_UNCOMPRESSED_DTYPES:
                raise FrameValidationError(
                    f"Unsupported FP8 original dtype '{self.original_dtype}', allowed: {sorted(SUPPORTED_UNCOMPRESSED_DTYPES)}"
                )
            if self.original_dtype != manifest.dtype:
                raise FrameValidationError(
                    f"Quantized frame original_dtype '{self.original_dtype}' does not match manifest dtype '{manifest.dtype}'"
                )
            if len(self.original_shape) != 4:
                raise FrameValidationError(
                    f"Quantized original shape must be 4D [batch, seq_len, num_heads, head_dim], got {self.original_shape}"
                )
            if self.original_shape[0] != 1:
                raise FrameValidationError(
                    f"Batch dimension {self.original_shape[0]} not supported; codec requires batch size 1"
                )
            if self.original_shape[1] != manifest.seq_len:
                raise FrameValidationError(
                    f"Quantized frame original seq_len ({self.original_shape[1]}) does not match manifest seq_len ({manifest.seq_len})"
                )
            if self.original_shape[2] != declared_num_heads:
                raise FrameValidationError(
                    f"Quantized frame original heads ({self.original_shape[2]}) does not match declared heads ({declared_num_heads})"
                )
            if self.original_shape[3] != manifest.head_dim:
                raise FrameValidationError(
                    f"Quantized frame head_dim ({self.original_shape[3]}) does not match manifest head_dim ({manifest.head_dim})"
                )
            if self.block_size_tokens != manifest.block_size_tokens:
                raise FrameValidationError(
                    f"Quantized frame block_size ({self.block_size_tokens}) does not match manifest ({manifest.block_size_tokens})"
                )

            # Validate FP8 scale factor geometry
            block_size = manifest.block_size_tokens
            expected_num_blocks = (manifest.seq_len + block_size - 1) // block_size
            expected_scale_shape = [expected_num_blocks]
            if list(self.scale_shape) != expected_scale_shape:
                raise FrameValidationError(
                    f"Invalid FP8 scale shape {self.scale_shape}, expected {expected_scale_shape}"
                )

            expected_quant_shape = [expected_num_blocks, block_size * declared_num_heads * manifest.head_dim]
            if list(self.shape) != expected_quant_shape:
                raise FrameValidationError(
                    f"Invalid FP8 quantized data shape {self.shape}, expected {expected_quant_shape}"
                )

            quant_elem_size = DTYPE_BYTE_SIZES.get(self.dtype, 1)
            quant_bytes = math.prod(self.shape) * quant_elem_size
            scale_elem_size = DTYPE_BYTE_SIZES.get(self.scale_dtype, 4)
            scale_bytes = math.prod(self.scale_shape) * scale_elem_size

            expected_bytes = quant_bytes + scale_bytes
            if self.body_bytes != expected_bytes:
                raise FrameValidationError(
                    f"Quantized frame body_bytes ({self.body_bytes}) does not match quant + scale expected bytes ({expected_bytes})"
                )

            # Check decoded tensor size against allocation limit
            orig_elem_size = DTYPE_BYTE_SIZES.get(self.original_dtype, 4)
            decoded_bytes = math.prod(self.original_shape) * orig_elem_size
            if decoded_bytes > max_tensor_bytes:
                raise FrameValidationError(
                    f"Decoded tensor size {decoded_bytes} exceeds limit {max_tensor_bytes}"
                )


def recv_exact(
    sock: socket.socket,
    n_bytes: int,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> bytes:
    """
    Read exactly n_bytes from socket with strict deadline enforcement and allocation cap.
    Raises IncompleteTransferError if connection terminates prematurely.
    Raises TransportTimeoutError if deadline expires.
    Raises FrameValidationError if requested allocation exceeds max_bytes.
    """
    if n_bytes < 0:
        raise ValueError(f"n_bytes must be non-negative, got {n_bytes}")
    if n_bytes > max_bytes:
        raise FrameValidationError(
            f"Requested allocation of {n_bytes} bytes exceeds configured limit ({max_bytes} bytes)"
        )
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
    header_bytes = recv_exact(sock, 8, timeout=timeout, max_bytes=8)
    magic = header_bytes[:4]
    if magic != MAGIC_MANIFEST:
        raise CorruptedFrameError(f"Invalid manifest magic: expected {MAGIC_MANIFEST}, got {magic}")

    meta_len = int.from_bytes(header_bytes[4:8], byteorder="big")
    if meta_len <= 0 or meta_len > MAX_METADATA_BYTES:
        raise ManifestValidationError(f"Manifest length {meta_len} out of allowed bounds [1, {MAX_METADATA_BYTES}]")

    json_bytes = recv_exact(sock, meta_len, timeout=timeout, max_bytes=MAX_METADATA_BYTES)
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        manifest = RequestManifest(**data)
        manifest.validate()
    except (ManifestValidationError, CorruptedFrameError):
        raise
    except Exception as e:
        raise ManifestValidationError(f"Failed to decode or validate RequestManifest: {e}") from e

    accounting = WireAccounting(
        framing_bytes=8,
        metadata_bytes=meta_len,
        body_bytes=0,
    )
    return manifest, accounting


def send_handshake_msg(sock: socket.socket, magic: bytes, payload_dict: Dict[str, Any]) -> WireAccounting:
    """Sends a 4-byte magic + 4-byte big-endian length + JSON payload handshake packet."""
    if len(magic) != 4:
        raise ValueError(f"Magic must be exactly 4 bytes, got {magic}")
    json_bytes = json.dumps(payload_dict).encode("utf-8")
    meta_len = len(json_bytes)
    if meta_len > MAX_METADATA_BYTES:
        raise FrameValidationError(f"Handshake JSON exceeds max size: {meta_len} > {MAX_METADATA_BYTES}")
    header = magic + meta_len.to_bytes(4, byteorder="big")
    sock.sendall(header + json_bytes)
    return WireAccounting(
        framing_bytes=len(header),
        metadata_bytes=meta_len,
        body_bytes=0,
    )


def recv_handshake_msg(
    sock: socket.socket,
    expected_magic: Optional[bytes] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bytes, Dict[str, Any], WireAccounting]:
    """
    Receives a 4-byte magic + 4-byte length + JSON payload packet with bounded exact-length reads
    and a single absolute deadline for the entire handshake.
    Handles fragmented TCP headers and slow-drip delivery seamlessly.
    """
    deadline = time.monotonic() + timeout

    rem_header = deadline - time.monotonic()
    if rem_header <= 0:
        raise TransportTimeoutError("Handshake timed out before header read")
    header_bytes = recv_exact(sock, 8, timeout=rem_header, max_bytes=8)
    magic = header_bytes[:4]
    if expected_magic is not None and magic != expected_magic:
        raise CorruptedFrameError(f"Handshake magic mismatch: expected {expected_magic}, got {magic}")

    meta_len = int.from_bytes(header_bytes[4:8], byteorder="big")
    if meta_len <= 0 or meta_len > MAX_METADATA_BYTES:
        raise FrameValidationError(f"Handshake length {meta_len} out of allowed bounds [1, {MAX_METADATA_BYTES}]")

    rem_body = deadline - time.monotonic()
    if rem_body <= 0:
        raise TransportTimeoutError("Handshake timed out before body read")
    body_bytes = recv_exact(sock, meta_len, timeout=rem_body, max_bytes=MAX_METADATA_BYTES)

    try:
        payload_data = json.loads(body_bytes.decode("utf-8"))
    except Exception as e:
        raise CorruptedFrameError(f"Failed to decode handshake JSON: {e}") from e

    accounting = WireAccounting(
        framing_bytes=8,
        metadata_bytes=meta_len,
        body_bytes=0,
    )
    return magic, payload_data, accounting


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


class ByteCountingSink:
    """Socket-like destination that tallies serialized wire bytes without network I/O."""

    def __init__(self) -> None:
        self.bytes_sent: int = 0

    def sendall(self, data: bytes) -> None:
        self.bytes_sent += len(data)


def recv_frame(
    sock: socket.socket,
    manifest: RequestManifest,
    timeout: float = DEFAULT_TIMEOUT,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    cumulative_bytes: int = 0,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    plan: Optional[KVPartitionPlan] = None,
) -> Tuple[FrameMetadata, bytes, WireAccounting]:
    """Receive individual tensor frame from socket with bounded validation."""
    header_bytes = recv_exact(sock, 8, timeout=timeout, max_bytes=8)
    magic = header_bytes[:4]
    if magic != MAGIC_FRAME:
        raise CorruptedFrameError(f"Invalid frame magic: expected {MAGIC_FRAME}, got {magic}")

    meta_len = int.from_bytes(header_bytes[4:8], byteorder="big")
    if meta_len <= 0 or meta_len > MAX_METADATA_BYTES:
        raise FrameValidationError(f"Frame metadata length {meta_len} out of bounds")

    json_bytes = recv_exact(sock, meta_len, timeout=timeout, max_bytes=MAX_METADATA_BYTES)
    try:
        data = json.loads(json_bytes.decode("utf-8"))
        meta = FrameMetadata(**data)
        meta.validate(manifest, plan=plan, max_payload_bytes=max_payload_bytes)
    except (MismatchedRequestError, FrameValidationError, ManifestValidationError, CorruptedFrameError):
        raise
    except Exception as e:
        raise FrameValidationError(f"Failed to decode FrameMetadata: {e}") from e

    # Explicit check before allocating buffer
    if meta.body_bytes > max_payload_bytes:
        raise FrameValidationError(
            f"Frame body size {meta.body_bytes} exceeds maximum allowed payload limit {max_payload_bytes}"
        )
    if cumulative_bytes + meta.body_bytes > max_request_bytes:
        raise FrameValidationError(
            f"Cumulative transfer size {cumulative_bytes + meta.body_bytes} exceeds request limit {max_request_bytes}"
        )

    # Read binary body with bounded allocation
    body = recv_exact(sock, meta.body_bytes, timeout=timeout, max_bytes=max_payload_bytes)

    accounting = WireAccounting(
        framing_bytes=8,
        metadata_bytes=meta_len,
        body_bytes=len(body),
    )
    return meta, body, accounting


class KVCacheFrameAccumulator:
    """
    Accumulates, validates, and reassembles incoming KV frames for a request.
    Strictly verifies partition plan ownership, destination, complete head coverage,
    sequence dimensions, dtypes, and FP8 scale geometry against KVPartitionPlan.
    """

    def __init__(
        self,
        manifest: RequestManifest,
        max_request_bytes: int = MAX_REQUEST_BYTES,
    ):
        self.manifest = manifest
        self.max_request_bytes = max_request_bytes
        self.plan = KVPartitionPlan.create(manifest.num_kv_heads, manifest.prefill_tp, manifest.decode_tp)
        self.target_decode_rank = manifest.decode_rank
        self.expected_slice = self.plan.decode_head_slices[self.target_decode_rank]
        self.received_frames: Dict[Tuple[int, str, int, int, int], Tuple[FrameMetadata, bytes]] = {}
        self.accounting = WireAccounting()
        self.cumulative_body_bytes = 0

    def add_frame(self, meta: FrameMetadata, body: bytes, accounting: WireAccounting) -> None:
        # Validate frame against manifest and plan
        meta.validate(self.manifest, plan=self.plan)

        if meta.decode_rank != self.target_decode_rank:
            raise FrameValidationError(
                f"Frame decode_rank {meta.decode_rank} does not match accumulator target {self.target_decode_rank}"
            )

        if self.cumulative_body_bytes + meta.body_bytes > self.max_request_bytes:
            raise FrameValidationError(
                f"Cumulative request body bytes {self.cumulative_body_bytes + meta.body_bytes} "
                f"exceeds maximum allowed limit {self.max_request_bytes}"
            )

        # Check for duplicate frame
        key = (meta.layer_idx, meta.kv_type, meta.prefill_rank, meta.start_head_idx, meta.end_head_idx)
        if key in self.received_frames:
            raise FrameValidationError(
                f"Duplicate frame received for layer={meta.layer_idx}, type={meta.kv_type}, "
                f"rank={meta.prefill_rank}, heads=[{meta.start_head_idx}:{meta.end_head_idx}]"
            )

        # Check for overlapping head ranges in the same (layer_idx, kv_type)
        new_slice = KVHeadSlice(start_head_idx=meta.start_head_idx, end_head_idx=meta.end_head_idx)
        for (l, kv, r, s, e) in self.received_frames.keys():
            if l == meta.layer_idx and kv == meta.kv_type:
                existing_slice = KVHeadSlice(start_head_idx=s, end_head_idx=e)
                if new_slice.overlaps_with(existing_slice):
                    raise FrameValidationError(
                        f"Overlapping head slice [{meta.start_head_idx}:{meta.end_head_idx}] from rank {meta.prefill_rank} "
                        f"overlaps with existing frame [{s}:{e}] from rank {r} on layer {l} ({kv})"
                    )

        self.received_frames[key] = (meta, body)
        self.accounting = self.accounting.add(accounting)
        self.cumulative_body_bytes += meta.body_bytes

    def is_complete(self) -> bool:
        return len(self.received_frames) == self.manifest.expected_frames

    def validate_completeness(self) -> None:
        """Verify that every layer and kv_type has complete, contiguous head coverage with no gaps."""
        if not self.is_complete():
            raise FrameValidationError(
                f"Incomplete KV cache: received {len(self.received_frames)} of {self.manifest.expected_frames} frames"
            )

        for l in range(self.manifest.num_layers):
            for kv in ("key", "value"):
                layer_frames = [
                    (meta, body)
                    for (layer_idx, kv_type, _, _, _), (meta, body) in self.received_frames.items()
                    if layer_idx == l and kv_type == kv
                ]
                layer_frames.sort(key=lambda item: item[0].start_head_idx)

                curr_head = self.expected_slice.start_head_idx
                for meta, _ in layer_frames:
                    if meta.start_head_idx != curr_head:
                        raise FrameValidationError(
                            f"Head range gap on layer {l} ({kv}): expected head {curr_head}, got {meta.start_head_idx}"
                        )
                    curr_head = meta.end_head_idx

                if curr_head != self.expected_slice.end_head_idx:
                    raise FrameValidationError(
                        f"Incomplete head coverage on layer {l} ({kv}): ended at head {curr_head}, "
                        f"expected {self.expected_slice.end_head_idx}"
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
                layer_frames = [
                    (meta, body)
                    for (layer_idx, kv, rank, start, end), (meta, body) in self.received_frames.items()
                    if layer_idx == l and kv == kv_type
                ]
                layer_frames.sort(key=lambda item: item[0].start_head_idx)

                shards = []
                for meta, body in layer_frames:
                    if meta.is_quantized:
                        quant_elem_size = DTYPE_BYTE_SIZES.get(meta.dtype, 1)
                        quant_numel = math.prod(meta.shape)
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

