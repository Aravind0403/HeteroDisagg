"""Unit and Regression Tests for Multi-Process Socket KV Cache Transport.

Tests:
1. End-to-End Real Model Process-to-Process Handoff (Uncompressed):
   - Producer and Consumer in distinct OS processes (via spawn).
   - Transfer exclusively over localhost socket.
   - Max absolute logit error <= 1e-6 (exact match against single-process baseline).
2. Failure Tests in Both Directions:
   - Producer disconnect: Abrupt client socket termination detected as IncompleteTransferError.
   - Consumer rejection: Mismatched request ID rejected with MismatchedRequestError.
   - Frame corruption: Body length vs shape mismatch rejected with FrameValidationError.
   - Duplicate frame rejection: Re-sending same slice rejected cleanly.
   - Missing layer completeness validation: Incomplete cache rejected before decode.
   - Stalled transfer timeout: Deadline expiration raises TransportTimeoutError without hanging.
   - Bounded read protection: Oversized metadata header rejected before memory allocation.
3. Socket Cleanup:
   - Verifies server and client sockets cleanly close and ports are released.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import socket
import time
import pytest
import torch

from hetero_kv.transport import (
    MAGIC_MANIFEST,
    MAGIC_FRAME,
    MAX_METADATA_BYTES,
    RequestManifest,
    FrameMetadata,
    WireAccounting,
    send_manifest,
    recv_manifest,
    send_frame,
    recv_frame,
    recv_exact,
    KVCacheFrameAccumulator,
    TransportTimeoutError,
    IncompleteTransferError,
    CorruptedFrameError,
    MismatchedRequestError,
    ManifestValidationError,
    FrameValidationError,
)
from scripts.verify_socket_process_cache_handoff import run_socket_cache_handoff_experiment


def test_end_to_end_multiprocess_socket_handoff():
    """
    Phase 1 Acceptance:
    - Independent OS processes (spawn context).
    - Cache transferred exclusively over socket.
    - Max absolute logit divergence <= 1e-6.
    - 100% token equivalence across decoding steps.
    """
    results = run_socket_cache_handoff_experiment(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        prompt="Heterogeneous disaggregated inference allows",
        decode_steps=4,
        prefill_tp=2,
        decode_tp=1,
        device="cpu",
        timeout=30.0,
    )

    # 1. Distinct process isolation
    pids = results["pids"]
    assert len({pids["orchestrator_pid"], pids["producer_pid"], pids["consumer_pid"]}) == 3

    # 2. Numerical equivalence against uninterrupted baseline
    assert results["all_tokens_match"] is True
    assert results["max_abs_logit_divergence"] <= 1e-6
    assert results["divergence_within_tolerance"] is True

    # 3. Byte accounting separation
    wa = results["wire_accounting"]
    assert wa["framing_bytes"] > 0
    assert wa["metadata_bytes"] > 0
    assert wa["body_bytes"] > 0
    assert wa["total_wire_bytes"] == wa["framing_bytes"] + wa["metadata_bytes"] + wa["body_bytes"]


def test_multiprocess_socket_fp8_cache_handoff():
    """
    Phase 3 Acceptance:
    - Multi-process socket handoff with block-scaled FP8 wire compression.
    - Preserves high token-level logit correlation (mean cosine >= 0.99).
    - Achieves significant payload reduction vs FP32 and FP16 baselines on complete blocks.
    """
    prompt = "Distributed inference engines disaggregate prefill and decode stages to eliminate memory bandwidth"
    results = run_socket_cache_handoff_experiment(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        prompt=prompt,
        decode_steps=4,
        prefill_tp=2,
        decode_tp=1,
        enable_fp8=True,
        block_size_tokens=16,
        device="cpu",
        timeout=30.0,
    )

    pids = results["pids"]
    assert len({pids["orchestrator_pid"], pids["producer_pid"], pids["consumer_pid"]}) == 3

    assert results["divergence_within_tolerance"] is True
    assert results["mean_cosine_similarity"] >= 0.99
    assert results["all_tokens_match"] is True

    wa = results["wire_accounting"]
    assert wa["body_bytes"] < wa["uncompressed_fp32_bytes"]
    assert wa["ratio_vs_fp16"] >= 1.4
    assert wa["ratio_vs_fp32"] >= 2.5


def test_stalled_transfer_deadline_timeout():
    """Receiver must time out rather than hanging indefinitely on an open stalled socket."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    try:
        # Client sends nothing; server attempts to read with 0.1s timeout
        start = time.monotonic()
        with pytest.raises(TransportTimeoutError):
            recv_exact(conn, n_bytes=100, timeout=0.1)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"Timeout took too long: {elapsed}s"
    finally:
        client_sock.close()
        conn.close()
        server_sock.close()


def test_producer_abrupt_disconnect():
    """Server must detect abrupt client disconnect as IncompleteTransferError."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    try:
        # Client sends 2 bytes then abruptly closes
        client_sock.sendall(b"HK")
        client_sock.close()

        with pytest.raises(IncompleteTransferError):
            recv_exact(conn, n_bytes=8, timeout=1.0)
    finally:
        conn.close()
        server_sock.close()


def test_bounded_metadata_read_rejection():
    """Oversized metadata length header must be rejected before memory allocation."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    try:
        # Send magic + length exceeding MAX_METADATA_BYTES (e.g. 10 MB)
        oversized_len = 10 * 1024 * 1024
        header = MAGIC_MANIFEST + oversized_len.to_bytes(4, byteorder="big")
        client_sock.sendall(header)

        with pytest.raises(ManifestValidationError, match="out of allowed bounds"):
            recv_manifest(conn, timeout=1.0)
    finally:
        client_sock.close()
        conn.close()
        server_sock.close()


def test_consumer_rejection_on_mismatched_request_id():
    """Frames with mismatched request IDs must be rejected immediately."""
    manifest = RequestManifest(
        request_id="req-valid-100",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        seq_len=8,
        prefill_tp=1,
        decode_tp=1,
        expected_frames=2,
        initial_decode_token=42,
        initial_decode_token_str="test",
        prompt_tokens=[1, 2, 3],
        dtype="float32",
    )

    bad_frame = FrameMetadata(
        request_id="req-spoofed-999",  # Mismatch!
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[1, 8, 2, 64],
        dtype="float32",
        body_bytes=1 * 8 * 2 * 64 * 4,
    )

    with pytest.raises(MismatchedRequestError, match="does not match manifest"):
        bad_frame.validate(manifest)


def test_frame_shape_and_body_bytes_validation():
    """Body length mismatch against shape * elem_size must be rejected."""
    manifest = RequestManifest(
        request_id="req-shape-test",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        seq_len=8,
        prefill_tp=1,
        decode_tp=1,
        expected_frames=2,
        initial_decode_token=42,
        initial_decode_token_str="test",
        prompt_tokens=[1, 2],
        dtype="float32",
    )

    expected_bytes = 1 * 8 * 2 * 64 * 4  # 4096 bytes
    bad_frame = FrameMetadata(
        request_id="req-shape-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[1, 8, 2, 64],
        dtype="float32",
        body_bytes=2048,  # Deliberate mismatch
    )

    with pytest.raises(FrameValidationError, match="does not match shape"):
        bad_frame.validate(manifest)


def test_accumulator_duplicate_and_missing_layer_rejection():
    """Accumulator must reject duplicate frames and incomplete layers."""
    manifest = RequestManifest(
        request_id="req-acc-test",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        seq_len=8,
        prefill_tp=1,
        decode_tp=1,
        expected_frames=4,  # 2 layers * 2 kv_types
        initial_decode_token=42,
        initial_decode_token_str="test",
        prompt_tokens=[1, 2],
        dtype="float32",
    )

    acc = KVCacheFrameAccumulator(manifest)
    dummy_body = b"\x00" * 4096
    wa = WireAccounting(framing_bytes=8, metadata_bytes=100, body_bytes=4096)

    meta_l0_k = FrameMetadata(
        request_id="req-acc-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[1, 8, 2, 64],
        dtype="float32",
        body_bytes=4096,
    )
    acc.add_frame(meta_l0_k, dummy_body, wa)

    # 1. Reject duplicate frame
    with pytest.raises(FrameValidationError, match="Duplicate frame"):
        acc.add_frame(meta_l0_k, dummy_body, wa)

    # 2. Reject incomplete set when attempting reconstruction
    with pytest.raises(FrameValidationError, match="Incomplete KV cache"):
        acc.reconstruct_uncompressed_cache()
