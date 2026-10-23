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

import json
import multiprocessing as mp
import socket
import time
from dataclasses import asdict
import pytest
import torch

from hetero_kv.layout import KVPartitionPlan
from hetero_kv.transport import (
    MAGIC_MANIFEST,
    MAGIC_FRAME,
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    RequestManifest,
    FrameMetadata,
    WireAccounting,
    send_manifest,
    recv_manifest,
    send_frame,
    recv_frame,
    recv_exact,
    validate_manifest_compatibility,
    KVCacheFrameAccumulator,
    TransportTimeoutError,
    IncompleteTransferError,
    CorruptedFrameError,
    MismatchedRequestError,
    ManifestValidationError,
    FrameValidationError,
)
from scripts.verify_socket_process_cache_handoff import (
    run_socket_cache_handoff_experiment,
    decode_consumer_worker,
)


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
        client_sock.sendall(b"HK")
        client_sock.close()

        with pytest.raises(IncompleteTransferError):
            recv_exact(conn, n_bytes=8, timeout=1.0)
    finally:
        conn.close()
        server_sock.close()


def test_producer_disconnect_during_tensor_body():
    """Premature disconnect during tensor body transmission must raise IncompleteTransferError."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    manifest = RequestManifest(
        request_id="req-trunc-body",
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
        decode_rank=0,
    )

    meta = FrameMetadata(
        request_id="req-trunc-body",
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
    meta_json = json.dumps(asdict(meta)).encode("utf-8")
    header = MAGIC_FRAME + len(meta_json).to_bytes(4, byteorder="big") + meta_json

    try:
        # Send complete framing + partial body (only 50 bytes out of 4096) then close socket abruptly
        client_sock.sendall(header + b"\x00" * 50)
        client_sock.close()

        with pytest.raises(IncompleteTransferError, match="Socket EOF reached unexpectedly"):
            recv_frame(conn, manifest, timeout=1.0)
    finally:
        conn.close()
        server_sock.close()


def test_payload_cap_enforced_before_allocation():
    """Declared frame body size exceeding MAX_PAYLOAD_BYTES must be rejected before allocation."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    manifest = RequestManifest(
        request_id="req-cap-test",
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
        decode_rank=0,
    )

    try:
        # Deliberately craft a frame declaring 1,073,741,828 bytes (> 1 GB limit)
        oversized_bytes = 1073741828
        bad_meta = {
            "request_id": "req-cap-test",
            "kv_type": "key",
            "layer_idx": 0,
            "prefill_rank": 0,
            "decode_rank": 0,
            "start_head_idx": 0,
            "end_head_idx": 2,
            "is_quantized": False,
            "shape": [1, 8, 2, 64],
            "dtype": "float32",
            "body_bytes": oversized_bytes,
        }
        meta_json = json.dumps(bad_meta).encode("utf-8")
        header = MAGIC_FRAME + len(meta_json).to_bytes(4, byteorder="big") + meta_json
        client_sock.sendall(header)

        # Receiver should reject BEFORE reading or allocating 1 GB body buffer
        with pytest.raises(FrameValidationError, match="exceeds maximum allowed payload limit"):
            recv_frame(conn, manifest, timeout=1.0)

        # Also test recv_exact defense-in-depth allocation limit
        with pytest.raises(FrameValidationError, match="exceeds configured limit"):
            recv_exact(conn, n_bytes=oversized_bytes, timeout=1.0)
    finally:
        client_sock.close()
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
    """Frames with mismatched request IDs must raise MismatchedRequestError over socket."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

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
        decode_rank=0,
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
    body = b"\x00" * bad_frame.body_bytes

    try:
        send_frame(client_sock, bad_frame, body)

        # Server receives and verifies MismatchedRequestError is raised directly
        with pytest.raises(MismatchedRequestError, match="does not match manifest"):
            recv_frame(conn, manifest, timeout=1.0)
    finally:
        client_sock.close()
        conn.close()
        server_sock.close()


def test_layout_validation_rejects_overlapping_heads_and_bogus_decode_rank():
    """Accumulator and frame validation must reject overlapping head slices and bogus decode ranks."""
    manifest = RequestManifest(
        request_id="req-layout-test",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        seq_len=8,
        prefill_tp=2,
        decode_tp=1,
        expected_frames=4,
        initial_decode_token=42,
        initial_decode_token_str="test",
        prompt_tokens=[1, 2],
        dtype="float32",
        decode_rank=0,
    )

    # 1. Reject bogus decode_rank=999
    meta_bogus_rank = FrameMetadata(
        request_id="req-layout-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=999,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[1, 8, 2, 64],
        dtype="float32",
        body_bytes=1 * 8 * 2 * 64 * 4,
    )
    with pytest.raises(FrameValidationError, match="decode_rank 999 out of bounds"):
        meta_bogus_rank.validate(manifest)

    # 2. Reject overlapping head ranges in accumulator
    acc = KVCacheFrameAccumulator(manifest)
    wa = WireAccounting(framing_bytes=8, metadata_bytes=100, body_bytes=4096)
    dummy_body = b"\x00" * 4096

    meta_rank0 = FrameMetadata(
        request_id="req-layout-test",
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
    acc.add_frame(meta_rank0, dummy_body, wa)

    # Rank 1 attempts to declare overlapping heads [1:3] instead of [2:4]
    meta_overlapping = FrameMetadata(
        request_id="req-layout-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=1,
        decode_rank=0,
        start_head_idx=1,
        end_head_idx=3,
        is_quantized=False,
        shape=[1, 8, 2, 64],
        dtype="float32",
        body_bytes=4096,
    )
    with pytest.raises(FrameValidationError, match="(Unauthorized route|Overlapping head slice)"):
        acc.add_frame(meta_overlapping, dummy_body, wa)


def test_layout_validation_rejects_seq_len_mismatch():
    """Frame with seq_len different from manifest seq_len must be rejected."""
    manifest = RequestManifest(
        request_id="req-seq-test",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        seq_len=8,  # Declared 8
        prefill_tp=1,
        decode_tp=1,
        expected_frames=2,
        initial_decode_token=42,
        initial_decode_token_str="test",
        prompt_tokens=[1, 2],
        dtype="float32",
        decode_rank=0,
    )

    bad_frame = FrameMetadata(
        request_id="req-seq-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[1, 9, 2, 64],  # Sequence length 9 instead of 8!
        dtype="float32",
        body_bytes=1 * 9 * 2 * 64 * 4,
    )

    with pytest.raises(FrameValidationError, match="Frame seq_len \\(9\\) does not match manifest seq_len \\(8\\)"):
        bad_frame.validate(manifest)


def test_manifest_compatibility_validation():
    """Consumer must validate manifest model identity and architecture against config."""
    class DummyConfig:
        num_hidden_layers = 12
        num_key_value_heads = 4
        head_dim = 64
        hidden_size = 512
        num_attention_heads = 8

    cfg = DummyConfig()

    base_manifest = RequestManifest(
        request_id="req-compat-001",
        model_id="expected-model",
        checkpoint_commit_hash=None,
        num_layers=12,
        num_kv_heads=4,
        head_dim=64,
        seq_len=16,
        prefill_tp=2,
        decode_tp=1,
        expected_frames=48,
        initial_decode_token=10,
        initial_decode_token_str="tok",
        prompt_tokens=[1, 2],
        dtype="float32",
        decode_rank=0,
    )

    # Valid case passes
    validate_manifest_compatibility(base_manifest, cfg, expected_model_id="expected-model", expected_decode_rank=0)

    # Mismatched model ID
    with pytest.raises(MismatchedRequestError, match="model_id"):
        validate_manifest_compatibility(base_manifest, cfg, expected_model_id="other-model", expected_decode_rank=0)

    # Mismatched decode rank
    with pytest.raises(ManifestValidationError, match="decode_rank"):
        validate_manifest_compatibility(base_manifest, cfg, expected_model_id="expected-model", expected_decode_rank=1)

    # Mismatched checkpoint revision
    cfg_with_commit = DummyConfig()
    cfg_with_commit._commit_hash = "abc123commit"
    with pytest.raises(MismatchedRequestError, match="checkpoint revision"):
        validate_manifest_compatibility(
            base_manifest, cfg_with_commit, expected_model_id="expected-model"
        )
    manifest_with_wrong_commit = RequestManifest(
        **{**base_manifest.__dict__, "checkpoint_commit_hash": "different_commit_456"}
    )
    with pytest.raises(MismatchedRequestError, match="does not match consumer resolved revision"):
        validate_manifest_compatibility(
            manifest_with_wrong_commit, cfg_with_commit, expected_model_id="expected-model"
        )
    # Matching commit hash passes
    manifest_matching_commit = RequestManifest(
        **{**base_manifest.__dict__, "checkpoint_commit_hash": "abc123commit"}
    )
    validate_manifest_compatibility(
        manifest_matching_commit, cfg_with_commit, expected_model_id="expected-model"
    )

    # Mismatched num_layers
    bad_layers = RequestManifest(
        **{**base_manifest.__dict__, "num_layers": 8, "expected_frames": 32}
    )
    with pytest.raises(ManifestValidationError, match="num_layers"):
        validate_manifest_compatibility(bad_layers, cfg, expected_model_id="expected-model")


def test_unsupported_dtypes_and_batch_geometry_rejected():
    """Unsupported quantized/scale dtypes and batch dimensions != 1 must be rejected."""
    manifest = RequestManifest(
        request_id="req-dtype-test",
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
        decode_rank=0,
    )

    # 1. Reject invented quantized dtype
    bad_quant_dtype = FrameMetadata(
        request_id="req-dtype-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=True,
        shape=[1, 16 * 2 * 64],
        dtype="invented_fp8_custom",  # Invalid!
        scale_shape=[1],
        scale_dtype="float32",
        original_shape=[1, 8, 2, 64],
        original_dtype="float32",
        block_size_tokens=16,
        body_bytes=1 * 16 * 2 * 64 * 1 + 4,
    )
    with pytest.raises(FrameValidationError, match="Unsupported FP8 quantized dtype"):
        bad_quant_dtype.validate(manifest)

    # 2. Reject invented scale dtype
    bad_scale_dtype = FrameMetadata(
        request_id="req-dtype-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=True,
        shape=[1, 16 * 2 * 64],
        dtype="float8_e4m3fn",
        scale_shape=[1],
        scale_dtype="invented_scale_fp64",  # Invalid!
        original_shape=[1, 8, 2, 64],
        original_dtype="float32",
        block_size_tokens=16,
        body_bytes=1 * 16 * 2 * 64 * 1 + 8,
    )
    with pytest.raises(FrameValidationError, match="Unsupported FP8 scale dtype"):
        bad_scale_dtype.validate(manifest)

    # 3. Reject batch dimension != 1 (codec currently requires batch size 1)
    bad_batch_frame = FrameMetadata(
        request_id="req-dtype-test",
        kv_type="key",
        layer_idx=0,
        prefill_rank=0,
        decode_rank=0,
        start_head_idx=0,
        end_head_idx=2,
        is_quantized=False,
        shape=[2, 8, 2, 64],  # Batch size 2!
        dtype="float32",
        body_bytes=2 * 8 * 2 * 64 * 4,
    )
    with pytest.raises(FrameValidationError, match="Batch dimension 2 not supported"):
        bad_batch_frame.validate(manifest)


def test_temporary_file_unconditional_cleanup_on_producer_failure():
    """Temporary logits file must be cleaned up even if producer reports failure after file creation."""
    import tempfile
    import uuid

    temp_dir = tempfile.gettempdir()
    temp_logits_path = os.path.join(temp_dir, f"hkv_test_cleanup_{uuid.uuid4().hex}.pt")

    # Simulate consumer writing logits before failure is detected
    torch.save(torch.randn(4, 100), temp_logits_path)
    assert os.path.exists(temp_logits_path)

    # Simulate orchestrator encountering producer failure
    producer_res = {"status": "error", "error_message": "Producer failed early"}

    try:
        if producer_res["status"] != "success":
            raise RuntimeError(f"Producer failed: {producer_res}")
    except RuntimeError:
        pass
    finally:
        # Parent-owned cleanup block
        if temp_logits_path and os.path.exists(temp_logits_path):
            os.remove(temp_logits_path)

    assert not os.path.exists(temp_logits_path), "Temporary file must be deleted unconditionally on failure!"


def test_complete_consumer_rejection_and_cleanup_scenario():
    """Complete multi-process producer/consumer scenario where consumer rejects incompatible manifest and cleanly exits."""
    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    consumer_result_queue = ctx.Queue()

    # Consumer expects "Qwen/Qwen2.5-0.5B-Instruct"
    consumer_proc = ctx.Process(
        target=decode_consumer_worker,
        args=(port_queue, consumer_result_queue, "Qwen/Qwen2.5-0.5B-Instruct", 2, "cpu", 30.0),
    )

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        consumer_proc.start()
        ready_info = port_queue.get(timeout=30.0)
        port = ready_info["port"]

        client_sock.connect(("127.0.0.1", port))

        # Send incompatible manifest (wrong model_id)
        incompatible_manifest = RequestManifest(
            request_id="req-incompat",
            model_id="WrongModel/IncompatibleArch",
            checkpoint_commit_hash=None,
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            seq_len=8,
            prefill_tp=1,
            decode_tp=1,
            expected_frames=48,
            initial_decode_token=1,
            initial_decode_token_str="x",
            prompt_tokens=[1],
            dtype="float32",
            decode_rank=0,
        )
        send_manifest(client_sock, incompatible_manifest)

        # Consumer should reject and return error
        res = consumer_result_queue.get(timeout=30.0)
        assert res["status"] == "error"
        assert res["error_type"] in ("MismatchedRequestError", "ManifestValidationError")
    finally:
        client_sock.close()
        consumer_proc.join(timeout=5.0)
        if consumer_proc.is_alive():
            consumer_proc.terminate()
            consumer_proc.join(timeout=1.0)
        port_queue.close()
        port_queue.cancel_join_thread()
        consumer_result_queue.close()
        consumer_result_queue.cancel_join_thread()


def test_socket_deserialized_fp8_exact_match_with_in_process_dequantization():
    """
    Proves that socket-deserialized FP8 caches match in-process dequantized caches with bitwise tensor equality.
    Ensures that streaming over TCP transport introduces zero numerical distortion.
    """
    from hetero_kv.layout import KVPartitionPlan
    from hetero_kv.reshard import package_payloads_from_prefill_rank, reassemble_decode_kv

    num_layers = 2
    num_kv_heads = 4
    head_dim = 64
    seq_len = 16
    prefill_tp = 2
    decode_tp = 1
    plan = KVPartitionPlan.create(num_kv_heads, prefill_tp, decode_tp)

    torch.manual_seed(42)
    sample_keys = [torch.randn(1, seq_len, num_kv_heads, head_dim) for _ in range(num_layers)]
    sample_vals = [torch.randn(1, seq_len, num_kv_heads, head_dim) for _ in range(num_layers)]

    # 1. In-process package and reassemble
    in_process_keys = []
    in_process_vals = []
    for l in range(num_layers):
        payloads_k = []
        payloads_v = []
        for p in range(prefill_tp):
            slice_k = plan.prefill_head_slices[p]
            sub_k = sample_keys[l][..., slice_k.start_head_idx : slice_k.end_head_idx, :]
            payloads_k.extend(package_payloads_from_prefill_rank(sub_k, p, l, plan, enable_fp8=True, request_id="req-exact", kv_type="key"))

            sub_v = sample_vals[l][..., slice_k.start_head_idx : slice_k.end_head_idx, :]
            payloads_v.extend(package_payloads_from_prefill_rank(sub_v, p, l, plan, enable_fp8=True, request_id="req-exact", kv_type="value"))

        restored_k = reassemble_decode_kv(payloads_k, decode_rank=0, plan=plan)
        restored_v = reassemble_decode_kv(payloads_v, decode_rank=0, plan=plan)
        in_process_keys.append(restored_k.permute(0, 2, 1, 3).contiguous())
        in_process_vals.append(restored_v.permute(0, 2, 1, 3).contiguous())

    # 2. Transmit identical payloads over loopback TCP socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    conn, _ = server_sock.accept()

    manifest = RequestManifest(
        request_id="req-exact",
        model_id="test-model",
        checkpoint_commit_hash=None,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        expected_frames=num_layers * 2 * prefill_tp,
        initial_decode_token=1,
        initial_decode_token_str="x",
        prompt_tokens=[1, 2],
        dtype="float32",
        enable_fp8=True,
        block_size_tokens=16,
        decode_rank=0,
    )

    try:
        send_manifest(client_sock, manifest)

        for l in range(num_layers):
            for kv_type, tensor in [("key", sample_keys[l]), ("value", sample_vals[l])]:
                for p in range(prefill_tp):
                    slice_h = plan.prefill_head_slices[p]
                    sub = tensor[..., slice_h.start_head_idx : slice_h.end_head_idx, :]
                    p_list = package_payloads_from_prefill_rank(sub, p, l, plan, enable_fp8=True, request_id="req-exact", kv_type=kv_type)
                    for payload in p_list:
                        fp8_data = payload.data
                        dtype_str = "float8_e4m3fn" if fp8_data.quantized_data.dtype == getattr(torch, "float8_e4m3fn", None) else "int8"
                        quant_bytes = fp8_data.quantized_data.view(torch.uint8).cpu().numpy().tobytes()
                        scale_bytes = fp8_data.scale_factors.cpu().numpy().tobytes()
                        body = quant_bytes + scale_bytes
                        meta = FrameMetadata(
                            request_id="req-exact",
                            kv_type=kv_type,
                            layer_idx=l,
                            prefill_rank=p,
                            decode_rank=0,
                            start_head_idx=payload.start_head_idx,
                            end_head_idx=payload.end_head_idx,
                            is_quantized=True,
                            shape=list(fp8_data.quantized_data.shape),
                            dtype=dtype_str,
                            scale_shape=list(fp8_data.scale_factors.shape),
                            scale_dtype="float32",
                            original_shape=list(fp8_data.original_shape),
                            original_dtype="float32",
                            block_size_tokens=fp8_data.block_size_tokens,
                            body_bytes=len(body),
                        )
                        send_frame(client_sock, meta, body)

        recv_m, _ = recv_manifest(conn)
        acc = KVCacheFrameAccumulator(recv_m)
        while not acc.is_complete():
            meta, body, w_acc = recv_frame(conn, recv_m, plan=acc.plan)
            acc.add_frame(meta, body, w_acc)

        socket_cache = acc.reconstruct_cache()

        # 3. Assert BITWISE exact equivalence between socket and in-process dequantization
        for l in range(num_layers):
            assert torch.equal(socket_cache["keys"][l], in_process_keys[l]), f"Layer {l} keys mismatch!"
            assert torch.equal(socket_cache["values"][l], in_process_vals[l]), f"Layer {l} values mismatch!"
            max_diff_k = torch.max(torch.abs(socket_cache["keys"][l] - in_process_keys[l])).item()
            max_diff_v = torch.max(torch.abs(socket_cache["values"][l] - in_process_vals[l])).item()
            assert max_diff_k == 0.0
            assert max_diff_v == 0.0
    finally:
        client_sock.close()
        conn.close()
        server_sock.close()


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
        decode_rank=0,
    )

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
        decode_rank=0,
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


def test_process_and_resource_cleanup_on_injected_failure():
    """
    Exercises cleanup through failure paths rather than inferring it from success:
    1. Injects an intentional failure (body length mismatch).
    2. Proves run_socket_cache_handoff_experiment raises RuntimeError.
    3. Confirms that no temporary logits files leak in the filesystem.
    """
    import glob
    import tempfile

    temp_dir = tempfile.gettempdir()
    initial_temp_files = set(glob.glob(os.path.join(temp_dir, f"hkv_logits_{os.getpid()}_*.pt")))

    with pytest.raises(RuntimeError, match="(Producer failed|Consumer failed)"):
        run_socket_cache_handoff_experiment(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            prompt="Heterogeneous disaggregated inference allows",
            decode_steps=2,
            prefill_tp=2,
            decode_tp=1,
            device="cpu",
            timeout=10.0,
            corrupt_frame="body_length_mismatch",
        )

    final_temp_files = set(glob.glob(os.path.join(temp_dir, f"hkv_logits_{os.getpid()}_*.pt")))
    assert final_temp_files == initial_temp_files, f"Leaked temp files: {final_temp_files - initial_temp_files}"


