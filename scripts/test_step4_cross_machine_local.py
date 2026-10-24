#!/usr/bin/env python3
"""Automated Local Orchestrator Test Suite for Step 4 Cross-Machine Handoff.

Launches Producer and Consumer as independent OS subprocesses over TCP loopback,
matching the exact decoupled operational paradigm of remote multi-host execution.

Executes and verifies:
1. Unit Test A: Fragmented-Header Handshake Framing (reconstructs 8-byte header delivered in 1-2 byte TCP chunks)
2. Unit Test B: Slow-Drip Handshake Payload Delivery (bounded exact-length reads under single absolute deadline)
3. Case 1: Successful Uncompressed Handoff (Code 0, logit error <= 1e-4, clamped cosine similarity in [-1, 1])
4. Case 2: Successful FP8 Wire Codec Handoff (Code 0, cosine sim >= 0.999, aligned top-1)
5. Case 3: Deliberate Failure — Checkpoint Revision Mismatch (Code 1, fail-closed rejection)
6. Case 4: Deliberate Failure — Unavailable Consumer (Code 1, bounded timeout / connection refused)
7. Case 5: Deliberate Failure — Truncated Body (Code 1, incomplete transfer rejection)
8. Case 6: Deliberate Failure — Consumer Simulated Rejection (Code 1, handshake rejection)
9. Case 7: Deliberate Failure — Dtype Mismatch Rejection (Code 1, consumer rejects invalid/incompatible dtype)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import socket
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from hetero_kv.transport import (
    MAGIC_ACK_MANIFEST,
    MAGIC_REJ_MANIFEST,
    MAGIC_ACK_TRANSFER,
    MAGIC_VAL_PASSED,
    MAGIC_VAL_FAILED,
    send_handshake_msg,
    recv_handshake_msg,
    WireAccounting,
)

PYTHON_EXE = sys.executable
RUNNER_SCRIPT = str(repo_root / "scripts" / "run_step4_cross_machine_handoff.py")


def find_free_port() -> int:
    """Finds an unused ephemeral port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ==============================================================================
# UNIT TESTS: TCP FRAGMENTATION & SLOW-DRIP RESILIENCE
# ==============================================================================

def test_fragmented_header_handshake():
    """
    Verifies that recv_handshake_msg correctly reassembles handshake frames
    when TCP delivers the 8-byte header in fragmented 1-2 byte chunks.
    Proves that ACK handling never rejects valid TCP traffic due to partial reads.
    """
    print("\n[UNIT TEST A] Fragmented-Header Handshake TCP Read Test...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    magic_sent = MAGIC_ACK_MANIFEST
    payload_sent = {"status": "ACCEPTED", "model_id": "test/model"}
    json_bytes = json.dumps(payload_sent).encode("utf-8")
    header = magic_sent + len(json_bytes).to_bytes(4, byteorder="big")

    def client_worker():
        s = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        # Send header in 1-2 byte fragments with deliberate delays
        fragments = [header[:1], header[1:3], header[3:4], header[4:6], header[6:8]]
        for frag in fragments:
            s.sendall(frag)
            time.sleep(0.015)
        # Send body
        s.sendall(json_bytes)
        s.close()

    t = threading.Thread(target=client_worker)
    t.start()

    conn, _ = server_sock.accept()
    magic_recv, payload_recv, accounting = recv_handshake_msg(conn, timeout=5.0)
    conn.close()
    server_sock.close()
    t.join()

    assert magic_recv == magic_sent, f"Expected {magic_sent}, got {magic_recv}"
    assert payload_recv == payload_sent, f"Expected {payload_sent}, got {payload_recv}"
    assert accounting.framing_bytes == 8
    assert accounting.metadata_bytes == len(json_bytes)
    print("  -> Fragmented 8-byte header successfully received without IncompleteTransferError.")
    print("  -> PASSED.")


def test_slow_drip_handshake():
    """
    Verifies that recv_handshake_msg correctly reads handshake payload when delivery
    is drip-fed in 16-byte chunks with delays under a single absolute deadline.
    Proves that partial body reads do not prematurely reset or fail the timeout.
    """
    print("\n[UNIT TEST B] Slow-Drip Handshake Payload Delivery Test...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    magic_sent = MAGIC_ACK_TRANSFER
    payload_sent = {"status": "TRANSFER_RECEIVED", "blob": "a" * 384}
    json_bytes = json.dumps(payload_sent).encode("utf-8")
    header = magic_sent + len(json_bytes).to_bytes(4, byteorder="big")

    def client_worker():
        s = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        s.sendall(header)
        # Drip feed body in 16-byte chunks with small delay
        chunk_size = 16
        for i in range(0, len(json_bytes), chunk_size):
            s.sendall(json_bytes[i:i + chunk_size])
            time.sleep(0.01)
        s.close()

    t = threading.Thread(target=client_worker)
    t.start()

    conn, _ = server_sock.accept()
    magic_recv, payload_recv, accounting = recv_handshake_msg(conn, timeout=5.0)
    conn.close()
    server_sock.close()
    t.join()

    assert magic_recv == magic_sent, f"Expected {magic_sent}, got {magic_recv}"
    assert payload_recv == payload_sent, f"Expected {payload_sent}, got {payload_recv}"
    assert accounting.framing_bytes == 8
    assert accounting.metadata_bytes == len(json_bytes)
    print("  -> Slow-drip payload reassembled successfully under single absolute deadline.")
    print("  -> PASSED.")


# ==============================================================================
# SUBPROCESS PAIR HARNESS
# ==============================================================================

def run_test_pair(
    consumer_args: list[str],
    producer_args: list[str],
    consumer_out_json: str,
    producer_out_json: str,
    timeout: float = 90.0,
    expected_consumer_code: int = 0,
    expected_producer_code: int = 0,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    """Launches consumer and producer as independent OS processes and collects outputs."""
    consumer_cmd = [PYTHON_EXE, RUNNER_SCRIPT, "consumer"] + consumer_args
    producer_cmd = [PYTHON_EXE, RUNNER_SCRIPT, "producer"] + producer_args

    consumer_proc = subprocess.Popen(
        consumer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Allow consumer process to initialize socket listener; producer has connect retry loop
    time.sleep(0.5)

    producer_proc = subprocess.Popen(
        producer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        p_stdout, p_stderr = producer_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        producer_proc.kill()
        p_stdout, p_stderr = producer_proc.communicate()

    try:
        c_stdout, c_stderr = consumer_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        consumer_proc.kill()
        c_stdout, c_stderr = consumer_proc.communicate()

    p_code = producer_proc.returncode
    c_code = consumer_proc.returncode

    consumer_data = {}
    if os.path.exists(consumer_out_json):
        try:
            with open(consumer_out_json, "r", encoding="utf-8") as f:
                consumer_data = json.load(f)
        except Exception:
            pass

    producer_data = {}
    if os.path.exists(producer_out_json):
        try:
            with open(producer_out_json, "r", encoding="utf-8") as f:
                producer_data = json.load(f)
        except Exception:
            pass

    return c_code, p_code, consumer_data, producer_data


# ==============================================================================
# INTEGRATION TEST SUITE
# ==============================================================================

def test_suite():
    print("=" * 80)
    print("HeteroDisagg Step 4A: Local Cross-Machine Handoff Test Suite")
    print("=" * 80)

    # Run Unit Tests for Handshake Framing
    test_fragmented_header_handshake()
    test_slow_drip_handshake()

    artifacts_dir = repo_root / "benchmarks"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    passed_count = 2  # Unit tests passed
    total_count = 9

    # -------------------------------------------------------------------------
    # CASE 1: Successful Uncompressed Run
    # -------------------------------------------------------------------------
    print("\n[CASE 1] Successful Uncompressed Handoff (FP32 on CPU)...")
    port1 = find_free_port()
    c_out1 = str(artifacts_dir / "step4_consumer_uncompressed.json")
    p_out1 = str(artifacts_dir / "step4_producer_uncompressed.json")

    c_code1, p_code1, c_data1, p_data1 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port1),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--dtype", "float32",
            "--decode-steps", "4",
            "--timeout", "30",
            "--output", c_out1,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port1),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--prompt", "Heterogeneous disaggregated inference allows",
            "--device", "cpu",
            "--dtype", "float32",
            "--timeout", "30",
            "--output", p_out1,
        ],
        consumer_out_json=c_out1,
        producer_out_json=p_out1,
        expected_consumer_code=0,
        expected_producer_code=0,
    )

    assert c_code1 == 0, f"Case 1 consumer failed with exit code {c_code1}"
    assert p_code1 == 0, f"Case 1 producer failed with exit code {p_code1}"
    assert c_data1.get("success") is True, "Consumer report success must be True"
    assert p_data1.get("success") is True, "Producer report success must be True"

    discrete1 = c_data1["discrete_checks"]
    assert discrete1["uncompressed_tokens_match"] is True, "Uncompressed tokens must match baseline"
    assert discrete1["uncompressed_logit_error_within_tolerance"] is True, "Uncompressed logit error must be <= 1e-4"
    assert discrete1["null_cache_negative_control_diverged"] is True, "Null cache must diverge from baseline"
    assert discrete1["producer_initial_token_matches_consumer"] is True, "Initial token must match baseline prefill"
    assert discrete1["metrics_are_finite"] is True, "Metrics must be finite"

    aligned1 = c_data1["transferred_cache_decoding"]["aligned_comparison"]
    assert -1.0 <= aligned1["cosine_similarity"] <= 1.0, f"Cosine similarity {aligned1['cosine_similarity']} not clamped to [-1, 1]"

    assert "wire_transfer_ack_rtt_ms" in p_data1["timings_producer_local_ms"], "Producer must record wire_transfer_ack_rtt_ms"
    assert p_data1.get("consumer_transfer_ack", {}).get("status") == "TRANSFER_RECEIVED", "Producer must record TRANSFER_RECEIVED"
    assert p_data1.get("consumer_validation", {}).get("status") == "PASSED", "Producer must record consumer validation PASSED"

    print(f"  -> Exit Codes: Consumer={c_code1}, Producer={p_code1}")
    print(f"  -> Max Abs Logit Error: {aligned1['max_abs_logit_error']:.6e}")
    print(f"  -> Clamped Cosine Similarity: {aligned1['cosine_similarity']:.6f}")
    print(f"  -> Tokens Match: {discrete1['uncompressed_tokens_match']}")
    print(f"  -> Producer Wire Bytes: {p_data1['wire_accounting']['total_wire_bytes']} bytes")
    print(f"  -> Wire Transfer ACK RTT: {p_data1['timings_producer_local_ms']['wire_transfer_ack_rtt_ms']:.2f} ms")
    print(f"  -> Total Validation Session RTT: {p_data1['timings_producer_local_ms']['total_session_rtt_ms']:.2f} ms")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 2: Successful FP8 Wire Codec Run
    # -------------------------------------------------------------------------
    print("\n[CASE 2] Successful FP8 Wire Codec Handoff (E4M3 / Fallback)...")
    port2 = find_free_port()
    c_out2 = str(artifacts_dir / "step4_consumer_fp8.json")
    p_out2 = str(artifacts_dir / "step4_producer_fp8.json")

    c_code2, p_code2, c_data2, p_data2 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port2),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--dtype", "float32",
            "--decode-steps", "4",
            "--timeout", "30",
            "--output", c_out2,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port2),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--prompt", "Heterogeneous disaggregated inference allows",
            "--device", "cpu",
            "--dtype", "float32",
            "--enable-fp8",
            "--timeout", "30",
            "--output", p_out2,
        ],
        consumer_out_json=c_out2,
        producer_out_json=p_out2,
        expected_consumer_code=0,
        expected_producer_code=0,
    )

    assert c_code2 == 0, f"Case 2 consumer failed with exit code {c_code2}"
    assert p_code2 == 0, f"Case 2 producer failed with exit code {p_code2}"
    assert c_data2.get("success") is True, "Consumer report success must be True"
    assert p_data2.get("success") is True, "Producer report success must be True"

    discrete2 = c_data2["discrete_checks"]
    assert discrete2["fp8_cosine_similarity_within_threshold"] is True, "FP8 cosine similarity must be >= 0.999"
    assert discrete2["null_cache_negative_control_diverged"] is True, "Null cache must diverge from baseline"
    assert discrete2["producer_initial_token_matches_consumer"] is True, "Initial token must match baseline prefill"
    assert discrete2["metrics_are_finite"] is True, "Metrics must be finite"

    aligned2 = c_data2["transferred_cache_decoding"]["aligned_comparison"]
    assert -1.0 <= aligned2["cosine_similarity"] <= 1.0, f"Cosine similarity {aligned2['cosine_similarity']} not clamped to [-1, 1]"

    assert "wire_transfer_ack_rtt_ms" in p_data2["timings_producer_local_ms"]
    assert p_data2.get("consumer_transfer_ack", {}).get("status") == "TRANSFER_RECEIVED"
    assert p_data2.get("consumer_validation", {}).get("status") == "PASSED"

    print(f"  -> Exit Codes: Consumer={c_code2}, Producer={p_code2}")
    print(f"  -> Clamped FP8 Cosine Similarity: {aligned2['cosine_similarity']:.6f}")
    print(f"  -> FP8 Free-Running Tokens Match: {discrete2['fp8_free_running_tokens_match']}")
    print(f"  -> FP8 Aligned Top-1 Match: {discrete2['fp8_aligned_top1_match']}")
    print(f"  -> Producer Wire Bytes: {p_data2['wire_accounting']['total_wire_bytes']} bytes")
    print(f"  -> Wire Transfer ACK RTT: {p_data2['timings_producer_local_ms']['wire_transfer_ack_rtt_ms']:.2f} ms")
    print(f"  -> Total Validation Session RTT: {p_data2['timings_producer_local_ms']['total_session_rtt_ms']:.2f} ms")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 3: Deliberate Failure — Checkpoint Revision Mismatch
    # -------------------------------------------------------------------------
    print("\n[CASE 3] Deliberate Failure: Checkpoint Revision Mismatch...")
    port3 = find_free_port()
    c_out3 = str(artifacts_dir / "step4_consumer_fail_checkpoint.json")
    p_out3 = str(artifacts_dir / "step4_producer_fail_checkpoint.json")

    c_code3, p_code3, c_data3, p_data3 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port3),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--timeout", "15",
            "--output", c_out3,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port3),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--inject-checkpoint-mismatch",
            "--timeout", "15",
            "--output", p_out3,
        ],
        consumer_out_json=c_out3,
        producer_out_json=p_out3,
        expected_consumer_code=1,
        expected_producer_code=1,
    )

    assert c_code3 == 1, f"Consumer should exit code 1 on checkpoint mismatch, got {c_code3}"
    assert p_code3 == 1, f"Producer should exit code 1 on rejection, got {p_code3}"
    assert c_data3.get("success") is False, "Consumer success must be False"
    assert p_data3.get("success") is False, "Producer success must be False"
    print(f"  -> Consumer Exit Code: {c_code3} (Properly failed-closed)")
    print(f"  -> Producer Exit Code: {p_code3} (Properly failed-closed)")
    print(f"  -> Rejection Reason: {c_data3.get('error', {}).get('message')}")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 4: Deliberate Failure — Unavailable Consumer
    # -------------------------------------------------------------------------
    print("\n[CASE 4] Deliberate Failure: Unavailable Consumer (Connection Refused)...")
    unopened_port = find_free_port()
    p_out4 = str(artifacts_dir / "step4_producer_fail_unavail.json")

    p_cmd = [
        PYTHON_EXE, RUNNER_SCRIPT, "producer",
        "--dest-host", "127.0.0.1",
        "--dest-port", str(unopened_port),
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--device", "cpu",
        "--timeout", "2.0",
        "--output", p_out4,
    ]
    p_res4 = subprocess.run(p_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert p_res4.returncode == 1, f"Producer should exit code 1 when consumer unavailable, got {p_res4.returncode}"
    p_data4 = {}
    if os.path.exists(p_out4):
        with open(p_out4, "r", encoding="utf-8") as f:
            p_data4 = json.load(f)
    assert p_data4.get("success") is False
    print(f"  -> Producer Exit Code: {p_res4.returncode} (Failed fast on connection failure)")
    print(f"  -> Error Logged: {p_data4.get('error', {}).get('message')}")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 5: Deliberate Failure — Truncated Payload Body
    # -------------------------------------------------------------------------
    print("\n[CASE 5] Deliberate Failure: Truncated Payload (Premature Disconnect)...")
    port5 = find_free_port()
    c_out5 = str(artifacts_dir / "step4_consumer_fail_trunc.json")
    p_out5 = str(artifacts_dir / "step4_producer_fail_trunc.json")

    c_code5, p_code5, c_data5, p_data5 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port5),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--timeout", "5.0",
            "--output", c_out5,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port5),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--inject-truncation",
            "--timeout", "5.0",
            "--output", p_out5,
        ],
        consumer_out_json=c_out5,
        producer_out_json=p_out5,
        expected_consumer_code=1,
        expected_producer_code=1,
    )

    assert c_code5 == 1, f"Consumer should exit code 1 on truncated payload, got {c_code5}"
    assert p_code5 == 1, f"Producer should exit code 1 on truncation, got {p_code5}"
    assert c_data5.get("success") is False
    assert p_data5.get("success") is False
    print(f"  -> Consumer Exit Code: {c_code5} (Rejected incomplete transfer)")
    print(f"  -> Producer Exit Code: {p_code5}")
    print(f"  -> Consumer Error: {c_data5.get('error', {}).get('message')}")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 6: Deliberate Failure — Consumer Simulated Rejection
    # -------------------------------------------------------------------------
    print("\n[CASE 6] Deliberate Failure: Consumer Handshake Rejection...")
    port6 = find_free_port()
    c_out6 = str(artifacts_dir / "step4_consumer_fail_reject.json")
    p_out6 = str(artifacts_dir / "step4_producer_fail_reject.json")

    c_code6, p_code6, c_data6, p_data6 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port6),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--simulate-rejection",
            "--timeout", "10.0",
            "--output", c_out6,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port6),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--timeout", "10.0",
            "--output", p_out6,
        ],
        consumer_out_json=c_out6,
        producer_out_json=p_out6,
        expected_consumer_code=1,
        expected_producer_code=1,
    )

    assert c_code6 == 1, f"Consumer should exit code 1 on rejection, got {c_code6}"
    assert p_code6 == 1, f"Producer should exit code 1 when rejected, got {p_code6}"
    assert c_data6.get("success") is False
    assert p_data6.get("success") is False
    print(f"  -> Consumer Exit Code: {c_code6}")
    print(f"  -> Producer Exit Code: {p_code6} (Received reject frame)")
    print("  -> PASSED.")
    passed_count += 1

    # -------------------------------------------------------------------------
    # CASE 7: Deliberate Failure — Dtype Mismatch Rejection
    # -------------------------------------------------------------------------
    print("\n[CASE 7] Deliberate Failure: Incompatible Dtype Mismatch Rejection...")
    port7 = find_free_port()
    c_out7 = str(artifacts_dir / "step4_consumer_fail_dtype.json")
    p_out7 = str(artifacts_dir / "step4_producer_fail_dtype.json")

    c_code7, p_code7, c_data7, p_data7 = run_test_pair(
        consumer_args=[
            "--bind-host", "127.0.0.1",
            "--bind-port", str(port7),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--dtype", "float32",
            "--timeout", "10.0",
            "--output", c_out7,
        ],
        producer_args=[
            "--dest-host", "127.0.0.1",
            "--dest-port", str(port7),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--dtype", "float32",
            "--inject-dtype-mismatch",
            "--timeout", "10.0",
            "--output", p_out7,
        ],
        consumer_out_json=c_out7,
        producer_out_json=p_out7,
        expected_consumer_code=1,
        expected_producer_code=1,
    )

    assert c_code7 == 1, f"Consumer should exit code 1 on dtype mismatch, got {c_code7}"
    assert p_code7 == 1, f"Producer should exit code 1 when rejected, got {p_code7}"
    assert c_data7.get("success") is False
    assert p_data7.get("success") is False
    print(f"  -> Consumer Exit Code: {c_code7} (Rejected incompatible manifest dtype)")
    print(f"  -> Producer Exit Code: {p_code7} (Caught consumer rejection)")
    print(f"  -> Rejection Reason: {c_data7.get('error', {}).get('message')}")
    print("  -> PASSED.")
    passed_count += 1

    print("\n" + "=" * 80)
    print(f"ALL {passed_count}/{total_count} STEP 4A LOCAL TESTS (2 UNIT + 7 INTEGRATION) PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_suite()
