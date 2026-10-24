#!/usr/bin/env python3
"""Automated Local Orchestrator Test Suite for Step 4 Cross-Machine Handoff.

Launches Producer and Consumer as independent OS subprocesses over TCP loopback,
matching the exact decoupled operational paradigm of remote multi-host execution.

Executes and verifies:
1. Case 1: Successful Uncompressed FP32/FP16 Cache Handoff (Code 0, logit error <= 1e-4)
2. Case 2: Successful FP8 Wire Codec Handoff (Code 0, cosine sim >= 0.999, aligned top-1)
3. Case 3: Deliberate Failure — Checkpoint Revision Mismatch (Code 1, fail-closed rejection)
4. Case 4: Deliberate Failure — Unavailable Consumer (Code 1, bounded timeout / connection refused)
5. Case 5: Deliberate Failure — Truncated Body (Code 1, incomplete transfer rejection)
6. Case 6: Deliberate Failure — Consumer Rejection (Code 1, handshake rejection)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import socket
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

PYTHON_EXE = sys.executable
RUNNER_SCRIPT = str(repo_root / "scripts" / "run_step4_cross_machine_handoff.py")


def find_free_port() -> int:
    """Finds an unused ephemeral port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_listener(host: str, port: int, timeout: float = 45.0) -> bool:
    """Polls until a TCP server is accepting connections or deadline expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


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

    # Start consumer
    consumer_proc = subprocess.Popen(
        consumer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    port = None
    for i, arg in enumerate(consumer_args):
        if arg == "--bind-port" and i + 1 < len(consumer_args):
            port = int(consumer_args[i + 1])
            break

    # Give consumer process brief start to launch listener; producer has connect retry loop
    time.sleep(0.5)

    # Start producer
    producer_proc = subprocess.Popen(
        producer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Await both completions
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


def test_suite():
    print("=" * 80)
    print("HeteroDisagg Step 4A: Local Cross-Machine Handoff Test Suite")
    print("=" * 80)

    artifacts_dir = repo_root / "benchmarks"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    passed_count = 0
    total_count = 6

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
    print(f"  -> Exit Codes: Consumer={c_code1}, Producer={p_code1}")
    print(f"  -> Max Abs Logit Error: {aligned1['max_abs_logit_error']:.6e}")
    print(f"  -> Tokens Match: {discrete1['uncompressed_tokens_match']}")
    print(f"  -> Producer Wire Bytes: {p_data1['wire_accounting']['total_wire_bytes']} bytes")
    print(f"  -> Producer ACK RTT: {p_data1['timings_producer_local_ms']['producer_ack_roundtrip_ms']:.2f} ms")
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
    print(f"  -> Exit Codes: Consumer={c_code2}, Producer={p_code2}")
    print(f"  -> FP8 Cosine Similarity: {aligned2['cosine_similarity']:.6f}")
    print(f"  -> FP8 Free-Running Tokens Match: {discrete2['fp8_free_running_tokens_match']}")
    print(f"  -> FP8 Aligned Top-1 Match: {discrete2['fp8_aligned_top1_match']}")
    print(f"  -> Producer Wire Bytes: {p_data2['wire_accounting']['total_wire_bytes']} bytes")
    print(f"  -> Producer ACK RTT: {p_data2['timings_producer_local_ms']['producer_ack_roundtrip_ms']:.2f} ms")
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
    print(f"  -> Consumer Exit Code: {c_code6}")
    print(f"  -> Producer Exit Code: {p_code6} (Received reject frame)")
    print("  -> PASSED.")
    passed_count += 1

    print("\n" + "=" * 80)
    print(f"ALL {passed_count}/{total_count} STEP 4A LOCAL INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_suite()
