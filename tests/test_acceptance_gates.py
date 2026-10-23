"""Acceptance and Negative-Control Regression Tests for Step 0, 1, and 2 Milestones.

Validates that:
1. Step 0 readiness gate fails closed on error/divergence or non-finite values.
2. Step 1 GPU milestone gate fails closed on non-CUDA / emulation fallback.
3. Step 2 milestone gate fails closed on 99.0 logit error or non-CUDA execution.
4. True inter-token handoff gap accurately reflects delays injected after producer token readiness.
5. Process and resource lifecycle cleanup terminates and verifies child exit on failure.
"""

import os
import math
import time
import tempfile
import multiprocessing as mp
from unittest.mock import patch, MagicMock
import pytest
import torch

from hetero_kv.lifecycle import safe_cleanup_processes, verify_pid_dead
from scripts.verify_socket_process_cache_handoff import run_socket_cache_handoff_experiment
from scripts.run_local_step0_checks import run_step0_checks
from scripts.run_step1_single_gpu_validation import run_step1_suite
from scripts.run_step2_two_gpu_handoff import run_step2_suite, inspect_two_gpu_system


def test_step0_predicate_fails_closed_on_uncompressed_divergence():
    """Step 0 readiness MUST fail closed (step0_ready_for_gpu_rental=False) if uncompressed fails."""
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step0_test.json")
        
        # Mock run_socket_cache_handoff_experiment to simulate uncompressed failure
        real_fn = run_socket_cache_handoff_experiment
        def mock_handoff(*args, **kwargs):
            res = real_fn(*args, **kwargs)
            if not kwargs.get("enable_fp8", False):
                # Inject a failure
                res["divergence_within_tolerance"] = False
                res["all_tokens_match"] = False
                res["max_abs_logit_divergence"] = 99.0
            return res

        with patch("scripts.run_local_step0_checks.run_socket_cache_handoff_experiment", side_effect=mock_handoff):
            readiness = run_step0_checks(output_path=out_json)
            assert readiness["step0_ready_for_gpu_rental"] is False
            assert readiness["uncompressed_handoff"]["status"] == "FAIL"
            assert os.path.exists(out_json), "Diagnostic artifact must still be exported on failure"


def test_step0_predicate_fails_closed_on_non_finite_metric():
    """Step 0 readiness MUST fail closed if any numerical metric is NaN or Inf."""
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step0_nan_test.json")

        real_fn = run_socket_cache_handoff_experiment
        def mock_handoff(*args, **kwargs):
            res = real_fn(*args, **kwargs)
            if not kwargs.get("enable_fp8", False):
                res["max_abs_logit_divergence"] = float("nan")
            return res

        with patch("scripts.run_local_step0_checks.run_socket_cache_handoff_experiment", side_effect=mock_handoff):
            readiness = run_step0_checks(output_path=out_json)
            assert readiness["step0_ready_for_gpu_rental"] is False


def test_step1_predicate_fails_closed_without_physical_cuda():
    """Step 1 GPU milestone MUST fail closed (gpu_milestone_passed=False) when running on non-CUDA."""
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step1_test.json")
        
        # Test real entrypoint on CPU
        payload = run_step1_suite(device_str="cpu", output_path=out_json)
        crit = payload["completion_criteria"]
        
        assert crit["is_cuda_device"] is False
        assert crit["gpu_milestone_passed"] is False
        assert payload["step1_completion_status"] is False
        assert os.path.exists(out_json)


def test_step2_predicate_fails_closed_on_99_logit_error_injection():
    """
    Independent negative probe: Injecting 99.0 logit error into real Step 2 entrypoint
    MUST result in completion_criterion.passed=False and discrete check failure.
    """
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step2_error_injection.json")

        from scripts.run_step2_two_gpu_handoff import execute_two_gpu_handoff_case
        real_exec = execute_two_gpu_handoff_case

        def mock_execute(*args, **kwargs):
            res = real_exec(*args, **kwargs)
            if not kwargs.get("enable_fp8", False):
                # Inject 99.0 max error into uncompressed logits
                res["consumer_logits"] = res["consumer_logits"] + 99.0
            return res

        with patch("scripts.run_step2_two_gpu_handoff.execute_two_gpu_handoff_case", side_effect=mock_execute):
            results = run_step2_suite(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                prompt="Heterogeneous disaggregated inference allows",
                steps=2,
                prefill_device="cpu",
                decode_device="cpu",
                output_path=out_json,
            )

            crit = results["completion_criterion"]
            discrete = crit["discrete_checks"]

            assert crit["passed"] is False, "Milestone must fail on 99.0 logit error!"
            assert crit["two_gpu_milestone_passed"] is False
            assert crit["emulation_pipeline_passed"] is False
            assert discrete["uncompressed_logit_error_within_tolerance"] is False
            assert results["uncompressed_exact"]["max_abs_logit_error"] >= 99.0


def test_step2_predicate_fails_closed_without_two_cuda_gpus():
    """Step 2 milestone MUST require distinct physical CUDA devices."""
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step2_cuda_check.json")

        results = run_step2_suite(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            prompt="Heterogeneous disaggregated inference allows",
            steps=2,
            prefill_device="cpu",
            decode_device="cpu",
            output_path=out_json,
        )

        crit = results["completion_criterion"]
        discrete = crit["discrete_checks"]

        # CPU run cannot pass the physical two-GPU milestone
        assert discrete["distinct_physical_cuda_gpus"] is False
        assert crit["two_gpu_milestone_passed"] is False
        assert crit["passed"] is False


def test_inter_token_gap_tracks_injected_delay():
    """
    Validates that the true inter-token handoff gap (t_consumer_token_ready - t_producer_token_ready)
    faithfully reflects delays injected after producer token readiness.
    Proves that:
    1. A delay after producer token readiness is recorded in the monotonic host timeline (t_sender_wire_start - t_producer_token_ready >= injected_delay).
    2. The true inter-token handoff gap includes this delay in full:
       true_inter_token_handoff_gap_ms = (t_consumer_token_ready - t_producer_token_ready) * 1000.
    3. Old receiver-local timers that begin at socket connect/receive omit this interval.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Heterogeneous disaggregated inference"
    injected_delay_s = 0.250  # 250ms delay injected immediately after producer token readiness

    res = run_socket_cache_handoff_experiment(
        model_id=model_id,
        prompt=prompt,
        decode_steps=1,
        prefill_tp=2,
        decode_tp=1,
        device="cpu",
        injected_pre_send_delay_s=injected_delay_s,
    )

    timing = res["timing_decomposition"]
    raw_ts = timing["raw_event_timestamps"]
    t_prod_tok = raw_ts["t_producer_token_ready"]
    t_send_start = raw_ts["t_sender_wire_start"]
    t_cons_tok = raw_ts["t_consumer_token_ready"]

    # 1. Delay was captured on producer host monotonic clock
    pre_send_delay_observed_ms = (t_send_start - t_prod_tok) * 1000.0
    assert pre_send_delay_observed_ms >= (injected_delay_s * 1000.0 - 5.0), (
        f"Observed pre-send delay {pre_send_delay_observed_ms:.2f}ms < injected {injected_delay_s * 1000}ms"
    )

    # 2. True gap strictly spans from producer token ready to consumer token ready
    true_gap_ms = timing["true_inter_token_handoff_gap_ms"]
    expected_gap_ms = (t_cons_tok - t_prod_tok) * 1000.0
    assert abs(true_gap_ms - expected_gap_ms) < 1e-4

    # 3. Delay is strictly incorporated into the true gap
    post_send_latency_ms = (t_cons_tok - t_send_start) * 1000.0
    assert true_gap_ms >= pre_send_delay_observed_ms + post_send_latency_ms - 1e-4
    assert true_gap_ms >= (injected_delay_s * 1000.0)

    # 4. Old receiver-only latency omitted this delay
    receiver_recon_ms = timing["receiver_reconstruction_duration_ms"]
    assert true_gap_ms - receiver_recon_ms >= (injected_delay_s * 1000.0 - 5.0)


def test_process_lifecycle_cleanup_kills_and_reaps():
    """
    Exercises safe_cleanup_processes with a lingering background process:
    Confirms SIGTERM/SIGKILL escalation, verified process termination via os.kill(pid, 0),
    and absence of zombie/surviving children.
    """
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=time.sleep, args=(30,))
    proc.start()
    pid = proc.pid
    assert pid is not None
    assert not verify_pid_dead(pid), f"PID {pid} should be alive initially"

    temp_f = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_f.name
    temp_f.close()
    assert os.path.exists(temp_path)

    report = safe_cleanup_processes(
        processes=[(proc, pid)],
        temp_files=[temp_path],
        timeout_terminate=0.5,
        timeout_kill=0.5,
    )

    assert report["all_child_processes_reaped"] is True
    assert report["temporary_files_deleted"] is True
    assert verify_pid_dead(pid) is True
    assert not os.path.exists(temp_path)
