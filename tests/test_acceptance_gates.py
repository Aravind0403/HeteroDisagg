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


def test_step2_allow_emulation_fails_closed_when_emulation_fails():
    """
    --allow-emulation MUST return success ONLY when emulation correctness passes.
    If emulation correctness fails (e.g. 99.0 logit error or token mismatch),
    the CLI MUST exit with code 1, even with --allow-emulation specified.
    """
    from click.testing import CliRunner
    from scripts.run_step2_two_gpu_handoff import main as step2_main

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step2_emul_fail.json")

        # Mock run_step2_suite to simulate emulation pipeline failure (emulation_pipeline_passed=False)
        def mock_suite(*args, **kwargs):
            return {
                "completion_criterion": {
                    "two_gpu_milestone_passed": False,
                    "emulation_pipeline_passed": False,
                    "discrete_checks": {
                        "distinct_physical_cuda_gpus": False,
                        "uncompressed_tokens_match": False,
                        "uncompressed_logit_error_within_tolerance": False,
                        "uncompressed_error_is_finite": True,
                        "fp8_tokens_match": False,
                        "fp8_cosine_similarity_within_threshold": False,
                        "fp8_cosine_is_finite": True,
                        "null_cache_negative_control_diverged": True,
                        "child_processes_reaped": True,
                        "temporary_files_deleted": True,
                    },
                },
                "hardware_telemetry": {"is_two_physical_cuda_gpus": False},
                "uncompressed_exact": {
                    "prefill_device": "cpu",
                    "decode_device": "cpu",
                    "encoding_used": "float32",
                    "tokens_match": False,
                    "max_abs_logit_error": 99.0,
                    "cosine_similarity": 0.1,
                    "wire_accounting": {"body_bytes": 100, "total_wire_bytes": 200},
                    "generated_text": "fail",
                    "timing_decomposition": {
                        "time_to_first_token_ready_ms": 10.0,
                        "transfer_and_reconstruction_latency_ms": 5.0,
                        "receiver_reconstruction_duration_ms": 2.0,
                        "true_inter_token_handoff_gap_ms": 15.0,
                        "median_subsequent_latency_ms": 3.0,
                    },
                },
                "fp8_wire_codec": {
                    "encoding_used": "float8_e4m3fn",
                    "is_emulated_fallback": True,
                    "tokens_match": False,
                    "max_abs_logit_error": 99.0,
                    "cosine_similarity": 0.1,
                    "wire_accounting": {"body_bytes": 100, "total_wire_bytes": 200},
                    "generated_text": "fail",
                    "timing_decomposition": {
                        "time_to_first_token_ready_ms": 10.0,
                        "transfer_and_reconstruction_latency_ms": 5.0,
                        "receiver_reconstruction_duration_ms": 2.0,
                        "true_inter_token_handoff_gap_ms": 15.0,
                        "median_subsequent_latency_ms": 3.0,
                    },
                },
                "empirical_negative_control": {
                    "diverged_from_baseline": True,
                    "generated_text": "null",
                },
                "expansion_breakdown": {
                    "tensor_body_expansion_pct": 0.0,
                    "application_wire_expansion_pct": 0.0,
                },
                "tolerances": {
                    "uncompressed_max_abs_logit_tolerance": 1e-5,
                    "fp8_cosine_similarity_threshold": 0.999,
                },
            }

        with patch("scripts.run_step2_two_gpu_handoff.run_step2_suite", side_effect=mock_suite):
            result = runner.invoke(step2_main, ["--allow-emulation", "--output", out_json])
            assert result.exit_code == 1, f"Expected exit code 1 when emulation fails, got {result.exit_code}. Output: {result.output}"


def test_step2_device_alias_resolution_and_validation():
    """
    CUDA devices must be resolved to explicit integer indices before checking that they differ.
    Aliases like 'cuda' and 'cuda:0' (or identical explicit indices 'cuda:1' and 'cuda:1')
    must resolve to the same device index and reject is_two_physical_cuda_gpus.
    """
    # Simulate a 2-GPU CUDA environment
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.current_device", return_value=0):

        # Case A: Alias 'cuda' and 'cuda:0' (both point to GPU 0)
        telemetry_alias = inspect_two_gpu_system("cuda", "cuda:0")
        assert telemetry_alias["resolved_prefill_device_index"] == 0
        assert telemetry_alias["resolved_decode_device_index"] == 0
        assert telemetry_alias["is_two_physical_cuda_gpus"] is False, (
            "Alias 'cuda' and 'cuda:0' both resolve to GPU 0 and must not be marked as two distinct GPUs!"
        )

        # Case B: Duplicate explicit indices 'cuda:1' and 'cuda:1'
        telemetry_dup = inspect_two_gpu_system("cuda:1", "cuda:1")
        assert telemetry_dup["resolved_prefill_device_index"] == 1
        assert telemetry_dup["resolved_decode_device_index"] == 1
        assert telemetry_dup["is_two_physical_cuda_gpus"] is False

        # Case C: Valid distinct physical GPUs 'cuda:0' and 'cuda:1'
        with patch("torch.cuda.get_device_properties") as mock_props, \
             patch("torch.cuda.can_device_access_peer", return_value=True):
            mock_prop = MagicMock()
            mock_prop.name = "NVIDIA RTX 4070 SUPER"
            mock_prop.total_memory = 12 * (1024**3)
            mock_prop.major = 8
            mock_prop.minor = 9
            mock_props.return_value = mock_prop

            telemetry_distinct = inspect_two_gpu_system("cuda:0", "cuda:1")
            assert telemetry_distinct["resolved_prefill_device_index"] == 0
            assert telemetry_distinct["resolved_decode_device_index"] == 1
            assert telemetry_distinct["is_two_physical_cuda_gpus"] is True


def test_step0_rejects_missing_or_failed_cleanup_observations():
    """
    Step 0 readiness MUST reject missing, empty, or failed cleanup observations.
    """
    from hetero_kv.lifecycle import validate_cleanup_report

    # Unit assertions on validate_cleanup_report
    assert validate_cleanup_report(None) is False
    assert validate_cleanup_report({}) is False
    assert validate_cleanup_report({"all_child_processes_reaped": True}) is False
    assert validate_cleanup_report({
        "all_child_processes_reaped": True,
        "temporary_files_deleted": False,
        "surviving_pids": [],
        "leaked_temp_files": [],
    }) is False
    assert validate_cleanup_report({
        "all_child_processes_reaped": True,
        "temporary_files_deleted": True,
        "surviving_pids": [12345],
        "leaked_temp_files": [],
    }) is False
    assert validate_cleanup_report({
        "all_child_processes_reaped": True,
        "temporary_files_deleted": True,
        "surviving_pids": [],
        "leaked_temp_files": ["/tmp/leaked.pt"],
    }) is False
    assert validate_cleanup_report({
        "all_child_processes_reaped": True,
        "temporary_files_deleted": True,
        "surviving_pids": [],
        "leaked_temp_files": [],
    }) is True

    # End-to-end Step 0 test: inject missing cleanup_observed into uncompressed run
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step0_cleanup_fail.json")

        real_fn = run_socket_cache_handoff_experiment
        def mock_handoff_missing_cleanup(*args, **kwargs):
            res = real_fn(*args, **kwargs)
            # Remove cleanup_observed
            res.pop("cleanup_observed", None)
            return res

        with patch("scripts.run_local_step0_checks.run_socket_cache_handoff_experiment", side_effect=mock_handoff_missing_cleanup):
            readiness = run_step0_checks(output_path=out_json)
            assert readiness["step0_ready_for_gpu_rental"] is False
            assert readiness["cleanup_status"]["all_child_processes_reaped"] is False


def test_step2_explicit_cuda_device_forwarding_with_parent_current_device_1():
    """
    Verifies that when parent's current_device=1 and requested devices are ('cuda', 'cuda:0'),
    the devices are converted to explicit 'cuda:1' and 'cuda:0' strings before baseline
    execution and worker launch, preventing spawned child workers from selecting GPU 0.
    """
    from scripts.run_step2_two_gpu_handoff import run_step2_suite

    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "step2_device_forwarding.json")

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.current_device", return_value=1), \
             patch("scripts.run_step2_two_gpu_handoff.run_baseline_and_ablation") as mock_baseline, \
             patch("scripts.run_step2_two_gpu_handoff.execute_two_gpu_handoff_case") as mock_exec, \
             patch("torch.cuda.get_device_properties") as mock_props, \
             patch("torch.cuda.can_device_access_peer", return_value=True):

            mock_prop = MagicMock()
            mock_prop.name = "NVIDIA RTX 4070 SUPER"
            mock_prop.total_memory = 12 * (1024**3)
            mock_prop.major = 8
            mock_prop.minor = 9
            mock_props.return_value = mock_prop

            dummy_logits = torch.zeros((2, 10))
            dummy_tokens = [1, 2]
            dummy_text = "test"
            mock_baseline.return_value = (dummy_logits, dummy_tokens, dummy_text, [3, 4], "diverged", "commit123")

            def mock_exec_fn(*args, **kwargs):
                return {
                    "consumer": {
                        "generated_tokens": dummy_tokens,
                        "generated_text": dummy_text,
                        "wire_accounting": {"body_bytes": 100, "total_wire_bytes": 200},
                        "manifest_dict": {},
                    },
                    "producer": {
                        "encoding_used": "float16",
                        "is_emulated_fallback": False,
                        "prompt_tokens": [10, 20],
                        "seq_len": 2,
                    },
                    "consumer_logits": dummy_logits,
                    "timing_decomposition": {
                        "time_to_first_token_ready_ms": 1.0,
                        "transfer_and_reconstruction_latency_ms": 1.0,
                        "receiver_reconstruction_duration_ms": 0.5,
                        "first_decode_step_latency_ms": 1.0,
                        "true_inter_token_handoff_gap_ms": 2.0,
                        "median_subsequent_latency_ms": 1.0,
                        "subsequent_step_latencies_ms": [1.0],
                    },
                    "cleanup_report": {
                        "all_child_processes_reaped": True,
                        "temporary_files_deleted": True,
                        "surviving_pids": [],
                        "leaked_temp_files": [],
                    },
                }

            mock_exec.side_effect = mock_exec_fn

            results = run_step2_suite(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                prompt="Heterogeneous disaggregated inference allows",
                steps=2,
                prefill_device="cuda",
                decode_device="cuda:0",
                output_path=out_json,
            )

            # 1. Assert worker launch received explicit canonical strings
            assert mock_exec.call_count == 2
            first_call_kwargs = mock_exec.call_args_list[0].kwargs
            assert first_call_kwargs["prefill_device"] == "cuda:1", (
                f"Worker must receive explicit 'cuda:1', got {first_call_kwargs['prefill_device']}"
            )
            assert first_call_kwargs["decode_device"] == "cuda:0", (
                f"Worker must receive explicit 'cuda:0', got {first_call_kwargs['decode_device']}"
            )

            # 2. Assert baseline received explicit canonical decode device
            baseline_call_args = mock_baseline.call_args[0]
            assert baseline_call_args[3] == "cuda:0"

            # 3. Assert artifact exported explicit canonical strings
            assert results["uncompressed_exact"]["prefill_device"] == "cuda:1"
            assert results["uncompressed_exact"]["decode_device"] == "cuda:0"
            assert results["fp8_wire_codec"]["prefill_device"] == "cuda:1"
            assert results["fp8_wire_codec"]["decode_device"] == "cuda:0"
            assert results["hardware_telemetry"]["prefill_device"] == "cuda:1"
            assert results["hardware_telemetry"]["decode_device"] == "cuda:0"
            assert results["hardware_telemetry"]["requested_prefill_device"] == "cuda"
            assert results["hardware_telemetry"]["requested_decode_device"] == "cuda:0"
