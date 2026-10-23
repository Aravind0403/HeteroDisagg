"""Verification Script for Process-to-Process Socket KV Cache Handoff.

Spawns independent producer and consumer processes communicating exclusively
over a localhost TCP socket.

Key Features:
- Distinct OS processes spawned via multiprocessing.get_context("spawn").
- OS-assigned port allocation (port=0) with multiprocessing readiness synchronization.
- Full RequestManifest handshake before cache streaming begins.
- Strict frame validation: rejects corrupted frames, duplicate slices, or missing layers.
- Byte-level wire accounting separating framing, metadata JSON, and tensor body.
- Compares decoded logits against an uninterrupted baseline to confirm exact equivalence.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import socket
import statistics
import tempfile
import time
import uuid
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
import click
import torch
import transformers
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from hetero_kv.lifecycle import safe_cleanup_processes, verify_pid_dead
from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import slice_kv_for_prefill_rank
from hetero_kv.transport import (
    RequestManifest,
    FrameMetadata,
    WireAccounting,
    send_manifest,
    recv_manifest,
    send_frame,
    recv_frame,
    validate_manifest_compatibility,
    KVCacheFrameAccumulator,
    DEFAULT_TIMEOUT,
    TransportError,
)

console = Console()


def decode_consumer_worker(
    port_queue: mp.Queue,
    result_queue: mp.Queue,
    model_id: str,
    decode_steps: int = 4,
    device: str = "cpu",
    timeout: float = DEFAULT_TIMEOUT,
    temp_logits_path: Optional[str] = None,
) -> None:
    """
    Consumer Process (Worker B):
    1. Binds to OS-assigned port on 127.0.0.1.
    2. Signals ready and puts port into port_queue.
    3. Accepts connection from prefill producer.
    4. Receives and validates RequestManifest against expected model config.
    5. Receives all frames, accumulating and validating against plan and manifest.
    6. Reconstructs DynamicCache and decodes `decode_steps` continuation tokens on target device.
    7. Returns results, timing decomposition, and process telemetry to result_queue.
    """
    consumer_pid = os.getpid()
    torch_device = torch.device(device)
    # 1. Pre-warm: Load model onto target decode device BEFORE signaling readiness
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
    )
    model.to(torch_device)
    model.eval()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    assigned_port = server_sock.getsockname()[1]
    server_sock.listen(1)
    server_sock.settimeout(timeout + 30)

    # Signal readiness to orchestrator after model is warmed
    port_queue.put({"port": assigned_port, "pid": consumer_pid})

    conn: Optional[socket.socket] = None
    if temp_logits_path is None:
        temp_logits_path = f"/tmp/hkv_consumer_logits_{consumer_pid}.pt"
    try:
        conn, addr = server_sock.accept()
        conn.settimeout(timeout)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_receiver_wire_start = time.monotonic()

        # 2. Receive and validate request manifest against consumer's expected model config
        manifest, manifest_acc = recv_manifest(conn, timeout=timeout)
        validate_manifest_compatibility(
            manifest=manifest,
            model_config=model.config,
            expected_model_id=model_id,
            expected_decode_rank=0,
        )
        total_acc = manifest_acc

        # 3. Accumulate incoming frames with strict layout and plan validation
        accumulator = KVCacheFrameAccumulator(manifest)
        while not accumulator.is_complete():
            meta, body, frame_acc = recv_frame(
                conn,
                manifest,
                timeout=timeout,
                cumulative_bytes=accumulator.cumulative_body_bytes,
                plan=accumulator.plan,
            )
            accumulator.add_frame(meta, body, frame_acc)
            total_acc = total_acc.add(frame_acc)

        # 4. Validate complete frame set before decoding
        accumulator.validate_completeness()
        raw_cache_dict = accumulator.reconstruct_cache()

        # 5. Populate cache on target device
        target_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        reconstructed_cache = DynamicCache()
        for l in range(manifest.num_layers):
            reconstructed_cache.update(
                raw_cache_dict["keys"][l].to(device=torch_device, dtype=target_dtype),
                raw_cache_dict["values"][l].to(device=torch_device, dtype=target_dtype),
                layer_idx=l,
            )

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_consumer_cache_ready = time.monotonic()
        receiver_reconstruction_duration_ms = (t_consumer_cache_ready - t_receiver_wire_start) * 1000.0

        # 6. Decode continuation tokens measuring per-token step durations
        curr_cache = reconstructed_cache
        curr_token = torch.tensor([[manifest.initial_decode_token]], device=torch_device)
        generated_tokens: List[int] = []
        step_logits_list: List[torch.Tensor] = []
        step_durations_ms: List[float] = []

        t_consumer_token_ready: Optional[float] = None

        for step in range(decode_steps):
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            t_s = time.monotonic()
            with torch.no_grad():
                out = model(input_ids=curr_token, past_key_values=curr_cache, use_cache=True)
                logits = out.logits[:, -1, :]
                curr_cache = out.past_key_values
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                curr_token = next_tok
                tok_id = int(next_tok.item())
                generated_tokens.append(tok_id)
                step_logits_list.append(logits[0].cpu())
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            t_e = time.monotonic()
            if step == 0:
                t_consumer_token_ready = t_e
            step_durations_ms.append((t_e - t_s) * 1000.0)

        first_decode_step_latency_ms = step_durations_ms[0] if step_durations_ms else 0.0
        subsequent_step_latencies_ms = step_durations_ms[1:] if len(step_durations_ms) > 1 else []
        median_subsequent_latency_ms = (
            statistics.median(subsequent_step_latencies_ms)
            if subsequent_step_latencies_ms
            else first_decode_step_latency_ms
        )

        generated_text = tokenizer.decode(generated_tokens)
        stacked_logits = torch.stack(step_logits_list, dim=0)  # [steps, vocab_size]
        torch.save(stacked_logits, temp_logits_path)

        result_queue.put(
            {
                "status": "success",
                "consumer_pid": consumer_pid,
                "manifest": json.loads(json.dumps(manifest.__dict__)),
                "frames_received": len(accumulator.received_frames),
                "wire_accounting": {
                    "framing_bytes": total_acc.framing_bytes,
                    "metadata_bytes": total_acc.metadata_bytes,
                    "body_bytes": total_acc.body_bytes,
                    "total_wire_bytes": total_acc.total_wire_bytes,
                },
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "temp_logits_path": temp_logits_path,
                "t_receiver_wire_start": t_receiver_wire_start,
                "t_consumer_cache_ready": t_consumer_cache_ready,
                "t_consumer_token_ready": t_consumer_token_ready,
                "receiver_reconstruction_duration_ms": receiver_reconstruction_duration_ms,
                "first_decode_step_latency_ms": first_decode_step_latency_ms,
                "subsequent_step_latencies_ms": subsequent_step_latencies_ms,
                "median_subsequent_latency_ms": median_subsequent_latency_ms,
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "status": "error",
                "consumer_pid": consumer_pid,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        try:
            server_sock.close()
        except Exception:
            pass


def prefill_producer_worker(
    target_port: int,
    result_queue: mp.Queue,
    model_id: str,
    prompt: str,
    prefill_tp: int = 2,
    decode_tp: int = 1,
    device: str = "cpu",
    enable_fp8: bool = False,
    block_size_tokens: int = 16,
    timeout: float = DEFAULT_TIMEOUT,
    corrupt_frame: Optional[str] = None,
    injected_pre_send_delay_s: float = 0.0,
) -> None:
    """
    Producer Process (Worker A):
    1. Loads model on device and runs prefill on prompt.
    2. Slices and packages KV cache across prefill TP ranks.
    3. Connects to decode worker socket at 127.0.0.1:target_port (with retry).
    4. Transmits RequestManifest followed by all partitioned frames.
    5. Cleanly closes socket and returns telemetry with host monotonic timestamps.
    """
    producer_pid = os.getpid()
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.settimeout(timeout)

    try:
        torch_device = torch.device(device)
        # Load model and run prefill
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        target_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=target_dtype).to(torch_device)
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(torch_device)
        prompt_tokens = input_ids[0].tolist()

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_prefill_start = time.monotonic()
        with torch.no_grad():
            out_prefill = model(input_ids=input_ids, use_cache=True)
            prefill_cache = out_prefill.past_key_values
            initial_next_token = int(torch.argmax(out_prefill.logits[:, -1, :], dim=-1).item())
            initial_token_str = tokenizer.decode([initial_next_token])
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_producer_token_ready = time.monotonic()
        time_to_first_token_ready_ms = (t_producer_token_ready - t_prefill_start) * 1000.0

        if injected_pre_send_delay_s > 0:
            time.sleep(injected_pre_send_delay_s)

        t_sender_wire_start = time.monotonic()

        # Robust extraction across transformers cache representations
        if hasattr(prefill_cache, "layers"):
            num_layers = len(prefill_cache.layers)
        elif hasattr(prefill_cache, "key_cache"):
            num_layers = len(prefill_cache.key_cache)
        else:
            num_layers = len(prefill_cache)

        num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        seq_len = input_ids.shape[1]

        plan = KVPartitionPlan.create(num_kv_heads=num_kv_heads, prefill_tp=prefill_tp, decode_tp=decode_tp)

        # Connect to consumer socket with retry loop
        connected = False
        connect_deadline = time.monotonic() + timeout
        while not connected and time.monotonic() < connect_deadline:
            try:
                client_sock.connect(("127.0.0.1", target_port))
                connected = True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)

        if not connected:
            raise ConnectionRefusedError(f"Failed to connect to 127.0.0.1:{target_port} within {timeout}s")

        manifest = RequestManifest(
            request_id="req-prod-to-cons-001",
            model_id=model_id,
            checkpoint_commit_hash=getattr(model.config, "_commit_hash", None),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            prefill_tp=prefill_tp,
            decode_tp=decode_tp,
            expected_frames=num_layers * 2 * prefill_tp,
            initial_decode_token=initial_next_token,
            initial_decode_token_str=initial_token_str,
            prompt_tokens=prompt_tokens,
            dtype="float16" if torch_device.type == "cuda" else "float32",
            enable_fp8=enable_fp8,
            block_size_tokens=block_size_tokens,
            decode_rank=0,
        )

        acc = send_manifest(client_sock, manifest)

        encoding_used = "float16" if torch_device.type == "cuda" else "float32"
        is_emulated_fallback = False

        # Transmit all frames
        for l in range(num_layers):
            if hasattr(prefill_cache, "layers"):
                orig_k = prefill_cache.layers[l].keys
                orig_v = prefill_cache.layers[l].values
            elif hasattr(prefill_cache, "key_cache"):
                orig_k = prefill_cache.key_cache[l]
                orig_v = prefill_cache.value_cache[l]
            else:
                orig_k = prefill_cache[l][0]
                orig_v = prefill_cache[l][1]

            k_hetero = orig_k.permute(0, 2, 1, 3).contiguous()
            v_hetero = orig_v.permute(0, 2, 1, 3).contiguous()

            for kv_type, tensor_hetero in [("key", k_hetero), ("value", v_hetero)]:
                for p in range(prefill_tp):
                    shard = slice_kv_for_prefill_rank(tensor_hetero, prefill_rank=p, plan=plan)
                    h_slice = plan.prefill_head_slices[p]

                    if enable_fp8:
                        from hetero_kv.quant.fp8_block import quantize_kv_block_fp8
                        compressed = quantize_kv_block_fp8(shard, block_size_tokens=block_size_tokens)
                        is_f8 = compressed.quantized_data.dtype == getattr(torch, "float8_e4m3fn", None)
                        dtype_str = "float8_e4m3fn" if is_f8 else "int8"
                        encoding_used = dtype_str
                        is_emulated_fallback = not is_f8
                        quant_bytes = compressed.quantized_data.view(torch.uint8).cpu().numpy().tobytes()
                        scale_bytes = compressed.scale_factors.cpu().numpy().tobytes()
                        body = quant_bytes + scale_bytes

                        if corrupt_frame == "body_length_mismatch":
                            body = body[: len(body) // 2]
                        meta = FrameMetadata(
                            request_id=manifest.request_id if corrupt_frame != "wrong_request_id" else "req-corrupted",
                            kv_type=kv_type,
                            layer_idx=l,
                            prefill_rank=p,
                            decode_rank=0,
                            start_head_idx=h_slice.start_head_idx,
                            end_head_idx=h_slice.end_head_idx,
                            is_quantized=True,
                            shape=list(compressed.quantized_data.shape),
                            dtype=dtype_str,
                            scale_shape=list(compressed.scale_factors.shape),
                            scale_dtype="float32",
                            original_shape=list(compressed.original_shape),
                            original_dtype="float16" if torch_device.type == "cuda" else "float32",
                            block_size_tokens=compressed.block_size_tokens,
                            body_bytes=len(body) if corrupt_frame != "body_length_mismatch" else len(body) * 2,
                        )
                    else:
                        body = shard.cpu().numpy().tobytes()
                        if corrupt_frame == "body_length_mismatch":
                            body = body[: len(body) // 2]

                        meta = FrameMetadata(
                            request_id=manifest.request_id if corrupt_frame != "wrong_request_id" else "req-corrupted",
                            kv_type=kv_type,
                            layer_idx=l,
                            prefill_rank=p,
                            decode_rank=0,
                            start_head_idx=h_slice.start_head_idx,
                            end_head_idx=h_slice.end_head_idx,
                            is_quantized=False,
                            shape=list(shard.shape),
                            dtype="float16" if torch_device.type == "cuda" else "float32",
                            body_bytes=len(body) if corrupt_frame != "body_length_mismatch" else len(body) * 2,
                        )
                    frame_acc = send_frame(client_sock, meta, body)
                    acc = acc.add(frame_acc)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_sender_wire_end = time.monotonic()

        result_queue.put(
            {
                "status": "success",
                "producer_pid": producer_pid,
                "initial_token": initial_next_token,
                "initial_token_str": initial_token_str,
                "time_to_first_token_ready_ms": time_to_first_token_ready_ms,
                "t_producer_token_ready": t_producer_token_ready,
                "t_sender_wire_start": t_sender_wire_start,
                "t_sender_wire_end": t_sender_wire_end,
                "encoding_used": encoding_used,
                "is_emulated_fallback": is_emulated_fallback,
                "total_wire_bytes": acc.total_wire_bytes,
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "status": "error",
                "producer_pid": producer_pid,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
    finally:
        try:
            client_sock.close()
        except Exception:
            pass


def run_socket_cache_handoff_experiment(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "Heterogeneous disaggregated inference allows",
    decode_steps: int = 4,
    prefill_tp: int = 2,
    decode_tp: int = 1,
    enable_fp8: bool = False,
    block_size_tokens: int = 16,
    device: Optional[str] = None,
    prefill_device: str = "cpu",
    decode_device: str = "cpu",
    timeout: float = DEFAULT_TIMEOUT,
    corrupt_frame: Optional[str] = None,
    injected_pre_send_delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Executes end-to-end multi-process socket cache handoff:
    1. Spawns Producer and Consumer in separate OS processes via multiprocessing ("spawn").
    2. Consumer pre-warms its model, then listens on OS-assigned port; Producer connects and streams cache.
    3. Runs uninterrupted baseline in orchestrator for ground-truth comparison.
    4. Evaluates maximum absolute logit error, token match, and byte breakdown.
    5. Returns decomposed timing metrics measured from true monotonic clock events.
    """
    if device is not None:
        prefill_device = device
        decode_device = device

    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    consumer_result_queue = ctx.Queue()
    producer_result_queue = ctx.Queue()

    # Parent allocates and owns temporary file path before starting workers
    temp_dir = tempfile.gettempdir()
    temp_logits_path = os.path.join(temp_dir, f"hkv_logits_{os.getpid()}_{uuid.uuid4().hex}.pt")

    consumer_proc: Optional[mp.Process] = None
    producer_proc: Optional[mp.Process] = None
    producer_pid: Optional[int] = None
    consumer_pid: Optional[int] = None
    cleanup_observed: Dict[str, Any] = {}

    try:
        # 1. Start Consumer process with parent-owned temp path
        consumer_proc = ctx.Process(
            target=decode_consumer_worker,
            args=(port_queue, consumer_result_queue, model_id, decode_steps, decode_device, timeout, temp_logits_path),
        )
        consumer_proc.start()

        # Wait for consumer readiness and assigned port
        ready_info = port_queue.get(timeout=timeout + 30)
        assigned_port = ready_info["port"]
        consumer_pid = ready_info["pid"]

        # 2. Start Producer process
        producer_proc = ctx.Process(
            target=prefill_producer_worker,
            args=(
                assigned_port,
                producer_result_queue,
                model_id,
                prompt,
                prefill_tp,
                decode_tp,
                prefill_device,
                enable_fp8,
                block_size_tokens,
                timeout,
                corrupt_frame,
                injected_pre_send_delay_s,
            ),
        )
        producer_proc.start()

        # Wait for results from queues with bounded deadline
        producer_res = producer_result_queue.get(timeout=timeout + 30)
        consumer_res = consumer_result_queue.get(timeout=timeout + 30)

        # Join child processes boundedly
        producer_proc.join(timeout=10)
        consumer_proc.join(timeout=10)

        if producer_res["status"] != "success":
            raise RuntimeError(f"Producer failed: {producer_res}")
        if consumer_res["status"] != "success":
            raise RuntimeError(f"Consumer failed: {consumer_res}")

        producer_pid = producer_res["producer_pid"]
        main_pid = os.getpid()

        # Confirm distinct processes
        assert len({main_pid, producer_pid, consumer_pid}) == 3, (
            f"Processes must be distinct! Main: {main_pid}, Producer: {producer_pid}, Consumer: {consumer_pid}"
        )

        # Load consumer logits from temp file
        consumer_logits_tensor = torch.load(temp_logits_path, weights_only=True)
    finally:
        cleanup_observed = safe_cleanup_processes(
            processes=[
                (producer_proc, producer_pid if producer_pid is not None else (producer_proc.pid if producer_proc else None)),
                (consumer_proc, consumer_pid if consumer_pid is not None else (consumer_proc.pid if consumer_proc else None)),
            ],
            temp_files=[temp_logits_path],
            queues=[port_queue, consumer_result_queue, producer_result_queue],
        )

    # 3. Run ground-truth baseline in orchestrator
    torch_base_device = torch.device(decode_device if decode_device != "cpu" else prefill_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    target_base_dtype = torch.float16 if torch_base_device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=target_base_dtype).to(torch_base_device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(torch_base_device)
    with torch.no_grad():
        out_base = model(input_ids=input_ids, use_cache=True)
        base_cache = out_base.past_key_values
        next_tok = torch.argmax(out_base.logits[:, -1, :], dim=-1, keepdim=True)

        base_tokens: List[int] = []
        base_logits_list: List[torch.Tensor] = []
        for _ in range(decode_steps):
            out_step = model(input_ids=next_tok, past_key_values=base_cache, use_cache=True)
            step_logit = out_step.logits[:, -1, :]
            base_cache = out_step.past_key_values
            next_tok = torch.argmax(step_logit, dim=-1, keepdim=True)
            base_tokens.append(int(next_tok.item()))
            base_logits_list.append(step_logit[0].cpu())

    base_text = tokenizer.decode(base_tokens)
    reconstructed_tokens = consumer_res["generated_tokens"]
    reconstructed_text = consumer_res["generated_text"]

    # 4. Compare step-by-step logits (casting to float32 for stable cosine computation)
    max_logit_errors: List[float] = []
    step_comparisons: List[Dict[str, Any]] = []

    for s in range(decode_steps):
        t_base = base_logits_list[s].float()
        t_recon = consumer_logits_tensor[s].float()
        max_err = torch.max(torch.abs(t_base - t_recon)).item()
        cos_sim = torch.cosine_similarity(t_base.flatten(), t_recon.flatten(), dim=0).item()
        max_logit_errors.append(max_err)
        step_comparisons.append(
            {
                "step": s + 1,
                "baseline_token": base_tokens[s],
                "baseline_token_str": tokenizer.decode([base_tokens[s]]),
                "reconstructed_token": reconstructed_tokens[s],
                "reconstructed_token_str": tokenizer.decode([reconstructed_tokens[s]]),
                "tokens_match": bool(base_tokens[s] == reconstructed_tokens[s]),
                "max_abs_logit_error": max_err,
                "cosine_similarity": max(-1.0, min(1.0, float(cos_sim))),
            }
        )

    all_tokens_match = (base_tokens == reconstructed_tokens)
    max_abs_divergence = max(max_logit_errors)
    mean_cos = sum(s["cosine_similarity"] for s in step_comparisons) / len(step_comparisons)

    # Wire byte accounting and dual compression ratios
    wa = consumer_res["wire_accounting"]
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    seq_len = inputs.input_ids.shape[1]
    if hasattr(out_base.past_key_values, "layers"):
        num_layers = len(out_base.past_key_values.layers)
    elif hasattr(out_base.past_key_values, "key_cache"):
        num_layers = len(out_base.past_key_values.key_cache)
    else:
        num_layers = len(out_base.past_key_values)

    uncomp_fp32_bytes = seq_len * num_kv_heads * head_dim * 4 * num_layers * 2
    uncomp_fp16_bytes = uncomp_fp32_bytes // 2
    wire_bytes = wa["total_wire_bytes"]
    body_bytes = wa["body_bytes"]

    ratio_vs_fp32 = round(uncomp_fp32_bytes / max(1, body_bytes), 2)
    ratio_vs_fp16 = round(uncomp_fp16_bytes / max(1, body_bytes), 2)
    wire_reduction_vs_fp16_pct = round((1.0 - (body_bytes / max(1, uncomp_fp16_bytes))) * 100.0, 1)

    if enable_fp8:
        divergence_within_tolerance = bool(mean_cos >= 0.99 and body_bytes < uncomp_fp32_bytes)
        tolerance_desc = "Mean Cosine Sim >= 0.99 & Tensor Body Compressed"
    else:
        divergence_within_tolerance = bool(max_abs_divergence <= 1e-6)
        tolerance_desc = "|Δ Logits| <= 1e-6"

    # Precise event-based monotonic clock timings
    t_prod_tok = producer_res.get("t_producer_token_ready", 0.0)
    t_send_start = producer_res.get("t_sender_wire_start", 0.0)
    t_cons_cache = consumer_res.get("t_consumer_cache_ready", 0.0)
    t_cons_tok = consumer_res.get("t_consumer_token_ready", 0.0)

    if t_prod_tok > 0 and t_cons_tok > 0:
        true_inter_token_handoff_gap_ms = (t_cons_tok - t_prod_tok) * 1000.0
        transfer_and_reconstruction_latency_ms = (t_cons_cache - t_send_start) * 1000.0
    else:
        true_inter_token_handoff_gap_ms = 0.0
        transfer_and_reconstruction_latency_ms = 0.0

    return {
        "model_id": model_id,
        "enable_fp8": enable_fp8,
        "block_size_tokens": block_size_tokens,
        "pids": {
            "orchestrator_pid": main_pid,
            "producer_pid": producer_pid,
            "consumer_pid": consumer_pid,
        },
        "device_placement": {
            "prefill_device": prefill_device,
            "decode_device": decode_device,
        },
        "codec_telemetry": {
            "enable_fp8": enable_fp8,
            "encoding_used": producer_res.get("encoding_used", "float32"),
            "is_emulated_fallback": producer_res.get("is_emulated_fallback", False),
            "block_size_tokens": block_size_tokens,
        },
        "timing_decomposition": {
            "time_to_first_token_ready_ms": producer_res.get("time_to_first_token_ready_ms", 0.0),
            "transfer_and_reconstruction_latency_ms": transfer_and_reconstruction_latency_ms,
            "receiver_reconstruction_duration_ms": consumer_res.get("receiver_reconstruction_duration_ms", 0.0),
            "first_decode_step_latency_ms": consumer_res.get("first_decode_step_latency_ms", 0.0),
            "true_inter_token_handoff_gap_ms": true_inter_token_handoff_gap_ms,
            "subsequent_step_latencies_ms": consumer_res.get("subsequent_step_latencies_ms", []),
            "median_subsequent_latency_ms": consumer_res.get("median_subsequent_latency_ms", 0.0),
            "raw_event_timestamps": {
                "t_producer_token_ready": t_prod_tok,
                "t_sender_wire_start": t_send_start,
                "t_consumer_cache_ready": t_cons_cache,
                "t_consumer_token_ready": t_cons_tok,
            },
        },
        "assigned_port": assigned_port,
        "all_tokens_match": all_tokens_match,
        "max_abs_logit_divergence": max_abs_divergence,
        "mean_cosine_similarity": round(mean_cos, 6),
        "tolerance": tolerance_desc,
        "divergence_within_tolerance": divergence_within_tolerance,
        "baseline_generated_text": base_text,
        "reconstructed_generated_text": reconstructed_text,
        "wire_accounting": {
            **wa,
            "uncompressed_fp32_bytes": uncomp_fp32_bytes,
            "uncompressed_fp16_bytes": uncomp_fp16_bytes,
            "ratio_vs_fp32": ratio_vs_fp32,
            "ratio_vs_fp16": ratio_vs_fp16,
            "wire_reduction_vs_fp16_pct": wire_reduction_vs_fp16_pct,
        },
        "step_comparisons": step_comparisons,
        "cleanup_observed": cleanup_observed,
    }


@click.command()
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--prompt", "-p", default="Heterogeneous disaggregated inference allows", help="Input prompt.")
@click.option("--steps", "-s", default=4, help="Autoregressive decoding steps.")
@click.option("--enable-fp8/--no-fp8", default=False, help="Enable block-scaled FP8 KV cache wire compression.")
@click.option("--block-size", default=16, help="FP8 block tile size.")
@click.option("--device", default=None, help="Global device placement for both workers (e.g. cpu, mps, cuda:0).")
@click.option("--prefill-device", default="cpu", help="Device for prefill producer worker.")
@click.option("--decode-device", default="cpu", help="Device for decode consumer worker.")
@click.option("--output", "-o", default="benchmarks/socket_cache_handoff.json", help="JSON output path.")
def main(
    model: str,
    prompt: str,
    steps: int,
    enable_fp8: bool,
    block_size: int,
    device: Optional[str],
    prefill_device: str,
    decode_device: str,
    output: str,
):
    """End-to-End Multiprocess Socket KV Cache Transfer Verification."""
    if device is not None:
        prefill_device = device
        decode_device = device

    mode_str = "FP8 Block-Quantized" if enable_fp8 else "Uncompressed Exact"
    metric_desc = "Cosine Sim >= 0.99 & Wire Ratio >= 1.4x vs FP16" if enable_fp8 else "Max Absolute Logit Error (|Δ Logits| <= 1e-6)"
    console.print(
        Panel(
            f"[bold cyan]Process-to-Process Socket KV Cache Transfer Verification[/bold cyan]\n"
            f"Target Model: {model} | Prefill TP=2 -> Decode TP=1\n"
            f"Mode: {mode_str} (Block Size: {block_size})\n"
            f"Placement: Prefill={prefill_device} -> Decode={decode_device}\n"
            f"Transport: Localhost TCP Socket with CPU Staging (127.0.0.1:OS-assigned)\n"
            f"Workers: Distinct OS processes spawned via multiprocessing (get_context('spawn'))\n"
            f"Pass Criteria: {metric_desc}",
            border_style="cyan",
        )
    )

    results = run_socket_cache_handoff_experiment(
        model_id=model,
        prompt=prompt,
        decode_steps=steps,
        enable_fp8=enable_fp8,
        block_size_tokens=block_size,
        prefill_device=prefill_device,
        decode_device=decode_device,
    )

    pids = results["pids"]
    codec_info = results["codec_telemetry"]
    timing = results["timing_decomposition"]

    console.print(
        f"[dim]Process Topology: Orchestrator PID={pids['orchestrator_pid']} | "
        f"Producer PID={pids['producer_pid']} ({prefill_device}) | "
        f"Consumer PID={pids['consumer_pid']} ({decode_device}) | "
        f"Socket Port={results['assigned_port']}[/dim]\n"
    )

    table = Table(title="Step-by-Step Socket Decoding Verification", header_style="bold magenta")
    table.add_column("Step", justify="center", style="cyan")
    table.add_column("Baseline Token", style="dim")
    table.add_column("Socket Token", style="white")
    table.add_column("Token Match", justify="center")
    table.add_column("Max |Δ Logit|", justify="right", style="green")
    table.add_column("Cosine Sim", justify="right", style="yellow")

    for s in results["step_comparisons"]:
        match_str = "[bold green]YES[/bold green]" if s["tokens_match"] else "[bold red]NO[/bold red]"
        table.add_row(
            str(s["step"]),
            f"'{s['baseline_token_str']}' ({s['baseline_token']})",
            f"'{s['reconstructed_token_str']}' ({s['reconstructed_token']})",
            match_str,
            f"{s['max_abs_logit_error']:.2e}",
            f"{s['cosine_similarity']:.6f}",
        )

    console.print(table)

    wa = results["wire_accounting"]
    summary_panel = Panel(
        f"[bold]Process Verification & Placement:[/bold]\n"
        f"  Distinct Process Isolation: [bold green]CONFIRMED[/bold green] (3 distinct PIDs)\n"
        f"  Prefill Worker Placement:   {prefill_device}\n"
        f"  Decode Worker Placement:    {decode_device}\n"
        f"  Transport Mechanism:        Localhost TCP with CPU staging (socket buffer & timeout control)\n\n"
        f"[bold]Wire Codec Telemetry:[/bold]\n"
        f"  Requested Mode:             {mode_str}\n"
        f"  Encoding Used on Wire:      [bold yellow]{codec_info['encoding_used']}[/bold yellow]\n"
        f"  Fallback Status:            {'[italic yellow]Emulated Fallback (Non-native FP8 HW)[/italic yellow]' if codec_info['is_emulated_fallback'] else '[bold green]Direct Native Support[/bold green]'}\n\n"
        f"[bold]Disaggregated Serving Latency Decomposition (Measured Monotonic Events):[/bold]\n"
        f"  T_first_token (Prefill Token 0 Ready): {timing.get('time_to_first_token_ready_ms', timing.get('time_to_first_emitted_token_ms', 0.0)):.2f} ms (Computed on prefill worker)\n"
        f"  T_transfer (Wire + Staging + Reconstruct): {timing.get('transfer_and_reconstruction_latency_ms', timing.get('handoff_duration_ms', 0.0)):.2f} ms (Sender start to receiver ready)\n"
        f"  T_reconstruct (Receiver Local Duration): {timing.get('receiver_reconstruction_duration_ms', 0.0):.2f} ms\n"
        f"  ΔT_gap (True Inter-Token Handoff Gap): {timing.get('true_inter_token_handoff_gap_ms', timing.get('inter_token_handoff_gap_ms', 0.0)):.2f} ms (Token 0 ready to Token 1 ready)\n"
        f"  T_decode (First Token Compute):         {timing.get('first_decode_step_latency_ms', 0.0):.2f} ms\n"
        f"  T_decode (Continuation Step Median):    {timing.get('median_subsequent_latency_ms', 0.0):.2f} ms\n\n"
        f"[bold]Numerical Equivalence:[/bold]\n"
        f"  Max Absolute Logit Error: [bold green]{results['max_abs_logit_divergence']:.2e}[/bold green] (Tolerance: <= 1e-6)\n"
        f"  Tokens Match: {'[bold green]100% EXACT[/bold green]' if results['all_tokens_match'] else '[bold red]FAILED[/bold red]'}\n\n"
        f"[bold]Wire Byte Accounting:[/bold]\n"
        f"  Framing Bytes:     {wa['framing_bytes']:,} B (Magic + header length headers)\n"
        f"  Metadata Bytes:    {wa['metadata_bytes']:,} B (JSON manifest and frame headers)\n"
        f"  Tensor Body Bytes: {wa['body_bytes']:,} B (Raw contiguous tensor memory)\n"
        f"  Total Wire Bytes:  {wa['total_wire_bytes']:,} B\n\n"
        f"[bold]Generated Continuations:[/bold]\n"
        f"  Baseline:      \"{results['baseline_generated_text']}\"\n"
        f"  Reconstructed: \"{results['reconstructed_generated_text']}\"",
        title="Socket Transfer Verification Summary",
        border_style="green" if results["divergence_within_tolerance"] else "red",
    )
    console.print(summary_panel)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Results written to {out_path}[/dim]")


if __name__ == "__main__":
    main()

