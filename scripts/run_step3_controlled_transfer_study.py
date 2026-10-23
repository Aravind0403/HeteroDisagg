#!/usr/bin/env python3
"""
Step 3: Controlled Cache Transfer Study — FP16 vs Always-On FP8 vs Adaptive Policy.

Executes physical multi-process KV cache handoff across two distinct GPUs on the same host:
Prefill Worker (GPU 0) -> Decode Worker (GPU 1) over host-staged localhost TCP socket (127.0.0.1).

Research Question:
Does GPU compression reduce complete handoff latency for any tested configuration,
and does the policy choose appropriately?

Study Structure:
1. Compares three modes across swept cache sizes (token sequence lengths):
   - Uncompressed FP16: Raw FP16 tensors transmitted without loss.
   - Always-On FP8: Block-scaled FP8 wire compression with per-block FP32 scale factors.
   - Adaptive Selection: Evaluates AdaptiveCompressionPolicy dynamically based on link bandwidth
     and cache dimensions, selecting the predicted lowest-latency mode.
2. Sweeps cache sizes:
   - Evaluates short, medium, and long sequences (e.g. 64, 128, 256, 512, 1024 tokens).
3. Warm-up and Timed Repetitions:
   - Warmed up prior to recording measurements to prime CUDA kernels, allocators, and socket buffers.
   - Executes N repetitions per condition, recording individual runs and computing summary statistics.
4. Transport & Bandwidth Options:
   - Evaluates unrestricted host-staged transport (localhost loopback TCP).
   - Supports calibrated bandwidth restriction via socket chunk pacing (--bandwidth-mbps),
     clearly labeling emulated conditions.
5. Export & Reproducibility:
   - Captures exact code revision (git commit hash, branch, dirty flag), environment, CLI invocation,
     individual run latencies, wire accounting, and empirical policy accuracy.
   - Exports results to benchmarks/step3_controlled_transfer_study.json.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import math
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import click
import torch
import transformers
import multiprocessing as mp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from hetero_kv.lifecycle import safe_cleanup_processes, verify_pid_dead, validate_cleanup_report
from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import slice_kv_for_prefill_rank
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    compute_cosine_similarity,
)
from hetero_kv.policy import AdaptiveCompressionPolicy, CompressionMode, PolicyEvaluationResult
from hetero_kv.transport import (
    RequestManifest,
    FrameMetadata,
    WireAccounting,
    send_manifest,
    recv_manifest,
    recv_frame,
    validate_manifest_compatibility,
    KVCacheFrameAccumulator,
    DEFAULT_TIMEOUT,
    MAGIC_FRAME,
    MAX_METADATA_BYTES,
)
from scripts.run_step2_two_gpu_handoff import resolve_canonical_device_str, inspect_two_gpu_system

console = Console()


def get_git_metadata() -> Dict[str, Any]:
    """Retrieves current git repository revision and working tree state."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
        commit_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_dirty": is_dirty,
            "commit_date": commit_date,
        }
    except Exception as e:
        return {
            "commit_hash": "unknown",
            "branch": "unknown",
            "is_dirty": False,
            "error": str(e),
        }


def generate_prompt_for_seq_len(tokenizer: Any, target_seq_len: int) -> str:
    """
    Constructs a deterministic prompt whose tokenized length is exactly target_seq_len.
    Uses realistic technical English sentences repeated and cleanly truncated.
    """
    base_text = (
        "Heterogeneous disaggregated inference decouples prefill and decode execution onto specialized compute nodes. "
        "Key-value cache transmission across high-speed interconnects or networks is the primary latency determinant. "
        "Block-scaled FP8 compression reduces wire bytes at the expense of quantization and dequantization compute overhead. "
    )
    tokens = tokenizer.encode(base_text)
    if not tokens:
        tokens = [100, 101, 102]
    multiplier = (target_seq_len // len(tokens)) + 2
    repeated_tokens = (tokens * multiplier)[:target_seq_len]
    return tokenizer.decode(repeated_tokens)


def send_frame_with_pacing(
    sock: socket.socket,
    meta: FrameMetadata,
    body: bytes,
    bandwidth_mbps: Optional[float] = None,
    chunk_size: int = 65536,
) -> WireAccounting:
    """
    Sends frame header, metadata JSON, and binary body over TCP socket.
    If bandwidth_mbps is provided, throttles transmission via calibrated chunk pacing.
    """
    meta_json = json.dumps(asdict(meta)).encode("utf-8")
    meta_len = len(meta_json)
    if meta_len > MAX_METADATA_BYTES:
        raise ValueError(f"Frame metadata exceeds limit: {meta_len} > {MAX_METADATA_BYTES}")
    if len(body) != meta.body_bytes:
        raise ValueError(f"Body length ({len(body)}) does not match metadata ({meta.body_bytes})")

    header = MAGIC_FRAME + meta_len.to_bytes(4, byteorder="big")
    full_payload = header + meta_json + body

    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        sock.sendall(full_payload)
    else:
        # Calibrated chunk pacing: target bytes per second
        target_bytes_per_sec = (bandwidth_mbps * 1e6) / 8.0
        total_len = len(full_payload)
        sent = 0
        while sent < total_len:
            t_chunk_start = time.monotonic()
            chunk = full_payload[sent : sent + chunk_size]
            sock.sendall(chunk)
            sent += len(chunk)

            desired_elapsed_sec = len(chunk) / target_bytes_per_sec
            actual_elapsed_sec = time.monotonic() - t_chunk_start
            sleep_time = desired_elapsed_sec - actual_elapsed_sec
            if sleep_time > 0:
                time.sleep(sleep_time)

    return WireAccounting(
        framing_bytes=len(header),
        metadata_bytes=meta_len,
        body_bytes=len(body),
    )


def _decode_consumer_worker_proc(
    port_queue: mp.Queue,
    result_queue: mp.Queue,
    model_id: str,
    decode_steps: int,
    device_str: str,
    timeout: float,
    temp_logits_path: str,
) -> None:
    """Consumer Worker: runs on decode device (e.g. cuda:1). Pre-warms model before signaling ready."""
    pid = os.getpid()
    torch_device = torch.device(device_str)

    # 1. Pre-warm: Load model onto target decode device BEFORE signaling socket readiness
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    target_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=target_dtype)
    model.to(torch_device)
    model.eval()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    assigned_port = server_sock.getsockname()[1]
    server_sock.listen(1)
    server_sock.settimeout(timeout + 30)

    # Signal readiness to orchestrator after model is resident in device memory
    port_queue.put({"port": assigned_port, "pid": pid})

    conn: Optional[socket.socket] = None
    try:
        conn, _ = server_sock.accept()
        conn.settimeout(timeout)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_receiver_wire_start = time.monotonic()

        # 2. Receive manifest
        manifest, manifest_acc = recv_manifest(conn, timeout=timeout)
        validate_manifest_compatibility(
            manifest=manifest,
            model_config=model.config,
            expected_model_id=model_id,
            expected_decode_rank=0,
        )
        total_acc = manifest_acc

        # 3. Accumulate frames
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

        accumulator.validate_completeness()
        raw_cache_dict = accumulator.reconstruct_cache()

        # 4. Transfer reconstructed cache onto target decode device
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

        # 5. Decode continuation tokens
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
        stacked_logits = torch.stack(step_logits_list, dim=0)
        torch.save(stacked_logits, temp_logits_path)

        result_queue.put(
            {
                "status": "success",
                "consumer_pid": pid,
                "device": device_str,
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
                "manifest_dict": json.loads(json.dumps(manifest.__dict__)),
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
                "consumer_pid": pid,
                "device": device_str,
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


def _prefill_producer_worker_proc(
    target_port: int,
    result_queue: mp.Queue,
    model_id: str,
    prompt: str,
    prefill_tp: int,
    decode_tp: int,
    device_str: str,
    enable_fp8: bool,
    block_size_tokens: int,
    timeout: float,
    bandwidth_mbps: Optional[float] = None,
) -> None:
    """Producer Worker: runs on prefill device (e.g. cuda:0). Records monotonic timestamps."""
    pid = os.getpid()
    torch_device = torch.device(device_str)
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.settimeout(timeout)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        target_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=target_dtype)
        model.to(torch_device)
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

        t_sender_wire_start = time.monotonic()

        # Extract KV dimensions
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

        # Connect to consumer socket
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
            request_id=f"step3-req-{uuid.uuid4().hex[:8]}",
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
                        compressed = quantize_kv_block_fp8(shard, block_size_tokens=block_size_tokens)
                        is_f8 = compressed.quantized_data.dtype == getattr(torch, "float8_e4m3fn", None)
                        dtype_str = "float8_e4m3fn" if is_f8 else "int8"
                        encoding_used = dtype_str
                        is_emulated_fallback = not is_f8
                        quant_bytes = compressed.quantized_data.view(torch.uint8).cpu().numpy().tobytes()
                        scale_bytes = compressed.scale_factors.cpu().numpy().tobytes()
                        body = quant_bytes + scale_bytes

                        meta = FrameMetadata(
                            request_id=manifest.request_id,
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
                            body_bytes=len(body),
                        )
                    else:
                        body = shard.cpu().numpy().tobytes()
                        meta = FrameMetadata(
                            request_id=manifest.request_id,
                            kv_type=kv_type,
                            layer_idx=l,
                            prefill_rank=p,
                            decode_rank=0,
                            start_head_idx=h_slice.start_head_idx,
                            end_head_idx=h_slice.end_head_idx,
                            is_quantized=False,
                            shape=list(shard.shape),
                            dtype="float16" if torch_device.type == "cuda" else "float32",
                            body_bytes=len(body),
                        )

                    frame_acc = send_frame_with_pacing(
                        sock=client_sock,
                        meta=meta,
                        body=body,
                        bandwidth_mbps=bandwidth_mbps,
                    )
                    acc = acc.add(frame_acc)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_sender_wire_end = time.monotonic()

        result_queue.put(
            {
                "status": "success",
                "producer_pid": pid,
                "device": device_str,
                "initial_token": initial_next_token,
                "initial_token_str": initial_token_str,
                "time_to_first_token_ready_ms": time_to_first_token_ready_ms,
                "t_prefill_start": t_prefill_start,
                "t_producer_token_ready": t_producer_token_ready,
                "t_sender_wire_start": t_sender_wire_start,
                "t_sender_wire_end": t_sender_wire_end,
                "sender_wire_duration_ms": (t_sender_wire_end - t_sender_wire_start) * 1000.0,
                "encoding_used": encoding_used,
                "is_emulated_fallback": is_emulated_fallback,
                "total_wire_bytes": acc.total_wire_bytes,
                "prompt_tokens": prompt_tokens,
                "seq_len": seq_len,
                "checkpoint_commit_hash": getattr(model.config, "_commit_hash", None),
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "status": "error",
                "producer_pid": pid,
                "device": device_str,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
    finally:
        try:
            client_sock.close()
        except Exception:
            pass


def execute_handoff_iteration(
    model_id: str,
    prompt: str,
    decode_steps: int,
    prefill_device: str,
    decode_device: str,
    enable_fp8: bool,
    bandwidth_mbps: Optional[float] = None,
    block_size_tokens: int = 16,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Executes a single timed handoff iteration from prefill_device to decode_device."""
    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    consumer_result_queue = ctx.Queue()
    producer_result_queue = ctx.Queue()

    temp_dir = tempfile.gettempdir()
    temp_logits_path = os.path.join(temp_dir, f"step3_logits_{os.getpid()}_{uuid.uuid4().hex}.pt")

    consumer_proc: Optional[mp.Process] = None
    producer_proc: Optional[mp.Process] = None
    producer_pid: Optional[int] = None
    consumer_pid: Optional[int] = None
    cleanup_report: Dict[str, Any] = {}

    try:
        consumer_proc = ctx.Process(
            target=_decode_consumer_worker_proc,
            args=(
                port_queue,
                consumer_result_queue,
                model_id,
                decode_steps,
                decode_device,
                timeout,
                temp_logits_path,
            ),
        )
        consumer_proc.start()

        ready_info = port_queue.get(timeout=timeout + 30)
        assigned_port = ready_info["port"]
        consumer_pid = ready_info["pid"]

        producer_proc = ctx.Process(
            target=_prefill_producer_worker_proc,
            args=(
                assigned_port,
                producer_result_queue,
                model_id,
                prompt,
                2,  # logical prefill_tp inside single-GPU producer
                1,  # decode_tp
                prefill_device,
                enable_fp8,
                block_size_tokens,
                timeout,
                bandwidth_mbps,
            ),
        )
        producer_proc.start()

        producer_res = producer_result_queue.get(timeout=timeout + 30)
        consumer_res = consumer_result_queue.get(timeout=timeout + 30)

        producer_proc.join(timeout=10)
        consumer_proc.join(timeout=10)

        if producer_res["status"] != "success":
            raise RuntimeError(f"Producer worker failed: {producer_res}")
        if consumer_res["status"] != "success":
            raise RuntimeError(f"Consumer worker failed: {consumer_res}")

        producer_pid = producer_res["producer_pid"]
        consumer_logits = torch.load(temp_logits_path, weights_only=True)
    finally:
        cleanup_report = safe_cleanup_processes(
            processes=[
                (producer_proc, producer_pid if producer_pid is not None else (producer_proc.pid if producer_proc else None)),
                (consumer_proc, consumer_pid if consumer_pid is not None else (consumer_proc.pid if consumer_proc else None)),
            ],
            temp_files=[temp_logits_path],
            queues=[port_queue, consumer_result_queue, producer_result_queue],
        )

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
        "producer": producer_res,
        "consumer": consumer_res,
        "consumer_logits": consumer_logits,
        "cleanup_report": cleanup_report,
        "timing": {
            "time_to_first_token_ready_ms": producer_res.get("time_to_first_token_ready_ms", 0.0),
            "sender_wire_duration_ms": producer_res.get("sender_wire_duration_ms", 0.0),
            "receiver_reconstruction_duration_ms": consumer_res.get("receiver_reconstruction_duration_ms", 0.0),
            "transfer_and_reconstruction_latency_ms": transfer_and_reconstruction_latency_ms,
            "true_inter_token_handoff_gap_ms": true_inter_token_handoff_gap_ms,
            "first_decode_step_latency_ms": consumer_res.get("first_decode_step_latency_ms", 0.0),
            "median_subsequent_latency_ms": consumer_res.get("median_subsequent_latency_ms", 0.0),
            "raw_event_timestamps": {
                "t_prefill_start": producer_res.get("t_prefill_start", 0.0),
                "t_producer_token_ready": t_prod_tok,
                "t_sender_wire_start": t_send_start,
                "t_sender_wire_end": producer_res.get("t_sender_wire_end", 0.0),
                "t_receiver_wire_start": consumer_res.get("t_receiver_wire_start", 0.0),
                "t_consumer_cache_ready": t_cons_cache,
                "t_consumer_token_ready": t_cons_tok,
            },
        },
    }


def compute_series_statistics(values: List[float]) -> Dict[str, float]:
    """Computes median, mean, standard deviation, min, and max for a metric series."""
    if not values:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    med = statistics.median(values)
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "median": round(med, 3),
        "mean": round(avg, 3),
        "std": round(std, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def run_baseline_continuation(
    model_id: str,
    prompt: str,
    decode_steps: int,
    device_str: str,
) -> Tuple[torch.Tensor, List[int], str]:
    """Runs uninterrupted baseline generation on target decode device for ground-truth comparison."""
    torch_device = torch.device(device_str)
    target_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=target_dtype)
    model.to(torch_device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(torch_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        baseline_cache = out.past_key_values
        init_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    curr_tok = init_token
    curr_cache = baseline_cache
    tokens: List[int] = []
    logits_list: List[torch.Tensor] = []

    for _ in range(decode_steps):
        with torch.no_grad():
            out_dec = model(input_ids=curr_tok, past_key_values=curr_cache, use_cache=True)
            logits = out_dec.logits[:, -1, :]
            curr_cache = out_dec.past_key_values
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            curr_tok = next_tok
            tokens.append(int(next_tok.item()))
            logits_list.append(logits[0].cpu())

    stacked_logits = torch.stack(logits_list, dim=0)
    text = tokenizer.decode(tokens)
    return stacked_logits, tokens, text


def run_step3_controlled_study(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    seq_lens: Optional[List[int]] = None,
    repetitions: int = 3,
    decode_steps: int = 2,
    prefill_device: Optional[str] = None,
    decode_device: Optional[str] = None,
    bandwidth_mbps: Optional[float] = None,
    output_path: str = "benchmarks/step3_controlled_transfer_study.json",
    warmup_runs: int = 1,
) -> Dict[str, Any]:
    """
    Executes the Step 3 Controlled Transfer Study:
    Sweeps cache sizes, warms up, measures individual runs, evaluates AdaptiveCompressionPolicy,
    and returns comprehensive results dictionary.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512, 1024]

    has_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if has_cuda else 0

    if prefill_device is None or decode_device is None:
        if has_cuda and cuda_count >= 2:
            p_dev = "cuda:0"
            d_dev = "cuda:1"
        elif has_cuda and cuda_count == 1:
            p_dev = "cuda:0"
            d_dev = "cuda:0"
        else:
            p_dev = "cpu"
            d_dev = "cpu"
    else:
        p_dev = prefill_device
        d_dev = decode_device

    p_dev = resolve_canonical_device_str(p_dev)
    d_dev = resolve_canonical_device_str(d_dev)

    hardware_info = inspect_two_gpu_system(p_dev, d_dev)
    git_meta = get_git_metadata()

    # Effective bandwidth for policy evaluation:
    # If bandwidth_mbps is specified, use exact value in bytes/sec.
    # If unrestricted, use estimated loopback throughput (~10.0 GB/s = 80 Gbps).
    if bandwidth_mbps is not None and bandwidth_mbps > 0:
        effective_policy_bandwidth_bps = (bandwidth_mbps * 1e6) / 8.0
        bandwidth_condition = f"emulated_rate_limit_{bandwidth_mbps:.0f}mbps"
    else:
        effective_policy_bandwidth_bps = 10.0 * 1024 * 1024 * 1024  # 10 GB/s loopback baseline
        bandwidth_condition = "unrestricted_loopback_tcp"

    policy = AdaptiveCompressionPolicy()

    # Pre-load tokenizer for prompt generation
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Cache model config dimensions
    dummy_model = AutoModelForCausalLM.from_pretrained(model_id)
    cfg = dummy_model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 24))
    num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 2))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    checkpoint_commit_hash = getattr(cfg, "_commit_hash", None)
    del dummy_model

    sweep_results: List[Dict[str, Any]] = []

    console.print(f"[bold cyan]Starting Step 3 Sweep across {len(seq_lens)} sequence lengths: {seq_lens}[/bold cyan]")
    console.print(f"[cyan]Repetitions per condition:[/cyan] {repetitions} (Warmup runs: {warmup_runs})")
    console.print(f"[cyan]Link Condition:[/cyan] {bandwidth_condition} (Pacing: {bandwidth_mbps} Mbps)")

    for target_seq_len in seq_lens:
        console.print(f"\n[bold yellow]── Evaluating Sequence Length S = {target_seq_len} tokens ──[/bold yellow]")
        prompt_text = generate_prompt_for_seq_len(tokenizer, target_seq_len)
        actual_token_count = len(tokenizer.encode(prompt_text))

        # Ground-truth baseline for logit & token equivalence
        base_logits, base_tokens, base_text = run_baseline_continuation(
            model_id=model_id,
            prompt=prompt_text,
            decode_steps=decode_steps,
            device_str=d_dev,
        )

        # 1. Warm-up runs (unrecorded)
        for _ in range(warmup_runs):
            execute_handoff_iteration(
                model_id=model_id,
                prompt=prompt_text,
                decode_steps=decode_steps,
                prefill_device=p_dev,
                decode_device=d_dev,
                enable_fp8=False,
                bandwidth_mbps=bandwidth_mbps,
            )
            execute_handoff_iteration(
                model_id=model_id,
                prompt=prompt_text,
                decode_steps=decode_steps,
                prefill_device=p_dev,
                decode_device=d_dev,
                enable_fp8=True,
                bandwidth_mbps=bandwidth_mbps,
            )

        # 2. Mode A: Uncompressed FP16
        uncomp_runs: List[Dict[str, Any]] = []
        for rep in range(repetitions):
            res = execute_handoff_iteration(
                model_id=model_id,
                prompt=prompt_text,
                decode_steps=decode_steps,
                prefill_device=p_dev,
                decode_device=d_dev,
                enable_fp8=False,
                bandwidth_mbps=bandwidth_mbps,
            )
            c_logits = res["consumer_logits"]
            diff = torch.abs(base_logits.float() - c_logits.float())
            max_err = float(diff.max().item())
            cos_sim = float(torch.nn.functional.cosine_similarity(base_logits.float().flatten(), c_logits.float().flatten(), dim=0).item())
            tok_match = bool(res["consumer"]["generated_tokens"] == base_tokens)

            run_entry = {
                "iteration": rep + 1,
                "timing": res["timing"],
                "wire_accounting": res["consumer"]["wire_accounting"],
                "encoding_used": res["producer"]["encoding_used"],
                "tokens_match": tok_match,
                "max_abs_logit_error": max_err,
                "cosine_similarity": cos_sim,
                "cleanup_verified": validate_cleanup_report(res["cleanup_report"]),
            }
            uncomp_runs.append(run_entry)

        # 3. Mode B: Always-On FP8
        fp8_runs: List[Dict[str, Any]] = []
        for rep in range(repetitions):
            res = execute_handoff_iteration(
                model_id=model_id,
                prompt=prompt_text,
                decode_steps=decode_steps,
                prefill_device=p_dev,
                decode_device=d_dev,
                enable_fp8=True,
                bandwidth_mbps=bandwidth_mbps,
            )
            c_logits = res["consumer_logits"]
            diff = torch.abs(base_logits.float() - c_logits.float())
            max_err = float(diff.max().item())
            cos_sim = float(torch.nn.functional.cosine_similarity(base_logits.float().flatten(), c_logits.float().flatten(), dim=0).item())
            tok_match = bool(res["consumer"]["generated_tokens"] == base_tokens)

            run_entry = {
                "iteration": rep + 1,
                "timing": res["timing"],
                "wire_accounting": res["consumer"]["wire_accounting"],
                "encoding_used": res["producer"]["encoding_used"],
                "is_emulated_fallback": res["producer"]["is_emulated_fallback"],
                "tokens_match": tok_match,
                "max_abs_logit_error": max_err,
                "cosine_similarity": cos_sim,
                "cleanup_verified": validate_cleanup_report(res["cleanup_report"]),
            }
            fp8_runs.append(run_entry)

        # 4. Mode C: Adaptive Policy Evaluation & Selection
        policy_eval: PolicyEvaluationResult = policy.evaluate(
            seq_len=target_seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            link_bandwidth_bytes_per_sec=effective_policy_bandwidth_bps,
            block_size_tokens=16,
            prefill_tp=2,
        )

        selected_mode_str = policy_eval.selected_mode.value
        policy_chose_fp8 = bool(policy_eval.selected_mode == CompressionMode.BLOCK_FP8)

        # Run repetitions under policy's selected mode
        adaptive_runs: List[Dict[str, Any]] = []
        for rep in range(repetitions):
            res = execute_handoff_iteration(
                model_id=model_id,
                prompt=prompt_text,
                decode_steps=decode_steps,
                prefill_device=p_dev,
                decode_device=d_dev,
                enable_fp8=policy_chose_fp8,
                bandwidth_mbps=bandwidth_mbps,
            )
            c_logits = res["consumer_logits"]
            diff = torch.abs(base_logits.float() - c_logits.float())
            max_err = float(diff.max().item())
            cos_sim = float(torch.nn.functional.cosine_similarity(base_logits.float().flatten(), c_logits.float().flatten(), dim=0).item())
            tok_match = bool(res["consumer"]["generated_tokens"] == base_tokens)

            run_entry = {
                "iteration": rep + 1,
                "timing": res["timing"],
                "wire_accounting": res["consumer"]["wire_accounting"],
                "encoding_used": res["producer"]["encoding_used"],
                "tokens_match": tok_match,
                "max_abs_logit_error": max_err,
                "cosine_similarity": cos_sim,
                "cleanup_verified": validate_cleanup_report(res["cleanup_report"]),
            }
            adaptive_runs.append(run_entry)

        # Summary statistics per mode
        uncomp_latencies = [r["timing"]["transfer_and_reconstruction_latency_ms"] for r in uncomp_runs]
        fp8_latencies = [r["timing"]["transfer_and_reconstruction_latency_ms"] for r in fp8_runs]
        adaptive_latencies = [r["timing"]["transfer_and_reconstruction_latency_ms"] for r in adaptive_runs]

        uncomp_gaps = [r["timing"]["true_inter_token_handoff_gap_ms"] for r in uncomp_runs]
        fp8_gaps = [r["timing"]["true_inter_token_handoff_gap_ms"] for r in fp8_runs]
        adaptive_gaps = [r["timing"]["true_inter_token_handoff_gap_ms"] for r in adaptive_runs]

        uncomp_stats = compute_series_statistics(uncomp_latencies)
        fp8_stats = compute_series_statistics(fp8_latencies)
        adaptive_stats = compute_series_statistics(adaptive_latencies)

        uncomp_gap_stats = compute_series_statistics(uncomp_gaps)
        fp8_gap_stats = compute_series_statistics(fp8_gaps)
        adaptive_gap_stats = compute_series_statistics(adaptive_gaps)

        # Empirical Best Determination: which mode achieved lower median complete handoff latency?
        empirical_best_mode = "block_fp8" if fp8_stats["median"] < uncomp_stats["median"] else "uncompressed_fp16"
        policy_made_optimal_choice = bool(selected_mode_str == empirical_best_mode)

        # Tradeoff analysis: did FP8 reduce latency?
        fp8_latency_diff_ms = fp8_stats["median"] - uncomp_stats["median"]
        fp8_speedup_factor = round(uncomp_stats["median"] / max(0.001, fp8_stats["median"]), 3)
        fp8_reduced_latency = bool(fp8_stats["median"] < uncomp_stats["median"])

        uncomp_wire_bytes = uncomp_runs[0]["wire_accounting"]["total_wire_bytes"]
        fp8_wire_bytes = fp8_runs[0]["wire_accounting"]["total_wire_bytes"]
        wire_ratio = round(uncomp_wire_bytes / max(1, fp8_wire_bytes), 3)

        condition_summary = {
            "target_seq_len": target_seq_len,
            "measured_prompt_tokens": actual_token_count,
            "bandwidth_condition": bandwidth_condition,
            "bandwidth_mbps": bandwidth_mbps,
            "uncompressed_fp16": {
                "individual_runs": uncomp_runs,
                "transfer_latency_stats_ms": uncomp_stats,
                "handoff_gap_stats_ms": uncomp_gap_stats,
                "wire_bytes": uncomp_wire_bytes,
                "tokens_match_all_runs": all(r["tokens_match"] for r in uncomp_runs),
            },
            "always_fp8": {
                "individual_runs": fp8_runs,
                "transfer_latency_stats_ms": fp8_stats,
                "handoff_gap_stats_ms": fp8_gap_stats,
                "wire_bytes": fp8_wire_bytes,
                "wire_compression_ratio": wire_ratio,
                "tokens_match_all_runs": all(r["tokens_match"] for r in fp8_runs),
            },
            "adaptive_selection": {
                "policy_evaluation": asdict(policy_eval),
                "selected_mode": selected_mode_str,
                "individual_runs": adaptive_runs,
                "transfer_latency_stats_ms": adaptive_stats,
                "handoff_gap_stats_ms": adaptive_gap_stats,
            },
            "empirical_tradeoff": {
                "fp8_reduced_latency": fp8_reduced_latency,
                "fp8_latency_diff_ms": round(fp8_latency_diff_ms, 3),
                "fp8_speedup_factor": fp8_speedup_factor,
                "empirical_best_mode": empirical_best_mode,
                "policy_made_optimal_choice": policy_made_optimal_choice,
            },
        }
        sweep_results.append(condition_summary)

        console.print(
            f"  [cyan]Uncompressed Median:[/cyan] {uncomp_stats['median']:.2f} ms ({uncomp_wire_bytes:,} B) | "
            f"[magenta]FP8 Median:[/magenta] {fp8_stats['median']:.2f} ms ({fp8_wire_bytes:,} B, {wire_ratio:.2f}x wire) | "
            f"[yellow]Policy Selected:[/yellow] {selected_mode_str} (Optimal: {policy_made_optimal_choice})"
        )

    # Core Research Question Answer synthesis
    fp8_won_any_config = any(s["empirical_tradeoff"]["fp8_reduced_latency"] for s in sweep_results)
    policy_correct_all_configs = all(s["empirical_tradeoff"]["policy_made_optimal_choice"] for s in sweep_results)

    overall_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_title": "Step 3: Controlled Transfer Study — FP16 vs Always-On FP8 vs Adaptive Policy",
        "core_research_question": (
            "Does GPU compression reduce complete handoff latency for any tested configuration, "
            "and does the policy choose appropriately?"
        ),
        "empirical_answer": {
            "compression_reduced_complete_handoff_latency": fp8_won_any_config,
            "policy_chose_appropriately_across_all_configs": policy_correct_all_configs,
            "summary_verdict": (
                "Compression reduced complete handoff latency in at least one tested configuration."
                if fp8_won_any_config
                else "Uncompressed FP16 outperformed FP8 across all tested configurations on this interconnect; "
                     "FP8 wire savings did not overcome codec and formatting overhead."
            ),
        },
        "cli_invocation": sys.argv,
        "git_metadata": git_meta,
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_toolkit_version": getattr(torch.version, "cuda", None),
            "os": platform.system(),
            "arch": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "hardware_telemetry": hardware_info,
        "model_metadata": {
            "model_id": model_id,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "checkpoint_commit_hash": checkpoint_commit_hash,
        },
        "study_configuration": {
            "swept_sequence_lengths": seq_lens,
            "repetitions_per_condition": repetitions,
            "warmup_runs": warmup_runs,
            "decode_steps": decode_steps,
            "prefill_device": p_dev,
            "decode_device": d_dev,
            "bandwidth_condition": bandwidth_condition,
            "bandwidth_mbps": bandwidth_mbps,
            "effective_policy_bandwidth_bps": effective_policy_bandwidth_bps,
        },
        "sweep_results": sweep_results,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    return overall_results


@click.command()
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID.")
@click.option("--seq-lens", default="64,128,256,512,1024", help="Comma-separated sequence lengths to sweep.")
@click.option("--reps", "-r", default=3, help="Timed repetitions per condition.")
@click.option("--warmup", default=1, help="Warmup iterations before timing.")
@click.option("--steps", "-s", default=2, help="Decode continuation steps per run.")
@click.option("--prefill-device", default=None, help="Device for GPU 0 prefill (defaults to cuda:0 if available).")
@click.option("--decode-device", default=None, help="Device for GPU 1 decode (defaults to cuda:1 if available).")
@click.option("--bandwidth-mbps", default=None, type=float, help="Calibrated link bandwidth restriction in Mbps (None = unrestricted).")
@click.option("--output", "-o", default="benchmarks/step3_controlled_transfer_study.json", help="Output JSON path.")
@click.option("--allow-emulation/--require-physical-two-gpu", default=False, help="Allow emulation mode without failing CLI exit.")
def main(
    model: str,
    seq_lens: str,
    reps: int,
    warmup: int,
    steps: int,
    prefill_device: Optional[str],
    decode_device: Optional[str],
    bandwidth_mbps: Optional[float],
    output: str,
    allow_emulation: bool,
):
    """Executes Step 3: Controlled Transfer Study."""
    parsed_seq_lens = [int(s.strip()) for s in seq_lens.split(",") if s.strip()]

    console.print(
        Panel(
            f"[bold cyan]Step 3: Controlled Transfer Study — FP16 vs Always-On FP8 vs Adaptive Policy[/bold cyan]\n"
            f"Model: {model} | Cache Sizes: {parsed_seq_lens}\n"
            f"Bandwidth: {'Unrestricted Host-Staged TCP' if bandwidth_mbps is None else f'{bandwidth_mbps:.1f} Mbps (Calibrated Pacing)'}\n"
            f"Repetitions: {reps} (Warmup: {warmup}) | Decode Steps: {steps}",
            border_style="cyan",
        )
    )

    results = run_step3_controlled_study(
        model_id=model,
        seq_lens=parsed_seq_lens,
        repetitions=reps,
        decode_steps=steps,
        prefill_device=prefill_device,
        decode_device=decode_device,
        bandwidth_mbps=bandwidth_mbps,
        output_path=output,
        warmup_runs=warmup,
    )

    hardware_info = results["hardware_telemetry"]
    is_two_physical_cuda = hardware_info.get("is_two_physical_cuda_gpus", False)

    # Present Summary Table
    table = Table(title="Step 3: Controlled Transfer Study Results Summary", header_style="bold magenta")
    table.add_column("Seq Len (Tokens)", justify="center", style="cyan")
    table.add_column("Uncompressed FP16", justify="center")
    table.add_column("Always-On FP8", justify="center")
    table.add_column("Wire Ratio", justify="center")
    table.add_column("Latency Diff", justify="center")
    table.add_column("Policy Choice", justify="center")
    table.add_column("Empirical Winner", justify="center")
    table.add_column("Policy Optimal?", justify="center")

    for s in results["sweep_results"]:
        s_len = s["target_seq_len"]
        u_med = s["uncompressed_fp16"]["transfer_latency_stats_ms"]["median"]
        f_med = s["always_fp8"]["transfer_latency_stats_ms"]["median"]
        ratio = s["always_fp8"]["wire_compression_ratio"]
        diff = s["empirical_tradeoff"]["fp8_latency_diff_ms"]
        diff_str = f"+{diff:.2f} ms" if diff > 0 else f"{diff:.2f} ms"
        policy_sel = s["adaptive_selection"]["selected_mode"]
        emp_win = s["empirical_tradeoff"]["empirical_best_mode"]
        optimal = s["empirical_tradeoff"]["policy_made_optimal_choice"]

        table.add_row(
            str(s_len),
            f"{u_med:.2f} ms",
            f"{f_med:.2f} ms",
            f"{ratio:.2f}x",
            f"[red]{diff_str}[/red]" if diff > 0 else f"[green]{diff_str}[/green]",
            f"[bold cyan]{policy_sel}[/bold cyan]",
            f"[bold]{emp_win}[/bold]",
            "[bold green]YES[/bold green]" if optimal else "[bold red]NO[/bold red]",
        )

    console.print(table)

    ans = results["empirical_answer"]
    summary_panel = Panel(
        f"[bold]Core Research Question:[/bold] {results['core_research_question']}\n\n"
        f"  [bold]Compression Reduced Latency:[/bold] {'[bold green]YES[/bold green]' if ans['compression_reduced_complete_handoff_latency'] else '[bold yellow]NO (FP16 was faster)[/bold yellow]'}\n"
        f"  [bold]Policy Accuracy:[/bold] {'[bold green]100% OPTIMAL CHOICES[/bold green]' if ans['policy_chose_appropriately_across_all_configs'] else '[bold yellow]Sub-optimal in some configs[/bold yellow]'}\n"
        f"  [bold]Verdict:[/bold] {ans['summary_verdict']}\n"
        f"  [bold]Hardware Interconnect:[/bold] {hardware_info.get('transport_architecture', 'Localhost TCP')}\n"
        f"  [bold]Two Physical CUDA GPUs:[/bold] {is_two_physical_cuda}",
        title="Step 3 Research Findings Summary",
        border_style="green" if is_two_physical_cuda else "yellow",
    )
    console.print(summary_panel)
    console.print(f"[dim]Artifact exported to {output}[/dim]")

    if not is_two_physical_cuda and not allow_emulation:
        console.print(Panel(
            "[bold red]Step 3 Physical Benchmark Gate Failed.[/bold red]\n"
            "Execution requires two distinct physical CUDA GPUs unless --allow-emulation is passed.",
            border_style="red",
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
