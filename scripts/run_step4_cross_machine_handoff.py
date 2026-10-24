#!/usr/bin/env python3
"""HeteroDisagg Step 4: Decoupled Cross-Machine KV Cache Handoff CLI.

Enables physical cross-machine execution between independent nodes (e.g. separate
cloud rental instances or physical GPUs across distinct hosts) communicating over TCP.

Core Architectural Guarantees:
1. Consumer-Local Baseline: Uninterrupted baseline runs on the consumer device
   itself, guaranteeing identical checkpoint revision, dtype, attention implementation,
   and hardware kernel arithmetic.
2. Aligned Comparisons: For FP8 (and uncompressed), force-feeds ground-truth baseline
   tokens to measure aligned logit fidelity, while independently reporting free-running
   greedy token agreement.
3. Cross-Machine Timing Isolation: Monotonic timestamps from separate hosts are NEVER
   subtracted. Host-local intervals are reported separately, and complete transfer
   duration is captured strictly via producer-side round-trip ACKs.
4. Explicit Decoupled Networking: Consumer bind address (--bind-host, --bind-port) is
   strictly separated from producer destination address (--dest-host, --dest-port).
5. Fail-Closed Validation: Checks model ID, checkpoint commit hash, layer counts, KV head
   counts, dtypes, and shapes before accepting frames. Non-matching checkpoints, missing
   metrics, non-finite values, or divergence immediately exit with code 1.
6. Bounded Deadlines: All socket operations enforce strict timeouts, cleanly reaping
   resources and sockets on timeout or abort.
7. Complete Provenance: Captures git commit hash, dirty status, software versions,
   hardware telemetry, exact token IDs, wire byte accounting, and discrete check outcomes.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import math
import uuid
import socket
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import click
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import slice_kv_for_prefill_rank
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
)
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
    MAGIC_MANIFEST,
    MAGIC_FRAME,
    MAX_METADATA_BYTES,
    TransportError,
    MismatchedRequestError,
    ManifestValidationError,
    FrameValidationError,
    IncompleteTransferError,
    CorruptedFrameError,
)

# Custom Handshake Framing
MAGIC_ACK_MANIFEST = b"ACKM"
MAGIC_REJ_MANIFEST = b"REJM"
MAGIC_ACK_COMPLETED = b"ACKD"


def send_handshake_msg(sock: socket.socket, magic: bytes, payload_dict: Dict[str, Any]) -> None:
    """Sends a 4-byte magic + 4-byte big-endian length + JSON payload handshake packet."""
    json_bytes = json.dumps(payload_dict).encode("utf-8")
    header = magic + len(json_bytes).to_bytes(4, byteorder="big")
    sock.sendall(header + json_bytes)


def recv_handshake_msg(sock: socket.socket, expected_magic: Optional[bytes] = None, timeout: float = DEFAULT_TIMEOUT) -> Tuple[bytes, Dict[str, Any]]:
    """Receives a 4-byte magic + 4-byte length + JSON payload packet with bounded timeout."""
    sock.settimeout(timeout)
    header = sock.recv(8)
    if len(header) < 8:
        raise IncompleteTransferError(f"Expected 8-byte handshake header, received {len(header)} bytes")
    magic = header[:4]
    if expected_magic is not None and magic != expected_magic:
        raise CorruptedFrameError(f"Handshake magic mismatch: expected {expected_magic}, got {magic}")
    length = int.from_bytes(header[4:8], byteorder="big")
    if length > MAX_METADATA_BYTES:
        raise CorruptedFrameError(f"Handshake payload length {length} exceeds maximum {MAX_METADATA_BYTES}")
    
    received = 0
    chunks = []
    while received < length:
        chunk = sock.recv(min(length - received, 4096))
        if not chunk:
            raise IncompleteTransferError(f"Connection closed during handshake body read: got {received}/{length}")
        chunks.append(chunk)
        received += len(chunk)
    payload_data = json.loads(b"".join(chunks).decode("utf-8"))
    return magic, payload_data


def get_git_metadata() -> Dict[str, Any]:
    """Retrieves current git repository revision and working tree dirty status."""
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
        commit_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_dirty": len(status_out) > 0,
            "commit_date": commit_date,
        }
    except Exception as e:
        return {
            "commit_hash": "unknown",
            "branch": "unknown",
            "is_dirty": False,
            "error": str(e),
        }


def get_hardware_telemetry(device_str: str) -> Dict[str, Any]:
    """Inspects environment, device architecture, and compute capabilities."""
    has_cuda = torch.cuda.is_available()
    dev = torch.device(device_str)
    
    device_info: Dict[str, Any] = {
        "device_str": device_str,
        "device_type": dev.type,
        "device_index": dev.index,
    }
    if dev.type == "cuda" and has_cuda:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        device_info.update({
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multi_processor_count": props.multi_processor_count,
        })
    elif dev.type == "cpu":
        device_info.update({
            "name": platform.processor() or "CPU",
            "cpu_count": os.cpu_count(),
        })

    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_toolkit_version": getattr(torch.version, "cuda", None),
        "cuda_available": has_cuda,
        "cuda_device_count": torch.cuda.device_count() if has_cuda else 0,
        "device": device_info,
    }


def resolve_torch_dtype(dtype_str: str, device_type: str) -> torch.dtype:
    """Maps string representation to torch dtype with device compatibility."""
    if dtype_str == "float16":
        return torch.float16 if device_type == "cuda" else torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16 if device_type == "cuda" else torch.float32
    elif dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


@click.group()
def cli():
    """HeteroDisagg Step 4: Cross-Machine Heterogeneous KV Cache Handoff Suite."""
    pass


# ==============================================================================
# PRODUCER ROLE
# ==============================================================================

@cli.command("producer")
@click.option("--dest-host", required=True, help="Destination IP or hostname of consumer.")
@click.option("--dest-port", type=int, required=True, help="Destination port of consumer.")
@click.option("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--prompt", default="Heterogeneous disaggregated inference allows", help="Input prompt text.")
@click.option("--device", default="cpu", help="Compute device for producer (e.g. cuda:0 or cpu).")
@click.option("--dtype", default="float32", type=click.Choice(["float16", "bfloat16", "float32"]), help="Base computation dtype.")
@click.option("--enable-fp8", is_flag=True, default=False, help="Enable in-flight FP8 wire codec.")
@click.option("--prefill-tp", type=int, default=1, help="Prefill tensor parallelism degree.")
@click.option("--decode-tp", type=int, default=1, help="Decode tensor parallelism degree.")
@click.option("--block-size", type=int, default=16, help="FP8 quantization block size in tokens.")
@click.option("--timeout", type=float, default=20.0, help="Socket and transfer timeout in seconds.")
@click.option("--output", default="benchmarks/step4_producer_evidence.json", help="Path to write producer JSON report.")
# Failure Injection Probes
@click.option("--inject-checkpoint-mismatch", is_flag=True, default=False, help="Inject corrupted checkpoint hash for fail-closed testing.")
@click.option("--inject-dtype-mismatch", is_flag=True, default=False, help="Inject invalid dtype for fail-closed testing.")
@click.option("--inject-truncation", is_flag=True, default=False, help="Disconnect mid-transfer after sending 1 frame.")
def run_producer(
    dest_host: str,
    dest_port: int,
    model: str,
    prompt: str,
    device: str,
    dtype: str,
    enable_fp8: bool,
    prefill_tp: int,
    decode_tp: int,
    block_size: int,
    timeout: float,
    output: str,
    inject_checkpoint_mismatch: bool,
    inject_dtype_mismatch: bool,
    inject_truncation: bool,
):
    """Executes the Producer node: prefills prompt, serializes KV cache, and transfers across TCP."""
    click.echo(f"=== [PRODUCER] Starting Execution ===")
    click.echo(f"Target: {dest_host}:{dest_port} | Device: {device} | Dtype: {dtype} | FP8: {enable_fp8}")

    git_meta = get_git_metadata()
    hw_telemetry = get_hardware_telemetry(device)
    torch_device = torch.device(device)
    torch_dtype = resolve_torch_dtype(dtype, torch_device.type)

    request_id = f"step4-{uuid.uuid4().hex[:12]}"
    report: Dict[str, Any] = {
        "role": "producer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": request_id,
        "model_id": model,
        "destination": {"host": dest_host, "port": dest_port},
        "git_provenance": git_meta,
        "hardware_telemetry": hw_telemetry,
        "configuration": {
            "prompt": prompt,
            "device": device,
            "dtype": dtype,
            "enable_fp8": enable_fp8,
            "prefill_tp": prefill_tp,
            "decode_tp": decode_tp,
            "block_size": block_size,
            "timeout_seconds": timeout,
        },
        "success": False,
        "error": None,
    }

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.settimeout(timeout)

    try:
        # 1. Load Model & Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        causal_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype).to(torch_device)
        causal_model.eval()

        resolved_commit_hash = getattr(causal_model.config, "_commit_hash", None)
        report["resolved_checkpoint_commit_hash"] = resolved_commit_hash

        # 2. Run Prefill Forward Pass
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(torch_device)
        prompt_tokens = input_ids[0].tolist()
        seq_len = input_ids.shape[1]

        report["prompt_tokens"] = prompt_tokens
        report["prompt_token_count"] = len(prompt_tokens)
        report["cached_seq_len"] = seq_len

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            out = causal_model(input_ids=input_ids, use_cache=True)
            prefill_logits = out.logits[:, -1, :]
            initial_token_id = int(torch.argmax(prefill_logits, dim=-1).item())
            initial_token_str = tokenizer.decode([initial_token_id])
            prefill_cache = out.past_key_values

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_prefill_end = time.perf_counter()
        prefill_duration_ms = (t_prefill_end - t_prefill_start) * 1000.0

        report["initial_token_id"] = initial_token_id
        report["initial_token_str"] = initial_token_str
        report["timings_producer_local_ms"] = {
            "prefill_duration_ms": prefill_duration_ms,
        }

        # 3. Connect to Consumer with Deadline
        click.echo(f"[PRODUCER] Connecting to consumer at {dest_host}:{dest_port}...")
        connect_deadline = time.monotonic() + timeout
        connected = False
        while not connected and time.monotonic() < connect_deadline:
            try:
                client_sock.connect((dest_host, dest_port))
                connected = True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.1)

        if not connected:
            raise ConnectionRefusedError(f"Failed to connect to consumer {dest_host}:{dest_port} within {timeout}s deadline")

        click.echo("[PRODUCER] Connected. Sending RequestManifest...")

        # 4. Formulate RequestManifest
        num_layers = causal_model.config.num_hidden_layers
        num_kv_heads = getattr(causal_model.config, "num_key_value_heads", causal_model.config.num_attention_heads)
        head_dim = causal_model.config.hidden_size // causal_model.config.num_attention_heads

        sent_commit_hash = "corrupted_fake_commit_hash" if inject_checkpoint_mismatch else resolved_commit_hash
        sent_dtype = "invalid_dtype_99" if inject_dtype_mismatch else ("float16" if torch_device.type == "cuda" else "float32")

        manifest = RequestManifest(
            request_id=request_id,
            model_id=model,
            checkpoint_commit_hash=sent_commit_hash,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            prefill_tp=prefill_tp,
            decode_tp=decode_tp,
            expected_frames=num_layers * 2 * prefill_tp,
            initial_decode_token=initial_token_id,
            initial_decode_token_str=initial_token_str,
            prompt_tokens=prompt_tokens,
            dtype=sent_dtype,
            enable_fp8=enable_fp8,
            block_size_tokens=block_size,
            decode_rank=0,
        )

        t_send_start = time.perf_counter()
        manifest_accounting = send_manifest(client_sock, manifest)

        # 5. Await Consumer Handshake Manifest ACK
        click.echo("[PRODUCER] Awaiting consumer manifest handshake...")
        magic, ack_data = recv_handshake_msg(client_sock, timeout=timeout)
        if magic == MAGIC_REJ_MANIFEST:
            raise ManifestValidationError(f"Consumer rejected manifest: {ack_data.get('reason')}")
        if magic != MAGIC_ACK_MANIFEST:
            raise CorruptedFrameError(f"Unexpected handshake response magic: {magic}")
        click.echo(f"[PRODUCER] Handshake accepted: {ack_data}")

        # 6. Slice, Quantize, and Stream Cache Frames
        click.echo("[PRODUCER] Streaming KV cache frames...")
        plan = KVPartitionPlan.create(num_kv_heads=num_kv_heads, prefill_tp=prefill_tp, decode_tp=decode_tp)
        total_accounting = manifest_accounting
        frames_sent = 0

        t_codec_start = time.perf_counter()
        for l in range(num_layers):
            if hasattr(prefill_cache, "layers") and len(prefill_cache.layers) > l:
                layer = prefill_cache.layers[l]
                orig_k = getattr(layer, "keys", getattr(layer, "key_states", None))
                orig_v = getattr(layer, "values", getattr(layer, "value_states", None))
            elif hasattr(prefill_cache, "key_cache") and len(prefill_cache.key_cache) > l:
                orig_k = prefill_cache.key_cache[l]
                orig_v = prefill_cache.value_cache[l]
            elif isinstance(prefill_cache, (list, tuple)) and len(prefill_cache) > l:
                orig_k = prefill_cache[l][0]
                orig_v = prefill_cache[l][1]
            else:
                raise ValueError(f"Unable to extract KV tensors from cache of type {type(prefill_cache)}")

            k_hetero = orig_k.permute(0, 2, 1, 3).contiguous()
            v_hetero = orig_v.permute(0, 2, 1, 3).contiguous()

            for kv_type, tensor_hetero in [("key", k_hetero), ("value", v_hetero)]:
                for p in range(prefill_tp):
                    shard = slice_kv_for_prefill_rank(tensor_hetero, prefill_rank=p, plan=plan)
                    h_slice = plan.prefill_head_slices[p]

                    if enable_fp8:
                        compressed = quantize_kv_block_fp8(shard, block_size_tokens=block_size)
                        is_f8 = compressed.quantized_data.dtype == getattr(torch, "float8_e4m3fn", None)
                        dtype_str = "float8_e4m3fn" if is_f8 else "int8"
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

                    frame_acc = send_frame(client_sock, meta, body)
                    total_accounting = total_accounting.add(frame_acc)
                    frames_sent += 1

                    if inject_truncation and frames_sent >= 1:
                        click.echo("[PRODUCER] Injected truncation: closing socket prematurely!")
                        client_sock.close()
                        report["error"] = "Injected truncation simulated"
                        _write_json(output, report)
                        sys.exit(1)

        t_stream_end = time.perf_counter()
        wire_send_ms = (t_stream_end - t_send_start) * 1000.0

        # 7. Await Final Completion ACK from Consumer
        click.echo("[PRODUCER] Awaiting consumer completion ACK...")
        magic_done, done_data = recv_handshake_msg(client_sock, expected_magic=MAGIC_ACK_COMPLETED, timeout=timeout)
        t_ack_recv = time.perf_counter()
        total_roundtrip_ms = (t_ack_recv - t_send_start) * 1000.0

        click.echo(f"[PRODUCER] Transfer completed successfully! Consumer reported: {done_data}")

        report["wire_accounting"] = {
            "framing_bytes": total_accounting.framing_bytes,
            "metadata_bytes": total_accounting.metadata_bytes,
            "body_bytes": total_accounting.body_bytes,
            "total_wire_bytes": total_accounting.total_wire_bytes,
            "frames_sent": frames_sent,
        }
        report["timings_producer_local_ms"].update({
            "wire_send_duration_ms": wire_send_ms,
            "producer_ack_roundtrip_ms": total_roundtrip_ms,
        })
        report["consumer_completion_ack"] = done_data
        report["success"] = True

        _write_json(output, report)
        click.echo(f"[PRODUCER] Output exported to {output}")
        sys.exit(0)

    except Exception as e:
        click.echo(f"[PRODUCER ERROR] {type(e).__name__}: {e}", err=True)
        report["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        report["success"] = False
        _write_json(output, report)
        sys.exit(1)
    finally:
        try:
            client_sock.close()
        except Exception:
            pass


# ==============================================================================
# CONSUMER ROLE
# ==============================================================================

@cli.command("consumer")
@click.option("--bind-host", default="0.0.0.0", help="IP address to bind the listener socket.")
@click.option("--bind-port", type=int, default=50051, help="Port to bind the listener socket.")
@click.option("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--device", default="cpu", help="Compute device for consumer (e.g. cuda:1 or cpu).")
@click.option("--dtype", default="float32", type=click.Choice(["float16", "bfloat16", "float32"]), help="Base computation dtype.")
@click.option("--decode-steps", type=int, default=4, help="Number of continuation tokens to decode.")
@click.option("--timeout", type=float, default=25.0, help="Listen and receive timeout in seconds.")
@click.option("--output", default="benchmarks/step4_consumer_evidence.json", help="Path to write consumer JSON report.")
# Failure Injection Probes
@click.option("--simulate-rejection", is_flag=True, default=False, help="Simulate consumer rejecting incoming request.")
@click.option("--simulate-stall", is_flag=True, default=False, help="Simulate network stall before receiving frames.")
def run_consumer(
    bind_host: str,
    bind_port: int,
    model: str,
    device: str,
    dtype: str,
    decode_steps: int,
    timeout: float,
    output: str,
    simulate_rejection: bool,
    simulate_stall: bool,
):
    """Executes the Consumer node: accepts KV cache, executes local baseline, and verifies aligned logit fidelity."""
    click.echo(f"=== [CONSUMER] Starting Execution ===")
    click.echo(f"Bind: {bind_host}:{bind_port} | Device: {device} | Dtype: {dtype} | Steps: {decode_steps}")

    git_meta = get_git_metadata()
    hw_telemetry = get_hardware_telemetry(device)
    torch_device = torch.device(device)
    torch_dtype = resolve_torch_dtype(dtype, torch_device.type)

    report: Dict[str, Any] = {
        "role": "consumer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model,
        "binding": {"host": bind_host, "port": bind_port},
        "git_provenance": git_meta,
        "hardware_telemetry": hw_telemetry,
        "configuration": {
            "device": device,
            "dtype": dtype,
            "decode_steps": decode_steps,
            "timeout_seconds": timeout,
        },
        "success": False,
        "error": None,
    }

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.settimeout(timeout)

    conn: Optional[socket.socket] = None

    try:
        # 1. Bind and Listen Immediately
        server_sock.bind((bind_host, bind_port))
        server_sock.listen(1)
        click.echo(f"[CONSUMER] Listening on {bind_host}:{bind_port} (timeout={timeout}s)...")

        # 2. Pre-warm Consumer Model and Tokenizer
        click.echo("[CONSUMER] Loading consumer model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model)
        causal_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype).to(torch_device)
        causal_model.eval()

        consumer_commit_hash = getattr(causal_model.config, "_commit_hash", None)
        report["resolved_checkpoint_commit_hash"] = consumer_commit_hash

        # Accept connection from producer (robust against ephemeral probes)
        accept_deadline = time.monotonic() + timeout
        manifest = None
        manifest_acc = None
        t_wire_recv_start = time.perf_counter()

        while time.monotonic() < accept_deadline:
            conn, addr = server_sock.accept()
            conn.settimeout(timeout)
            click.echo(f"[CONSUMER] Connection accepted from {addr[0]}:{addr[1]}")
            t_wire_recv_start = time.perf_counter()
            try:
                manifest, manifest_acc = recv_manifest(conn, timeout=timeout)
                break
            except IncompleteTransferError:
                click.echo("[CONSUMER] Ephemeral probe connection closed without manifest, awaiting producer...")
                conn.close()
                conn = None
                continue

        if manifest is None or manifest_acc is None or conn is None:
            raise IncompleteTransferError("Consumer timed out waiting for valid manifest connection")

        report["request_id"] = manifest.request_id
        report["manifest"] = manifest.__dict__

        # Validate Compatibility
        try:
            if simulate_rejection:
                raise ManifestValidationError("Simulated rejection triggered via CLI flag")
            validate_manifest_compatibility(
                manifest=manifest,
                model_config=causal_model.config,
                expected_model_id=model,
                expected_decode_rank=0,
                expected_checkpoint_commit_hash=consumer_commit_hash,
            )
        except Exception as e:
            # Send Rejection Handshake Packet before aborting
            send_handshake_msg(conn, MAGIC_REJ_MANIFEST, {"status": "REJECTED", "reason": str(e)})
            raise

        # Send Acceptance Handshake Packet
        send_handshake_msg(conn, MAGIC_ACK_MANIFEST, {"status": "ACCEPTED", "model_id": model})

        if simulate_stall:
            click.echo("[CONSUMER] Simulating stall: sleeping beyond timeout...")
            time.sleep(timeout + 2.0)

        # 4. Receive Cache Frames into Accumulator
        click.echo(f"[CONSUMER] Receiving {manifest.expected_frames} frames...")
        accumulator = KVCacheFrameAccumulator(manifest)
        accumulator.accounting = accumulator.accounting.add(manifest_acc)

        for frame_idx in range(manifest.expected_frames):
            meta, body, frame_acc = recv_frame(
                sock=conn,
                manifest=manifest,
                timeout=timeout,
                cumulative_bytes=accumulator.cumulative_body_bytes,
                plan=accumulator.plan,
            )
            accumulator.add_frame(meta, body, frame_acc)

        accumulator.validate_completeness()
        t_wire_recv_end = time.perf_counter()
        wire_recv_duration_ms = (t_wire_recv_end - t_wire_recv_start) * 1000.0

        # Send Completion ACK back to Producer
        send_handshake_msg(
            conn,
            MAGIC_ACK_COMPLETED,
            {
                "status": "COMPLETED",
                "received_frames": len(accumulator.received_frames),
                "total_wire_bytes": accumulator.accounting.total_wire_bytes,
            },
        )

        # 5. Reconstruct Cache onto Consumer Device
        click.echo("[CONSUMER] Reconstructing cache tensors...")
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_recon_start = time.perf_counter()

        raw_cache = accumulator.reconstruct_cache()
        transferred_cache = DynamicCache()
        for l in range(manifest.num_layers):
            transferred_cache.update(
                raw_cache["keys"][l].to(device=torch_device, dtype=torch_dtype),
                raw_cache["values"][l].to(device=torch_device, dtype=torch_dtype),
                layer_idx=l,
            )

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t_recon_end = time.perf_counter()
        reconstruction_duration_ms = (t_recon_end - t_recon_start) * 1000.0

        # ======================================================================
        # 6. Consumer-Local Baseline & Ground-Truth Execution
        # ======================================================================
        click.echo("[CONSUMER] Executing consumer-local uninterrupted baseline...")
        input_ids = torch.tensor([manifest.prompt_tokens], device=torch_device)

        with torch.no_grad():
            out_base = causal_model(input_ids=input_ids, use_cache=True)
            base_init_logits = out_base.logits[:, -1, :]
            consumer_init_tok_id = int(torch.argmax(base_init_logits, dim=-1).item())
            consumer_init_tok_str = tokenizer.decode([consumer_init_tok_id])
            baseline_cache = out_base.past_key_values

        producer_init_match = bool(consumer_init_tok_id == manifest.initial_decode_token)

        # Uninterrupted Baseline Continuation
        baseline_tokens: List[int] = []
        baseline_step_logits: List[torch.Tensor] = []
        curr_tok = torch.tensor([[consumer_init_tok_id]], device=torch_device)
        curr_cache = baseline_cache

        for _ in range(decode_steps):
            with torch.no_grad():
                dec_out = causal_model(input_ids=curr_tok, past_key_values=curr_cache, use_cache=True)
                step_log = dec_out.logits[:, -1, :]
                curr_cache = dec_out.past_key_values
                next_tok = torch.argmax(step_log, dim=-1, keepdim=True)
                curr_tok = next_tok
                baseline_tokens.append(int(next_tok.item()))
                baseline_step_logits.append(step_log[0].cpu())

        baseline_logits_tensor = torch.stack(baseline_step_logits, dim=0)

        # ======================================================================
        # 7. Empirical Negative Control: Null-Cache Ablation
        # ======================================================================
        click.echo("[CONSUMER] Executing null-cache negative control...")
        null_tokens: List[int] = []
        curr_tok = torch.tensor([[manifest.initial_decode_token]], device=torch_device)
        for _ in range(decode_steps):
            with torch.no_grad():
                dec_null = causal_model(input_ids=curr_tok, past_key_values=None, use_cache=True)
                null_next = torch.argmax(dec_null.logits[:, -1, :], dim=-1, keepdim=True)
                curr_tok = null_next
                null_tokens.append(int(null_next.item()))

        null_diverged = bool(null_tokens != baseline_tokens)

        # ======================================================================
        # 8. Transferred Cache: Free-Running Autoregressive Decode
        # ======================================================================
        click.echo("[CONSUMER] Executing free-running decode with transferred cache...")
        free_running_tokens: List[int] = []
        free_step_durations_ms: List[float] = []
        curr_tok = torch.tensor([[manifest.initial_decode_token]], device=torch_device)
        curr_cache = transferred_cache

        for s in range(decode_steps):
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            t_s = time.perf_counter()

            with torch.no_grad():
                out_trans = causal_model(input_ids=curr_tok, past_key_values=curr_cache, use_cache=True)
                step_log = out_trans.logits[:, -1, :]
                curr_cache = out_trans.past_key_values
                next_tok = torch.argmax(step_log, dim=-1, keepdim=True)
                curr_tok = next_tok
                free_running_tokens.append(int(next_tok.item()))

            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            t_e = time.perf_counter()
            free_step_durations_ms.append((t_e - t_s) * 1000.0)

        first_decode_step_ms = free_step_durations_ms[0] if free_step_durations_ms else 0.0
        subsequent_latencies_ms = free_step_durations_ms[1:] if len(free_step_durations_ms) > 1 else []
        median_subsequent_ms = (
            statistics_median(subsequent_latencies_ms)
            if subsequent_latencies_ms
            else first_decode_step_ms
        )

        free_tokens_match = bool(free_running_tokens == baseline_tokens)

        # ======================================================================
        # 9. Transferred Cache: Aligned Force-Fed Decode (for Logit Comparison)
        # ======================================================================
        click.echo("[CONSUMER] Executing aligned force-fed decode for logit fidelity...")
        aligned_step_logits: List[torch.Tensor] = []
        aligned_cache = DynamicCache()
        for l in range(manifest.num_layers):
            aligned_cache.update(
                raw_cache["keys"][l].to(device=torch_device, dtype=torch_dtype),
                raw_cache["values"][l].to(device=torch_device, dtype=torch_dtype),
                layer_idx=l,
            )

        for s in range(decode_steps):
            # Force-feed the EXACT baseline token from the previous step
            if s == 0:
                feed_tok = torch.tensor([[manifest.initial_decode_token]], device=torch_device)
            else:
                feed_tok = torch.tensor([[baseline_tokens[s - 1]]], device=torch_device)

            with torch.no_grad():
                aligned_out = causal_model(input_ids=feed_tok, past_key_values=aligned_cache, use_cache=True)
                aligned_log = aligned_out.logits[:, -1, :]
                aligned_cache = aligned_out.past_key_values
                aligned_step_logits.append(aligned_log[0].cpu())

        aligned_logits_tensor = torch.stack(aligned_step_logits, dim=0)

        # Compute Mathematical Metrics vs Baseline
        logit_diff = torch.abs(baseline_logits_tensor.float() - aligned_logits_tensor.float())
        max_abs_logit_error = float(logit_diff.max().item())
        mean_abs_logit_error = float(logit_diff.mean().item())
        cosine_sim = float(
            torch.nn.functional.cosine_similarity(
                baseline_logits_tensor.float().flatten(),
                aligned_logits_tensor.float().flatten(),
                dim=0,
            ).item()
        )
        aligned_top1_match = bool(
            (torch.argmax(baseline_logits_tensor, dim=-1) == torch.argmax(aligned_logits_tensor, dim=-1)).all().item()
        )

        # Discrete Check Tolerances
        uncompressed_tolerance = 1e-4
        fp8_cosine_threshold = 0.999

        finite_metrics = (
            math.isfinite(max_abs_logit_error)
            and math.isfinite(mean_abs_logit_error)
            and math.isfinite(cosine_sim)
        )

        if not manifest.enable_fp8:
            correctness_passed = (
                producer_init_match
                and null_diverged
                and free_tokens_match
                and aligned_top1_match
                and max_abs_logit_error <= uncompressed_tolerance
                and finite_metrics
            )
        else:
            correctness_passed = (
                producer_init_match
                and null_diverged
                and cosine_sim >= fp8_cosine_threshold
                and finite_metrics
            )

        report["wire_accounting"] = {
            "framing_bytes": accumulator.accounting.framing_bytes,
            "metadata_bytes": accumulator.accounting.metadata_bytes,
            "body_bytes": accumulator.accounting.body_bytes,
            "total_wire_bytes": accumulator.accounting.total_wire_bytes,
            "frames_received": len(accumulator.received_frames),
        }
        report["timings_consumer_local_ms"] = {
            "wire_receive_duration_ms": wire_recv_duration_ms,
            "reconstruction_duration_ms": reconstruction_duration_ms,
            "first_decode_step_latency_ms": first_decode_step_ms,
            "median_subsequent_latency_ms": median_subsequent_ms,
            "subsequent_latencies_ms": subsequent_latencies_ms,
        }
        report["baseline_execution"] = {
            "initial_token_id": consumer_init_tok_id,
            "initial_token_str": consumer_init_tok_str,
            "generated_tokens": baseline_tokens,
            "generated_text": tokenizer.decode(baseline_tokens),
        }
        report["negative_control_null_cache"] = {
            "null_tokens": null_tokens,
            "null_text": tokenizer.decode(null_tokens),
            "diverged_from_baseline": null_diverged,
        }
        report["transferred_cache_decoding"] = {
            "free_running_tokens": free_running_tokens,
            "free_running_text": tokenizer.decode(free_running_tokens),
            "free_running_tokens_match": free_tokens_match,
            "aligned_comparison": {
                "max_abs_logit_error": max_abs_logit_error,
                "mean_abs_logit_error": mean_abs_logit_error,
                "cosine_similarity": cosine_sim,
                "aligned_top1_match": aligned_top1_match,
            },
        }
        report["discrete_checks"] = {
            "producer_initial_token_matches_consumer": producer_init_match,
            "null_cache_negative_control_diverged": null_diverged,
            "metrics_are_finite": finite_metrics,
            "uncompressed_tokens_match": free_tokens_match if not manifest.enable_fp8 else None,
            "uncompressed_logit_error_within_tolerance": (max_abs_logit_error <= uncompressed_tolerance) if not manifest.enable_fp8 else None,
            "fp8_cosine_similarity_within_threshold": (cosine_sim >= fp8_cosine_threshold) if manifest.enable_fp8 else None,
            "fp8_free_running_tokens_match": free_tokens_match if manifest.enable_fp8 else None,
            "fp8_aligned_top1_match": aligned_top1_match if manifest.enable_fp8 else None,
        }
        report["success"] = correctness_passed

        _write_json(output, report)
        click.echo(f"[CONSUMER] Output exported to {output}")
        if correctness_passed:
            click.echo("[CONSUMER] All correctness criteria PASSED!")
            sys.exit(0)
        else:
            click.echo("[CONSUMER] Correctness criteria FAILED!", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"[CONSUMER ERROR] {type(e).__name__}: {e}", err=True)
        report["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        report["success"] = False
        _write_json(output, report)
        sys.exit(1)
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


def statistics_median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def _write_json(path_str: str, data: Dict[str, Any]) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    cli()
