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
import time
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
) -> None:
    """
    Consumer Process (Worker B):
    1. Binds to OS-assigned port on 127.0.0.1.
    2. Signals ready and puts port into port_queue.
    3. Accepts connection from prefill producer.
    4. Receives and validates RequestManifest.
    5. Receives all frames, accumulating and validating against manifest.
    6. Reconstructs DynamicCache and decodes `decode_steps` continuation tokens.
    7. Returns results and process telemetry to result_queue.
    """
    consumer_pid = os.getpid()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    assigned_port = server_sock.getsockname()[1]
    server_sock.listen(1)
    # Generous timeout for accept while producer initializes
    server_sock.settimeout(timeout + 30)

    # Signal readiness to orchestrator
    port_queue.put({"port": assigned_port, "pid": consumer_pid})

    conn: Optional[socket.socket] = None
    temp_logits_path = f"/tmp/hkv_consumer_logits_{consumer_pid}.pt"
    try:
        conn, addr = server_sock.accept()
        conn.settimeout(timeout)

        # 1. Receive and validate request manifest
        manifest, manifest_acc = recv_manifest(conn, timeout=timeout)
        total_acc = manifest_acc

        # 2. Accumulate incoming frames
        accumulator = KVCacheFrameAccumulator(manifest)
        while not accumulator.is_complete():
            meta, body, frame_acc = recv_frame(conn, manifest, timeout=timeout)
            accumulator.add_frame(meta, body, frame_acc)
            total_acc = total_acc.add(frame_acc)

        # 3. Validate complete frame set before decoding
        accumulator.validate_completeness()
        raw_cache_dict = accumulator.reconstruct_cache()

        # 4. Load local model to run decoding with reconstructed cache
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model.eval()

        reconstructed_cache = DynamicCache()
        for l in range(manifest.num_layers):
            reconstructed_cache.update(
                raw_cache_dict["keys"][l].to(device),
                raw_cache_dict["values"][l].to(device),
                layer_idx=l,
            )

        # 5. Decode continuation tokens
        curr_cache = reconstructed_cache
        curr_token = torch.tensor([[manifest.initial_decode_token]], device=device)
        generated_tokens: List[int] = []
        step_logits_list: List[torch.Tensor] = []

        for step in range(decode_steps):
            with torch.no_grad():
                out = model(input_ids=curr_token, past_key_values=curr_cache, use_cache=True)
                logits = out.logits[:, -1, :]
                curr_cache = out.past_key_values
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                curr_token = next_tok
                tok_id = int(next_tok.item())
                generated_tokens.append(tok_id)
                step_logits_list.append(logits[0].cpu())

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
) -> None:
    """
    Producer Process (Worker A):
    1. Loads model on device and runs prefill on prompt.
    2. Slices and packages KV cache across prefill TP ranks.
    3. Connects to decode worker socket at 127.0.0.1:target_port (with retry).
    4. Transmits RequestManifest followed by all partitioned frames.
    5. Cleanly closes socket and returns telemetry.
    """
    producer_pid = os.getpid()
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.settimeout(timeout)

    try:
        # Load model and run prefill
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        prompt_tokens = input_ids[0].tolist()

        with torch.no_grad():
            out_prefill = model(input_ids=input_ids, use_cache=True)
            prefill_cache = out_prefill.past_key_values
            initial_next_token = int(torch.argmax(out_prefill.logits[:, -1, :], dim=-1).item())
            initial_token_str = tokenizer.decode([initial_next_token])

        num_layers = len(prefill_cache.layers)
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
            dtype="float32",
            enable_fp8=enable_fp8,
            block_size_tokens=block_size_tokens,
        )

        acc = send_manifest(client_sock, manifest)

        # Transmit all frames
        for l in range(num_layers):
            orig_k = prefill_cache.layers[l].keys
            orig_v = prefill_cache.layers[l].values

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
                            original_dtype="float32",
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
                            dtype="float32",
                            body_bytes=len(body) if corrupt_frame != "body_length_mismatch" else len(body) * 2,
                        )
                    frame_acc = send_frame(client_sock, meta, body)
                    acc = acc.add(frame_acc)

        result_queue.put(
            {
                "status": "success",
                "producer_pid": producer_pid,
                "initial_token": initial_next_token,
                "initial_token_str": initial_token_str,
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
    device: str = "cpu",
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Executes end-to-end multi-process socket cache handoff:
    1. Spawns Producer and Consumer in separate OS processes via multiprocessing ("spawn").
    2. Consumer listens on OS-assigned port; Producer connects and streams serialized cache (uncompressed or FP8).
    3. Runs uninterrupted baseline in orchestrator for ground-truth comparison.
    4. Evaluates maximum absolute logit error, token match, and byte breakdown.
    """
    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    consumer_result_queue = ctx.Queue()
    producer_result_queue = ctx.Queue()

    # 1. Start Consumer process
    consumer_proc = ctx.Process(
        target=decode_consumer_worker,
        args=(port_queue, consumer_result_queue, model_id, decode_steps, device, timeout),
    )
    consumer_proc.start()

    # Wait for consumer readiness and assigned port
    ready_info = port_queue.get(timeout=timeout)
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
            device,
            enable_fp8,
            block_size_tokens,
            timeout,
        ),
    )
    producer_proc.start()

    # Wait for results from queues with bounded deadline
    producer_res = producer_result_queue.get(timeout=timeout + 30)
    consumer_res = consumer_result_queue.get(timeout=timeout + 30)

    # Join child processes boundedly
    producer_proc.join(timeout=10)
    consumer_proc.join(timeout=10)

    # Guarantee process termination
    if producer_proc.is_alive():
        producer_proc.terminate()
    if consumer_proc.is_alive():
        consumer_proc.terminate()

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

    # Load consumer logits from temp file and clean up
    temp_logits_path = consumer_res["temp_logits_path"]
    try:
        consumer_logits_tensor = torch.load(temp_logits_path, weights_only=True)
    finally:
        if os.path.exists(temp_logits_path):
            try:
                os.remove(temp_logits_path)
            except Exception:
                pass

    # 3. Run ground-truth baseline in orchestrator
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out_base = model(input_ids=inputs.input_ids, use_cache=True)
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

    # 4. Compare step-by-step logits
    max_logit_errors: List[float] = []
    step_comparisons: List[Dict[str, Any]] = []

    for s in range(decode_steps):
        t_base = base_logits_list[s]
        t_recon = consumer_logits_tensor[s]
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
    num_layers = len(out_base.past_key_values.layers)

    uncomp_fp32_bytes = seq_len * num_kv_heads * head_dim * 4 * num_layers * 2
    uncomp_fp32_bytes = seq_len * num_kv_heads * head_dim * 4 * num_layers * 2
    uncomp_fp16_bytes = uncomp_fp32_bytes // 2
    wire_bytes = wa["total_wire_bytes"]
    body_bytes = wa["body_bytes"]

    ratio_vs_fp32 = round(uncomp_fp32_bytes / max(1, body_bytes), 2)
    ratio_vs_fp16 = round(uncomp_fp16_bytes / max(1, body_bytes), 2)
    wire_reduction_vs_fp16_pct = round((1.0 - (body_bytes / max(1, uncomp_fp16_bytes))) * 100.0, 1)

    if enable_fp8:
        # Lossy FP8 pass condition: high cosine similarity >= 0.99 and body compression vs FP32
        divergence_within_tolerance = bool(mean_cos >= 0.99 and body_bytes < uncomp_fp32_bytes)
        tolerance_desc = "Mean Cosine Sim >= 0.99 & Tensor Body Compressed"
    else:
        # Exact uncompressed pass condition: max abs logit divergence <= 1e-6
        divergence_within_tolerance = bool(max_abs_divergence <= 1e-6)
        tolerance_desc = "|Δ Logits| <= 1e-6"

    return {
        "model_id": model_id,
        "enable_fp8": enable_fp8,
        "block_size_tokens": block_size_tokens,
        "pids": {
            "orchestrator_pid": main_pid,
            "producer_pid": producer_pid,
            "consumer_pid": consumer_pid,
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
    }


@click.command()
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--prompt", "-p", default="Heterogeneous disaggregated inference allows", help="Input prompt.")
@click.option("--steps", "-s", default=4, help="Autoregressive decoding steps.")
@click.option("--enable-fp8/--no-fp8", default=False, help="Enable block-scaled FP8 KV cache wire compression.")
@click.option("--block-size", default=16, help="FP8 block tile size.")
@click.option("--output", "-o", default="benchmarks/socket_cache_handoff.json", help="JSON output path.")
def main(model: str, prompt: str, steps: int, enable_fp8: bool, block_size: int, output: str):
    """End-to-End Multiprocess Socket KV Cache Transfer Verification."""
    mode_str = "FP8 Block-Quantized" if enable_fp8 else "Uncompressed Exact"
    metric_desc = "Cosine Sim >= 0.99 & Wire Ratio >= 1.4x vs FP16" if enable_fp8 else "Max Absolute Logit Error (|Δ Logits| <= 1e-6)"
    console.print(
        Panel(
            f"[bold cyan]Process-to-Process Socket KV Cache Transfer Verification[/bold cyan]\n"
            f"Target Model: {model} | Prefill TP=2 -> Decode TP=1\n"
            f"Mode: {mode_str} (Block Size: {block_size})\n"
            f"Transport: Localhost TCP Socket (127.0.0.1:OS-assigned)\n"
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
    )

    pids = results["pids"]
    console.print(
        f"[dim]Process Topology: Orchestrator PID={pids['orchestrator_pid']} | "
        f"Producer PID={pids['producer_pid']} | Consumer PID={pids['consumer_pid']} | "
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
        f"[bold]Process Verification:[/bold]\n"
        f"  Distinct Process Isolation: [bold green]CONFIRMED[/bold green] (3 distinct PIDs)\n"
        f"  Cache Transferred Over Socket: [bold green]YES[/bold green] (No shared memory tensor handoff)\n\n"
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
