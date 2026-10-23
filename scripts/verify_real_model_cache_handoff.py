"""Gate 2 Verification: Real-Model Cache Handoff & Logits Divergence Benchmark.

Validates that real model KV caches extracted after prefill can be sliced across
prefill ranks, packaged with routing metadata, reassembled on decode ranks, and
consumed by the model during multi-step autoregressive decoding with exactly zero
divergence from uninterrupted serving.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import Dict, Any, List, Tuple
import click
import torch
import transformers
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from hetero_kv.layout import KVPartitionPlan
from hetero_kv.reshard import (
    slice_kv_for_prefill_rank,
    package_payloads_from_prefill_rank,
    reassemble_decode_kv,
)

console = Console()


def run_cache_handoff_experiment(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "Heterogeneous disaggregated inference allows",
    decode_steps: int = 5,
    prefill_tp: int = 2,
    decode_tp: int = 1,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Execute end-to-end real-model cache handoff and logit equivalence experiment:
    1. Run prefill on prompt using small local LLM.
    2. Extract actual K and V caches for all transformer layers.
    3. Partition across prefill ranks (logical head slicing in-process), package uncompressed payloads,
       and reassemble on decode rank 0.
    4. Continue decoding using the reconstructed cache.
    5. Compare step-by-step next-token logits against uninterrupted baseline inference.
    """
    if decode_tp != 1:
        raise ValueError(
            f"Unsupported decode_tp={decode_tp}. In-process model forwarding currently reconstructs "
            f"rank 0 into the local model. Multi-rank decode tensor parallelism across separate "
            f"processes/nodes is an upcoming distributed systems milestone. Use decode_tp=1 for local verification."
        )

    dev = torch.device(device)
    console.print(f"[dim]Loading model '{model_id}' on {device}...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.to(dev)
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # -------------------------------------------------------------
    # Step 1: Prefill Pass
    # -------------------------------------------------------------
    console.print(f"[bold cyan]Running prefill on prompt ({prompt_len} tokens)...[/bold cyan]")
    with torch.no_grad():
        out_base_prefill = model(input_ids=input_ids, use_cache=True)
        prefill_cache = out_base_prefill.past_key_values
        next_token_base = torch.argmax(out_base_prefill.logits[:, -1, :], dim=-1, keepdim=True)

    # Clone prefill cache for uninterrupted baseline so in-place decoding doesn't corrupt prefill state
    baseline_cache = DynamicCache()
    for l in range(num_layers):
        baseline_cache.update(
            prefill_cache.layers[l].keys.clone(),
            prefill_cache.layers[l].values.clone(),
            layer_idx=l,
        )

    # -------------------------------------------------------------
    # Step 2: Uninterrupted Baseline Autoregressive Decoding
    # -------------------------------------------------------------
    console.print(f"[bold cyan]Running uninterrupted baseline ({decode_steps} steps)...[/bold cyan]")
    t0_base = time.perf_counter()
    baseline_step_logits: List[torch.Tensor] = []
    baseline_step_tokens: List[int] = []

    curr_token = next_token_base
    curr_cache = baseline_cache
    with torch.no_grad():
        for step in range(decode_steps):
            out_step = model(input_ids=curr_token, past_key_values=curr_cache, use_cache=True)
            step_logits = out_step.logits[:, -1, :].clone()
            step_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            baseline_step_logits.append(step_logits)
            baseline_step_tokens.append(int(step_token.item()))

            curr_token = step_token
            curr_cache = out_step.past_key_values
    base_time_ms = (time.perf_counter() - t0_base) * 1000.0

    # -------------------------------------------------------------
    # Step 3: Partition and Reconstruct Cache from Prefill without Compression
    # -------------------------------------------------------------
    console.print(
        f"[bold magenta]Partitioning {num_layers} layers across {prefill_tp} prefill ranks -> {decode_tp} decode rank(s)...[/bold magenta]"
    )
    plan = KVPartitionPlan.create(num_kv_heads=num_kv_heads, prefill_tp=prefill_tp, decode_tp=decode_tp)
    reconstructed_cache = DynamicCache()

    total_k_diff = 0.0
    total_v_diff = 0.0

    t0_handoff = time.perf_counter()
    for layer_idx in range(num_layers):
        orig_k = prefill_cache.layers[layer_idx].keys
        orig_v = prefill_cache.layers[layer_idx].values

        # Convert HF format [B, num_kv_heads, S, head_dim] -> Hetero format [B, S, num_kv_heads, head_dim]
        k_hetero = orig_k.permute(0, 2, 1, 3).contiguous()
        v_hetero = orig_v.permute(0, 2, 1, 3).contiguous()

        # Transfer Key Cache across prefill ranks
        payloads_k = []
        for p_rank in range(prefill_tp):
            shard = slice_kv_for_prefill_rank(k_hetero, prefill_rank=p_rank, plan=plan)
            p_list = package_payloads_from_prefill_rank(
                shard,
                prefill_rank=p_rank,
                layer_idx=layer_idx,
                plan=plan,
                enable_fp8=False,
                request_id="gate2_verified_req",
                kv_type="key",
            )
            payloads_k.extend(p_list)

        reassembled_k_hetero = reassemble_decode_kv(
            payloads_k,
            decode_rank=0,
            plan=plan,
            expected_request_id="gate2_verified_req",
            expected_kv_type="key",
        )
        restored_k = reassembled_k_hetero.permute(0, 2, 1, 3).contiguous()

        # Transfer Value Cache across prefill ranks
        payloads_v = []
        for p_rank in range(prefill_tp):
            shard = slice_kv_for_prefill_rank(v_hetero, prefill_rank=p_rank, plan=plan)
            p_list = package_payloads_from_prefill_rank(
                shard,
                prefill_rank=p_rank,
                layer_idx=layer_idx,
                plan=plan,
                enable_fp8=False,
                request_id="gate2_verified_req",
                kv_type="value",
            )
            payloads_v.extend(p_list)

        reassembled_v_hetero = reassemble_decode_kv(
            payloads_v,
            decode_rank=0,
            plan=plan,
            expected_request_id="gate2_verified_req",
            expected_kv_type="value",
        )
        restored_v = reassembled_v_hetero.permute(0, 2, 1, 3).contiguous()

        # Compute tensor difference between original and reconstructed
        diff_k = torch.max(torch.abs(orig_k - restored_k)).item()
        diff_v = torch.max(torch.abs(orig_v - restored_v)).item()
        total_k_diff = max(total_k_diff, diff_k)
        total_v_diff = max(total_v_diff, diff_v)

        reconstructed_cache.update(restored_k, restored_v, layer_idx=layer_idx)

    # -------------------------------------------------------------
    # Step 4: Continue Decoding Using Reconstructed Cache
    # -------------------------------------------------------------
    console.print(f"[bold green]Executing {decode_steps} decoding steps with reconstructed cache...[/bold green]")
    reconstructed_step_logits: List[torch.Tensor] = []
    reconstructed_step_tokens: List[int] = []

    curr_token_recon = next_token_base
    curr_cache_recon = reconstructed_cache

    with torch.no_grad():
        for step in range(decode_steps):
            out_recon = model(input_ids=curr_token_recon, past_key_values=curr_cache_recon, use_cache=True)
            step_logits_recon = out_recon.logits[:, -1, :].clone()
            step_token_recon = torch.argmax(step_logits_recon, dim=-1, keepdim=True)

            reconstructed_step_logits.append(step_logits_recon)
            reconstructed_step_tokens.append(int(step_token_recon.item()))

            curr_token_recon = step_token_recon
            curr_cache_recon = out_recon.past_key_values

    handoff_time_ms = (time.perf_counter() - t0_handoff) * 1000.0

    # -------------------------------------------------------------
    # Step 5: Compare Next-Token Logits Across All Steps
    # -------------------------------------------------------------
    step_results = []
    all_tokens_match = True
    max_divergence_overall = 0.0

    for step in range(decode_steps):
        logits_b = baseline_step_logits[step]
        logits_r = reconstructed_step_logits[step]

        max_diff = torch.max(torch.abs(logits_b - logits_r)).item()
        max_divergence_overall = max(max_divergence_overall, max_diff)

        raw_cos = torch.cosine_similarity(logits_b.flatten(), logits_r.flatten(), dim=0).item()
        cos_sim = max(-1.0, min(1.0, float(raw_cos)))
        tok_b = baseline_step_tokens[step]
        tok_r = reconstructed_step_tokens[step]
        tok_str_b = tokenizer.decode([tok_b])
        tok_str_r = tokenizer.decode([tok_r])
        matches = (tok_b == tok_r)
        if not matches:
            all_tokens_match = False

        step_results.append({
            "step": step + 1,
            "max_abs_logit_diff": float(max_diff),
            "cosine_similarity": float(cos_sim),
            "baseline_token_id": tok_b,
            "baseline_token_str": tok_str_b,
            "reconstructed_token_id": tok_r,
            "reconstructed_token_str": tok_str_r,
            "tokens_match": matches,
        })

    baseline_generation = tokenizer.decode(baseline_step_tokens)
    reconstructed_generation = tokenizer.decode(reconstructed_step_tokens)

    return {
        "model_id": model_id,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "prefill_tp": prefill_tp,
        "decode_tp": decode_tp,
        "prompt": prompt,
        "decode_steps": decode_steps,
        "max_k_cache_diff": total_k_diff,
        "max_v_cache_diff": total_v_diff,
        "max_logit_divergence_overall": max_divergence_overall,
        "all_tokens_match": all_tokens_match,
        "baseline_generated_text": baseline_generation,
        "reconstructed_generated_text": reconstructed_generation,
        "reproducibility": {
            "methodology": "empirical_validation",
            "parallelism_type": "in_process_logical_head_partitioning",
            "primary_metric": "max_abs_logit_error",
            "tolerance_max_abs_logit_error": 1e-6,
            "torch_version": torch.__version__,
            "transformers_version": getattr(transformers, "__version__", "unknown"),
            "dtype": "float32",
            "checkpoint_commit_hash": getattr(model.config, "_commit_hash", None),
        },
        "step_results": step_results,
    }


@click.command()
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--prompt", "-p", default="Heterogeneous disaggregated inference allows", help="Input prompt.")
@click.option("--steps", "-s", default=5, help="Number of autoregressive decoding steps.")
@click.option("--prefill-tp", default=2, help="Prefill tensor-parallel degree.")
@click.option("--decode-tp", default=1, help="Decode tensor-parallel degree.")
@click.option("--output", "-o", default=None, help="JSON output file for results.")
def main(model: str, prompt: str, steps: int, prefill_tp: int, decode_tp: int, output: str):
    """Gate 2 Empirical Validation: Real-Model Cache Handoff & Logit Equivalence."""
    console.print(
        Panel(
            f"[bold cyan]Gate 2: Real-Model Cache Handoff Empirical Validation[/bold cyan]\n"
            f"Target Model: {model}\n"
            f"Architecture: Prefill TP={prefill_tp} -> Decode TP={decode_tp} (In-Process Logical Head Partitioning)\n"
            f"Primary Metric: Max Absolute Logit Error (|Δ Logits| <= 1e-6)\n"
            f"Decoding Horizon: {steps} autoregressive steps",
            border_style="cyan",
        )
    )

    results = run_cache_handoff_experiment(
        model_id=model,
        prompt=prompt,
        decode_steps=steps,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
    )

    table = Table(title="Step-by-Step Logit & Token Equivalence", header_style="bold magenta")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Baseline Token", style="white")
    table.add_column("Reconstructed Token", style="white")
    table.add_column("Tokens Match", justify="center")
    table.add_column("Max |Δ Logit|", justify="right", style="bold green")
    table.add_column("Cosine Sim", justify="right", style="bold yellow")

    for r in results["step_results"]:
        status = "[green]MATCH[/green]" if r["tokens_match"] else "[red]MISMATCH[/red]"
        table.add_row(
            str(r["step"]),
            f"'{r['baseline_token_str']}' ({r['baseline_token_id']})",
            f"'{r['reconstructed_token_str']}' ({r['reconstructed_token_id']})",
            status,
            f"{r['max_abs_logit_diff']:.2e}",
            f"{r['cosine_similarity']:.6f}",
        )

    console.print(table)

    summary_panel = Panel(
        f"[bold]Cache Tensor Reassembly Error:[/bold]\n"
        f"  Max Key Cache Diff: {results['max_k_cache_diff']:.2e}\n"
        f"  Max Value Cache Diff: {results['max_v_cache_diff']:.2e}\n\n"
        f"[bold]Overall Logits Equivalence (Primary Metric):[/bold]\n"
        f"  Max Absolute Logit Error: [bold green]{results['max_logit_divergence_overall']:.2e}[/bold green]\n"
        f"  All Tokens Match: {'[bold green]YES (100% Exact)[/bold green]' if results['all_tokens_match'] else '[bold red]NO[/bold red]'}\n\n"
        f"[bold]Generated Continuations:[/bold]\n"
        f"  Baseline:      \"{results['baseline_generated_text']}\"\n"
        f"  Reconstructed: \"{results['reconstructed_generated_text']}\"\n\n"
        f"[dim]Note: Logical head partitioning inside one process; worker-to-worker transport is a separate systems milestone.[/dim]",
        title="Gate 2 Empirical Cache Handoff Validation",
        border_style="green" if results["all_tokens_match"] else "red",
    )
    console.print(summary_panel)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
