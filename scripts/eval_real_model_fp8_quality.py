"""Real-Model FP8 Cache Quality Evaluation on Held-Out Text.

Measures quality preservation of HeteroDisagg's block-scaled FP8 KV cache quantization
against an uncompressed baseline on a real Transformer model (Qwen2.5-0.5B-Instruct).

Aligned Teacher-Forcing Methodology:
1. Prefill on context text: producer produces prefill logits and KV cache.
2. Boundary target token (step 0): evaluates producer prefill logits for both branches.
3. Continuation target tokens (step 1..M-1): feeds identical continuation tokens into both
   baseline and reconstructed FP8 caches; asserts equal cache lengths before every forward.
4. Reports metrics across both the compression-affected continuation steps and the full sequence.
5. Computes wire compression ratio against both FP32 and unpadded FP16/BF16 baselines explicitly.

Scope: Diagnostic smoke test across curated passages (48 target tokens) to confirm evaluator
alignment and lossy quantization stability. Does not replace broad benchmark sweeps.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
from typing import Dict, Any, List, Optional
import click
import torch
import torch.nn.functional as F
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

# Curated benchmark corpus spanning diverse domains (smoke test)
BENCHMARK_CORPUS = [
    {
        "category": "Systems & Distributed ML",
        "context": "Distributed inference engines disaggregate prefill and decode stages to",
        "target": " eliminate memory bandwidth interference and maximize accelerator compute saturation across heterogeneous server tiers.",
    },
    {
        "category": "Algorithmic Reasoning",
        "context": "To implement efficient block-level quantization, we partition high-dimensional tensors into",
        "target": " uniform tile blocks and compute separate dynamic scaling factors per sequence to prevent cross-request numerical contamination.",
    },
    {
        "category": "Software Architecture",
        "context": "The zero-fork adapter dynamically configures tensor parallelism degree and",
        "target": " chunked prefill parameters to match local GPU memory capacity without modifying engine kernels.",
    },
]

# Multi-context length held-out evaluation corpus spanning short to long prompt lengths
EXPANDED_BENCHMARK_CORPUS = [
    {
        "category": "Systems & Prefill (~16 tok)",
        "context": "Distributed inference engines disaggregate prefill and decode stages to eliminate memory bandwidth interference and maximize compute saturation across heterogeneous clusters.",
        "target": " By decoupling compute-bound prefill from memory-bound decode, clusters achieve near-optimal throughput and bounded latency.",
    },
    {
        "category": "Quantization Math (~32 tok)",
        "context": "Block-scaled quantization divides floating point activation tensors into contiguous tiles, computing a local maximum absolute scale factor for each block to minimize roundoff error while preserving dynamic numerical range across outliers.",
        "target": " When applied to key and value caches, block scaling bounds precision loss to isolated head dimensions.",
    },
    {
        "category": "Distributed Consensus (~64 tok)",
        "context": (
            "In distributed consensus protocols such as Raft and Paxos, replica state machines maintain identical execution histories "
            "by appending log entries proposed by an elected leader. When network partitions isolate the leader from a quorum, the cluster "
            "must prevent split-brain anomalies by rejecting uncommitted transitions and electing a successor with the highest term and index."
        ),
        "target": " This ensures linearizable consistency across write operations while allowing healthy minor partitions to continue serving reads.",
    },
    {
        "category": "Compiler Infrastructure (~128 tok)",
        "context": (
            "Modern deep learning compilers lower high-level computation graphs into intermediate representations to perform target-independent "
            "optimizations such as dead code elimination, constant folding, and operator fusion. During loop tiling and polyhedral scheduling, "
            "the compiler restructures nested iteration spaces to maximize cache reuse and vectorization efficiency. For tensor processing units, "
            "register allocation and memory planning must explicitly account for scratchpad memory banking conflicts, asynchronous DMA transfers, "
            "and systolic array dataflow latency before emitting device assembly code."
        ),
        "target": " Kernel fusion reduces memory traffic between off-chip DRAM and on-chip SRAM by keeping intermediate activation tiles resident in registers.",
    },
    {
        "category": "Computer Architecture (~256 tok)",
        "context": (
            "Modern multiprocessor memory subsystems rely on hierarchical cache organizations to bridge the widening latency gap between "
            "fast execution cores and high-capacity dynamic random-access memory. Each processing core incorporates dedicated Level 1 instruction "
            "and data caches backed by a non-inclusive or inclusive Level 2 cache, which in turn interface with a large, shared Level 3 cache "
            "connected via a low-latency coherent interconnect fabric. To preserve data consistency across multiple concurrent threads, hardware "
            "coherence controllers implement snooping or directory-based variants of the MESI protocol, invalidating obsolete cache lines or "
            "transitioning lines to modified states upon store instructions. When coherence traffic saturates the crossbar or mesh interconnect, "
            "false sharing and cache thrashing degrade scalable multithreaded performance. Hardware prefetchers monitor memory access patterns "
            "to prefetch cache lines into lower levels ahead of execution, but irregular stride patterns or pointer-chasing data structures can "
            "induce severe prefetch pollution, displacing working sets and exacerbating DRAM bandwidth contention."
        ),
        "target": " Disaggregated key-value cache architectures alleviate local memory pressure by streaming activations across high-bandwidth RDMA networks.",
    },
]

# Executable acceptance criteria shared across CLI, tests, and reporting
DEFAULT_FP8_THRESHOLDS = {
    "max_delta_nll": 0.05,            # nats (continuation steps)
    "min_top1_agreement_pct": 80.0,   # % (continuation steps)
    "min_top5_overlap_pct": 85.0,     # % (continuation steps)
    "min_cosine_similarity": 0.99,    # cosine similarity
    "min_compression_ratio_fp32": 2.5,
    "min_compression_ratio_fp16": 1.4,
}


def evaluate_passage_fp8_quality(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_text: str,
    target_text: str,
    prefill_tp: int = 2,
    decode_tp: int = 1,
    block_size_tokens: int = 16,
    enable_fp8: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate teacher-forced NLL, perplexity, and logit distribution metrics
    for a single passage comparing uncompressed baseline vs reconstructed cache.

    If enable_fp8 is False, serves as a compression-disabled identity control.
    """
    ctx_ids = tokenizer(context_text, return_tensors="pt").input_ids
    tgt_ids = tokenizer(target_text, return_tensors="pt").input_ids

    prompt_len = ctx_ids.shape[1]
    target_tokens = tgt_ids[0]
    num_targets = len(target_tokens)

    # 1. Prefill on context
    with torch.no_grad():
        out_prefill = model(input_ids=ctx_ids, use_cache=True)
        raw_prefill_cache = out_prefill.past_key_values
        prefill_logits = out_prefill.logits[:, -1, :]

    num_layers = len(raw_prefill_cache.layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    # Clone pristine uncompressed cache
    uncompressed_cache = DynamicCache()
    for l in range(num_layers):
        uncompressed_cache.update(
            raw_prefill_cache.layers[l].keys.clone(),
            raw_prefill_cache.layers[l].values.clone(),
            layer_idx=l,
        )

    # Build partitioned and reassembled cache (FP8 or exact identity)
    plan = KVPartitionPlan.create(num_kv_heads=num_kv_heads, prefill_tp=prefill_tp, decode_tp=decode_tp)
    eval_cache = DynamicCache()

    total_uncompressed_fp32_bytes = 0
    total_wire_bytes = 0

    for l in range(num_layers):
        orig_k = raw_prefill_cache.layers[l].keys
        orig_v = raw_prefill_cache.layers[l].values

        total_uncompressed_fp32_bytes += (orig_k.nbytes + orig_v.nbytes)

        # Convert HF [B, H, S, D] -> Hetero [B, S, H, D]
        k_hetero = orig_k.permute(0, 2, 1, 3).contiguous()
        v_hetero = orig_v.permute(0, 2, 1, 3).contiguous()

        # Key cache: slice, quantize/package, and reassemble
        payloads_k = []
        for p in range(prefill_tp):
            shard = slice_kv_for_prefill_rank(k_hetero, prefill_rank=p, plan=plan)
            p_list = package_payloads_from_prefill_rank(
                shard,
                prefill_rank=p,
                layer_idx=l,
                plan=plan,
                enable_fp8=enable_fp8,
                block_size_tokens=block_size_tokens,
                request_id="fp8_eval_req",
                kv_type="key",
            )
            payloads_k.extend(p_list)
        for p in payloads_k:
            total_wire_bytes += p.payload_bytes

        rec_k = reassemble_decode_kv(
            payloads_k, decode_rank=0, plan=plan, expected_request_id="fp8_eval_req", expected_kv_type="key"
        )

        # Value cache: slice, quantize/package, and reassemble
        payloads_v = []
        for p in range(prefill_tp):
            shard = slice_kv_for_prefill_rank(v_hetero, prefill_rank=p, plan=plan)
            p_list = package_payloads_from_prefill_rank(
                shard,
                prefill_rank=p,
                layer_idx=l,
                plan=plan,
                enable_fp8=enable_fp8,
                block_size_tokens=block_size_tokens,
                request_id="fp8_eval_req",
                kv_type="value",
            )
            payloads_v.extend(p_list)
        for p in payloads_v:
            total_wire_bytes += p.payload_bytes

        rec_v = reassemble_decode_kv(
            payloads_v, decode_rank=0, plan=plan, expected_request_id="fp8_eval_req", expected_kv_type="value"
        )

        eval_cache.update(
            rec_k.permute(0, 2, 1, 3).contiguous(),
            rec_v.permute(0, 2, 1, 3).contiguous(),
            layer_idx=l,
        )

    # 2. Teacher-forced token-by-token evaluation loop with strict length assertions
    base_nlls = []
    eval_nlls = []
    cos_sims = []
    abs_diffs = []
    top1_matches = []
    top5_overlaps = []

    curr_cache_base = uncompressed_cache
    curr_cache_eval = eval_cache

    for step in range(num_targets):
        # Assert equal cache length before every forward pass
        assert curr_cache_base.get_seq_length() == curr_cache_eval.get_seq_length(), (
            f"Cache length mismatch at step {step}: "
            f"baseline={curr_cache_base.get_seq_length()} vs eval={curr_cache_eval.get_seq_length()}"
        )

        if step == 0:
            # First target predicted directly from producer prefill logits in both branches
            lb = prefill_logits
            le = prefill_logits
        else:
            # Continuation target tokens fed into both caches identically
            prev_token = target_tokens[step - 1].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out_b = model(input_ids=prev_token, past_key_values=curr_cache_base, use_cache=True)
                out_e = model(input_ids=prev_token, past_key_values=curr_cache_eval, use_cache=True)
                lb = out_b.logits[:, -1, :]
                le = out_e.logits[:, -1, :]
                curr_cache_base = out_b.past_key_values
                curr_cache_eval = out_e.past_key_values

        target_tok = target_tokens[step]

        # Log probability of true target token (Negative Log-Likelihood)
        log_prob_b = F.log_softmax(lb, dim=-1)[0, target_tok].item()
        log_prob_e = F.log_softmax(le, dim=-1)[0, target_tok].item()

        base_nlls.append(-log_prob_b)
        eval_nlls.append(-log_prob_e)

        # Logit alignment metrics
        raw_cos = torch.cosine_similarity(lb.flatten(), le.flatten(), dim=0).item()
        cos_sims.append(max(-1.0, min(1.0, float(raw_cos))))
        abs_diffs.append(torch.mean(torch.abs(lb - le)).item())

        # Top-1 token agreement
        tok1_b = torch.argmax(lb, dim=-1).item()
        tok1_e = torch.argmax(le, dim=-1).item()
        top1_matches.append(1 if tok1_b == tok1_e else 0)

        # Top-5 set overlap
        top5_b = set(torch.topk(lb, k=5, dim=-1).indices[0].tolist())
        top5_e = set(torch.topk(le, k=5, dim=-1).indices[0].tolist())
        top5_overlaps.append(len(top5_b.intersection(top5_e)) / 5.0)

    # Continuation steps only (positions where attention attends to reconstructed/quantized cache)
    cont_base_nlls = base_nlls[1:] if num_targets > 1 else base_nlls
    cont_eval_nlls = eval_nlls[1:] if num_targets > 1 else eval_nlls
    cont_cos_sims = cos_sims[1:] if num_targets > 1 else cos_sims
    cont_abs_diffs = abs_diffs[1:] if num_targets > 1 else abs_diffs
    cont_top1_matches = top1_matches[1:] if num_targets > 1 else top1_matches
    cont_top5_overlaps = top5_overlaps[1:] if num_targets > 1 else top5_overlaps

    # Metrics on continuation steps
    cont_n = len(cont_base_nlls)
    cont_mean_nll_base = sum(cont_base_nlls) / cont_n
    cont_mean_nll_eval = sum(cont_eval_nlls) / cont_n
    cont_delta_nll = cont_mean_nll_eval - cont_mean_nll_base

    # Metrics on overall sequence (including boundary step 0)
    all_n = len(base_nlls)
    all_mean_nll_base = sum(base_nlls) / all_n
    all_mean_nll_eval = sum(eval_nlls) / all_n
    all_delta_nll = all_mean_nll_eval - all_mean_nll_base

    total_uncompressed_fp16_bytes = total_uncompressed_fp32_bytes // 2
    comp_ratio_fp32 = total_uncompressed_fp32_bytes / max(1, total_wire_bytes)
    comp_ratio_fp16 = total_uncompressed_fp16_bytes / max(1, total_wire_bytes)
    wire_reduction_fp16_pct = (1.0 - (total_wire_bytes / max(1, total_uncompressed_fp16_bytes))) * 100.0

    return {
        "context_length": prompt_len,
        "target_tokens": num_targets,
        "continuation_tokens": cont_n,
        "compression": {
            "uncompressed_fp32_bytes": total_uncompressed_fp32_bytes,
            "uncompressed_fp16_bytes": total_uncompressed_fp16_bytes,
            "wire_payload_bytes": total_wire_bytes,
            "ratio_vs_fp32": round(comp_ratio_fp32, 2),
            "ratio_vs_fp16": round(comp_ratio_fp16, 2),
            "wire_reduction_vs_fp16_pct": round(wire_reduction_fp16_pct, 1),
        },
        "boundary_step": {
            "target_token": int(target_tokens[0]),
            "delta_nll": 0.0,
            "top1_match": True,
            "cosine_similarity": 1.0,
            "mean_absolute_logit_error": 0.0,
        },
        "continuation_steps": {
            "delta_nll": round(cont_delta_nll, 4),
            "nll_uncompressed": round(cont_mean_nll_base, 4),
            "nll_eval": round(cont_mean_nll_eval, 4),
            "top1_agreement_pct": round((sum(cont_top1_matches) / cont_n) * 100.0, 1),
            "top5_overlap_pct": round((sum(cont_top5_overlaps) / cont_n) * 100.0, 1),
            "mean_logit_cosine_similarity": round(sum(cont_cos_sims) / cont_n, 5),
            "mean_absolute_logit_error": round(sum(cont_abs_diffs) / cont_n, 4),
        },
        "overall_sequence": {
            "delta_nll": round(all_delta_nll, 4),
            "nll_uncompressed": round(all_mean_nll_base, 4),
            "nll_eval": round(all_mean_nll_eval, 4),
            "top1_agreement_pct": round((sum(top1_matches) / all_n) * 100.0, 1),
            "top5_overlap_pct": round((sum(top5_overlaps) / all_n) * 100.0, 1),
            "mean_logit_cosine_similarity": round(sum(cos_sims) / all_n, 5),
            "mean_absolute_logit_error": round(sum(abs_diffs) / all_n, 4),
        },
    }


def run_full_fp8_quality_benchmark(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    corpus: List[Dict[str, str]] = BENCHMARK_CORPUS,
    block_size: int = 16,
    enable_fp8: bool = True,
    thresholds: Dict[str, float] = DEFAULT_FP8_THRESHOLDS,
) -> Dict[str, Any]:
    """Execute evaluation over benchmark corpus with strict threshold checks."""
    console.print(f"[dim]Loading '{model_id}' on CPU for evaluation...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    passage_results = []
    for idx, item in enumerate(corpus):
        res = evaluate_passage_fp8_quality(
            model=model,
            tokenizer=tokenizer,
            context_text=item["context"],
            target_text=item["target"],
            prefill_tp=2,
            decode_tp=1,
            block_size_tokens=block_size,
            enable_fp8=enable_fp8,
        )
        res["category"] = item["category"]
        passage_results.append(res)

    n = len(passage_results)
    # Averages over continuation steps (positions affected by compression)
    avg_cont_delta_nll = sum(p["continuation_steps"]["delta_nll"] for p in passage_results) / n
    avg_cont_top1 = sum(p["continuation_steps"]["top1_agreement_pct"] for p in passage_results) / n
    avg_cont_top5 = sum(p["continuation_steps"]["top5_overlap_pct"] for p in passage_results) / n
    avg_cont_cos = sum(p["continuation_steps"]["mean_logit_cosine_similarity"] for p in passage_results) / n
    avg_cont_mae = sum(p["continuation_steps"]["mean_absolute_logit_error"] for p in passage_results) / n

    # Token-weighted averages across all continuation tokens
    total_cont_tokens = sum(p["continuation_tokens"] for p in passage_results)
    weighted_cont_delta_nll = (
        sum(p["continuation_steps"]["delta_nll"] * p["continuation_tokens"] for p in passage_results) / total_cont_tokens
        if total_cont_tokens > 0 else 0.0
    )
    weighted_cont_top1 = (
        sum(p["continuation_steps"]["top1_agreement_pct"] * p["continuation_tokens"] for p in passage_results) / total_cont_tokens
        if total_cont_tokens > 0 else 0.0
    )
    weighted_cont_top5 = (
        sum(p["continuation_steps"]["top5_overlap_pct"] * p["continuation_tokens"] for p in passage_results) / total_cont_tokens
        if total_cont_tokens > 0 else 0.0
    )
    weighted_cont_cos = (
        sum(p["continuation_steps"]["mean_logit_cosine_similarity"] * p["continuation_tokens"] for p in passage_results) / total_cont_tokens
        if total_cont_tokens > 0 else 1.0
    )
    weighted_cont_mae = (
        sum(p["continuation_steps"]["mean_absolute_logit_error"] * p["continuation_tokens"] for p in passage_results) / total_cont_tokens
        if total_cont_tokens > 0 else 0.0
    )

    # Worst-case metrics across passages
    worst_case_metrics = {
        "max_abs_delta_nll": max(abs(p["continuation_steps"]["delta_nll"]) for p in passage_results),
        "min_top1_agreement_pct": min(p["continuation_steps"]["top1_agreement_pct"] for p in passage_results),
        "min_top5_overlap_pct": min(p["continuation_steps"]["top5_overlap_pct"] for p in passage_results),
        "min_cosine_similarity": min(p["continuation_steps"]["mean_logit_cosine_similarity"] for p in passage_results),
        "max_mean_absolute_logit_error": max(p["continuation_steps"]["mean_absolute_logit_error"] for p in passage_results),
    }

    # Overall averages (including boundary step 0)
    all_n = len(passage_results)
    avg_all_delta_nll = sum(p["overall_sequence"]["delta_nll"] for p in passage_results) / all_n
    avg_all_top1 = sum(p["overall_sequence"]["top1_agreement_pct"] for p in passage_results) / all_n
    avg_all_top5 = sum(p["overall_sequence"]["top5_overlap_pct"] for p in passage_results) / all_n
    avg_all_cos = sum(p["overall_sequence"]["mean_logit_cosine_similarity"] for p in passage_results) / all_n

    # Compression averages
    avg_comp_fp32 = sum(p["compression"]["ratio_vs_fp32"] for p in passage_results) / n
    avg_comp_fp16 = sum(p["compression"]["ratio_vs_fp16"] for p in passage_results) / n
    avg_reduction_fp16 = sum(p["compression"]["wire_reduction_vs_fp16_pct"] for p in passage_results) / n

    # Check against criteria
    criteria_checks = {
        "delta_nll_passed": bool(abs(avg_cont_delta_nll) <= thresholds["max_delta_nll"]),
        "top1_agreement_passed": bool(avg_cont_top1 >= thresholds["min_top1_agreement_pct"]),
        "top5_overlap_passed": bool(avg_cont_top5 >= thresholds["min_top5_overlap_pct"]),
        "cosine_similarity_passed": bool(avg_cont_cos >= thresholds["min_cosine_similarity"]),
        "compression_fp32_passed": bool(avg_comp_fp32 >= thresholds["min_compression_ratio_fp32"]),
        "compression_fp16_passed": bool(avg_comp_fp16 >= thresholds["min_compression_ratio_fp16"]),
    }
    all_passed = bool(all(criteria_checks.values()))

    is_expanded = (corpus == EXPANDED_BENCHMARK_CORPUS)
    eval_scope = "multi_context_length_evaluation" if is_expanded else "diagnostic_smoke_test_curated_passages"
    scope_disclaimer = (
        "Multi-context length evaluation across diverse held-out passages (short to 256+ tokens) "
        "reporting macro-averages, token-weighted aggregates, and worst-case metrics."
        if is_expanded
        else (
            "Diagnostic smoke test across 3 curated passages (48 target tokens) designed to verify "
            "evaluator alignment and detect severe quantization degradation. Does not establish "
            "broad LLM benchmark preservation."
        )
    )

    return {
        "model_id": model_id,
        "evaluation_scope": eval_scope,
        "scope_disclaimer": scope_disclaimer,
        "enable_fp8": enable_fp8,
        "block_size_tokens": block_size,
        "torch_version": torch.__version__,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "dtype": "float32",
        "checkpoint_commit_hash": getattr(model.config, "_commit_hash", None),
        "thresholds": thresholds,
        "criteria_checks": criteria_checks,
        "all_criteria_passed": all_passed,
        "summary": {
            "continuation_steps": {
                "avg_delta_nll": round(avg_cont_delta_nll, 4),
                "avg_top1_agreement_pct": round(avg_cont_top1, 1),
                "avg_top5_overlap_pct": round(avg_cont_top5, 1),
                "avg_cosine_similarity": round(avg_cont_cos, 5),
                "avg_mean_absolute_logit_error": round(avg_cont_mae, 4),
            },
            "token_weighted_continuation_steps": {
                "total_continuation_tokens": total_cont_tokens,
                "weighted_delta_nll": round(weighted_cont_delta_nll, 4),
                "weighted_top1_agreement_pct": round(weighted_cont_top1, 1),
                "weighted_top5_overlap_pct": round(weighted_cont_top5, 1),
                "weighted_cosine_similarity": round(weighted_cont_cos, 5),
                "weighted_mean_absolute_logit_error": round(weighted_cont_mae, 4),
            },
            "worst_case_continuation_steps": {
                "max_abs_delta_nll": round(worst_case_metrics["max_abs_delta_nll"], 4),
                "min_top1_agreement_pct": round(worst_case_metrics["min_top1_agreement_pct"], 1),
                "min_top5_overlap_pct": round(worst_case_metrics["min_top5_overlap_pct"], 1),
                "min_cosine_similarity": round(worst_case_metrics["min_cosine_similarity"], 5),
                "max_mean_absolute_logit_error": round(worst_case_metrics["max_mean_absolute_logit_error"], 4),
            },
            "overall_sequence": {
                "avg_delta_nll": round(avg_all_delta_nll, 4),
                "avg_top1_agreement_pct": round(avg_all_top1, 1),
                "avg_top5_overlap_pct": round(avg_all_top5, 1),
                "avg_cosine_similarity": round(avg_all_cos, 5),
            },
            "compression": {
                "avg_ratio_vs_fp32": round(avg_comp_fp32, 2),
                "avg_ratio_vs_fp16": round(avg_comp_fp16, 2),
                "avg_wire_reduction_vs_fp16_pct": round(avg_reduction_fp16, 1),
            },
        },
        "passages": passage_results,
    }


@click.command()
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID.")
@click.option("--block-size", "-b", default=16, help="FP8 quantization tile block size.")
@click.option("--mode", type=click.Choice(["smoke", "expanded"]), default="smoke", help="Evaluation mode: curated smoke test (3 passages) or expanded multi-context evaluation.")
@click.option("--output", "-o", default=None, help="JSON output file.")
def main(model: str, block_size: int, mode: str, output: Optional[str]):
    """Evaluate FP8 KV Cache Quality on Real Model via Aligned Teacher-Forced NLL."""
    th = DEFAULT_FP8_THRESHOLDS
    corpus = EXPANDED_BENCHMARK_CORPUS if mode == "expanded" else BENCHMARK_CORPUS
    if output is None:
        output = "benchmarks/expanded_fp8_quality.json" if mode == "expanded" else "benchmarks/fp8_real_model_quality.json"

    console.print(
        Panel(
            f"[bold cyan]Real-Model FP8 KV Cache Quality Evaluation ({mode.upper()} MODE)[/bold cyan]\n"
            f"Target Model: {model} | Block Size: {block_size} tokens\n"
            f"Corpus: {len(corpus)} passages spanning diverse domains and context lengths\n"
            f"Executable Pass Criteria:\n"
            f"  • Continuation ΔNLL <= {th['max_delta_nll']} nats\n"
            f"  • Continuation Top-1 Agreement >= {th['min_top1_agreement_pct']}%\n"
            f"  • Continuation Top-5 Overlap >= {th['min_top5_overlap_pct']}%\n"
            f"  • Continuation Cosine Sim >= {th['min_cosine_similarity']}\n"
            f"  • Wire Compression >= {th['min_compression_ratio_fp32']}x vs FP32 (>= {th['min_compression_ratio_fp16']}x vs FP16)",
            border_style="cyan",
        )
    )

    results = run_full_fp8_quality_benchmark(model_id=model, corpus=corpus, block_size=block_size)

    table = Table(
        title="Passage-by-Passage Metrics (Continuation Steps / Positions Attending to Reconstructed Cache)",
        header_style="bold magenta",
    )
    table.add_column("Category", style="cyan", width=25)
    table.add_column("Context", justify="right", style="dim")
    table.add_column("Tokens", justify="right", style="white")
    table.add_column("NLL (Base)", justify="right", style="dim")
    table.add_column("NLL (FP8)", justify="right", style="white")
    table.add_column("Δ NLL", justify="right", style="bold green")
    table.add_column("Top-1 Match", justify="right", style="white")
    table.add_column("Top-5 Overlap", justify="right", style="white")
    table.add_column("Cosine Sim", justify="right", style="yellow")
    table.add_column("Ratio (FP32)", justify="right", style="bold cyan")
    table.add_column("Ratio (FP16)", justify="right", style="bold cyan")

    for p in results["passages"]:
        cs = p["continuation_steps"]
        cp = p["compression"]
        delta_str = f"{cs['delta_nll']:+.4f}"
        table.add_row(
            p["category"],
            f"{p['context_length']}",
            f"{p['continuation_tokens']}",
            f"{cs['nll_uncompressed']:.3f}",
            f"{cs['nll_eval']:.3f}",
            delta_str,
            f"{cs['top1_agreement_pct']:.1f}%",
            f"{cs['top5_overlap_pct']:.1f}%",
            f"{cs['mean_logit_cosine_similarity']:.5f}",
            f"{cp['ratio_vs_fp32']:.2f}x",
            f"{cp['ratio_vs_fp16']:.2f}x",
        )

    console.print(table)

    s_cont = results["summary"]["continuation_steps"]
    s_weighted = results["summary"]["token_weighted_continuation_steps"]
    s_worst = results["summary"]["worst_case_continuation_steps"]
    s_all = results["summary"]["overall_sequence"]
    s_comp = results["summary"]["compression"]
    all_passed = results["all_criteria_passed"]

    summary_content = (
        f"[bold]Continuation Steps Quality (Positions Attending to Reconstructed Cache):[/bold]\n"
        f"  [cyan]Macro-Average (Unweighted):[/cyan]\n"
        f"    Average ΔNLL:              [bold green]{s_cont['avg_delta_nll']:+.4f} nats[/bold green] (Threshold: <= {th['max_delta_nll']})\n"
        f"    Top-1 Greedy Agreement:    [bold cyan]{s_cont['avg_top1_agreement_pct']:.1f}%[/bold cyan] (Threshold: >= {th['min_top1_agreement_pct']}%)\n"
        f"    Top-5 Distribution Overlap: [bold cyan]{s_cont['avg_top5_overlap_pct']:.1f}%[/bold cyan] (Threshold: >= {th['min_top5_overlap_pct']}%)\n"
        f"    Logit Cosine Similarity:   [bold yellow]{s_cont['avg_cosine_similarity']:.5f}[/bold yellow] (Threshold: >= {th['min_cosine_similarity']})\n"
        f"    Mean Absolute Logit Error: [white]{s_cont['avg_mean_absolute_logit_error']:.4f}[/white]\n\n"
        f"  [magenta]Token-Weighted Aggregate ({s_weighted['total_continuation_tokens']} tokens):[/magenta]\n"
        f"    Weighted ΔNLL:             [bold green]{s_weighted['weighted_delta_nll']:+.4f} nats[/bold green]\n"
        f"    Weighted Top-1 Agreement:  [bold cyan]{s_weighted['weighted_top1_agreement_pct']:.1f}%[/bold cyan]\n"
        f"    Weighted Top-5 Overlap:    [bold cyan]{s_weighted['weighted_top5_overlap_pct']:.1f}%[/bold cyan]\n"
        f"    Weighted Cosine Sim:       [bold yellow]{s_weighted['weighted_cosine_similarity']:.5f}[/bold yellow]\n"
        f"    Weighted Logit MAE:        [white]{s_weighted['weighted_mean_absolute_logit_error']:.4f}[/white]\n\n"
        f"  [yellow]Worst-Case Passage Metrics:[/yellow]\n"
        f"    Max |ΔNLL|:                [bold yellow]{s_worst['max_abs_delta_nll']:.4f} nats[/bold yellow]\n"
        f"    Min Top-1 Agreement:       [bold yellow]{s_worst['min_top1_agreement_pct']:.1f}%[/bold yellow]\n"
        f"    Min Top-5 Overlap:         [bold yellow]{s_worst['min_top5_overlap_pct']:.1f}%[/bold yellow]\n"
        f"    Min Cosine Similarity:     [bold yellow]{s_worst['min_cosine_similarity']:.5f}[/bold yellow]\n"
        f"    Max Logit MAE:             [white]{s_worst['max_mean_absolute_logit_error']:.4f}[/white]\n\n"
        f"[bold]Wire Compression Ratios (Explicit Denominators):[/bold]\n"
        f"  Relative to in-memory FP32 tensor cache:   [bold green]{s_comp['avg_ratio_vs_fp32']:.2f}x[/bold green]\n"
        f"  Relative to unpadded FP16/BF16 wire cache: [bold green]{s_comp['avg_ratio_vs_fp16']:.2f}x[/bold green] ({s_comp['avg_wire_reduction_vs_fp16_pct']:.1f}% wire payload reduction)\n\n"
        f"[bold]Acceptance Status:[/bold] {'[bold green]ALL CRITERIA PASSED[/bold green]' if all_passed else '[bold red]FAILED THRESHOLDS[/bold red]'}\n"
    )

    panel = Panel(
        summary_content,
        title=f"FP8 Quality Benchmark Summary ({mode.upper()})",
        border_style="green" if all_passed else "red",
    )
    console.print(panel)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Results written to {out_path}[/dim]")


if __name__ == "__main__":
    main()
