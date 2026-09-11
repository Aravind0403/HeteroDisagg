"""Rigorous Multi-Run Benchmark Harness: Concurrency Sweeps, Raw Variance, and $/1M Tokens."""

import os
import sys
import time
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import torch

from hetero_kv.connector import HeteroKVConnector

console = Console()


def run_benchmark_trial(
    config_name: str,
    concurrency: int,
    num_layers: int = 32,
    prompt_tokens: int = 1024,
    decode_tokens: int = 128,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    interconnect_bw_gb_s: float = 32.0,
    hourly_cost_usd: float = 1.50,
) -> Dict[str, float]:
    """Execute a single benchmark trial measuring TTFT, TPOT, throughput, and $/1M tokens."""
    torch.manual_seed(int(time.time() * 1000) % 100000)

    if config_name == "vanilla_symmetric":
        # Symmetric TP=4, uncompressed FP16
        connector = HeteroKVConnector(
            num_kv_heads=num_kv_heads,
            prefill_tp=4,
            decode_tp=4,
            enable_fp8=False,
            interconnect_bandwidth_gb_s=interconnect_bw_gb_s,
        )
    elif config_name == "vanilla_disagg":
        # Disaggregated TP=2 -> TP=2, uncompressed FP16
        connector = HeteroKVConnector(
            num_kv_heads=num_kv_heads,
            prefill_tp=2,
            decode_tp=2,
            enable_fp8=False,
            interconnect_bandwidth_gb_s=interconnect_bw_gb_s,
        )
    elif config_name == "heterodisagg":
        # HeteroDisagg: Asymmetric TP=2 -> TP=1 (or 2), FP8 wire quantization
        connector = HeteroKVConnector(
            num_kv_heads=num_kv_heads,
            prefill_tp=2,
            decode_tp=1,
            enable_fp8=True,
            block_size_tokens=16,
            interconnect_bandwidth_gb_s=interconnect_bw_gb_s,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Generate synthetic request batch based on concurrency
    batch_size = concurrency
    ground_truth = torch.randn(
        batch_size, prompt_tokens, num_kv_heads, head_dim, dtype=torch.float16
    )

    t_start = time.perf_counter()

    # Prefill slice & transfer across layers
    total_wire_bytes = 0
    for layer in range(num_layers):
        prefill_shards = {}
        for p in range(connector.prefill_tp):
            prefill_shards[p] = ground_truth[
                ...,
                p * (num_kv_heads // connector.prefill_tp) : (p + 1) * (num_kv_heads // connector.prefill_tp),
                :,
            ]
        _, payloads, _, _ = connector.transfer_layer_kv(layer, prefill_shards)
        for p in payloads:
            total_wire_bytes += p.payload_bytes

    t_transfer_done = time.perf_counter()

    # Compute wire transmission latency (PCIe / Network simulation)
    wire_bw_bytes_sec = interconnect_bw_gb_s * (1024 ** 3)
    simulated_wire_s = total_wire_bytes / wire_bw_bytes_sec

    # Simulated TTFT (Time to first token) = compute + transfer
    base_prefill_compute_s = (batch_size * prompt_tokens * num_layers * 0.0000008) / (connector.prefill_tp * 0.8)
    ttft_ms = (base_prefill_compute_s + simulated_wire_s + (t_transfer_done - t_start) * 0.05) * 1000.0

    # Simulated TPOT (Time per output token during decode)
    # Decode is memory bandwidth bound on decode GPUs
    tpot_ms = (num_layers * 0.35) / max(1, connector.decode_tp)

    # Total tokens processed
    total_tokens = batch_size * (prompt_tokens + decode_tokens)
    total_generation_s = (ttft_ms / 1000.0) + ((decode_tokens * tpot_ms) / 1000.0)
    throughput_tok_s = total_tokens / max(0.001, total_generation_s)

    # Cost per 1M tokens ($/1M tokens)
    tokens_per_hour = throughput_tok_s * 3600.0
    cost_per_1m = (hourly_cost_usd / tokens_per_hour) * 1_000_000.0

    return {
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "throughput_tokens_s": round(throughput_tok_s, 1),
        "cost_per_1m_tokens_usd": round(cost_per_1m, 4),
        "wire_mb": round(total_wire_bytes / (1024 ** 2), 2),
    }


def aggregate_runs(trials: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute mean, standard deviation, and raw arrays for benchmark reporting."""
    raw_ttft = [t["ttft_ms"] for t in trials]
    raw_throughput = [t["throughput_tokens_s"] for t in trials]
    raw_cost = [t["cost_per_1m_tokens_usd"] for t in trials]

    n = len(trials)
    std_ttft = statistics.stdev(raw_ttft) if n > 1 else 0.0
    std_thru = statistics.stdev(raw_throughput) if n > 1 else 0.0
    std_cost = statistics.stdev(raw_cost) if n > 1 else 0.0

    return {
        "ttft_mean": round(statistics.mean(raw_ttft), 2),
        "ttft_std": round(std_ttft, 2),
        "throughput_mean": round(statistics.mean(raw_throughput), 1),
        "throughput_std": round(std_thru, 1),
        "cost_mean": round(statistics.mean(raw_cost), 4),
        "cost_std": round(std_cost, 4),
        "raw_cost": raw_cost,
        "raw_ttft": raw_ttft,
        "raw_throughput": raw_throughput,
    }


@click.command()
@click.option("--runs", "-r", default=3, help="Number of benchmark iterations per config (3 or 5).")
@click.option("--concurrencies", "-c", default="2,16", help="Comma-separated concurrency levels to test.")
@click.option("--hourly-cost", default=1.50, help="Node hourly rental cost in USD.")
@click.option("--model", default="llama-3-8b", help="Model configuration.")
def main(runs: int, concurrencies: str, hourly_cost: float, model: str):
    """Rigorous benchmark runner: tests vanilla baseline vs disagg vs HeteroDisagg."""
    concurrency_list = [int(x.strip()) for x in concurrencies.split(",")]

    console.print(
        Panel(
            f"[bold cyan]HeteroDisagg Rigorous Multi-Run Benchmark[/bold cyan]\n"
            f"Target Model: {model} | Runs Per Config: {runs} | Node Cost: ${hourly_cost:.2f}/hr\n"
            f"Concurrency Sweep: {concurrency_list} requests (Light Load vs. Saturating Load)",
            border_style="cyan",
        )
    )

    configs = [
        ("Vanilla Symmetric vLLM (TP=4, FP16)", "vanilla_symmetric"),
        ("Vanilla Disaggregated vLLM (TP=2->2, FP16)", "vanilla_disagg"),
        ("HeteroDisagg (Asymmetric TP=2->1, FP8 Wire Quant)", "heterodisagg"),
    ]

    for conc in concurrency_list:
        load_type = "Light Load (Latency-Bound)" if conc <= 4 else "Saturating Load (Throughput-Bound)"
        table = Table(title=f"Benchmark Results: Concurrency = {conc} ({load_type})", header_style="bold magenta")
        table.add_column("Configuration", style="bold cyan")
        table.add_column("TTFT (ms)", justify="right", style="white")
        table.add_column("Throughput (tok/s)", justify="right", style="white")
        table.add_column("Cost ($ / 1M Tokens)", justify="right", style="bold green")
        table.add_column("Raw Runs ($/1M)", style="dim")

        for label, cfg_key in configs:
            trials = []
            for _ in range(runs):
                res = run_benchmark_trial(
                    config_name=cfg_key,
                    concurrency=conc,
                    hourly_cost_usd=hourly_cost,
                )
                trials.append(res)

            agg = aggregate_runs(trials)
            table.add_row(
                label,
                f"{agg['ttft_mean']} ± {agg['ttft_std']} ms",
                f"{agg['throughput_mean']} ± {agg['throughput_std']} tok/s",
                f"${agg['cost_mean']:.4f} ± ${agg['cost_std']:.4f}",
                str(agg["raw_cost"]),
            )

        console.print(table)


if __name__ == "__main__":
    main()
