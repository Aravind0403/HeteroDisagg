"""Command line interface for HeteroDisagg cluster planner and engine adapter (hetero-plan)."""

import json
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from hetero_policy.schema import POPULAR_MODELS, ModelMetadata, ClusterPlacementPlan
from hetero_policy.planner import load_cluster_inventory, plan_cluster_placement
from hetero_policy.vllm_adapter import generate_vllm_command
from hetero_policy.sglang_adapter import generate_sglang_command

console = Console()


def _resolve_model(model_identifier: str) -> ModelMetadata:
    """Resolve model from preset key or construct standard metadata."""
    key = model_identifier.lower()
    if key in POPULAR_MODELS:
        return POPULAR_MODELS[key]
    # Default to 8 KV heads Llama-like config
    return ModelMetadata(
        model_name=model_identifier,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
    )


@click.group()
def main():
    """HeteroDisagg: Hardware-Adaptive Cluster Planner & Engine Configurator."""
    pass


@main.command("plan")
@click.option(
    "--cluster", "-c", default="l40s_and_3090.json", help="Cluster configuration JSON path."
)
@click.option(
    "--model", "-m", default="llama-3-8b", help="Model name (e.g. llama-3-8b, llama-3-70b, qwen-2.5-7b)."
)
def plan_cmd(cluster: str, model: str):
    """Evaluate cluster inventory and determine optimal Prefill vs. Decode role placement."""
    try:
        cluster_name, inventory = load_cluster_inventory(cluster)
        model_meta = _resolve_model(model)
        plan = plan_cluster_placement(inventory, model_meta, cluster_name=cluster_name)
    except Exception as e:
        console.print(f"[bold red]Error planning placement:[/bold red] {e}")
        return

    console.print(
        Panel(
            f"[bold]Target Model:[/bold] {plan.model_name} (KV Heads: {plan.num_kv_heads})\n"
            f"[bold]Total Cluster Cost:[/bold] ${plan.total_cluster_cost_per_hour_usd:.2f} / hr\n\n"
            f"[dim]{plan.rationale}[/dim]",
            title=f"Cluster Placement Plan: {plan.cluster_name}",
            border_style="magenta",
        )
    )

    # Role Assignment Table
    table = Table(title="Hardware Pool Allocations", header_style="bold cyan")
    table.add_column("Role", style="bold yellow")
    table.add_column("Accelerator Tier", style="white")
    table.add_column("GPUs", justify="right", style="cyan")
    table.add_column("Tensor Parallelism", justify="right", style="green")
    table.add_column("Chunk Tokens", justify="right", style="magenta")
    table.add_column("Hourly Cost", justify="right", style="white")

    table.add_row(
        "PREFILL",
        plan.prefill_assignment.device_name,
        str(plan.prefill_assignment.num_gpus),
        f"prefill_tp = {plan.prefill_tp}",
        f"{plan.prefill_assignment.optimal_chunk_size_tokens}",
        f"${plan.prefill_assignment.hourly_cost_usd:.2f}/hr",
    )
    table.add_row(
        "DECODE",
        plan.decode_assignment.device_name,
        str(plan.decode_assignment.num_gpus),
        f"decode_tp = {plan.decode_tp}",
        f"{plan.decode_assignment.optimal_chunk_size_tokens}",
        f"${plan.decode_assignment.hourly_cost_usd:.2f}/hr",
    )

    console.print(table)


@main.command("generate")
@click.option(
    "--cluster", "-c", default="l40s_and_3090.json", help="Cluster configuration JSON path."
)
@click.option(
    "--model", "-m", default="llama-3-8b", help="Model name."
)
@click.option(
    "--engine", "-e", default="vllm", type=click.Choice(["vllm", "sglang"], case_sensitive=False),
    help="Target inference engine."
)
@click.option(
    "--output-dir", "-o", default="launch_scripts", help="Directory to save generated launch scripts."
)
def generate_cmd(cluster: str, model: str, engine: str, output_dir: str):
    """Generate production launch bash scripts for target inference engine."""
    try:
        cluster_name, inventory = load_cluster_inventory(cluster)
        model_meta = _resolve_model(model)
        plan = plan_cluster_placement(inventory, model_meta, cluster_name=cluster_name)
    except Exception as e:
        console.print(f"[bold red]Error generating launch commands:[/bold red] {e}")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if engine.lower() == "vllm":
        prefill_cmd = generate_vllm_command(plan, role="prefill", host="0.0.0.0", port=8000)
        decode_cmd = generate_vllm_command(plan, role="decode", host="0.0.0.0", port=8001)
    else:
        prefill_cmd = generate_sglang_command(plan, role="prefill", host="0.0.0.0", port=30000)
        decode_cmd = generate_sglang_command(plan, role="decode", host="0.0.0.0", port=30001)

    prefill_script = out_path / f"launch_prefill_{engine.lower()}.sh"
    decode_script = out_path / f"launch_decode_{engine.lower()}.sh"

    with open(prefill_script, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n# HeteroDisagg Auto-Generated Prefill Launch Script\n")
        f.write(f"# Target: {plan.prefill_assignment.device_name} (prefill_tp={plan.prefill_tp})\n\n")
        f.write(prefill_cmd + "\n")

    with open(decode_script, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n# HeteroDisagg Auto-Generated Decode Launch Script\n")
        f.write(f"# Target: {plan.decode_assignment.device_name} (decode_tp={plan.decode_tp})\n\n")
        f.write(decode_cmd + "\n")

    console.print(f"[bold green]Successfully generated {engine.upper()} launch scripts in {out_path}/[/bold green]\n")
    console.print(f"[bold cyan]Prefill Command ({prefill_script.name}):[/bold cyan]\n  {prefill_cmd}\n")
    console.print(f"[bold cyan]Decode Command ({decode_script.name}):[/bold cyan]\n  {decode_cmd}\n")


if __name__ == "__main__":
    main()
