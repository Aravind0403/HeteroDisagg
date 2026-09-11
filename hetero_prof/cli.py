"""Command line interface for HeteroDisagg Roofline Profiler (hetero-prof)."""

import json
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from hetero_prof.device_detector import (
    detect_device,
    load_spec_from_file,
    get_available_reference_specs,
)
from hetero_prof.roofline import (
    evaluate_roofline,
    calculate_transformer_operational_intensity,
    recommend_chunk_size,
)
from hetero_prof.schema import AcceleratorSpec

console = Console()


def _resolve_spec(spec_path: Optional[str], device_index: int) -> AcceleratorSpec:
    """Helper to resolve an AcceleratorSpec from a path or auto-detection."""
    if spec_path:
        return load_spec_from_file(spec_path)
    return detect_device(device_index=device_index)


@click.group()
def main():
    """HeteroDisagg: Hardware-Aware Profiler and Roofline Analyzer."""
    pass


@main.command("list-specs")
def list_specs():
    """List all available reference accelerator hardware profiles."""
    specs = get_available_reference_specs()
    table = Table(title="Available Reference Accelerator Profiles", header_style="bold cyan")
    table.add_column("Filename", style="green")
    table.add_column("Device Model", style="white")

    for s in sorted(specs):
        try:
            loaded = load_spec_from_file(s)
            table.add_row(s, loaded.device_name)
        except Exception:
            table.add_row(s, "Unknown")

    console.print(table)


@main.command("inspect")
@click.option("--spec", "-s", default=None, help="Path or name of reference hardware JSON spec.")
@click.option("--device", "-d", default=0, help="CUDA device index if detecting live GPU.")
def inspect_device(spec: Optional[str], device: int):
    """Inspect accelerator capabilities: SMs, memory bandwidth, compute, and interconnect."""
    try:
        accel = _resolve_spec(spec, device)
    except Exception as e:
        console.print(f"[bold red]Error resolving accelerator spec:[/bold red] {e}")
        return

    table = Table(title=f"Accelerator Specification: {accel.device_name}", header_style="bold magenta")
    table.add_column("Attribute", style="bold cyan", width=30)
    table.add_column("Value", style="white")

    table.add_row("Microarchitecture", accel.architecture)
    table.add_row("Compute Capability", accel.compute_capability)
    table.add_row("Streaming Multiprocessors (SMs)", str(accel.num_sms))
    table.add_row("Memory Capacity", f"{accel.memory_capacity_gb:.1f} GB")
    table.add_row("Memory Bus Width", f"{accel.memory_bus_width_bits} bits")
    table.add_row("Peak Memory Bandwidth", f"{accel.memory_bandwidth_gb_s:.1f} GB/s")
    table.add_row("FP32 Peak Compute", f"{accel.compute_tflops.fp32:.1f} TFLOPS")
    table.add_row("FP16 Peak Compute", f"{accel.compute_tflops.fp16:.1f} TFLOPS")
    table.add_row("BF16 Peak Compute", f"{accel.compute_tflops.bf16:.1f} TFLOPS")
    table.add_row(
        "FP8 Peak Compute",
        f"{accel.compute_tflops.fp8:.1f} TFLOPS" if accel.compute_tflops.fp8 else "Not Supported",
    )
    table.add_row("Interconnect Type", accel.interconnect.type)
    table.add_row(
        "Interconnect Bandwidth (Unidir)",
        f"{accel.interconnect.unidirectional_bandwidth_gb_s:.1f} GB/s",
    )
    table.add_row(
        "Interconnect Bandwidth (Bidir)",
        f"{accel.interconnect.bidirectional_bandwidth_gb_s:.1f} GB/s",
    )

    console.print(table)


@main.command("roofline")
@click.option("--spec", "-s", default=None, help="Path or name of reference hardware JSON spec.")
@click.option("--device", "-d", default=0, help="CUDA device index if detecting live GPU.")
@click.option(
    "--dtype",
    default="fp16",
    type=click.Choice(["fp32", "fp16", "bf16", "fp8"], case_sensitive=False),
    help="Precision format for roofline analysis.",
)
def roofline_cmd(spec: Optional[str], device: int, dtype: str):
    """Calculate Roofline inflection point and evaluate execution regimes."""
    try:
        accel = _resolve_spec(spec, device)
        point = evaluate_roofline(accel, precision=dtype)
    except Exception as e:
        console.print(f"[bold red]Error running roofline evaluation:[/bold red] {e}")
        return

    table = Table(
        title=f"Roofline Analysis: {accel.device_name} ({dtype.upper()})",
        header_style="bold green",
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Peak Compute", f"{point.peak_tflops:.1f} TFLOPS")
    table.add_row("Peak Memory Bandwidth", f"{point.memory_bandwidth_gb_s:.1f} GB/s")
    table.add_row(
        "Inflection Point",
        f"[bold yellow]{point.inflection_point_flops_per_byte:.1f} FLOPs / Byte[/bold yellow]",
    )

    console.print(table)

    # Display operational intensity across chunk sizes
    tokens_table = Table(
        title=f"Transformer Operational Intensity by Chunk Size (Llama-3-8B Config, {dtype.upper()})",
        header_style="bold blue",
    )
    tokens_table.add_column("Chunk Tokens", justify="right", style="cyan")
    tokens_table.add_column("Operational Intensity", justify="right", style="white")
    tokens_table.add_column("Execution Regime", style="bold")

    bytes_per_elem = 1 if dtype.lower() == "fp8" else 2
    for tokens in [128, 256, 512, 1024, 2048, 4096, 8192]:
        intensity = calculate_transformer_operational_intensity(
            tokens, hidden_size=4096, num_layers=32, bytes_per_elem=bytes_per_elem
        )
        if intensity < point.inflection_point_flops_per_byte:
            regime = "[red]Memory-Bandwidth Bound[/red]"
        else:
            regime = "[green]Compute Bound (Saturated)[/green]"
        tokens_table.add_row(f"{tokens}", f"{intensity:.1f} FLOPs/Byte", regime)

    console.print(tokens_table)


@main.command("recommend")
@click.option("--spec", "-s", default=None, help="Path or name of reference hardware JSON spec.")
@click.option("--device", "-d", default=0, help="CUDA device index if detecting live GPU.")
@click.option(
    "--dtype",
    default="fp16",
    type=click.Choice(["fp16", "bf16", "fp8"], case_sensitive=False),
    help="Precision format.",
)
@click.option("--hidden-size", default=4096, help="Transformer hidden size.")
@click.option("--num-layers", default=32, help="Number of transformer layers.")
def recommend_cmd(spec: Optional[str], device: int, dtype: str, hidden_size: int, num_layers: int):
    """Derive hardware-adaptive optimal chunked prefill parameters."""
    try:
        accel = _resolve_spec(spec, device)
        rec = recommend_chunk_size(
            accel, hidden_size=hidden_size, num_layers=num_layers, dtype=dtype
        )
    except Exception as e:
        console.print(f"[bold red]Error calculating chunk recommendation:[/bold red] {e}")
        return

    panel = Panel(
        f"[bold]Accelerator:[/bold] {rec.accelerator_name}\n"
        f"[bold]Precision:[/bold] {rec.precision.upper()}\n"
        f"[bold]Roofline Inflection Point:[/bold] {rec.inflection_point_flops_per_byte} FLOPs/Byte\n\n"
        f"[bold green]Recommended Chunk Size:[/bold green] [bold cyan]{rec.recommended_chunk_size_tokens} tokens[/bold cyan]\n"
        f"[bold]Minimum Saturation Tokens:[/bold] {rec.minimum_saturation_chunk_tokens} tokens\n\n"
        f"[dim]{rec.rationale}[/dim]",
        title="Hardware-Adaptive Chunk Recommendation",
        border_style="green",
    )
    console.print(panel)


@main.command("export")
@click.option("--spec", "-s", default=None, help="Path or name of reference hardware JSON spec.")
@click.option("--device", "-d", default=0, help="CUDA device index if detecting live GPU.")
@click.option("--output", "-o", default="accelerator_spec.json", help="Destination JSON path.")
def export_spec(spec: Optional[str], device: int, output: str):
    """Export validated Accelerator Capability Descriptor to a JSON file."""
    try:
        accel = _resolve_spec(spec, device)
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(accel.model_dump_json(indent=2))
        console.print(
            f"[bold green]Successfully exported ACD for {accel.device_name} to {output_path}[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Error exporting spec:[/bold red] {e}")


if __name__ == "__main__":
    main()
