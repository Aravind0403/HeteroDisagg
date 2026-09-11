"""Hardware interrogation and accelerator specification loader."""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
import torch

from hetero_prof.schema import AcceleratorSpec, ComputeTflops, InterconnectSpec


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "hardware"

REFERENCE_ALIASES: Dict[str, str] = {
    "h100": "nvidia_h100_sxm5.json",
    "h100_sxm": "nvidia_h100_sxm5.json",
    "l40s": "nvidia_l40s.json",
    "rtx4090": "nvidia_rtx_4090.json",
    "4090": "nvidia_rtx_4090.json",
    "rtx3090": "nvidia_rtx_3090.json",
    "3090": "nvidia_rtx_3090.json",
    "a10g": "nvidia_a10g.json",
}


def load_spec_from_file(path_or_filename: str | Path) -> AcceleratorSpec:
    """Load an AcceleratorSpec from a JSON file path or known config filename."""
    path = Path(path_or_filename)
    if not path.exists():
        # Try checking in the configs/hardware directory
        candidate = CONFIG_DIR / path_or_filename
        if candidate.exists():
            path = candidate
        else:
            # Check aliases
            alias_key = str(path_or_filename).lower().replace(" ", "").replace("-", "")
            if alias_key in REFERENCE_ALIASES:
                path = CONFIG_DIR / REFERENCE_ALIASES[alias_key]
            else:
                raise FileNotFoundError(f"Accelerator specification file not found: {path_or_filename}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AcceleratorSpec.model_validate(data)


def get_available_reference_specs() -> List[str]:
    """List all available reference accelerator JSON specs."""
    if not CONFIG_DIR.exists():
        return []
    return [p.name for p in CONFIG_DIR.glob("*.json")]


def detect_device(device_index: int = 0, fallback_spec: str = "nvidia_rtx_4090.json") -> AcceleratorSpec:
    """
    Detect the active NVIDIA GPU at the specified device index.
    If CUDA is unavailable (e.g. running on macOS local development),
    gracefully fall back to the reference specification.
    """
    if not torch.cuda.is_available():
        fallback_path = CONFIG_DIR / fallback_spec
        if fallback_path.exists():
            return load_spec_from_file(fallback_path)
        # Fallback to the first available config file
        configs = list(CONFIG_DIR.glob("*.json"))
        if configs:
            return load_spec_from_file(configs[0])
        raise RuntimeError("CUDA is not available and no reference hardware configs were found.")

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise ValueError(
            f"Device index {device_index} requested, but only {device_count} CUDA device(s) found."
        )

    props = torch.cuda.get_device_properties(device_index)
    device_name = props.name
    major, minor = props.major, props.minor
    compute_capability = f"{major}.{minor}"
    num_sms = props.multi_processor_count
    total_memory_gb = round(props.total_memory / (1024 ** 3), 2)

    # Check if we have a calibrated reference profile matching this device name
    for config_file in CONFIG_DIR.glob("*.json"):
        spec = load_spec_from_file(config_file)
        if spec.device_name.lower() in device_name.lower() or device_name.lower() in spec.device_name.lower():
            # Update memory capacity with actual measured bytes
            spec_dict = spec.model_dump()
            spec_dict["memory_capacity_gb"] = total_memory_gb
            return AcceleratorSpec.model_validate(spec_dict)

    # If not in reference library, estimate from architecture & SM count
    arch_name = "Unknown"
    fp16_tflops = 100.0
    memory_bw_gb_s = 800.0
    bus_width_bits = 384
    has_fp8 = False

    if major == 9:  # Hopper
        arch_name = "Hopper"
        fp16_tflops = round(num_sms * 7.5, 1)
        memory_bw_gb_s = 3000.0
        bus_width_bits = 5120
        has_fp8 = True
    elif major == 8 and minor == 9:  # Ada Lovelace
        arch_name = "Ada Lovelace"
        fp16_tflops = round(num_sms * 1.3, 1)
        memory_bw_gb_s = 1000.0
        bus_width_bits = 384
        has_fp8 = True
    elif major == 8:  # Ampere
        arch_name = "Ampere"
        fp16_tflops = round(num_sms * 0.9, 1)
        memory_bw_gb_s = 900.0
        bus_width_bits = 384
        has_fp8 = False
    elif major == 7:  # Volta / Turing
        arch_name = "Volta/Turing"
        fp16_tflops = round(num_sms * 0.4, 1)
        memory_bw_gb_s = 600.0
        bus_width_bits = 384
        has_fp8 = False

    fp8_tflops = (fp16_tflops * 2.0) if has_fp8 else None

    return AcceleratorSpec(
        device_name=device_name,
        architecture=arch_name,
        compute_capability=compute_capability,
        num_sms=num_sms,
        memory_capacity_gb=total_memory_gb,
        memory_bus_width_bits=bus_width_bits,
        memory_bandwidth_gb_s=memory_bw_gb_s,
        compute_tflops=ComputeTflops(
            fp32=round(fp16_tflops / 2.0, 1),
            fp16=fp16_tflops,
            bf16=fp16_tflops,
            fp8=fp8_tflops,
        ),
        interconnect=InterconnectSpec(
            type="PCIe Gen4 x16",
            bidirectional_bandwidth_gb_s=64.0,
            unidirectional_bandwidth_gb_s=32.0,
        ),
    )
