"""Tests for device detection and accelerator specification schema."""

import pytest
from pathlib import Path
from hetero_prof.device_detector import (
    load_spec_from_file,
    get_available_reference_specs,
    detect_device,
)
from hetero_prof.schema import AcceleratorSpec


def test_reference_specs_exist():
    specs = get_available_reference_specs()
    assert len(specs) >= 5
    assert "nvidia_h100_sxm5.json" in specs
    assert "nvidia_l40s.json" in specs
    assert "nvidia_rtx_4090.json" in specs
    assert "nvidia_rtx_3090.json" in specs
    assert "nvidia_a10g.json" in specs


def test_load_h100_spec():
    spec = load_spec_from_file("nvidia_h100_sxm5.json")
    assert isinstance(spec, AcceleratorSpec)
    assert "H100" in spec.device_name
    assert spec.num_sms == 132
    assert spec.memory_capacity_gb == 80.0
    assert spec.memory_bandwidth_gb_s == 3350.0
    assert spec.compute_tflops.fp16 == 989.0
    assert spec.compute_tflops.fp8 == 1979.0
    assert "NVLink" in spec.interconnect.type


def test_load_spec_by_alias():
    spec_4090 = load_spec_from_file("4090")
    assert "4090" in spec_4090.device_name
    assert spec_4090.compute_tflops.fp8 is not None

    spec_3090 = load_spec_from_file("3090")
    assert "3090" in spec_3090.device_name
    assert spec_3090.compute_tflops.fp8 is None  # Ampere has no FP8 Tensor Cores


def test_detect_device_fallback():
    # When running locally (e.g. on Mac or non-CUDA machine), detect_device should fall back cleanly
    spec = detect_device(device_index=0)
    assert isinstance(spec, AcceleratorSpec)
    assert spec.memory_bandwidth_gb_s > 0
    assert spec.compute_tflops.fp16 > 0
