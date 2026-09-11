"""Tests for Roofline calculations and hardware-adaptive chunk sizing."""

import pytest
from hetero_prof.device_detector import load_spec_from_file
from hetero_prof.roofline import (
    calculate_inflection_point,
    evaluate_roofline,
    calculate_transformer_operational_intensity,
    recommend_chunk_size,
)


def test_calculate_inflection_point():
    # 100 TFLOPS and 1000 GB/s -> (100 * 1000) / 1000 = 100 FLOPs/Byte
    inflection = calculate_inflection_point(peak_tflops=100.0, memory_bandwidth_gb_s=1000.0)
    assert inflection == 100.0

    with pytest.raises(ValueError):
        calculate_inflection_point(peak_tflops=100.0, memory_bandwidth_gb_s=0.0)


def test_roofline_h100():
    h100 = load_spec_from_file("nvidia_h100_sxm5.json")
    # FP16: 989 TFLOPS, 3350 GB/s -> ~295.2 FLOPs/Byte
    fp16_point = evaluate_roofline(h100, precision="fp16")
    assert pytest.approx(fp16_point.inflection_point_flops_per_byte, rel=1e-2) == 295.2

    # FP8: 1979 TFLOPS, 3350 GB/s -> ~590.7 FLOPs/Byte
    fp8_point = evaluate_roofline(h100, precision="fp8")
    assert pytest.approx(fp8_point.inflection_point_flops_per_byte, rel=1e-2) == 590.7


def test_roofline_unsupported_fp8_on_ampere():
    rtx3090 = load_spec_from_file("nvidia_rtx_3090.json")
    with pytest.raises(ValueError, match="not supported"):
        evaluate_roofline(rtx3090, precision="fp8")


def test_transformer_operational_intensity_monotonic():
    # Operational intensity should increase as chunk tokens increase (amortizing weight loads)
    intensity_128 = calculate_transformer_operational_intensity(num_tokens=128)
    intensity_512 = calculate_transformer_operational_intensity(num_tokens=512)
    intensity_2048 = calculate_transformer_operational_intensity(num_tokens=2048)

    assert intensity_128 < intensity_512 < intensity_2048


def test_recommend_chunk_size_alignment():
    l40s = load_spec_from_file("nvidia_l40s.json")
    rec = recommend_chunk_size(l40s, block_size=16)

    # Chunk tokens must be a multiple of block_size
    assert rec.recommended_chunk_size_tokens % 16 == 0
    assert rec.recommended_chunk_size_tokens >= 512
    assert "L40S" in rec.accelerator_name
    assert rec.inflection_point_flops_per_byte > 400.0  # L40S is compute-dense
