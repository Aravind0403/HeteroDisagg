"""Tests for in-flight FP8 wire quantization and dequantization."""

import pytest
import torch
from hetero_kv.quant.fp8_block import (
    quantize_kv_block_fp8,
    dequantize_kv_block_fp8,
    compute_cosine_similarity,
    compute_mean_relative_error,
)


def test_quantize_dequantize_basic():
    torch.manual_seed(42)
    # Shape: (4 blocks, 16 tokens, 8 heads, 128 dim)
    original = torch.randn(4, 16, 8, 128, dtype=torch.float32)

    payload = quantize_kv_block_fp8(original, block_size_tokens=16)
    restored = dequantize_kv_block_fp8(payload)

    assert restored.shape == original.shape
    assert restored.dtype == original.dtype

    # Check high fidelity (cosine similarity > 0.999)
    sim = compute_cosine_similarity(original, restored)
    assert sim > 0.999


def test_wire_payload_reduction():
    # 1024 tokens of FP16 (2 bytes each)
    original = torch.randn(1, 1024, 8, 128, dtype=torch.float16)
    uncompressed_bytes = original.nbytes

    payload = quantize_kv_block_fp8(original, block_size_tokens=16)
    compressed_bytes = payload.payload_bytes

    # Wire payload should be roughly ~50% of the original uncompressed bytes
    ratio = uncompressed_bytes / compressed_bytes
    assert 1.95 <= ratio <= 2.05


def test_outlier_resilience():
    # Create tensor with a few large outliers (e.g. 100.0)
    original = torch.randn(2, 16, 8, 128, dtype=torch.float32) * 0.1
    # Inject large outlier in block 0
    original[0, 5, 2, 40] = 120.0

    payload = quantize_kv_block_fp8(original, block_size_tokens=16)
    restored = dequantize_kv_block_fp8(payload)

    # Cosine similarity must remain high even with massive outliers
    sim = compute_cosine_similarity(original, restored)
    assert sim > 0.995

    # Check that block 1 was NOT squashed to zero by block 0's outlier
    block_1_orig = original[1]
    block_1_rest = restored[1]
    block_1_sim = compute_cosine_similarity(block_1_orig, block_1_rest)
    assert block_1_sim > 0.999
