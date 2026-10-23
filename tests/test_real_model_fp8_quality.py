"""Tests for Real-Model FP8 Cache Quality Evaluation.

Verifies that block-scaled FP8 KV cache quantization satisfies quality preservation
criteria against an uncompressed baseline on a real model (Qwen2.5-0.5B-Instruct).

Test coverage:
1. test_compression_disabled_exact_identity: Proves that disabling compression produces
   mathematically exact logit equality (0.0 error, 1.0 cosine, 100% agreement), confirming
   zero alignment or token-history leakage in the teacher-forcing evaluation harness.
2. test_real_model_fp8_passage_quality: Tests aligned FP8 lossy cache handoff, asserting
   bounded continuation ΔNLL, high top-1/top-5 preservation, and wire compression vs both
   FP32 and FP16 baselines.
3. test_fp8_benchmark_artifact_reproducibility: Verifies saved smoke test benchmark artifact,
   its executable threshold consistency, and explicit compression denominators.
"""

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_real_model_fp8_quality import (
    evaluate_passage_fp8_quality,
    DEFAULT_FP8_THRESHOLDS,
)


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def test_compression_disabled_exact_identity(loaded_model_and_tokenizer):
    """
    Control Test: When enable_fp8 is False, reconstructed cache must produce
    exact identity with the uncompressed baseline (ΔNLL=0, MAE=0, Top-1=100%, Cosine=1.0).
    This confirms that the evaluation harness has zero token-history or position mismatch.
    """
    model, tokenizer = loaded_model_and_tokenizer

    context = "Distributed inference engines disaggregate prefill and decode stages to"
    target = " eliminate memory bandwidth interference and maximize accelerator compute saturation."

    res = evaluate_passage_fp8_quality(
        model=model,
        tokenizer=tokenizer,
        context_text=context,
        target_text=target,
        prefill_tp=2,
        decode_tp=1,
        block_size_tokens=16,
        enable_fp8=False,
    )

    cs = res["continuation_steps"]
    assert cs["delta_nll"] == 0.0, f"Expected 0.0 ΔNLL when uncompressed, got {cs['delta_nll']}"
    assert cs["mean_absolute_logit_error"] == 0.0, f"Expected 0.0 MAE, got {cs['mean_absolute_logit_error']}"
    assert cs["top1_agreement_pct"] == 100.0, f"Expected 100% top-1, got {cs['top1_agreement_pct']}%"
    assert cs["top5_overlap_pct"] == 100.0, f"Expected 100% top-5, got {cs['top5_overlap_pct']}%"
    assert cs["mean_logit_cosine_similarity"] == 1.0, f"Expected 1.0 cosine, got {cs['mean_logit_cosine_similarity']}"


def test_real_model_fp8_passage_quality(loaded_model_and_tokenizer):
    """
    Evaluates lossy FP8 cache handoff quality on real text passage.
    Confirms bounded continuation ΔNLL, high top-1/top-5 distributional preservation,
    and wire compression against both FP32 and FP16 baselines.
    """
    model, tokenizer = loaded_model_and_tokenizer
    th = DEFAULT_FP8_THRESHOLDS

    context = "Distributed inference engines disaggregate prefill and decode stages to"
    target = " eliminate memory bandwidth interference and maximize accelerator compute saturation across heterogeneous server tiers."

    res = evaluate_passage_fp8_quality(
        model=model,
        tokenizer=tokenizer,
        context_text=context,
        target_text=target,
        prefill_tp=2,
        decode_tp=1,
        block_size_tokens=16,
        enable_fp8=True,
    )

    # 1. Boundary step verification (shared prefill logits from producer)
    bs = res["boundary_step"]
    assert bs["delta_nll"] == 0.0
    assert bs["cosine_similarity"] == 1.0
    assert bs["top1_match"] is True

    # 2. Continuation steps verification (positions attending to reconstructed FP8 cache)
    cs = res["continuation_steps"]
    assert abs(cs["delta_nll"]) <= th["max_delta_nll"], f"Expected |ΔNLL| <= {th['max_delta_nll']}, got {cs['delta_nll']}"
    assert cs["top1_agreement_pct"] >= th["min_top1_agreement_pct"], (
        f"Expected top-1 agreement >= {th['min_top1_agreement_pct']}%, got {cs['top1_agreement_pct']}%"
    )
    assert cs["top5_overlap_pct"] >= th["min_top5_overlap_pct"], (
        f"Expected top-5 overlap >= {th['min_top5_overlap_pct']}%, got {cs['top5_overlap_pct']}%"
    )
    assert cs["mean_logit_cosine_similarity"] >= th["min_cosine_similarity"], (
        f"Expected cosine similarity >= {th['min_cosine_similarity']}, got {cs['mean_logit_cosine_similarity']}"
    )

    # 3. Confirm lossy nature: exact logits are NOT identical
    assert cs["mean_absolute_logit_error"] > 0.0, "Expected non-zero MAE for lossy FP8 compression"

    # 4. Explicit dual-denominator compression ratios
    comp = res["compression"]
    assert comp["ratio_vs_fp32"] >= th["min_compression_ratio_fp32"], (
        f"Expected ratio vs FP32 >= {th['min_compression_ratio_fp32']}x, got {comp['ratio_vs_fp32']}x"
    )
    assert comp["ratio_vs_fp16"] >= th["min_compression_ratio_fp16"], (
        f"Expected ratio vs FP16 >= {th['min_compression_ratio_fp16']}x, got {comp['ratio_vs_fp16']}x"
    )


def test_fp8_benchmark_artifact_reproducibility():
    """Verify saved benchmark artifact contains recorded environment metadata and meets executable criteria."""
    artifact_path = Path("benchmarks/fp8_real_model_quality.json")
    assert artifact_path.exists(), "Benchmark artifact missing"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Scope and reproducibility metadata
    assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert data["evaluation_scope"] == "diagnostic_smoke_test_curated_passages"
    assert "scope_disclaimer" in data
    assert "checkpoint_commit_hash" in data and len(data["checkpoint_commit_hash"]) == 40
    assert "torch_version" in data
    assert "transformers_version" in data
    assert data["dtype"] == "float32"

    # Executable threshold consistency
    assert data["all_criteria_passed"] is True
    checks = data["criteria_checks"]
    assert checks["delta_nll_passed"] is True
    assert checks["top1_agreement_passed"] is True
    assert checks["top5_overlap_passed"] is True
    assert checks["cosine_similarity_passed"] is True
    assert checks["compression_fp32_passed"] is True
    assert checks["compression_fp16_passed"] is True

    # Dual compression metrics
    comp = data["summary"]["compression"]
    assert comp["avg_ratio_vs_fp32"] >= 2.5
    assert comp["avg_ratio_vs_fp16"] >= 1.4
    assert comp["avg_wire_reduction_vs_fp16_pct"] >= 30.0
