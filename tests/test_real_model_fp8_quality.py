"""Tests for Real-Model FP8 Cache Quality Evaluation.

Verifies that block-scaled FP8 KV cache quantization satisfies quality preservation
criteria against an uncompressed baseline on a real model (Qwen2.5-0.5B-Instruct).
Explicitly checks teacher-forced token NLL, sequence perplexity, top-1 agreement,
and top-5 distributional overlap, while acknowledging that lossy compression does not
produce exact logit equality.
"""

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_real_model_fp8_quality import evaluate_passage_fp8_quality


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def test_real_model_fp8_passage_quality(loaded_model_and_tokenizer):
    """
    Evaluates lossy FP8 cache handoff quality on real text passage.
    Confirms bounded ΔNLL, high top-1/top-5 distributional preservation,
    and wire compression while verifying that exact logit equality is NOT expected.
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
    )

    # 1. Bounded Teacher-Forced NLL change (<= 0.05 nats)
    assert abs(res["delta_nll"]) <= 0.05, f"Expected |ΔNLL| <= 0.05, got {res['delta_nll']}"

    # 2. Token-level agreement and distributional overlap
    assert res["top1_agreement_pct"] >= 80.0, f"Expected top-1 agreement >= 80%, got {res['top1_agreement_pct']}%"
    assert res["top5_overlap_pct"] >= 85.0, f"Expected top-5 overlap >= 85%, got {res['top5_overlap_pct']}%"

    # 3. Logit distribution correlation
    assert res["mean_logit_cosine_similarity"] >= 0.95, (
        f"Expected cosine similarity >= 0.95, got {res['mean_logit_cosine_similarity']}"
    )

    # 4. Confirm lossy nature: exact logits are NOT identical
    assert res["mean_absolute_logit_error"] > 0.0, "Expected non-zero MAE for lossy FP8 compression"

    # 5. Effective wire compression ratio
    assert res["compression_ratio"] >= 2.0, f"Expected compression ratio >= 2.0x, got {res['compression_ratio']}"


def test_fp8_benchmark_artifact_reproducibility():
    """Verify saved benchmark artifact contains recorded environment metadata and meets summary thresholds."""
    artifact_path = Path("benchmarks/fp8_real_model_quality.json")
    assert artifact_path.exists(), "Benchmark artifact missing"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reproducibility metadata
    assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "checkpoint_commit_hash" in data and len(data["checkpoint_commit_hash"]) == 40
    assert "torch_version" in data
    assert "transformers_version" in data
    assert data["dtype"] == "float32"

    # Aggregate metric thresholds
    summary = data["summary"]
    assert abs(summary["avg_delta_nll"]) <= 0.02, f"Expected |avg ΔNLL| <= 0.02, got {summary['avg_delta_nll']}"
    assert summary["avg_top1_agreement_pct"] >= 80.0
    assert summary["avg_top5_overlap_pct"] >= 90.0
    assert summary["avg_cosine_similarity"] >= 0.98
    assert summary["avg_compression_ratio"] >= 2.5
