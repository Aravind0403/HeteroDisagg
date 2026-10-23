"""Tests for Gate 2: Real-Model Cache Handoff & Logits Divergence."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from scripts.verify_real_model_cache_handoff import run_cache_handoff_experiment


def test_real_model_cache_handoff_and_logits_equivalence():
    """
    Gate 2 Verification:
    1. Run prefill on Qwen2.5-0.5B.
    2. Extract actual multi-layer K and V caches.
    3. Partition across prefill ranks (TP=2) and reconstruct on decode rank (TP=1).
    4. Decode multiple steps with reconstructed cache.
    5. Assert exactly zero logit divergence and 100% token match against uninterrupted baseline.
    """
    results = run_cache_handoff_experiment(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        prompt="Heterogeneous disaggregated inference allows",
        decode_steps=4,
        prefill_tp=2,
        decode_tp=1,
        device="cpu",
    )

    # 1. Cache tensor exact restoration
    assert results["max_k_cache_diff"] == 0.0, f"Expected 0.0 K diff, got {results['max_k_cache_diff']}"
    assert results["max_v_cache_diff"] == 0.0, f"Expected 0.0 V diff, got {results['max_v_cache_diff']}"

    # 2. Exact token match
    assert results["all_tokens_match"] is True
    assert results["baseline_generated_text"] == results["reconstructed_generated_text"]

    # 3. Exactly zero logit divergence across all steps
    assert results["max_logit_divergence_overall"] <= 1e-7, (
        f"Expected near-zero logit divergence, got {results['max_logit_divergence_overall']}"
    )

    for r in results["step_results"]:
        assert r["tokens_match"] is True
        assert r["max_abs_logit_diff"] <= 1e-7
        assert r["cosine_similarity"] >= 0.9999999
