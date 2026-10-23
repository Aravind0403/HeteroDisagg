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
        assert r["cosine_similarity"] >= 0.999999


def test_reject_unsupported_decode_tp():
    """Script must reject decode_tp != 1 with explicit explanatory error."""
    with pytest.raises(ValueError, match="Unsupported decode_tp=2"):
        run_cache_handoff_experiment(decode_tp=2)


def test_real_model_cache_corruption_negative_control():
    """
    Negative Control:
    Zeroing or corrupting the reconstructed cache must cause large logit divergence
    (confirming that model decoding is sensitive to and actually consumes the cache).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    prompt = "Heterogeneous disaggregated inference allows"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out_prefill = model(input_ids=inputs.input_ids, use_cache=True)
        intact_cache = out_prefill.past_key_values
        next_tok = torch.argmax(out_prefill.logits[:, -1, :], dim=-1, keepdim=True)

        # Baseline 1-step decode
        out_base = model(input_ids=next_tok, past_key_values=intact_cache, use_cache=True)
        logits_base = out_base.logits[:, -1, :]

        # Corrupted cache (zeroed keys and values)
        corrupted_cache = DynamicCache()
        for l in range(len(intact_cache.layers)):
            zero_k = torch.zeros_like(intact_cache.layers[l].keys)
            zero_v = torch.zeros_like(intact_cache.layers[l].values)
            corrupted_cache.update(zero_k, zero_v, layer_idx=l)

        out_corrupt = model(input_ids=next_tok, past_key_values=corrupted_cache, use_cache=True)
        logits_corrupt = out_corrupt.logits[:, -1, :]

        max_divergence = torch.max(torch.abs(logits_base - logits_corrupt)).item()
        # Verify that corrupted cache causes massive logit change (> 15.0)
        assert max_divergence > 15.0, f"Expected large logit divergence on corrupted cache, got {max_divergence}"

