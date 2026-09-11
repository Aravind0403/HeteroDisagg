"""Quality Guardrail: Evaluate Perplexity and Attention Degradation for FP8 Wire Quantization."""

import os
import sys
import math
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from hetero_kv.quant.fp8_block import quantize_kv_block_fp8, dequantize_kv_block_fp8

console = Console()


def evaluate_attention_divergence(
    batch_size: int = 2,
    seq_len: int = 1024,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size_tokens: int = 16,
    seed: int = 42,
) -> dict:
    """
    Simulate Multi-Head Attention forward pass comparing uncompressed FP16 KV cache
    against HeteroDisagg FP8 wire-quantized KV cache.

    Measures:
    1. Output Attention Tensor Cosine Similarity
    2. Attention Probability Distribution KL-Divergence
    3. Simulated Cross-Entropy Loss & Perplexity Shift (Delta PPL)
    """
    torch.manual_seed(seed)

    # Generate synthetic query, key, value activations with realistic LLM outlier distribution
    # (Gaussian base + 0.1% high-magnitude channel outliers)
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
    k_fp16 = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
    v_fp16 = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

    # Inject realistic attention activation outliers
    outlier_mask = (torch.rand_like(k_fp16) < 0.005)
    k_fp16[outlier_mask] = k_fp16[outlier_mask] * 25.0
    v_fp16[outlier_mask] = v_fp16[outlier_mask] * 25.0

    # Repeat KV heads to match Query heads (Grouped Query Attention expansion)
    gqa_ratio = num_heads // num_kv_heads
    k_fp16_expanded = k_fp16.repeat_interleave(gqa_ratio, dim=1)
    v_fp16_expanded = v_fp16.repeat_interleave(gqa_ratio, dim=1)

    # Baseline Attention Computation (FP16 / FP32 unquantized)
    scale = 1.0 / math.sqrt(head_dim)
    scores_fp16 = torch.matmul(q, k_fp16_expanded.transpose(-1, -2)) * scale
    attn_probs_fp16 = F.softmax(scores_fp16, dim=-1)
    output_fp16 = torch.matmul(attn_probs_fp16, v_fp16_expanded)

    # HeteroDisagg: Quantize K and V to FP8 with per-block scaling
    # Reshape for block quantization: (batch * heads, seq_len, head_dim)
    k_payload = quantize_kv_block_fp8(k_fp16.permute(0, 2, 1, 3), block_size_tokens=block_size_tokens)
    v_payload = quantize_kv_block_fp8(v_fp16.permute(0, 2, 1, 3), block_size_tokens=block_size_tokens)

    # Dequantize (as done when received on the decode GPU)
    k_fp8 = dequantize_kv_block_fp8(k_payload).permute(0, 2, 1, 3)
    v_fp8 = dequantize_kv_block_fp8(v_payload).permute(0, 2, 1, 3)

    k_fp8_expanded = k_fp8.repeat_interleave(gqa_ratio, dim=1)
    v_fp8_expanded = v_fp8.repeat_interleave(gqa_ratio, dim=1)

    scores_fp8 = torch.matmul(q, k_fp8_expanded.transpose(-1, -2)) * scale
    attn_probs_fp8 = F.softmax(scores_fp8, dim=-1)
    output_fp8 = torch.matmul(attn_probs_fp8, v_fp8_expanded)

    # 1. Cosine similarity of attention outputs
    dot = torch.sum(output_fp16 * output_fp8)
    norm_a = torch.norm(output_fp16)
    norm_b = torch.norm(output_fp8)
    cosine_sim = float((dot / (norm_a * norm_b)).item())

    # 2. Mean Absolute Error on attention probabilities
    prob_mae = float(torch.mean(torch.abs(attn_probs_fp16 - attn_probs_fp8)).item())

    # 3. KL Divergence between attention distributions (information loss)
    kl_div = float(F.kl_div(
        F.log_softmax(scores_fp8, dim=-1),
        attn_probs_fp16,
        reduction="batchmean",
    ).item())

    # 4. Simulated Perplexity Impact
    # Synthetic cross-entropy over vocabulary projection
    vocab_size = 32000
    proj_weight = torch.randn(vocab_size, head_dim)
    logits_fp16 = torch.matmul(output_fp16.squeeze(2), proj_weight.t())
    logits_fp8 = torch.matmul(output_fp8.squeeze(2), proj_weight.t())

    target = torch.randint(0, vocab_size, (batch_size, num_heads))
    loss_fp16 = float(F.cross_entropy(logits_fp16.view(-1, vocab_size), target.view(-1)).item())
    loss_fp8 = float(F.cross_entropy(logits_fp8.view(-1, vocab_size), target.view(-1)).item())

    ppl_fp16 = math.exp(min(loss_fp16, 20.0))
    ppl_fp8 = math.exp(min(loss_fp8, 20.0))
    delta_ppl = ppl_fp8 - ppl_fp16

    return {
        "seq_len": seq_len,
        "cosine_similarity": cosine_sim,
        "prob_mae": prob_mae,
        "kl_divergence": kl_div,
        "loss_fp16": loss_fp16,
        "loss_fp8": loss_fp8,
        "ppl_fp16": ppl_fp16,
        "ppl_fp8": ppl_fp8,
        "delta_ppl": delta_ppl,
    }


def main():
    console.print(
        Panel(
            "[bold green]Quality Guardrail: FP8 Wire Quantization Accuracy & Perplexity Audit[/bold green]\n"
            "Testing Attention degradation across token sequence lengths with injected channel outliers.\n"
            "Architecture: Llama-3-8B Config (32 Query Heads, 8 KV Heads, GQA 4:1, Head Dim 128)",
            border_style="green",
        )
    )

    table = Table(title="Perplexity & Attention Fidelity Evaluation", header_style="bold cyan")
    table.add_column("Sequence Length", justify="right", style="cyan")
    table.add_column("Cosine Sim", justify="right", style="white")
    table.add_column("Attn Prob MAE", justify="right", style="white")
    table.add_column("KL Divergence", justify="right", style="white")
    table.add_column("FP16 Loss", justify="right", style="magenta")
    table.add_column("FP8 Loss", justify="right", style="magenta")
    table.add_column("Δ Perplexity", justify="right", style="bold green")
    table.add_column("RFC Quality Status", style="bold")

    for seq_len in [512, 1024, 2048, 4096]:
        metrics = evaluate_attention_divergence(seq_len=seq_len)
        status = "[green]PASS (Δ PPL < 0.05)[/green]" if abs(metrics["delta_ppl"]) < 0.05 else "[yellow]WATCH[/yellow]"
        table.add_row(
            f"{seq_len} tokens",
            f"{metrics['cosine_similarity']:.5f}",
            f"{metrics['prob_mae']:.6f}",
            f"{metrics['kl_divergence']:.6f}",
            f"{metrics['loss_fp16']:.4f}",
            f"{metrics['loss_fp8']:.4f}",
            f"{metrics['delta_ppl']:+.4f}",
            status,
        )

    console.print(table)


if __name__ == "__main__":
    main()
