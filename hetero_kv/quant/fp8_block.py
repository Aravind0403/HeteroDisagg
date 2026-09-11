"""In-flight FP8 (E4M3) wire quantization with per-block scale factors."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


# Maximum representable finite value in FP8 E4M3 standard
FP8_E4M3_MAX: float = 448.0


@dataclass
class FP8BlockPayload:
    """Compressed wire payload for a KV cache block tensor."""
    quantized_data: torch.Tensor  # 1-byte per element (int8 / float8)
    scale_factors: torch.Tensor   # FP32 scale factor per block
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    block_size_tokens: int

    @property
    def payload_bytes(self) -> int:
        """Total memory in bytes including quantized data and scale factors."""
        return self.quantized_data.nbytes + self.scale_factors.nbytes


def quantize_kv_block_fp8(
    kv_tensor: torch.Tensor,
    block_size_tokens: int = 16,
    eps: float = 1e-8,
) -> FP8BlockPayload:
    """
    Quantize a KV cache tensor from FP16/BF16/FP32 to 8-bit using per-block scaling.

    Expected tensor shapes:
    - PagedAttention: (num_blocks, block_size_tokens, num_heads, head_dim)
    - Linear: (batch_size, seq_len, num_heads, head_dim)
    """
    original_shape = kv_tensor.shape
    original_dtype = kv_tensor.dtype

    # Ensure tensor is contiguous and float format
    tensor_float = kv_tensor.contiguous().float()

    # If tensor is 4D (e.g. [num_blocks, block_size, num_heads, head_dim])
    # and already grouped by block_size, we compute scale per block
    if len(original_shape) == 4 and original_shape[1] == block_size_tokens:
        num_blocks, tokens_per_block, num_heads, head_dim = original_shape
        # Flatten within each block across (tokens, heads, head_dim)
        reshaped = tensor_float.view(num_blocks, -1)
    else:
        # Flatten leading dimensions into tokens
        # Reshape to (num_blocks, block_size_tokens, -1)
        total_elements = tensor_float.numel()
        elements_per_head_token = original_shape[-2] * original_shape[-1]
        elements_per_token_block = block_size_tokens * elements_per_head_token

        # Pad if tokens not perfectly divisible by block_size_tokens
        remainder = total_elements % elements_per_token_block
        if remainder != 0:
            padding_len = elements_per_token_block - remainder
            tensor_flat = torch.cat([tensor_float.view(-1), torch.zeros(padding_len, device=tensor_float.device)])
        else:
            tensor_flat = tensor_float.view(-1)

        num_blocks = tensor_flat.numel() // elements_per_token_block
        reshaped = tensor_flat.view(num_blocks, elements_per_token_block)

    # Step 1: Find max absolute value per block
    max_vals = torch.amax(torch.abs(reshaped), dim=-1, keepdim=True)

    # Step 2: Compute scale factor per block (clamped above eps)
    # scale_factor = max_val / 448.0
    scale_factors = torch.clamp(max_vals / FP8_E4M3_MAX, min=eps)

    # Step 3: Scale values into FP8 range [-448.0, 448.0]
    scaled_values = reshaped / scale_factors
    clamped_values = torch.clamp(scaled_values, -FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Step 4: Map to 8-bit wire representation
    # Check if native torch.float8_e4m3fn is supported on this platform
    try:
        quantized_data = clamped_values.to(torch.float8_e4m3fn)
    except (TypeError, RuntimeError, AttributeError):
        # Portable fallback: Quantize to 8-bit signed integer [-128, 127]
        # (normalized by 448.0 / 127.0)
        quantized_data = torch.round(clamped_values * (127.0 / FP8_E4M3_MAX)).to(torch.int8)

    return FP8BlockPayload(
        quantized_data=quantized_data,
        scale_factors=scale_factors.squeeze(-1),
        original_shape=original_shape,
        original_dtype=original_dtype,
        block_size_tokens=block_size_tokens,
    )


def dequantize_kv_block_fp8(payload: FP8BlockPayload) -> torch.Tensor:
    """
    Dequantize an FP8BlockPayload back to the original floating-point precision.
    """
    quantized = payload.quantized_data
    scale_factors = payload.scale_factors.unsqueeze(-1)

    if quantized.dtype == torch.float8_e4m3fn:
        dequantized = quantized.to(torch.float32) * scale_factors
    else:
        # Dequantize from int8 fallback
        dequantized = (quantized.to(torch.float32) * (FP8_E4M3_MAX / 127.0)) * scale_factors

    # Unpad / reshape back to original shape
    original_numel = 1
    for dim in payload.original_shape:
        original_numel *= dim

    flattened = dequantized.view(-1)[:original_numel]
    return flattened.view(payload.original_shape).to(payload.original_dtype)


def compute_cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """Calculate cosine similarity between two tensors."""
    flat_a = tensor_a.flatten().float()
    flat_b = tensor_b.flatten().float()
    dot = torch.dot(flat_a, flat_b)
    norm_a = torch.linalg.norm(flat_a)
    norm_b = torch.linalg.norm(flat_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(dot / (norm_a * norm_b))


def compute_mean_relative_error(original: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-6) -> float:
    """Calculate mean relative error between original and reconstructed tensors."""
    orig_flat = original.flatten().float()
    recon_flat = reconstructed.flatten().float()
    diff = torch.abs(orig_flat - recon_flat)
    denom = torch.clamp(torch.abs(orig_flat), min=eps)
    return float(torch.mean(diff / denom))
