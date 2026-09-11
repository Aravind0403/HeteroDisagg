"""Data models for cluster inventory, models, and placement policies."""

from typing import List, Optional
from pydantic import BaseModel, Field
from hetero_prof.schema import AcceleratorSpec


class GPUInventoryItem(BaseModel):
    """An inventory tier of identical GPUs with hourly rental cost."""
    spec: AcceleratorSpec
    count: int = Field(..., description="Number of GPUs in this tier")
    cost_per_hour_usd: float = Field(..., description="Hourly rental cost per GPU in USD")

    @property
    def compute_efficiency_tflops_per_dollar(self) -> float:
        """Peak FP16 TFLOPS per dollar per hour."""
        if self.cost_per_hour_usd <= 0:
            return 0.0
        return self.spec.compute_tflops.fp16 / self.cost_per_hour_usd

    @property
    def bandwidth_efficiency_gb_s_per_dollar(self) -> float:
        """Peak memory bandwidth (GB/s) per dollar per hour."""
        if self.cost_per_hour_usd <= 0:
            return 0.0
        return self.spec.memory_bandwidth_gb_s / self.cost_per_hour_usd


class ModelMetadata(BaseModel):
    """Architecture metadata for target transformer model."""
    model_name: str
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# Predefined models for convenient testing and CLI selection
POPULAR_MODELS = {
    "llama-3-8b": ModelMetadata(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
    ),
    "llama-3-70b": ModelMetadata(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,
    ),
    "qwen-2.5-7b": ModelMetadata(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        hidden_size=3584,
        num_layers=28,
        num_attention_heads=28,
        num_kv_heads=4,
    ),
    "mistral-7b": ModelMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
    ),
}


class RoleAssignment(BaseModel):
    """Specification of role, hardware, parallelism, and chunking for a pool."""
    role: str = Field(..., description="'prefill' or 'decode'")
    device_name: str
    num_gpus: int
    tp_degree: int
    optimal_chunk_size_tokens: int
    hourly_cost_usd: float


class ClusterPlacementPlan(BaseModel):
    """Complete cluster role assignment and configuration plan."""
    cluster_name: str
    model_name: str
    prefill_assignment: RoleAssignment
    decode_assignment: RoleAssignment
    total_cluster_cost_per_hour_usd: float
    prefill_tp: int
    decode_tp: int
    num_kv_heads: int
    rationale: str
