"""Data schemas for Accelerator Capability Descriptor (ACD) and Roofline analysis."""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class ComputeTflops(BaseModel):
    """Peak theoretical compute throughput across precision formats in TFLOPS."""
    fp32: float = Field(..., description="Single-precision FP32 TFLOPS")
    fp16: float = Field(..., description="Half-precision FP16 Tensor Core TFLOPS")
    bf16: float = Field(..., description="Bfloat16 Tensor Core TFLOPS")
    fp8: Optional[float] = Field(None, description="8-bit FP8 Tensor Core TFLOPS (if supported)")


class InterconnectSpec(BaseModel):
    """Host or intra-cluster interconnect specifications."""
    type: str = Field(..., description="Interconnect type, e.g. 'NVLink 4' or 'PCIe Gen4 x16'")
    bidirectional_bandwidth_gb_s: float = Field(..., description="Bidirectional transfer rate in GB/s")
    unidirectional_bandwidth_gb_s: float = Field(..., description="Unidirectional transfer rate in GB/s")


class AcceleratorSpec(BaseModel):
    """Standardized Accelerator Capability Descriptor (ACD)."""
    device_name: str = Field(..., description="Human-readable accelerator name")
    architecture: str = Field(..., description="Microarchitecture family, e.g. Hopper, Ada Lovelace, Ampere")
    compute_capability: str = Field(..., description="CUDA compute capability, e.g. '9.0'")
    num_sms: int = Field(..., description="Number of Streaming Multiprocessors (SMs)")
    memory_capacity_gb: float = Field(..., description="Total device memory capacity in gigabytes")
    memory_bus_width_bits: int = Field(..., description="Memory bus width in bits")
    memory_bandwidth_gb_s: float = Field(..., description="Peak memory bandwidth in gigabytes per second")
    compute_tflops: ComputeTflops = Field(..., description="Peak compute specifications")
    interconnect: InterconnectSpec = Field(..., description="Interconnect specifications")

    def get_inflection_point_flops_per_byte(self, precision: str = "fp16") -> float:
        """
        Calculate the Roofline Inflection Point for a given precision:
        inflection_point = (peak_tflops * 1000) / memory_bandwidth_gb_s
        
        Returns the arithmetic intensity boundary in FLOPs per Byte.
        """
        tflops_val: Optional[float] = None
        precision_lower = precision.lower()
        if precision_lower == "fp32":
            tflops_val = self.compute_tflops.fp32
        elif precision_lower == "fp16":
            tflops_val = self.compute_tflops.fp16
        elif precision_lower == "bf16":
            tflops_val = self.compute_tflops.bf16
        elif precision_lower == "fp8":
            tflops_val = self.compute_tflops.fp8
        else:
            raise ValueError(f"Unsupported precision: {precision}. Choose fp32, fp16, bf16, or fp8.")

        if tflops_val is None or tflops_val <= 0:
            raise ValueError(f"Precision '{precision}' is not supported on {self.device_name}.")

        return (tflops_val * 1000.0) / self.memory_bandwidth_gb_s


class RooflinePoint(BaseModel):
    """Represents an arithmetic intensity evaluation on a specific accelerator."""
    precision: str
    peak_tflops: float
    memory_bandwidth_gb_s: float
    inflection_point_flops_per_byte: float
    workload_operational_intensity: Optional[float] = None
    execution_regime: Optional[str] = Field(
        None, description="'memory_bandwidth_bound' or 'compute_bound'"
    )


class ChunkSizingRecommendation(BaseModel):
    """Hardware-derived optimal chunked prefill parameters."""
    accelerator_name: str
    precision: str
    inflection_point_flops_per_byte: float
    recommended_chunk_size_tokens: int
    minimum_saturation_chunk_tokens: int
    rationale: str
