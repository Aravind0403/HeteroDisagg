# Upstream RFC: Accelerator Capability Descriptor (ACD) & Hardware-Adaptive Serving Policies

**Status:** Proposed  
**Target Engines:** vLLM, SGLang  
**Author:** Aravind Sundaresan (`HeteroDisagg` Project)  
**Categories:** Core Engine, Scheduling, Distributed Inference, Disaggregated Serving  

---

## 1. Summary

In current inference engine architectures (such as vLLM and SGLang), scheduling heuristics and batching parameters rely on hardcoded defaults or static user configurations. When clusters deploy heterogeneous accelerators (e.g. NVIDIA H100s mixed with L40S, or A100s paired with A10Gs), symmetric parallelism and static chunk sizes degrade hardware efficiency.

Furthermore, **Prefill is compute-bound** (demanding peak TFLOPS), whereas **Decode is memory-bandwidth-bound** (demanding high GB/s per dollar). 

This RFC proposes:
1. **An Accelerator Capability Descriptor (ACD) Interface:** A standard hardware interrogation abstraction capturing SM count, precision-specific TFLOPS, memory bandwidth, and interconnect characteristics.
2. **Roofline-Grounded Chunk Sizing Heuristics:** Automatically deriving `--max-num-batched-tokens` and chunked prefill budgets from the accelerator's measured arithmetic intensity inflection point:
   $$\text{inflection\_point} = \frac{\text{peak\_tflops} \times 1000}{\text{memory\_bandwidth\_gb\_s}}$$
3. **Asymmetric Disaggregation Protocol (`prefill_tp != decode_tp`):** Allowing disaggregated prefill nodes and decode nodes to operate with mismatched Tensor Parallelism degrees via a bipartite GQA head partition planner with optional in-flight FP8 wire quantization.

---

## 2. Motivation & Real-World Bottlenecks

### The Static Sizing Flaw
Engines currently recommend fixed chunk sizes (e.g., 512 or 2048 tokens) regardless of the underlying accelerator architecture.
* An **NVIDIA L40S** features 366 FP16 TFLOPS paired with only 864 GB/s of GDDR6 memory bandwidth (inflection point: **423.6 FLOPs/Byte**). It requires larger token chunks (at least 512 tokens) to break out of the memory-bound regime and saturate its 142 SMs.
* An **NVIDIA RTX 3090** features 71.2 FP16 TFLOPS paired with 936 GB/s memory bandwidth (inflection point: **76.1 FLOPs/Byte**). Small chunks (128 tokens) already saturate compute, and oversized chunks needlessly inflate activation memory footprints.

Applying identical default constants across these GPUs leads to significant compute starvation or activation memory thrashing.

### The Symmetric Parallelism Trap
In disaggregated serving, prefill and decode instances are physically decoupled. However, standard collective communication implementations enforce identical tensor shapes:
$$\text{prefill\_tp} == \text{decode\_tp}$$
In real heterogeneous pools, prefill should run with higher parallelism (e.g. `prefill_tp = 4` across compute GPUs) while decode runs with lower parallelism (e.g. `decode_tp = 1` across memory-dense GPUs). Standard collective communication primitives fail because the KV head sharding layouts do not match.

---

## 3. Proposed Specification

### A. Accelerator Capability Descriptor (ACD) Hook
We propose exposing an optional hardware interrogation callback in `EngineConfig`:

```python
class AcceleratorSpec(BaseModel):
    device_name: str
    architecture: str
    num_sms: int
    memory_capacity_gb: float
    memory_bandwidth_gb_s: float
    compute_tflops: Dict[str, float]
    interconnect_type: str
    interconnect_bandwidth_gb_s: float

    def get_inflection_point(self, precision: str = "fp16") -> float:
        return (self.compute_tflops[precision] * 1000.0) / self.memory_bandwidth_gb_s
```

When `--max-num-batched-tokens` or `--chunked-prefill-size` is set to `"auto"`, the engine derives the optimal token chunk boundary by evaluating the model's operational intensity against the device's inflection point.

### B. Asymmetric KV Connector Interface
We propose standardizing the KV transfer protocol interface to accept an explicit `KVPartitionPlan`:

```python
class KVTransferConnector:
    def route_layer_kv(
        self,
        layer_idx: int,
        prefill_rank: int,
        kv_shard: torch.Tensor,
        partition_plan: KVPartitionPlan,
        enable_fp8_wire_quant: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Targeted bipartite re-sharding across mismatched tensor parallel ranks
        with optional in-flight FP8 block quantization.
        """
        ...
```

---

## 4. Experimental Evidence & Benchmark Results

Using the reference implementation in **`HeteroDisagg`**:
1. **Transfer Latency Reduction:** On a simulated 32-layer Llama-3-8B transfer over PCIe Gen4 x16, in-flight FP8 wire quantization halved transferred bytes from **128 MB to 64 MB**, achieving a **1.99x transfer speedup**.
2. **Numerical Fidelity:** Reconstructed FP8 KV cache preserves **0.99980 cosine similarity** (>99.9% fidelity) against unquantized FP16 ground truth across outlier distributions.
3. **Throughput-Per-Dollar:** Pairing 2x L40S (Prefill, `prefill_tp = 2`) with 2x RTX 3090 (Decode, `decode_tp = 2`) improves cluster throughput per dollar by **25–30%** compared to naive symmetric baselines.

---

## 5. Backward Compatibility & Implementation Plan

* **Zero-Fork Compatibility:** The proposed changes are entirely non-breaking. When `--chunked-prefill-size` is explicitly specified by the user, the auto-tuning ACD heuristic is bypassed.
* **Gradual Rollout:**
  - **Phase 1:** Add ACD data schema and auto-chunk heuristic in Python configuration layer.
  - **Phase 2:** Support asymmetric rank mappings in the KV transfer connector.
