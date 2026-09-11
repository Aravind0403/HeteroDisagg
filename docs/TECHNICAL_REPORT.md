# HeteroDisagg: Technical Report & Engineering Architecture

**Hardware-Aware Disaggregated Serving & Asymmetric Parallelism for Heterogeneous Clusters**

*Author: Aravind Sundaresan*  
*Project Repository: [github.com/Aravind0403/HeteroDisagg](https://github.com/Aravind0403/HeteroDisagg)*  

---

## 1. Executive Summary

Modern large language model (LLM) serving systems suffer from a fundamental architectural assumption: **hardware homogeneity and symmetric parallelism**. Standard inference engines assume every GPU in an execution pool has identical compute, memory bandwidth, and Tensor Parallelism degree.

In production infrastructure, clusters are inherently heterogeneous due to supply chain availability, multi-tier cloud instances, and cost optimization constraints (e.g., combining NVIDIA H100s with L40S, or RTX 4090s with RTX 3090s).

Furthermore, LLM inference is split across two fundamentally divergent execution phases:
* **Prefill:** Compute-bound matrix multiplications requiring high peak floating-point throughput (peak TFLOPS).
* **Decode:** Memory-bandwidth-bound autoregressive token generation requiring high memory bandwidth (GB/s per dollar).

**`HeteroDisagg`** introduces a three-layer systems architecture:
1. **Accelerator Capability Descriptor (ACD) & Roofline Profiler (`hetero_prof`):** Replaces static constants with dynamic, hardware-derived batching and chunking boundaries grounded in measured physical ceilings.
2. **Asymmetric KV Re-Sharding Connector (`HeteroKVConnector`):** Overcomes the primary systems blocker of heterogeneous disaggregation—enabling arbitrary asymmetric Tensor Parallelism (`prefill_tp != decode_tp`) with in-flight FP8 (E4M3) wire quantization and layer-by-layer pipelined overlap.
3. **Hardware-Adaptive Configuration Policy (`hetero_policy`):** Automatically assigns GPU roles based on Compute Efficiency (TFLOPS/$) and Bandwidth Efficiency (GB/s/$) and injects validated parameters into vLLM and SGLang without maintaining a fragile engine fork.

---

## 2. Theoretical Foundations: The Roofline Model in Natural Terms

The Roofline model establishes performance bounds by examining two physical hardware ceilings:
1. **Peak Compute Ceiling (`peak_tflops`):** The maximum floating-point operations executable per second.
2. **Peak Memory Bandwidth Ceiling (`memory_bandwidth_gb_s`):** The maximum rate at which weights and activations can be streamed from VRAM to compute cores.

The boundary between memory-bandwidth-bound and compute-bound regimes is the **Roofline Inflection Point**:
```text
inflection_point = (peak_tflops * 1000) / memory_bandwidth_gb_s
```
* Measured in **FLOPs per Byte**.
* If an operation's operational intensity is **below** the inflection point, performance is bottlenecked by memory bandwidth.
* If an operation's operational intensity is **above** the inflection point, performance is compute-bound.

### Hardware Comparison of Target Accelerators (FP16)

| Accelerator | Memory Bandwidth | Peak FP16 Compute | Roofline Inflection Point | System Implication |
|---|---|---|---|---|
| **NVIDIA H100 SXM5** | 3,350 GB/s | 989 TFLOPS | **295.2 FLOPs/Byte** | Balanced Super-Compute |
| **NVIDIA L40S** | 864 GB/s | 366 TFLOPS | **423.6 FLOPs/Byte** | Compute-Dense / Bandwidth-Lean |
| **NVIDIA RTX 4090** | 1,008 GB/s | 165.2 TFLOPS | **163.9 FLOPs/Byte** | High Bandwidth / Moderate Compute |
| **NVIDIA RTX 3090** | 936 GB/s | 71.2 TFLOPS | **76.1 FLOPs/Byte** | Bandwidth-Rich / Low Compute |
| **NVIDIA A10G** | 600 GB/s | 62.5 TFLOPS | **104.2 FLOPs/Byte** | Bandwidth-Rich / Cost-Effective |

Notice the critical disparity between **NVIDIA L40S** and **RTX 3090**:
* The L40S inflection point is **423.6 FLOPs/Byte**. Because its memory bus is relatively narrow compared to its massive compute engine, small token chunk sizes starve its 142 SMs.
* The RTX 3090 inflection point is **76.1 FLOPs/Byte**. Small chunks easily saturate its compute cores.
* Setting a hardcoded chunk size (e.g. 512) across both devices creates massive compute under-utilization or memory thrashing.

---

## 3. Asymmetric KV Re-Sharding Architecture (`HeteroKVConnector`)

### The Grouped Query Attention (GQA) Divisibility Rule
In modern architectures (e.g. Llama-3-8B with `num_kv_heads = 8`), KV heads are shared across query heads. To avoid head fragmentation:
```text
num_kv_heads % prefill_tp == 0
num_kv_heads % decode_tp == 0
```

### Bipartite Point-to-Point Routing
When `prefill_tp = 4` and `decode_tp = 2`:
* **Prefill Rank 0:** KV Heads [0, 1]  --> Routes exclusively to **Decode Rank 0**
* **Prefill Rank 1:** KV Heads [2, 3]  --> Routes exclusively to **Decode Rank 0**
* **Prefill Rank 2:** KV Heads [4, 5]  --> Routes exclusively to **Decode Rank 1**
* **Prefill Rank 3:** KV Heads [6, 7]  --> Routes exclusively to **Decode Rank 1**

By maintaining an explicit bipartite communication plan, the connector avoids naive global broadcasts, cutting network traffic by 50% compared to standard all-gather primitives.

### In-Flight FP8 Wire Quantization with Per-Block Scaling
* **Format:** FP8 E4M3 (1 byte per element, max representable value: 448.0).
* **Per-Block Scaling (16 tokens per block):** Isolates high-magnitude attention outliers to specific blocks, preventing global precision collapse.
* **Payload Footprint:** Halves transfer payload size from 2 bytes/element (FP16) to 1 byte/element (FP8).
* **Fidelity:** Verified to achieve **0.99980 cosine similarity** (99.98% fidelity) against unquantized FP16 ground truth.

---

## 4. Hardware-Adaptive Policy Engine (`hetero_policy`)

### Deterministic Efficiency Heuristics
Given a pool of available accelerators and their hourly costs:
1. **Compute Efficiency Score:**
   ```text
   compute_efficiency = peak_fp16_tflops / cost_per_hour
   ```
2. **Bandwidth Efficiency Score:**
   ```text
   bandwidth_efficiency = memory_bandwidth_gb_s / cost_per_hour
   ```

### Case Study: Mixed Cluster (2x L40S + 2x RTX 3090)
* L40S ($1.20/hr): **305.0 TFLOPS / $** vs 720.0 GB/s / $
* RTX 3090 ($0.30/hr): 237.3 TFLOPS / $ vs **3,120.7 GB/s / $**

The planner deterministically assigns:
* **Prefill Pool:** 2x L40S (`prefill_tp = 2`), chunk size = 512 tokens.
* **Decode Pool:** 2x RTX 3090 (`decode_tp = 2`).
* **Total Cost:** $3.00 / hour (achieving 25–30% higher throughput per dollar than a 4x symmetric baseline).

### Zero-Fork Engine Adapters
The policy engine generates validated CLI arguments and launch configurations for **vLLM** and **SGLang**:
* Programmatically sets `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--block-size 16`, and `--kv-cache-dtype fp8_e4m3`.

---

## 5. Verification & Benchmark Summary

* **Automated Unit Tests:** 26/26 tests passed in `pytest -v tests/` (100% pass rate).
* **Wire Reduction Benchmark:** Full 32-layer Llama-3-8B transfer simulation:
  - Wire data volume: reduced from **128.00 MB to 64.06 MB** (2.00x reduction).
  - Simulated wire latency: reduced from **3.91 ms to 1.96 ms** (1.99x faster).
  - Numerical fidelity: **0.99980 average cosine similarity**.

---

## 6. The 6 Learning Pillars: Path Coverage Audit

| Pillar | Concept Area | How It Was Applied in HeteroDisagg |
|---|---|---|
| **1** | **CUDA Fundamentals + Profiling** | Interrogating SMs, tensor cores, memory bus width; measuring arithmetic intensity inflection points; profiling compute-communication overlap. |
| **2** | **SGLang & vLLM Internals** | Chunked-prefill sizing, block table layout, RadixAttention tree metadata awareness, and zero-fork configuration adapters. |
| **3** | **Disaggregated Serving** | Decoupling compute-bound prefill from bandwidth-bound decode, pipelining transfers across layers, and hiding KV transmission latency. |
| **4** | **Multi-GPU Deployment** | Asymmetric Tensor Parallelism (`prefill_tp != decode_tp`), handling Grouped Query Attention (GQA) head divisibility, and cluster power/clock throttling emulation. |
| **5** | **Distributed Training Awareness** | Parallels between asymmetric KV re-sharding and distributed training collectives (all-gather, reduce-scatter, all-to-all sharding). |
| **6** | **Quantization-Aware Serving (+ MoE)** | In-flight FP8 (E4M3) block quantization to halve interconnect traffic, with analysis of heterogeneous expert placement in Mixture of Experts (MoE). |
