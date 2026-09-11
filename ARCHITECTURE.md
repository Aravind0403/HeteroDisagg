# HeteroDisagg Architecture Specification

## Overview & Core Thesis
Modern inference engines like vLLM and SGLang are engineered with the implicit assumption that all GPUs in an execution pool are identical:
1. **Symmetric Parallelism:** Tensor Parallelism degree is uniform across all stages.
2. **Fixed Engine Constants:** Parameters such as `max_num_batched_tokens` and chunked prefill budgets are manually configured or set to arbitrary static defaults.
3. **Hardware Agnosticism:** Scheduling loops treat an H100, L40S, and A10G identically, ignoring physical memory bandwidth and compute limitations.

In heterogeneous environments, **Prefill** (compute-bound matrix multiplications) requires high compute throughput (peak TFLOPS), whereas **Decode** (memory-bound token generation) requires high memory bandwidth (GB/s). 

HeteroDisagg bridges this gap through three decoupled, production-grade layers:
* **Layer 1 (ACD & Profiler):** Measures hardware bounds and derives the compute-to-memory inflection point.
* **Layer 2 (HeteroKVConnector):** Manages asymmetric KV cache transfer (`prefill_tp != decode_tp`) with FP8 wire quantization.
* **Layer 3 (Hardware-Adaptive Policy):** Translates measured hardware specs into optimal engine launch configurations.

---

## Phase 1: Accelerator Capability Descriptor (ACD) & Roofline Profiler

### 1. The Roofline Model in Natural Terms
The Roofline model defines the maximum achievable performance of a GPU kernel based on two physical ceilings:
1. **Peak Compute Ceiling:** The maximum theoretical floating-point operations the GPU cores can execute per second (`peak_tflops`).
2. **Peak Memory Bandwidth Ceiling:** The maximum speed at which data can be transferred from high-bandwidth memory (HBM/VRAM) to the compute cores (`memory_bandwidth_gb_s`).

The boundary between memory-bound and compute-bound execution is the **Roofline Inflection Point**:
```text
inflection_point = (peak_tflops * 1000) / memory_bandwidth_gb_s
```
* Measured in **FLOPs per Byte**.
* If a workload's **arithmetic intensity** (FLOPs executed divided by Bytes accessed) is **below** the inflection point, the kernel is **memory-bandwidth-bound**.
* If a workload's arithmetic intensity is **above** the inflection point, the kernel is **compute-bound**.

### 2. Hardware Inflection Points of Target Accelerators (FP16/BF16)

| Accelerator | VRAM Capacity | Memory Bandwidth | Peak FP16 Compute | Roofline Inflection Point | Classification |
|---|---|---|---|---|---|
| **NVIDIA H100 SXM5** | 80 GB HBM3 | 3,350 GB/s | 989 TFLOPS | **295.2 FLOPs/Byte** | Balanced Super-Compute |
| **NVIDIA L40S** | 48 GB GDDR6 | 864 GB/s | 366 TFLOPS | **423.6 FLOPs/Byte** | Extremely Compute-Heavy |
| **NVIDIA RTX 4090** | 24 GB GDDR6X | 1,008 GB/s | 165.2 TFLOPS | **163.9 FLOPs/Byte** | High Bandwidth / Moderate Compute |
| **NVIDIA RTX 3090** | 24 GB GDDR6X | 936 GB/s | 71.2 TFLOPS | **76.1 FLOPs/Byte** | Bandwidth-Rich / Low Compute |
| **NVIDIA A10G** | 24 GB GDDR6 | 600 GB/s | 62.5 TFLOPS | **104.2 FLOPs/Byte** | Bandwidth-Rich / Cost-Effective |

#### Critical System Insight
Notice the contrast between **NVIDIA L40S** and **RTX 4090**:
* The L40S has an inflection point of **423.6 FLOPs/Byte**. Because its memory bandwidth is relatively narrow (864 GB/s) compared to its massive compute (366 TFLOPS), it requires very large prefill chunks before its compute units are saturated.
* The RTX 3090 has an inflection point of only **76.1 FLOPs/Byte**. Small batch sizes and small chunks will already saturate its compute capability.
* Applying an identical default chunk size (e.g. 512 or 2048) across these accelerators leads to severe under-utilization on compute-dense cards or memory thrashing on bandwidth-lean cards.

### 3. Hardware-Adaptive Chunk Sizing Heuristic
To determine the optimal prefill chunk size (`chunk_size_tokens`), the profiler evaluates the arithmetic intensity of the attention and linear projection layers:
```text
arithmetic_intensity_tokens(tokens) = (2 * tokens * hidden_size * num_layers) / 
                                      (bytes_per_weight * total_model_weights + bytes_per_activation * tokens * hidden_size)
```
The optimal chunk size is the smallest number of tokens that brings the operational intensity above the accelerator's `inflection_point`:
```text
optimal_chunk_tokens = minimum tokens where arithmetic_intensity >= inflection_point
```
This is clamped to power-of-two page boundaries aligned with the engine's KV cache block size (typically 16 or 32 tokens).

---

## Phase 2: Asymmetric KV Re-Sharding Connector (`HeteroKVConnector`)

### 1. The Asymmetric Tensor Parallelism Problem
In standard disaggregated serving:
* Prefill workers run with `prefill_tp = 4`.
* Decode workers run with `decode_tp = 4`.
Because the Tensor Parallelism degree matches, rank $i$ in prefill can directly stream its KV cache block to rank $i$ in decode.

In heterogeneous clusters:
* We want **Prefill** on high-compute GPUs with high parallelism (e.g. `prefill_tp = 4` across 4x L40S).
* We want **Decode** on high-bandwidth, cost-effective GPUs with low parallelism (e.g. `decode_tp = 1` or `decode_tp = 2` across RTX 3090s or A10Gs).

Under this setup, the KV head layout on the prefill side does not match the KV head layout on the decode side.

### 2. Grouped Query Attention (GQA) Head Partition Mapping
Consider **Llama 3 8B**, which has:
* 32 Query Heads
* **8 KV Heads**
* Hidden dimension = 4096, Head dimension = 128

#### Scenario: Prefill Tensor Parallelism = 4, Decode Tensor Parallelism = 1
* **Prefill Rank 0:** Owns KV Heads [0, 1]
* **Prefill Rank 1:** Owns KV Heads [2, 3]
* **Prefill Rank 2:** Owns KV Heads [4, 5]
* **Prefill Rank 3:** Owns KV Heads [6, 7]
* **Decode Rank 0:** Requires **all 8 KV Heads [0, 1, 2, 3, 4, 5, 6, 7]**

The connector performs an **All-Gather / Scatter-Gather** over the KV head dimension:
Each prefill rank packages its local head slices, attaches block table indices, and transmits them to the designated decode ranks.

#### Scenario: Prefill Tensor Parallelism = 4, Decode Tensor Parallelism = 2
* **Decode Rank 0:** Needs KV Heads [0, 1, 2, 3] (Supplied by Prefill Rank 0 and Prefill Rank 1)
* **Decode Rank 1:** Needs KV Heads [4, 5, 6, 7] (Supplied by Prefill Rank 2 and Prefill Rank 3)

The connector builds a **bipartite communication plan**:
* Prefill Rank 0 sends to Decode Rank 0
* Prefill Rank 1 sends to Decode Rank 0
* Prefill Rank 2 sends to Decode Rank 1
* Prefill Rank 3 sends to Decode Rank 1
No cross-communication between Prefill Rank 0 and Decode Rank 1 is needed, eliminating 50% of network traffic compared to a naive full all-gather.

### 3. In-Flight FP8 Wire Quantization
To halve the bandwidth consumed across PCIe or network links:
* Prefill KV cache blocks in FP16/BF16 are converted to **FP8 (E4M3)** in-flight before transmission.
* Quantization is computed with **per-block scale factors** (storing one FP32 scale factor per 16 or 32 tokens) to prevent outlier degradation.
* The decode worker receives the FP8 blocks and directly writes them into its PagedAttention FP8 KV cache pool.

---

## Phase 3: Hardware-Adaptive Engine Policy & Heuristic Planner

### 1. Cluster Placement Planner
Given an arbitrary pool of accelerators (e.g. 2x L40S + 2x RTX 3090):
1. **Compute Efficiency Score:** `peak_tflops / cost_per_hour`
2. **Bandwidth Efficiency Score:** `memory_bandwidth_gb_s / cost_per_hour`
3. The planner assigns GPUs with high Compute Efficiency to **Prefill roles** (with higher tensor parallelism), and GPUs with high Bandwidth Efficiency to **Decode roles** (with lower tensor parallelism).

### 2. Non-Invasive Engine Configuration
Instead of maintaining an out-of-tree scheduler fork:
* The policy engine generates validated CLI flags and configuration files for vLLM and SGLang.
* It sets `--max-num-batched-tokens`, `--block-size`, `--kv-cache-dtype`, and `--gpu-memory-utilization` tailored to the specific GPU's hardware signature.
