# HeteroDisagg

**Hardware-Aware Disaggregated Serving & Asymmetric Parallelism for Heterogeneous Clusters**

---

## Executive Summary

Modern LLM inference engines (such as vLLM and SGLang) assume homogeneous hardware and symmetric parallelism: every GPU across prefill and decode is assumed to have identical compute power, memory bandwidth, and Tensor Parallelism degree.

In reality, data center clusters are heterogeneous (e.g. mixing H100s with L40S, A100s with A10Gs, or consumer RTX 4090s with 3090s). Furthermore:
* **Prefill is compute-bound:** demands high peak TFLOPS to crunch prompt tokens rapidly.
* **Decode is memory-bandwidth-bound:** demands high memory bandwidth (GB/s) per dollar to step through generation tokens.

**HeteroDisagg** solves this hardware-workload mismatch through three systems:
1. **Accelerator Capability Descriptor (ACD) & Roofline Profiler:** Ground engine scheduling heuristics in real, measured hardware characteristics instead of hardcoded defaults.
2. **HeteroKVConnector (Asymmetric Tensor Parallelism):** Re-shards KV caches between mismatched prefill and decode ranks (e.g., prefill tensor parallelism = 4 running on compute GPUs transferring to decode tensor parallelism = 1 running on memory GPUs) with in-flight FP8 wire quantization.
3. **Hardware-Adaptive Configuration Policy & Placement Planner:** Programmatically calculates and injects optimal batch sizes, chunked-prefill token budgets, and GPU roles into vLLM/SGLang without maintaining a fragile internal fork.

---

## System Architecture

```text
+-------------------------------------------------------------------------+
|                  HeteroDisagg Architecture Overview                     |
+-------------------------------------------------------------------------+

  [Hardware Discovery]
           |
           v
  +---------------------------------------------------------------------+
  | Phase 1: Accelerator Capability Descriptor (ACD) & Roofline Profiler|
  | - Interrogates SM count, peak TFLOPS, memory bandwidth, interconnect|
  | - Calculates: inflection_point = peak_tflops / memory_bandwidth     |
  | - Determines: optimal_chunk_size_tokens based on hardware limits    |
  | - Emits: accelerator_spec.json                                      |
  +---------------------------------------------------------------------+
           |
           v
  +---------------------------------------------------------------------+
  | Phase 3: Hardware-Adaptive Policy & Heuristic Placement Planner     |
  | - Inspects available cluster inventory (e.g., 2x L40S + 2x RTX 3090)|
  | - Assigns roles: High TFLOPS -> Prefill, High BW/$ -> Decode        |
  | - Configures vLLM / SGLang flags (chunk tokens, max batched tokens) |
  +---------------------------------------------------------------------+
           |
           | Schedules Prefill & Decode Workers
           v
  +---------------------------------------------------------------------+
  | Phase 2: HeteroKVConnector (Asymmetric KV Re-Sharder)               |
  |                                                                     |
  |   Prefill Pool (prefill_tp = 4)                                     |
  |   [Rank 0: Heads 0-1] [Rank 1: Heads 2-3]                           |
  |   [Rank 2: Heads 4-5] [Rank 3: Heads 6-7]                           |
  |                          |                                          |
  |   In-Flight Wire Quant: FP16/BF16 -> FP8 (E4M3 block quantization)  |
  |   Re-sharding & Transfer: All-Gather / Point-to-Point               |
  |                          v                                          |
  |   Decode Pool (decode_tp = 1 or decode_tp = 2)                      |
  |   [Rank 0: All Heads 0-7]                                           |
  +---------------------------------------------------------------------+
```

---

## Quickstart

### Installation
```bash
# Clone the repository
git clone https://github.com/Aravind0403/HeteroDisagg.git
cd HeteroDisagg

# Install in editable mode
pip install -e ".[dev]"
```

### 1. Hardware Profiling & Roofline Analysis (Phase 1)
```bash
# Inspect active GPU (or auto-detect fallback)
hetero-prof inspect

# Inspect a reference hardware spec
hetero-prof inspect --spec configs/hardware/nvidia_h100_sxm5.json

# Calculate Roofline inflection and optimal chunk sizing for Llama 3 8B
hetero-prof roofline --spec configs/hardware/nvidia_l40s.json --dtype fp16

# Get hardware-derived chunk sizing recommendation
hetero-prof recommend --spec configs/hardware/nvidia_l40s.json --dtype fp16

# Export a standardized hardware capability descriptor
hetero-prof export --spec configs/hardware/nvidia_rtx_4090.json --output accelerator_spec.json
```

### 2. Asymmetric KV Transfer & FP8 Wire Quantization (Phase 2)
```bash
# Run end-to-end 32-layer Llama-3-8B transfer comparison (FP16 vs. FP8 wire quantization)
python scripts/demo_hetero_kv.py
```

### 3. Cluster Placement Planning & Zero-Fork Engine Launch (Phase 3)
```bash
# Evaluate a mixed cluster (2x L40S + 2x RTX 3090) for Llama-3-8B
hetero-plan plan --cluster l40s_and_3090.json --model llama-3-8b

# Evaluate a cloud hybrid cluster (2x H100 + 4x A10G) for Llama-3-70B
hetero-plan plan --cluster h100_and_a10g.json --model llama-3-70b

# Generate native vLLM launch scripts (launch_prefill_vllm.sh & launch_decode_vllm.sh)
hetero-plan generate --cluster l40s_and_3090.json --model llama-3-8b --engine vllm

# Generate native SGLang launch scripts
hetero-plan generate --cluster l40s_and_3090.json --model llama-3-8b --engine sglang
```

---

## Repository Structure

* [ARCHITECTURE.md](file:///Users/aravindsundaresan/Development/HeteroDisagg/ARCHITECTURE.md): Progressive architectural decisions and deep-dive technical specs.
* [TRADEOFFS.md](file:///Users/aravindsundaresan/Development/HeteroDisagg/TRADEOFFS.md): Explicit record of design trade-offs, rejected alternatives, and rationale.
* [BENCHMARKS.md](file:///Users/aravindsundaresan/Development/HeteroDisagg/BENCHMARKS.md): Local mock simulations vs. real GPU cluster benchmark runs.
* [docs/UPSTREAM_RFC.md](file:///Users/aravindsundaresan/Development/HeteroDisagg/docs/UPSTREAM_RFC.md): Formal proposal for vLLM / SGLang communities.
* [docs/TECHNICAL_REPORT.md](file:///Users/aravindsundaresan/Development/HeteroDisagg/docs/TECHNICAL_REPORT.md): Comprehensive staff-level engineering report.
* [configs/hardware/](file:///Users/aravindsundaresan/Development/HeteroDisagg/configs/hardware/): Reference hardware specifications for modern accelerators.
* [configs/clusters/](file:///Users/aravindsundaresan/Development/HeteroDisagg/configs/clusters/): Reference heterogeneous cluster inventory profiles.
* `hetero_prof/`: Phase 1 profiler, hardware detector, and roofline analysis engine.
* `hetero_kv/`: Phase 2 asymmetric KV re-sharding connector with in-flight FP8 quantization.
* `hetero_policy/`: Phase 3 hardware-adaptive engine adapter and cluster placement planner.
* `scripts/`: Benchmark and demonstration runners.
* `tests/`: Automated unit and integration test suite.
