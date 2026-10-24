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

## Staged Hardware Validation Progression (Steps 0 – 4)

HeteroDisagg follows a strict 5-stage empirical validation ladder on real physical silicon, moving systematically from single-host readiness and physical multi-GPU handoff to controlled transfer studies and full cross-machine networked execution across disparate GPU architectures:

```text
[Step 0: Local Readiness] 
      │ (CPU protocol, fail-closed gates, process/socket lifecycle verification)
      ▼
[Step 1: Single-GPU Validation] 
      │ (NVIDIA Ada/Ampere silicon smoke test, wire codec scaling & reconstruction)
      ▼
[Step 2: Two-GPU Same-Host Handoff] 
      │ (Separate physical GPUs: cuda:0 -> cuda:1, CUDA sync, null-cache ablation)
      ▼
[Step 3: Controlled Transfer Study]
      │ (Prompt length sweeps: S in [64, 1024], uncompressed FP16 vs FP8 vs Adaptive Policy)
      ▼
[Step 4: Cross-Machine Heterogeneous Cloud Handoff]
        (Separate physical nodes & IPs: Ampere SM 8.6 RTX 3090 -> Ada Lovelace SM 8.9 RTX 4090)
```

### Milestone Summary & Artifacts

| Stage | Hardware Testbed | Core Question / Focus | Key Empirical Result | Verified Artifact |
|---|---|---|---|---|
| **Step 0: Local Readiness** | Localhost (CPU) | Protocol validity & fail-closed gates | CPU end-to-end socket handoff; zero process/resource leakage | [`step0_local_readiness.json`](benchmarks/step0_local_readiness.json) |
| **Step 1: Single GPU** | 1x NVIDIA RTX 4070 (Ada CC 8.9) | Wire codec correctness on silicon | Scaling, packing, and reconstruction verified without conflating wire codec with FP8 GEMM | [`step1_single_gpu_validation.json`](benchmarks/step1_single_gpu_validation.json) |
| **Step 2: Two-GPU Handoff** | 2x NVIDIA RTX 4070 (`cuda:0` $\to$ `cuda:1`) | Physical multi-GPU process handoff | Null-cache ablation proves cache consumption; FP16 logit error = 0.0, FP8 cosine sim = 0.9994 | [`step2_two_gpu_handoff.json`](benchmarks/step2_two_gpu_handoff.json) |
| **Step 3: Controlled Study** | 2x NVIDIA RTX 4070 (`cuda:0` $\to$ `cuda:1`) | Does compression reduce handoff latency, and does policy choose appropriately? | **Loopback:** FP16 wins (Policy 100% optimal). <br>**500 Mbps:** FP8 wins at $S=1024$ (**+25.73 ms speedup**, 1.104x). | [`step3_controlled_transfer_study.json`](benchmarks/step3_controlled_transfer_study.json) |
| **Step 4A: Network Protocol & Failure Gates** | Localhost (CPU) | Decoupled endpoints, bounded monotonic TCP framing, two-phase validation ACK sequencing, dynamic dtypes (FP16/BF16/FP32) | 9/9 automated protocol tests pass; zero socket leakage; fail-closed rejection on checkpoint mismatch, truncated frames, server unavailable, and dtype mismatch | [`step4_producer_uncompressed.json`](benchmarks/step4_producer_uncompressed.json) <br> [`step4_consumer_uncompressed.json`](benchmarks/step4_consumer_uncompressed.json) |
| **Step 4B: Cross-Machine Cloud Handoff** | 1x RTX 3090 (Ampere SM 8.6) $\to$ 1x RTX 4090 (Ada SM 8.9) across independent cloud hosts | Disaggregated cache handoff across disparate GPU microarchitectures and un-synchronized monotonic clocks | Null-cache negative control diverged; **100% free-running token match**; **99.999% cosine similarity**; bounded logit error (0.0439 $\le 0.05$); zero orphaned instances | [`step4b_producer_evidence.json`](benchmarks/step4b_producer_evidence.json) <br> [`step4b_consumer_evidence.json`](benchmarks/step4b_consumer_evidence.json) |

### Step 3 Empirical Findings: Compression Crossover

The Step 3 controlled transfer study evaluated `meta-llama/Meta-Llama-3-8B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` cache transport across prompt sequence lengths $S \in [64, 128, 256, 512, 1024]$ on $2\times$ NVIDIA RTX 4070 GPUs under two interconnect regimes:

#### 1. Unrestricted Localhost TCP Socket (Loopback)
* **Transport Architecture:** Independent OS processes communicating over `127.0.0.1` TCP socket with host-memory staging.
* **Result:** Uncompressed FP16 is strictly faster across all sequence lengths. Because loopback transfer is sub-10 ms, the ~60–83 ms CPU/GPU quantization and packing codec overhead cannot be amortized.
* **Policy Verdict:** `AdaptiveCompressionPolicy` achieved **100% optimal selection**, selecting `uncompressed_fp16` across all sequence lengths and incurring 0 ms regret.

#### 2. Calibrated Bandwidth Link (500 Mbps)
* **Result:** Wire transmission time scales with raw payload size ($86.0\text{ KB}$ for $S=64$ up to $6.26\text{ MB}$ for $S=1024$).
* **At $S=64$ and $S=256$:** Codec overhead outweighs small wire savings; uncompressed FP16 remains optimal.
* **At $S=1024$ (The Crossover Frontier):** FP8 wire compression achieves an end-to-end handoff latency speedup of **$25.73\text{ ms}$** ($247.73\text{ ms}$ vs $273.46\text{ ms}$, **$1.104\times$ speedup**), demonstrating that physical compression delivers net latency savings whenever the interconnect bandwidth-delay product crosses the codec compute threshold.

### Step 4 Empirical Findings: Cross-Node Network Handoff & Heterogeneous Microarchitectures

The Step 4 cross-machine deployment evaluated disaggregated serving across physically disparate cloud instances (NVIDIA RTX 3090 Ampere SM 8.6 prefill node $\to$ NVIDIA RTX 4090 Ada Lovelace SM 8.9 decode node) running `Qwen/Qwen2.5-0.5B-Instruct`:

#### 1. Transport Layer Hardening & Two-Phase Synchronization
* **Exact-Length Monotonic Framing:** Solved TCP segmentation false timeouts over WAN and tunnel relays by enforcing exact-byte reads (`recv_exact`) under a single absolute monotonic deadline across both header and payload frames.
* **Two-Phase Handshake Sequencing:** Decoupled physical wire reception (`MAGIC_ACK_TRANSFER` = `b"ACKT"`) from consumer decoding verification (`MAGIC_VAL_PASSED` = `b"VALP"` / `MAGIC_VAL_FAILED` = `b"VALF"`). This ensures network transport RTT is cleanly isolated from local consumer decoding time.
* **Dynamic Dtype Representation:** Replaced static FP16 wire assumptions with dynamic tensor-derived metadata supporting `float16`, `bfloat16`, and `float32` via raw contiguous byte views (`shard.contiguous().view(torch.uint8).cpu().numpy().tobytes()`), strictly validating consumer dtype compatibility before allocation.

#### 2. Cross-Architecture Numerical Equivalence
* **Negative Control Divergence:** An empty/null cache baseline generated `" the code/API/API"`, immediately diverging from uninterrupted baseline generation.
* **Token Identity:** With the transferred cache, the consumer generated `" the use of different"` (`[279, 990, 315, 2155]`), achieving **100% token agreement** with the consumer-local uninterrupted baseline.
* **Microarchitecture Variance:** Across identical hardware (or CPU $\to$ CPU), uncompressed logit error is exactly `0.0`. Across different physical GPU microarchitectures (Ampere SM 8.6 with 82 SMs vs. Ada Lovelace SM 8.9 with 128 SMs), cuBLAS floating-point accumulation ordering causes a minor numerical delta (`max_abs_logit_error = 0.0439` at logit values $\sim 16$, mean absolute error `0.0052`). The output preserved **99.999% cosine similarity** (`0.999990`) and **100% aligned top-1 token agreement**.

#### 3. Decoupled Local Timing
Because independent nodes have unsynchronized hardware clocks, cross-machine monotonic timestamps are never subtracted:
* **Producer Prefill Time (RTX 3090):** `289.80 ms`
* **Wire Transfer ACK RTT:** `2,873.00 ms` (103,815 wire bytes across 48 frames)
* **Total Validation Session RTT:** `3,284.65 ms`
* **Consumer Cache Reconstruction (RTX 4090):** `1.72 ms`
* **Consumer First Decode Step Latency:** `14.03 ms`
* **Consumer Subsequent Step Median:** `14.21 ms`

### Reproducing the Validation Ladder

```bash
# Step 0: Run local readiness predicates and lifecycle cleanup checks (CPU)
python scripts/run_local_step0_checks.py --model Qwen/Qwen2.5-0.5B-Instruct

# Step 1: Run single-GPU silicon smoke and wire codec verification
python scripts/run_step1_single_gpu_validation.py --device cuda:0 --model Qwen/Qwen2.5-0.5B-Instruct

# Step 2: Run two-GPU physical same-host cache handoff with null-cache ablation
python scripts/run_step2_two_gpu_handoff.py --prefill-device cuda:0 --decode-device cuda:1 --model Qwen/Qwen2.5-0.5B-Instruct

# Step 3: Run controlled transfer study (uncompressed vs FP8 vs adaptive policy)
python scripts/run_step3_controlled_transfer_study.py --seq-lens 64,128,256,512,1024 --bandwidth-mbps 500

# Step 4A: Run automated local protocol and failure-injection test suite (9 tests)
python scripts/test_step4_cross_machine_local.py

# Step 4A (Manual): Run standalone consumer and producer endpoints locally
python scripts/run_step4_cross_machine_handoff.py --role consumer --bind-host 127.0.0.1 --bind-port 50051 --device cpu --dtype float32 &
python scripts/run_step4_cross_machine_handoff.py --role producer --dest-host 127.0.0.1 --dest-port 50051 --device cpu --dtype float32

# Step 4B: Run automated remote cloud execution across heterogeneous GPUs (Vast.ai)
python scripts/run_step4b_remote_handoff.py
```

---

## Historical Exploratory Study (v0.1.0 Baseline — September 2026)

Prior to establishing the formal October 2026 5-stage validation ladder, an initial exploratory benchmark evaluated isolated GPU kernel timings and container networking constraints on physical Ampere silicon (Datacenter Host ID `399360`, California, US) running `meta-llama/Meta-Llama-3-8B-Instruct`:

* **Prefill Node:** 1x NVIDIA A100-SXM4-40GB ($0.4889/hr, 312 TFLOPS FP16, 1,555 GB/s HBM2)
* **Decode Node:** 1x NVIDIA RTX 3090-24GB ($0.1822/hr, 142 TFLOPS FP16, 936 GB/s GDDR6X)
* **Total Disaggregated Cluster Cost:** **$0.6711/hr**

### Key Takeaways from the Exploratory Phase:
1. **Architectural Workload-Hardware Match:** The A100 delivered **2.20x higher compute density** for compute-bound prefill, while the RTX 3090 yielded **5,137 GB/s per dollar** (**+61.5% higher memory bandwidth efficiency per dollar**) for memory-bound token decoding.
2. **In-Flight FP8 Wire Quantization:** Halved wire payload volume (128 MB to 64 MB at $C=2$) while preserving **99.98% cosine fidelity** and passing the strict zero-loss threshold ($\Delta\text{PPL} \le 0.0018$).
3. **Container Virtualization Reality:** Rented containers across separate chassis on the same provider experienced multi-tenant NAT/bridging bottlenecks (~0.1 MB/s), demonstrating that physical disaggregation requires dedicated L2/L3 networking or tunnel relays, which directly motivated the robust framing and two-phase transport engineered in **Step 4**.

> [!NOTE]
> For complete historical logs, socket RTT distributions, raw `iperf3` container bridge results, and analytical fabric projections from this exploratory phase, see [archive/SAME_HOST_BENCHMARKS.md](archive/SAME_HOST_BENCHMARKS.md).

---

## Repository Structure

* [ARCHITECTURE.md](ARCHITECTURE.md): Progressive architectural decisions, Roofline inflection formulas, and deep-dive technical specs.
* [TRADEOFFS.md](TRADEOFFS.md): Explicit record of design trade-offs, rejected alternatives, and systems rationale.
* [BENCHMARKS.md](BENCHMARKS.md): Formal validation protocol, statistical standards, quality guardrails ($\Delta\text{PPL}$), and benchmark progression.
* [docs/UPSTREAM_RFC.md](docs/UPSTREAM_RFC.md): Formal proposal for vLLM / SGLang communities.
* [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md): Comprehensive staff-level engineering report.
* [archive/SAME_HOST_BENCHMARKS.md](archive/SAME_HOST_BENCHMARKS.md): Archived September 2026 exploratory study on Vast.ai Host ID `399360`.
* [configs/hardware/](configs/hardware/): Reference hardware specifications for modern accelerators.
* [configs/clusters/](configs/clusters/): Reference heterogeneous cluster inventory profiles.
* `hetero_prof/`: Phase 1 profiler, hardware detector, and roofline analysis engine.
* `hetero_kv/`: Phase 2 asymmetric KV re-sharding connector with in-flight FP8 quantization.
* `hetero_policy/`: Phase 3 hardware-adaptive engine adapter and cluster placement planner.
* `scripts/`: Benchmark, profiling, and staged hardware validation runners (Steps 0–4).
* `benchmarks/`: Empirical JSON validation artifacts, sweep traces, and execution logs.
* `tests/`: Automated unit, integration, and fail-closed milestone acceptance gates.



