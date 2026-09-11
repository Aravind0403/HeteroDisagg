# HeteroDisagg Benchmarks & Validation Log

This document tracks benchmark experiments, comparing local simulated benchmarks with actual multi-GPU hardware runs.

---

## Benchmark Metrics & Definitions

* **Time to First Token (TTFT) [ms]:** Latency from request submission to emission of the first generated token (dominated by prefill duration and KV transfer).
* **Time Per Output Token (TPOT) [ms]:** Inter-token latency during the autoregressive generation phase (dominated by decode memory bandwidth).
* **Serving Throughput [tokens/sec]:** Total tokens processed and generated per second across the cluster.
* **Throughput Per Dollar [tokens/sec/$]:** Serving throughput divided by total hourly cluster rental cost.
* **KV Transfer Latency [ms]:** Time required to serialize, quantize, transmit, and re-shard KV cache from prefill ranks to decode ranks.

---

## 1. Local Simulation Benchmarks (Mac / CPU Harness)

The local test harness simulates multi-rank transfers, FP8 quantization overhead, and chunked prefill scheduling across simulated heterogeneous profiles.

### Simulated Cluster: 2x NVIDIA L40S (Prefill, prefill_tp = 2) + 2x NVIDIA RTX 3090 (Decode, decode_tp = 1)
* **Model:** Llama 3 8B (Prompt: 2,048 tokens, Generation: 256 tokens)
* **Simulated Interconnect:** PCIe Gen4 x16 (32 GB/s unidirectional bandwidth)

| Run Configuration | KV Cache Wire Format | Re-Sharding Strategy | Simulated KV Transfer Latency (ms) | Speedup vs. Baseline |
|---|---|---|---|---|
| Static Baseline (Naive All-Gather) | FP16 (2 bytes/elem) | Full Broadcast to all ranks | 32.8 ms | 1.00x |
| Targeted Bipartite Communication | FP16 (2 bytes/elem) | Direct rank-to-rank slices | 16.4 ms | 2.00x |
| HeteroKVConnector (Targeted + Quant) | FP8 E4M3 (1 byte/elem) | Direct rank-to-rank + FP8 | **8.3 ms** | **3.95x** |

*Key Takeaway:* Combining targeted bipartite communication with in-flight FP8 wire quantization reduces simulated KV transfer latency by **~75%** (a 3.95x speedup) compared to naive full all-gather in FP16.

---

## 2. Real Hardware Benchmarks (To Be Executed in Final Phase)

Hardware experiments will be conducted on a rented multi-GPU instance (e.g. 4x RTX 3090/4090 on Vast.ai or mixed GPU cluster).

### Cluster Setup & Emulation Plan
* **Emulated Heterogeneity:** 
  - GPUs 0 & 1: Full power / maximum clock (`nvidia-smi -pm 1 -pl <max_power>`) -> Assigned to **Prefill**.
  - GPUs 2 & 3: Throttled power / limited memory clock (`nvidia-smi -pl <low_power> -lgc <low_clock>`) -> Emulating cost-effective **Decode** nodes.
* **Parallelism:** `prefill_tp = 2` -> `decode_tp = 1` (or `prefill_tp = 4` -> `decode_tp = 1`).

### Hardware Results Table (Template)

| Test ID | Model | Cluster Hardware | Config Strategy | TTFT (ms) | TPOT (ms) | Cluster Throughput (tokens/s) | Throughput / Dollar (tokens/s/$) |
|---|---|---|---|---|---|---|---|
| HW-01 | Llama-3-8B | 4x GPU (Symmetric TP=2) | Naive Default (vLLM) | TBD | TBD | TBD | TBD |
| HW-02 | Llama-3-8B | 4x GPU (Asymmetric TP=2->1) | HeteroDisagg (FP16) | TBD | TBD | TBD | TBD |
| HW-03 | Llama-3-8B | 4x GPU (Asymmetric TP=2->1) | HeteroDisagg (FP8) | TBD | TBD | TBD | TBD |
