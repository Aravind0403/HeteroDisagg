# Empirical Validation: Heterogeneous Disaggregation (A100 Prefill $\to$ RTX 3090 Decode)

This document records the empirical network baseline, isolated GPU compute benchmarks, live wire transfer measurements, and cross-fabric analytical projections on genuine physical hardware rented under Vast.ai Host ID `399360` (California, US).

---

## 1. Hardware & Cluster Architecture

Two physical nodes were provisioned under a single datacenter provider (`host_id=399360`, California, US):

| Node Role | Silicon Specification | VRAM | Memory Bandwidth | Compute Peak | Host / Machine ID | Internal IP | Direct Endpoint | Cost ($/hr) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefill Tier** | **1x NVIDIA A100-SXM4-40GB** (Ampere CC 8.0) | 40 GB HBM2 | 1,555 GB/s | 312 TFLOPS (FP16) | `host: 399360` <br>`mach: 148097` | `192.168.100.212`<br>`overlay: 100.100.185.98` | `38.49.42.120:54296` | **$0.4889 / hr** |
| **Decode Tier** | **1x NVIDIA RTX 3090-24GB** (Ampere CC 8.6) | 24 GB GDDR6X | 936 GB/s | 142 TFLOPS (FP16) | `host: 399360` <br>`mach: 135788` | `192.168.100.138`<br>`overlay: 100.81.85.47` | `154.64.230.67:26514` | **$0.1822 / hr** |
| **Total Cluster** | **Heterogeneous Disaggregated Pair** | **64 GB** | **2,491 GB/s** | **454 TFLOPS** | **Same Datacenter Facility** | — | — | **$0.6711 / hr** |

### Silicon Disparity & Architectural Fit:
* **Compute Ratio (Prefill Advantage):** A100 SXM4 delivers **2.20x** higher FP16 tensor core throughput (312 TFLOPS vs 142 TFLOPS), critical for compute-bound quadratic prefill context processing.
* **Memory Bandwidth Efficiency (Decode Advantage):** RTX 3090 delivers 936 GB/s at $0.1822/hr = **5,137 GB/s per dollar**, compared to A100's 1,555 GB/s at $0.4889/hr = **3,180 GB/s per dollar** (**+61.5% memory bandwidth efficiency on consumer silicon**).

---

## 2. Network Baseline & Virtualization Reality (Empirical Ground Truth)

All network baseline measurements were captured directly between the A100 prefill container and RTX 3090 decode container.

### A. TCP Handshake Round-Trip Latency (Overlay Network: `100.81.85.47:22`)
Measured via high-resolution socket timing (`time.perf_counter()` over 10 consecutive connection attempts):

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Sample Count ($N$)** | 10 | trials |
| **Minimum RTT** | **124.01** | ms |
| **Average RTT** | **252.32** | ms |
| **Maximum RTT** | **513.71** | ms |
| **Sample Standard Deviation** | **151.79** | ms |

**Raw Handshake RTT Array (ms):**
```json
[126.02, 124.17, 486.01, 371.75, 188.44, 513.71, 236.81, 124.01, 124.16, 228.08]
```

### B. Direct Host NAT Latency (`154.64.230.67:26514`)
Measured across the external direct port mapping of Host `399360`:

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Sample Count ($N$)** | 10 | trials |
| **Minimum RTT** | **163.51** | ms |
| **Average RTT** | **905.65** | ms |
| **Maximum RTT** | **4304.03** | ms |
| **Sample Standard Deviation** | **1290.35** | ms |

**Raw Direct NAT RTT Array (ms):**
```json
[165.22, 4304.03, 163.51, 191.32, 191.06, 1208.63, 191.03, 1224.05, 191.69, 1226.00]
```

### C. Raw `iperf3` Throughput (Direct NAT Port `26518`, Zero SSH Tunneling)
Measured directly between A100 (`machine: 148097`) and RTX 3090 (`machine: 135788`) over published port `154.64.230.67:26518`:
* **Receiver Bitrate:** **831,242 bps (0.831 Mbps / 0.104 MB/s)**
* **TCP Retransmissions:** **257 retransmits** in 5 seconds
* **Mean Measured RTT:** **181.89 ms** (min 178.9 ms, max 191.2 ms)
* **Congestion Algorithm:** Linux TCP CUBIC

### D. Direct Raw Python Socket Transfer (Direct NAT Port `26576`, 1 MB Slice)
* **Trial 1:** 10.873 s (0.0920 MB/s)
* **Trial 2:** 15.074 s (0.0663 MB/s)
* **Trial 3:** 15.701 s (0.0637 MB/s)
* **Mean Transfer Time:** **13.883 s (0.0720 MB/s)**

### E. Architectural Finding on Multi-Tenant Cloud Virtualization:
* Even when two instances share the exact same provider (`host_id=399360`, California, US), they occupy distinct physical server chassis (`machine_id: 148097` vs `135788`).
* Vast.ai executes tenant containers inside isolated Docker network namespaces without flat inter-chassis L2 bridges or physical NIC passthrough.
* **The ~0.1 MB/s throughput is NOT an SSH tunnel artifact:** testing over directly published TCP NAT ports yields the exact same ~0.1 MB/s throughput ($0.104\text{ MB/s}$ on `iperf3`, $0.072\text{ MB/s}$ across 3 direct socket trials).
* **Causal Isolation:** The 257 TCP retransmissions in 5 seconds and low throughput are **consistent with either host-level NAT/bridging overhead or active traffic shaping / QoS policing between tenant containers on separate chassis — we did not isolate which**.
* Disaggregated serving in enterprise settings requires dedicated intra-datacenter networks (10 GbE / 100 GbE RoCEv2 / InfiniBand), analyzed analytically in Section 4.

---

## 3. Decoupled Empirical Benchmarks: Isolated GPU Kernel Timing

To remove network distortion and measure pure compute overhead, the FP8 quantization and asymmetric resharding kernels were wrapped in `torch.cuda.Event(enable_timing=True)` on the A100 SXM4 GPU.

* **Target Model:** `meta-llama/Meta-Llama-3-8B-Instruct` (32 layers, 8 KV heads, $d_{\text{head}}=128$)
* **Workload Dimensions:** Prompt Length = 1,024 tokens, Decode Length = 128 tokens
* **Statistical Rigor:** 3 independent runs per configuration with cooldown intervals; reporting **Mean ± StdDev** and **Raw Run Arrays**.

### A. Concurrency = 2 (Batch Size = 2, Latency-Bound Regime)
| Configuration | Isolated GPU Kernel (ms) | Raw Runs (ms) | Wire Payload Volume | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla Symmetric vLLM (TP=4, FP16)\*** | 7.48 ± 0.93 ms | `[8.55, 6.97, 6.93]` | 0.00 MB (Collocated) | — |
| **Vanilla Disaggregated vLLM (TP=2$\to$2, FP16)** | 4.52 ± 0.10 ms | `[4.58, 4.58, 4.40]` | 128.00 MB | 1.00x (Baseline) |
| **HeteroDisagg (Asymmetric TP=2$\to$1, FP8 Quant)** | 15.49 ± 0.33 ms | `[15.83, 15.45, 15.17]` | 64.03 MB | **2.00x reduction** |

### B. Concurrency = 16 (Batch Size = 16, Throughput-Bound Regime)
| Configuration | Isolated GPU Kernel (ms) | Raw Runs (ms) | Wire Payload Volume | Compression Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla Symmetric vLLM (TP=4, FP16)\*** | 8.99 ± 0.84 ms | `[9.96, 8.44, 8.56]` | 0.00 MB (Collocated) | — |
| **Vanilla Disaggregated vLLM (TP=2$\to$2, FP16)** | 7.24 ± 0.06 ms | `[7.21, 7.21, 7.31]` | 1024.00 MB (1.00 GB) | 1.00x (Baseline) |
| **HeteroDisagg (Asymmetric TP=2$\to$1, FP8 Quant)** | 38.88 ± 1.40 ms | `[39.79, 39.59, 37.27]` | 512.25 MB (0.50 GB) | **2.00x reduction** |

> [!NOTE]
> **\* Comparability Disclosure on "Vanilla Symmetric":** In standard collocated vLLM serving, prefill directly hands off KV pointers in GPU memory with zero repackaging. In this benchmark, the "Vanilla Symmetric" row passed symmetric TP=4 through the harness's partition plan loop (slicing across 4 prefill ranks and validating against 4 decode ranks in PyTorch). It is timed here solely to verify harness behavior across rank counts, **not** as a representative collocated forward pass.
>
> **The Real Disaggregated Compute Delta:** Comparing Vanilla Disaggregated (2 ranks, uncompressed FP16 slicing) against HeteroDisagg (2 ranks, dynamic block-FP8 quantization + asymmetric re-packing), the isolated compute cost of compression is:
> * At $C=2$: $15.49\text{ ms} - 4.52\text{ ms} = \mathbf{+10.97\text{ ms}}$
> * At $C=16$: $38.88\text{ ms} - 7.24\text{ ms} = \mathbf{+31.64\text{ ms}}$

---

## 4. Analytical Wire Latency Projections (Derived from Measured Payloads)

To evaluate the trade-off across production fabrics, wire transfer times are derived analytically from the **empirically measured payload sizes** ($64.03\text{ MB}$ vs $128.00\text{ MB}$ for $C=2$; $512.25\text{ MB}$ vs $1024.00\text{ MB}$ for $C=16$):

$$\text{Wire Latency (ms)} = \frac{\text{Measured Payload (MB)}}{\text{Nominal Fabric Bandwidth (MB/s)}} \times 1000$$

### Pure Wire-Time Delta vs. Isolated GPU Compute Overhead
*(Computed with exact byte accounting: $134,217,728\text{ B}$ for FP16 vs. $67,141,632\text{ B}$ for FP8 at $C=2$; $8\times$ scaling at $C=16$. Bus speeds like PCIe are binary $32\text{ GiB/s} = 34.36\times 10^9\text{ B/s}$; network lines are decimal standard $100\text{ Gbps} = 12.5\times 10^9\text{ B/s}$.)*

| Workload | Fabric Specification | Fabric Bandwidth | Vanilla Disagg Wire (FP16) | HeteroDisagg Wire (FP8) | Wire Time Saved ($\Delta T_{\text{wire}}$) | Compute Overhead ($\Delta T_{\text{compute}}$) | Net Latency Impact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$C=2$** (Latency-Bound) | **NVLink / PCIe 4.0** | 32.0 GiB/s | 3.91 ms | 1.95 ms | **1.95 ms** | +10.97 ms | Compute overhead dominates (-9.02 ms) |
| | **100 GbE RoCE / RDMA** | 100 Gbps (12.5 GB/s) | 10.74 ms | 5.37 ms | **5.37 ms** | +10.97 ms | Near parity (-5.60 ms) |
| | **10 GbE Datacenter LAN**| 10 Gbps (1.25 GB/s) | 107.37 ms | 53.71 ms | **53.66 ms** | +10.97 ms | **HeteroDisagg wins by +42.69 ms** |
| | **WAN (Campus / Edge)** | 100 Mbps (12.5 MB/s) | 10,737.42 ms | 5,371.33 ms | **5,366.09 ms** | +10.97 ms | **HeteroDisagg wins by +5,355.12 ms (2.0x)** |
| **$C=16$** (Throughput-Bound) | **NVLink / PCIe 4.0** | 32.0 GiB/s | 31.25 ms | 15.63 ms | **15.62 ms** | +31.64 ms | Compute overhead dominates (-16.02 ms) |
| | **100 GbE RoCE / RDMA** | 100 Gbps (12.5 GB/s) | 85.90 ms | 42.97 ms | **42.93 ms** | +31.64 ms | **HeteroDisagg wins by +11.29 ms** |
| | **10 GbE Datacenter LAN**| 10 Gbps (1.25 GB/s) | 858.99 ms | 429.71 ms | **429.29 ms** | +31.64 ms | **HeteroDisagg wins by +397.65 ms** |
| | **WAN (Campus / Edge)** | 100 Mbps (12.5 MB/s) | 85,899.35 ms | 42,970.64 ms | **42,928.70 ms** | +31.64 ms | **HeteroDisagg wins by +42,897.06 ms (2.0x)** |

---

## 5. What Was NOT Tested & Why

1. **Dedicated RoCEv2 (100/200/400 GbE) & InfiniBand Fabrics:**
   - **Reason:** Vast.ai provides containerized instances on commodity host setups. Containers are restricted to Docker bridge networks without access to physical NICs, SR-IOV, or RDMA verbs.
   - **Status:** All RoCE and NVLink network latencies are strictly labeled as **analytical projections** derived from empirical payload sizes.

2. **Direct Socket Transfer to an Exposed Custom TCP Port:**
   - **Reason:** The decode instance was rented with only port 22 mapped externally. The live test routed through an SSH-tunneled port, introducing channel window throttling.
   - **Status:** Direct un-tunneled socket benchmarking requires spinning up instances with pre-configured `-p <port>:<port>` flags, scheduled for the next hardware run.

---

## 6. Artifact & Run Trace Summary

* **Raw JSON Sweep:** [`benchmarks/empirical_decoupled_sweep.json`](file:///Users/aravindsundaresan/Development/HeteroDisagg/benchmarks/empirical_decoupled_sweep.json)
* **Raw Execution Log:** [`benchmarks/empirical_decoupled_sweep.log`](file:///Users/aravindsundaresan/Development/HeteroDisagg/benchmarks/empirical_decoupled_sweep.log)
* **Harness Script:** [`scripts/run_decoupled_empirical_benchmark.py`](file:///Users/aravindsundaresan/Development/HeteroDisagg/scripts/run_decoupled_empirical_benchmark.py)
* **Demarcation Statement:** All GPU kernel measurements and payload sizes are 100% empirically measured on A100 SXM4 silicon; cross-fabric network timings are mathematical projections based strictly on those empirical payload volumes.
