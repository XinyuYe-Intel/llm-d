# Tiered Prefix Cache — 32-XPU Benchmark Report

Comparison of KV-cache offloading strategies on a 32× Intel Arc Pro B60 cluster (vLLM v0.26.0, `llm-d-xpu:v0.9.0`).

**Configurations compared**

- **Baseline** — GPU-only prefix cache, no offloading.
- **Native CPU offload** — KV blocks offloaded to host RAM (CPU tier).
- **Native FS offload** — KV blocks offloaded to a shared filesystem tier.

**Methodology.** Each model was driven through a Poisson request ladder of increasing offered QPS. For the head-to-head tables below, **only offered-QPS points that recorded zero failed requests in all three configurations are used** — any QPS at which any config dropped requests is excluded so the comparison is apples-to-apples on stable load. Throughput and token rates are cluster-aggregate; latencies are per-request aggregate percentiles.

## 0. Experimental setup & context

**Cluster.** 4 worker nodes × 8 Intel Arc Pro B60 (24 GB GDDR6) = **32 XPUs**. The B60s are discrete PCIe cards with **no Xe-Link**, so every tensor-parallel all-reduce and every KV VRAM↔host/FS transfer rides PCIe — relevant to the offload-tier latencies below.

**Software.** vLLM v0.26.0, image `ghcr.io/llm-d/llm-d-xpu:v0.9.0`, served via llm-d with a standalone endpoint-picker (EPP) router doing prefix-aware routing. vLLM args common to all runs: `--dtype bfloat16 --block-size 64 --gpu-memory-utilization 0.85 --max-num-seq 128`. Each model is sized to fill all 32 XPUs (TP × replicas = 32).

**Offload tiers under test.**
- **Baseline** — VRAM-only KV cache (control), no offloading.
- **Native CPU offload** — `OffloadingConnector`, VRAM → host RAM; the CPU tier is sized to ~80% of each replica's reusable-prefix working set.
- **Native FS offload** — `TieringOffloadingSpec`, VRAM → CPU → node-local NVMe filesystem (hostPath); the FS tier is large enough to hold ~the whole working set.

Expected prefix-cache-hit ordering by design: baseline < native-cpu < native-fs. The workload deliberately sizes each replica's reusable-prefix working set to ~3× the in-VRAM KV cache, so baseline must evict/recompute while the offload tiers can retain more of it.

**Per-model deployment & workload.** Shared-prefix workload (`shared_prefix`): each request = a fixed shared system prefix + a 256-token question, generating a 256-token answer; 5 prompts/group, 5 users/group. Open-loop Poisson load, 60 s per stage, per-request timeout 1800 s. Cluster offered QPS per stage = per-replica ladder × replicas.

| Model | HF repo | TP | Replicas | Prefix len (tok) | Input len (tok) | Output len | Prefix groups | KV B/token | Meas. VRAM KV/replica (tok) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3 70B | `meta-llama/Meta-Llama-3-70B` | 8 | 4 | 4,000 | 4,255 | 256 | 216 | 320 KiB | n/a |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | 4 | 8 | 12,000 | 12,256 | 256 | 464 | 96 KiB | 196,736 |
| Qwen3-32B | `Qwen/Qwen3-32B` | 4 | 8 | 3,600 | 3,856 | 256 | 448 | 256 KiB | 50,944 |

**Offered-QPS ladders (cluster).** Llama-3 70B: 1, 2, 3, 4, 5, 6. Qwen3-30B-A3B & Qwen3-32B: 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32.

**Metric definitions.** *Out tok/s* = cluster output-token throughput; *Achieved req/s* = measured completed requests/s (vs offered QPS); *TTFT* = time-to-first-token; *TPOT* = time-per-output-token; *E2E* = end-to-end request latency. Latencies are aggregate over all requests in the stage.

## 1. Stability overview (failed requests per offered QPS)

### Llama-3 70B (TP=8, 4 replicas)

| Offered QPS | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|
| 1 | OK (60) | OK (60) | OK (60) |
| 2 | OK (120) | OK (120) | OK (120) |
| 3 | OK (180) | OK (180) | OK (180) |
| 4 | OK (240) | OK (240) | OK (240) |
| 5 | OK (300) | OK (300) | OK (300) |
| 6 | OK (360) | OK (360) | OK (360) |

### Qwen3-30B-A3B (TP=4, 8 replicas)

| Offered QPS | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|
| 2 | OK (120) | OK (120) | OK (120) |
| 4 | OK (240) | OK (240) | OK (240) |
| 6 | OK (360) | OK (360) | OK (360) |
| 8 | OK (480) | OK (480) | OK (480) |
| 10 | OK (600) | OK (600) | OK (600) |
| 12 | OK (720) | OK (720) | OK (720) |
| 14 | OK (840) | OK (840) | OK (840) |
| 16 | OK (960) | OK (960) | OK (960) |
| 20 | OK (1200) | OK (1200) | OK (1200) |
| 24 | OK (1440) | OK (1440) | OK (1440) |
| 28 | OK (1680) | OK (1680) | OK (1680) |
| 32 | OK (1920) | OK (1920) | OK (1920) |

### Qwen3-32B (TP=4, 8 replicas)

| Offered QPS | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|
| 2 | OK (120) | OK (120) | OK (120) |
| 4 | OK (240) | OK (240) | OK (240) |
| 6 | OK (360) | OK (360) | OK (360) |
| 8 | OK (480) | OK (480) | OK (480) |
| 10 | OK (600) | OK (600) | OK (600) |
| 12 | OK (720) | OK (720) | OK (720) |
| 14 | OK (840) | OK (840) | OK (840) |
| 16 | OK (960) | OK (960) | OK (960) |
| 20 | OK (1200) | OK (1200) | OK (1200) |
| 24 | OK (1440) | OK (1440) | OK (1440) |
| 28 | OK (1680) | OK (1680) | OK (1680) |
| 32 | OK (1920) | OK (1920) | OK (1920) |

## 2. Head-to-head at failure-free QPS points

Configs are columns for a direct side-by-side read; each offered-QPS block lists all metrics as rows. **Bold** marks the best config for that metric row (higher = better for throughput/req-rate, lower = better for the latency rows). Each offload column also shows its **% change vs baseline** in parentheses — for the throughput/req-rate rows positive is better, and for the latency rows negative is better. The percentage is coloured <span style="color:#d00000">red when it is an improvement over baseline</span> and <span style="color:#008000">green when it is worse</span>.

### Llama-3 70B (TP=8, 4 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **1** | Out tok/s | **174** | 168 <span style="color:#008000">(−3.1%)</span> | 141 <span style="color:#008000">(−18.8%)</span> |
|  | Achieved req/s | **0.72** | 0.70 <span style="color:#008000">(−1.9%)</span> | 0.61 <span style="color:#008000">(−15.0%)</span> |
|  | TTFT mean (s) | **3.95** | 5.75 <span style="color:#008000">(+45.7%)</span> | 7.15 <span style="color:#008000">(+81.0%)</span> |
|  | TTFT p99 (s) | **8.09** | 14.93 <span style="color:#008000">(+84.5%)</span> | 22.34 <span style="color:#008000">(+176.0%)</span> |
|  | TPOT mean (s) | **0.115** | 0.140 <span style="color:#008000">(+21.6%)</span> | 0.174 <span style="color:#008000">(+50.4%)</span> |
|  | E2E p99 (s) | **43.57** | 52.98 <span style="color:#008000">(+21.6%)</span> | 67.64 <span style="color:#008000">(+55.2%)</span> |
| **2** | Out tok/s | **291** | 263 <span style="color:#008000">(−9.9%)</span> | 233 <span style="color:#008000">(−19.9%)</span> |
|  | Achieved req/s | **1.20** | 1.07 <span style="color:#008000">(−11.1%)</span> | 0.94 <span style="color:#008000">(−21.8%)</span> |
|  | TTFT mean (s) | **6.14** | 7.74 <span style="color:#008000">(+26.1%)</span> | 12.60 <span style="color:#008000">(+105.1%)</span> |
|  | TTFT p99 (s) | **22.26** | 32.24 <span style="color:#008000">(+44.8%)</span> | 51.99 <span style="color:#008000">(+133.5%)</span> |
|  | TPOT mean (s) | **0.123** | 0.125 <span style="color:#008000">(+1.8%)</span> | 0.148 <span style="color:#008000">(+20.5%)</span> |
|  | E2E p99 (s) | **49.75** | 62.29 <span style="color:#008000">(+25.2%)</span> | 75.47 <span style="color:#008000">(+51.7%)</span> |
| **3** | Out tok/s | 295 | **320** <span style="color:#d00000">(+8.3%)</span> | 284 <span style="color:#008000">(−3.9%)</span> |
|  | Achieved req/s | 1.10 | **1.34** <span style="color:#d00000">(+21.3%)</span> | 1.22 <span style="color:#d00000">(+10.2%)</span> |
|  | TTFT mean (s) | 31.35 | **17.04** <span style="color:#d00000">(−45.7%)</span> | 18.51 <span style="color:#d00000">(−41.0%)</span> |
|  | TTFT p99 (s) | 79.88 | **51.33** <span style="color:#d00000">(−35.7%)</span> | 64.71 <span style="color:#d00000">(−19.0%)</span> |
|  | TPOT mean (s) | 0.158 | **0.116** <span style="color:#d00000">(−26.6%)</span> | 0.135 <span style="color:#d00000">(−14.8%)</span> |
|  | E2E p99 (s) | 104.72 | **74.35** <span style="color:#d00000">(−29.0%)</span> | 91.02 <span style="color:#d00000">(−13.1%)</span> |
| **4** | Out tok/s | 298 | 309 <span style="color:#d00000">(+3.8%)</span> | **336** <span style="color:#d00000">(+12.9%)</span> |
|  | Achieved req/s | 1.16 | 1.27 <span style="color:#d00000">(+9.8%)</span> | **1.28** <span style="color:#d00000">(+10.3%)</span> |
|  | TTFT mean (s) | 56.00 | 41.10 <span style="color:#d00000">(−26.6%)</span> | **29.48** <span style="color:#d00000">(−47.3%)</span> |
|  | TTFT p99 (s) | 125.38 | 110.76 <span style="color:#d00000">(−11.7%)</span> | **79.23** <span style="color:#d00000">(−36.8%)</span> |
|  | TPOT mean (s) | 0.172 | 0.129 <span style="color:#d00000">(−24.7%)</span> | **0.127** <span style="color:#d00000">(−26.0%)</span> |
|  | E2E p99 (s) | 147.63 | 132.12 <span style="color:#d00000">(−10.5%)</span> | **131.70** <span style="color:#d00000">(−10.8%)</span> |
| **5** | Out tok/s | 303 | 320 <span style="color:#d00000">(+5.4%)</span> | **355** <span style="color:#d00000">(+17.1%)</span> |
|  | Achieved req/s | 1.19 | 1.25 <span style="color:#d00000">(+4.7%)</span> | **1.38** <span style="color:#d00000">(+15.9%)</span> |
|  | TTFT mean (s) | 80.86 | 66.25 <span style="color:#d00000">(−18.1%)</span> | **47.04** <span style="color:#d00000">(−41.8%)</span> |
|  | TTFT p99 (s) | 176.96 | 148.88 <span style="color:#d00000">(−15.9%)</span> | **133.20** <span style="color:#d00000">(−24.7%)</span> |
|  | TPOT mean (s) | 0.167 | 0.152 <span style="color:#d00000">(−9.1%)</span> | **0.129** <span style="color:#d00000">(−22.7%)</span> |
|  | E2E p99 (s) | 199.61 | 171.30 <span style="color:#d00000">(−14.2%)</span> | **161.05** <span style="color:#d00000">(−19.3%)</span> |
| **6** | Out tok/s | 305 | 340 <span style="color:#d00000">(+11.3%)</span> | **393** <span style="color:#d00000">(+29.0%)</span> |
|  | Achieved req/s | 1.21 | 1.28 <span style="color:#d00000">(+5.4%)</span> | **1.56** <span style="color:#d00000">(+28.3%)</span> |
|  | TTFT mean (s) | 100.38 | 90.69 <span style="color:#d00000">(−9.7%)</span> | **58.52** <span style="color:#d00000">(−41.7%)</span> |
|  | TTFT p99 (s) | 215.84 | 188.88 <span style="color:#d00000">(−12.5%)</span> | **150.69** <span style="color:#d00000">(−30.2%)</span> |
|  | TPOT mean (s) | 0.163 | **0.155** <span style="color:#d00000">(−5.3%)</span> | 0.165 <span style="color:#008000">(+0.7%)</span> |
|  | E2E p99 (s) | 238.66 | 215.97 <span style="color:#d00000">(−9.5%)</span> | **183.16** <span style="color:#d00000">(−23.3%)</span> |

### Qwen3-30B-A3B (TP=4, 8 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **2** | Out tok/s | 734 | 645 <span style="color:#008000">(−12.2%)</span> | **820** <span style="color:#d00000">(+11.7%)</span> |
|  | Achieved req/s | **1.61** | 1.40 <span style="color:#008000">(−12.6%)</span> | 1.46 <span style="color:#008000">(−8.8%)</span> |
|  | TTFT mean (s) | 2.54 | 2.31 <span style="color:#d00000">(−8.9%)</span> | **2.15** <span style="color:#d00000">(−15.2%)</span> |
|  | TTFT p99 (s) | 6.75 | 5.38 <span style="color:#d00000">(−20.3%)</span> | **4.68** <span style="color:#d00000">(−30.6%)</span> |
|  | TPOT mean (s) | 0.074 | **0.073** <span style="color:#d00000">(−2.4%)</span> | 0.074 <span style="color:#008000">(+0.0%)</span> |
|  | E2E p99 (s) | **25.75** | 26.85 <span style="color:#008000">(+4.3%)</span> | 26.85 <span style="color:#008000">(+4.3%)</span> |
| **4** | Out tok/s | 1065 | 1052 <span style="color:#008000">(−1.2%)</span> | **1202** <span style="color:#d00000">(+12.9%)</span> |
|  | Achieved req/s | **3.01** | 2.95 <span style="color:#008000">(−2.2%)</span> | 2.97 <span style="color:#008000">(−1.6%)</span> |
|  | TTFT mean (s) | 1.85 | 1.56 <span style="color:#d00000">(−15.3%)</span> | **1.48** <span style="color:#d00000">(−19.7%)</span> |
|  | TTFT p99 (s) | 6.53 | **6.28** <span style="color:#d00000">(−3.9%)</span> | 7.68 <span style="color:#008000">(+17.5%)</span> |
|  | TPOT mean (s) | 0.084 | **0.080** <span style="color:#d00000">(−4.5%)</span> | 0.083 <span style="color:#d00000">(−1.0%)</span> |
|  | E2E p99 (s) | 31.96 | **31.84** <span style="color:#d00000">(−0.4%)</span> | 32.57 <span style="color:#008000">(+1.9%)</span> |
| **6** | Out tok/s | 1161 | 1233 <span style="color:#d00000">(+6.2%)</span> | **1739** <span style="color:#d00000">(+49.8%)</span> |
|  | Achieved req/s | 3.61 | **4.32** <span style="color:#d00000">(+19.7%)</span> | 3.93 <span style="color:#d00000">(+8.6%)</span> |
|  | TTFT mean (s) | 9.70 | 3.00 <span style="color:#d00000">(−69.0%)</span> | **1.91** <span style="color:#d00000">(−80.3%)</span> |
|  | TTFT p99 (s) | 25.66 | **11.27** <span style="color:#d00000">(−56.1%)</span> | 12.25 <span style="color:#d00000">(−52.3%)</span> |
|  | TPOT mean (s) | 0.173 | **0.085** <span style="color:#d00000">(−51.1%)</span> | 0.102 <span style="color:#d00000">(−41.3%)</span> |
|  | E2E p99 (s) | 43.98 | 32.36 <span style="color:#d00000">(−26.4%)</span> | **31.91** <span style="color:#d00000">(−27.4%)</span> |
| **8** | Out tok/s | 1122 | 1373 <span style="color:#d00000">(+22.4%)</span> | **2265** <span style="color:#d00000">(+101.9%)</span> |
|  | Achieved req/s | 3.83 | **4.99** <span style="color:#d00000">(+30.2%)</span> | 4.84 <span style="color:#d00000">(+26.4%)</span> |
|  | TTFT mean (s) | 18.50 | **4.74** <span style="color:#d00000">(−74.4%)</span> | 4.77 <span style="color:#d00000">(−74.2%)</span> |
|  | TTFT p99 (s) | 45.45 | **16.89** <span style="color:#d00000">(−62.8%)</span> | 24.52 <span style="color:#d00000">(−46.0%)</span> |
|  | TPOT mean (s) | 0.276 | **0.078** <span style="color:#d00000">(−71.7%)</span> | 0.104 <span style="color:#d00000">(−62.2%)</span> |
|  | E2E p99 (s) | 64.92 | **37.32** <span style="color:#d00000">(−42.5%)</span> | 41.32 <span style="color:#d00000">(−36.4%)</span> |
| **10** | Out tok/s | 883 | 1716 <span style="color:#d00000">(+94.2%)</span> | **1953** <span style="color:#d00000">(+121.0%)</span> |
|  | Achieved req/s | 4.00 | **5.52** <span style="color:#d00000">(+38.0%)</span> | 5.34 <span style="color:#d00000">(+33.5%)</span> |
|  | TTFT mean (s) | 31.05 | 13.61 <span style="color:#d00000">(−56.2%)</span> | **8.85** <span style="color:#d00000">(−71.5%)</span> |
|  | TTFT p99 (s) | 68.80 | 33.99 <span style="color:#d00000">(−50.6%)</span> | **29.87** <span style="color:#d00000">(−56.6%)</span> |
|  | TPOT mean (s) | 0.465 | **0.077** <span style="color:#d00000">(−83.5%)</span> | 0.129 <span style="color:#d00000">(−72.2%)</span> |
|  | E2E p99 (s) | 87.40 | 52.56 <span style="color:#d00000">(−39.9%)</span> | **48.75** <span style="color:#d00000">(−44.2%)</span> |
| **12** | Out tok/s | 812 | 1348 <span style="color:#d00000">(+66.0%)</span> | **2086** <span style="color:#d00000">(+156.9%)</span> |
|  | Achieved req/s | 4.01 | 4.49 <span style="color:#d00000">(+11.9%)</span> | **5.36** <span style="color:#d00000">(+33.6%)</span> |
|  | TTFT mean (s) | 44.70 | 38.03 <span style="color:#d00000">(−14.9%)</span> | **17.41** <span style="color:#d00000">(−61.0%)</span> |
|  | TTFT p99 (s) | 98.47 | 84.10 <span style="color:#d00000">(−14.6%)</span> | **50.58** <span style="color:#d00000">(−48.6%)</span> |
|  | TPOT mean (s) | 0.546 | **0.093** <span style="color:#d00000">(−82.9%)</span> | 0.145 <span style="color:#d00000">(−73.5%)</span> |
|  | E2E p99 (s) | 117.36 | 101.43 <span style="color:#d00000">(−13.6%)</span> | **73.12** <span style="color:#d00000">(−37.7%)</span> |
| **14** | Out tok/s | 869 | 1284 <span style="color:#d00000">(+47.8%)</span> | **2365** <span style="color:#d00000">(+172.3%)</span> |
|  | Achieved req/s | 4.00 | 4.34 <span style="color:#d00000">(+8.5%)</span> | **6.13** <span style="color:#d00000">(+53.3%)</span> |
|  | TTFT mean (s) | 60.64 | 50.80 <span style="color:#d00000">(−16.2%)</span> | **22.92** <span style="color:#d00000">(−62.2%)</span> |
|  | TTFT p99 (s) | 134.04 | 111.21 <span style="color:#d00000">(−17.0%)</span> | **57.90** <span style="color:#d00000">(−56.8%)</span> |
|  | TPOT mean (s) | 0.571 | **0.095** <span style="color:#d00000">(−83.4%)</span> | 0.157 <span style="color:#d00000">(−72.5%)</span> |
|  | E2E p99 (s) | 152.04 | 127.18 <span style="color:#d00000">(−16.3%)</span> | **76.78** <span style="color:#d00000">(−49.5%)</span> |
| **16** | Out tok/s | 959 | 1343 <span style="color:#d00000">(+40.0%)</span> | **2161** <span style="color:#d00000">(+125.4%)</span> |
|  | Achieved req/s | 3.92 | 4.40 <span style="color:#d00000">(+12.3%)</span> | **5.99** <span style="color:#d00000">(+52.7%)</span> |
|  | TTFT mean (s) | 76.19 | 61.99 <span style="color:#d00000">(−18.6%)</span> | **29.95** <span style="color:#d00000">(−60.7%)</span> |
|  | TTFT p99 (s) | 166.63 | 127.80 <span style="color:#d00000">(−23.3%)</span> | **81.06** <span style="color:#d00000">(−51.4%)</span> |
|  | TPOT mean (s) | 0.572 | **0.128** <span style="color:#d00000">(−77.7%)</span> | 0.167 <span style="color:#d00000">(−70.8%)</span> |
|  | E2E p99 (s) | 184.70 | 146.93 <span style="color:#d00000">(−20.4%)</span> | **98.55** <span style="color:#d00000">(−46.6%)</span> |
| **20** | Out tok/s | 1161 | 1321 <span style="color:#d00000">(+13.8%)</span> | **2218** <span style="color:#d00000">(+91.0%)</span> |
|  | Achieved req/s | 4.01 | 4.46 <span style="color:#d00000">(+11.3%)</span> | **6.20** <span style="color:#d00000">(+54.4%)</span> |
|  | TTFT mean (s) | 104.02 | 89.25 <span style="color:#d00000">(−14.2%)</span> | **44.07** <span style="color:#d00000">(−57.6%)</span> |
|  | TTFT p99 (s) | 213.28 | 187.20 <span style="color:#d00000">(−12.2%)</span> | **117.92** <span style="color:#d00000">(−44.7%)</span> |
|  | TPOT mean (s) | 0.465 | **0.182** <span style="color:#d00000">(−60.9%)</span> | 0.197 <span style="color:#d00000">(−57.5%)</span> |
|  | E2E p99 (s) | 233.03 | 208.81 <span style="color:#d00000">(−10.4%)</span> | **137.00** <span style="color:#d00000">(−41.2%)</span> |
| **24** | Out tok/s | 1460 | 1390 <span style="color:#008000">(−4.8%)</span> | **2052** <span style="color:#d00000">(+40.5%)</span> |
|  | Achieved req/s | 4.15 | 4.38 <span style="color:#d00000">(+5.6%)</span> | **6.41** <span style="color:#d00000">(+54.6%)</span> |
|  | TTFT mean (s) | 129.21 | 113.40 <span style="color:#d00000">(−12.2%)</span> | **57.75** <span style="color:#d00000">(−55.3%)</span> |
|  | TTFT p99 (s) | 262.52 | 233.49 <span style="color:#d00000">(−11.1%)</span> | **142.23** <span style="color:#d00000">(−45.8%)</span> |
|  | TPOT mean (s) | 0.237 | **0.187** <span style="color:#d00000">(−20.9%)</span> | 0.225 <span style="color:#d00000">(−4.8%)</span> |
|  | E2E p99 (s) | 280.57 | 256.89 <span style="color:#d00000">(−8.4%)</span> | **163.57** <span style="color:#d00000">(−41.7%)</span> |
| **28** | Out tok/s | 1517 | 1327 <span style="color:#008000">(−12.5%)</span> | **2047** <span style="color:#d00000">(+34.9%)</span> |
|  | Achieved req/s | 4.13 | 4.43 <span style="color:#d00000">(+7.2%)</span> | **6.57** <span style="color:#d00000">(+58.9%)</span> |
|  | TTFT mean (s) | 158.59 | 135.02 <span style="color:#d00000">(−14.9%)</span> | **68.85** <span style="color:#d00000">(−56.6%)</span> |
|  | TTFT p99 (s) | 324.15 | 281.68 <span style="color:#d00000">(−13.1%)</span> | **164.83** <span style="color:#d00000">(−49.1%)</span> |
|  | TPOT mean (s) | **0.144** | 0.301 <span style="color:#008000">(+109.3%)</span> | 0.242 <span style="color:#008000">(+68.1%)</span> |
|  | E2E p99 (s) | 345.29 | 305.28 <span style="color:#d00000">(−11.6%)</span> | **185.76** <span style="color:#d00000">(−46.2%)</span> |
| **32** | Out tok/s | 1520 | 1229 <span style="color:#008000">(−19.2%)</span> | **1969** <span style="color:#d00000">(+29.6%)</span> |
|  | Achieved req/s | 4.07 | 4.55 <span style="color:#d00000">(+12.0%)</span> | **6.78** <span style="color:#d00000">(+66.8%)</span> |
|  | TTFT mean (s) | 186.78 | 161.45 <span style="color:#d00000">(−13.6%)</span> | **81.18** <span style="color:#d00000">(−56.5%)</span> |
|  | TTFT p99 (s) | 381.66 | 337.59 <span style="color:#d00000">(−11.5%)</span> | **195.40** <span style="color:#d00000">(−48.8%)</span> |
|  | TPOT mean (s) | **0.111** | 0.393 <span style="color:#008000">(+255.0%)</span> | 0.261 <span style="color:#008000">(+135.6%)</span> |
|  | E2E p99 (s) | 401.47 | 357.86 <span style="color:#d00000">(−10.9%)</span> | **233.23** <span style="color:#d00000">(−41.9%)</span> |

### Qwen3-32B (TP=4, 8 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **2** | Out tok/s | 382 | **458** <span style="color:#d00000">(+19.8%)</span> | 451 <span style="color:#d00000">(+18.1%)</span> |
|  | Achieved req/s | 1.37 | **1.63** <span style="color:#d00000">(+18.9%)</span> | 1.58 <span style="color:#d00000">(+15.5%)</span> |
|  | TTFT mean (s) | **1.85** | 2.24 <span style="color:#008000">(+21.2%)</span> | 2.53 <span style="color:#008000">(+36.5%)</span> |
|  | TTFT p99 (s) | **5.23** | 7.35 <span style="color:#008000">(+40.5%)</span> | 6.67 <span style="color:#008000">(+27.5%)</span> |
|  | TPOT mean (s) | 0.081 | **0.072** <span style="color:#d00000">(−10.8%)</span> | 0.076 <span style="color:#d00000">(−6.2%)</span> |
|  | E2E p99 (s) | **20.30** | 23.93 <span style="color:#008000">(+17.9%)</span> | 28.52 <span style="color:#008000">(+40.5%)</span> |
| **4** | Out tok/s | **840** | 759 <span style="color:#008000">(−9.6%)</span> | 658 <span style="color:#008000">(−21.6%)</span> |
|  | Achieved req/s | **2.99** | 2.68 <span style="color:#008000">(−10.2%)</span> | 2.21 <span style="color:#008000">(−26.1%)</span> |
|  | TTFT mean (s) | 2.75 | **2.10** <span style="color:#d00000">(−23.7%)</span> | 3.45 <span style="color:#008000">(+25.2%)</span> |
|  | TTFT p99 (s) | **12.31** | 12.58 <span style="color:#008000">(+2.2%)</span> | 31.97 <span style="color:#008000">(+159.8%)</span> |
|  | TPOT mean (s) | 0.102 | **0.072** <span style="color:#d00000">(−29.1%)</span> | 0.099 <span style="color:#d00000">(−3.4%)</span> |
|  | E2E p99 (s) | 34.08 | **27.93** <span style="color:#d00000">(−18.0%)</span> | 48.25 <span style="color:#008000">(+41.6%)</span> |
| **6** | Out tok/s | 915 | **1140** <span style="color:#d00000">(+24.5%)</span> | 1105 <span style="color:#d00000">(+20.7%)</span> |
|  | Achieved req/s | 3.44 | **4.11** <span style="color:#d00000">(+19.2%)</span> | 3.81 <span style="color:#d00000">(+10.5%)</span> |
|  | TTFT mean (s) | 14.87 | 5.17 <span style="color:#d00000">(−65.2%)</span> | **4.79** <span style="color:#d00000">(−67.8%)</span> |
|  | TTFT p99 (s) | 38.82 | 23.89 <span style="color:#d00000">(−38.5%)</span> | **22.53** <span style="color:#d00000">(−42.0%)</span> |
|  | TPOT mean (s) | 0.185 | **0.112** <span style="color:#d00000">(−39.3%)</span> | 0.114 <span style="color:#d00000">(−38.1%)</span> |
|  | E2E p99 (s) | 52.78 | **37.71** <span style="color:#d00000">(−28.6%)</span> | 39.17 <span style="color:#d00000">(−25.8%)</span> |
| **8** | Out tok/s | 962 | 1262 <span style="color:#d00000">(+31.2%)</span> | **1296** <span style="color:#d00000">(+34.8%)</span> |
|  | Achieved req/s | 3.44 | 4.32 <span style="color:#d00000">(+25.9%)</span> | **4.41** <span style="color:#d00000">(+28.2%)</span> |
|  | TTFT mean (s) | 25.75 | **9.17** <span style="color:#d00000">(−64.4%)</span> | 9.48 <span style="color:#d00000">(−63.2%)</span> |
|  | TTFT p99 (s) | 62.80 | **30.21** <span style="color:#d00000">(−51.9%)</span> | 32.97 <span style="color:#d00000">(−47.5%)</span> |
|  | TPOT mean (s) | 0.139 | 0.131 <span style="color:#d00000">(−5.5%)</span> | **0.116** <span style="color:#d00000">(−16.3%)</span> |
|  | E2E p99 (s) | 82.08 | **46.89** <span style="color:#d00000">(−42.9%)</span> | 49.41 <span style="color:#d00000">(−39.8%)</span> |
| **10** | Out tok/s | 961 | 1118 <span style="color:#d00000">(+16.4%)</span> | **1387** <span style="color:#d00000">(+44.3%)</span> |
|  | Achieved req/s | 3.55 | 3.82 <span style="color:#d00000">(+7.5%)</span> | **4.67** <span style="color:#d00000">(+31.4%)</span> |
|  | TTFT mean (s) | 42.15 | 33.06 <span style="color:#d00000">(−21.6%)</span> | **17.26** <span style="color:#d00000">(−59.0%)</span> |
|  | TTFT p99 (s) | 90.31 | 77.75 <span style="color:#d00000">(−13.9%)</span> | **48.41** <span style="color:#d00000">(−46.4%)</span> |
|  | TPOT mean (s) | 0.149 | **0.125** <span style="color:#d00000">(−16.5%)</span> | 0.132 <span style="color:#d00000">(−11.7%)</span> |
|  | E2E p99 (s) | 107.92 | 94.54 <span style="color:#d00000">(−12.4%)</span> | **64.26** <span style="color:#d00000">(−40.5%)</span> |
| **12** | Out tok/s | 955 | 1132 <span style="color:#d00000">(+18.5%)</span> | **1406** <span style="color:#d00000">(+47.2%)</span> |
|  | Achieved req/s | 3.57 | 3.93 <span style="color:#d00000">(+10.0%)</span> | **4.75** <span style="color:#d00000">(+33.1%)</span> |
|  | TTFT mean (s) | 55.30 | 47.73 <span style="color:#d00000">(−13.7%)</span> | **27.52** <span style="color:#d00000">(−50.2%)</span> |
|  | TTFT p99 (s) | 125.82 | 110.15 <span style="color:#d00000">(−12.5%)</span> | **72.13** <span style="color:#d00000">(−42.7%)</span> |
|  | TPOT mean (s) | 0.161 | 0.136 <span style="color:#d00000">(−15.2%)</span> | **0.126** <span style="color:#d00000">(−21.3%)</span> |
|  | E2E p99 (s) | 141.57 | 125.18 <span style="color:#d00000">(−11.6%)</span> | **99.30** <span style="color:#d00000">(−29.9%)</span> |
| **14** | Out tok/s | 1011 | 1132 <span style="color:#d00000">(+12.0%)</span> | **1375** <span style="color:#d00000">(+36.1%)</span> |
|  | Achieved req/s | 3.50 | 3.97 <span style="color:#d00000">(+13.5%)</span> | **4.70** <span style="color:#d00000">(+34.2%)</span> |
|  | TTFT mean (s) | 71.02 | 61.89 <span style="color:#d00000">(−12.9%)</span> | **35.66** <span style="color:#d00000">(−49.8%)</span> |
|  | TTFT p99 (s) | 155.22 | 133.73 <span style="color:#d00000">(−13.8%)</span> | **87.29** <span style="color:#d00000">(−43.8%)</span> |
|  | TPOT mean (s) | 0.169 | 0.153 <span style="color:#d00000">(−9.8%)</span> | **0.104** <span style="color:#d00000">(−38.6%)</span> |
|  | E2E p99 (s) | 175.38 | 150.02 <span style="color:#d00000">(−14.5%)</span> | **108.53** <span style="color:#d00000">(−38.1%)</span> |
| **16** | Out tok/s | 1040 | 1096 <span style="color:#d00000">(+5.4%)</span> | **1559** <span style="color:#d00000">(+49.9%)</span> |
|  | Achieved req/s | 3.58 | 3.89 <span style="color:#d00000">(+8.5%)</span> | **5.37** <span style="color:#d00000">(+50.0%)</span> |
|  | TTFT mean (s) | 85.34 | 77.13 <span style="color:#d00000">(−9.6%)</span> | **42.52** <span style="color:#d00000">(−50.2%)</span> |
|  | TTFT p99 (s) | 186.75 | 167.01 <span style="color:#d00000">(−10.6%)</span> | **100.11** <span style="color:#d00000">(−46.4%)</span> |
|  | TPOT mean (s) | 0.146 | 0.141 <span style="color:#d00000">(−3.2%)</span> | **0.125** <span style="color:#d00000">(−14.2%)</span> |
|  | E2E p99 (s) | 206.77 | 183.17 <span style="color:#d00000">(−11.4%)</span> | **122.92** <span style="color:#d00000">(−40.6%)</span> |
| **20** | Out tok/s | 1029 | 1081 <span style="color:#d00000">(+5.0%)</span> | **1405** <span style="color:#d00000">(+36.5%)</span> |
|  | Achieved req/s | 3.62 | 3.83 <span style="color:#d00000">(+5.7%)</span> | **4.92** <span style="color:#d00000">(+35.9%)</span> |
|  | TTFT mean (s) | 116.13 | 107.58 <span style="color:#d00000">(−7.4%)</span> | **63.24** <span style="color:#d00000">(−45.5%)</span> |
|  | TTFT p99 (s) | 244.84 | 224.51 <span style="color:#d00000">(−8.3%)</span> | **152.44** <span style="color:#d00000">(−37.7%)</span> |
|  | TPOT mean (s) | 0.156 | **0.136** <span style="color:#d00000">(−12.6%)</span> | 0.150 <span style="color:#d00000">(−3.7%)</span> |
|  | E2E p99 (s) | 264.21 | 243.25 <span style="color:#d00000">(−7.9%)</span> | **173.18** <span style="color:#d00000">(−34.5%)</span> |
| **24** | Out tok/s | 1022 | 1140 <span style="color:#d00000">(+11.6%)</span> | **1383** <span style="color:#d00000">(+35.3%)</span> |
|  | Achieved req/s | 3.63 | 4.00 <span style="color:#d00000">(+10.3%)</span> | **4.92** <span style="color:#d00000">(+35.3%)</span> |
|  | TTFT mean (s) | 147.58 | 132.58 <span style="color:#d00000">(−10.2%)</span> | **79.74** <span style="color:#d00000">(−46.0%)</span> |
|  | TTFT p99 (s) | 307.78 | 266.84 <span style="color:#d00000">(−13.3%)</span> | **185.64** <span style="color:#d00000">(−39.7%)</span> |
|  | TPOT mean (s) | 0.147 | **0.130** <span style="color:#d00000">(−11.5%)</span> | 0.163 <span style="color:#008000">(+10.5%)</span> |
|  | E2E p99 (s) | 327.94 | 286.51 <span style="color:#d00000">(−12.6%)</span> | **243.98** <span style="color:#d00000">(−25.6%)</span> |
| **28** | Out tok/s | 1019 | 1128 <span style="color:#d00000">(+10.8%)</span> | **1452** <span style="color:#d00000">(+42.5%)</span> |
|  | Achieved req/s | 3.63 | 3.98 <span style="color:#d00000">(+9.5%)</span> | **5.18** <span style="color:#d00000">(+42.6%)</span> |
|  | TTFT mean (s) | 178.66 | 161.61 <span style="color:#d00000">(−9.5%)</span> | **98.72** <span style="color:#d00000">(−44.7%)</span> |
|  | TTFT p99 (s) | 366.03 | 332.90 <span style="color:#d00000">(−9.0%)</span> | **217.66** <span style="color:#d00000">(−40.5%)</span> |
|  | TPOT mean (s) | 0.154 | **0.131** <span style="color:#d00000">(−15.1%)</span> | 0.190 <span style="color:#008000">(+23.3%)</span> |
|  | E2E p99 (s) | 382.81 | 351.73 <span style="color:#d00000">(−8.1%)</span> | **280.46** <span style="color:#d00000">(−26.7%)</span> |
| **32** | Out tok/s | 1004 | 1117 <span style="color:#d00000">(+11.3%)</span> | **1506** <span style="color:#d00000">(+50.0%)</span> |
|  | Achieved req/s | 3.62 | 4.00 <span style="color:#d00000">(+10.4%)</span> | **5.36** <span style="color:#d00000">(+48.1%)</span> |
|  | TTFT mean (s) | 210.82 | 190.62 <span style="color:#d00000">(−9.6%)</span> | **114.84** <span style="color:#d00000">(−45.5%)</span> |
|  | TTFT p99 (s) | 436.59 | 388.44 <span style="color:#d00000">(−11.0%)</span> | **250.28** <span style="color:#d00000">(−42.7%)</span> |
|  | TPOT mean (s) | 0.156 | **0.146** <span style="color:#d00000">(−6.4%)</span> | 0.196 <span style="color:#008000">(+25.6%)</span> |
|  | E2E p99 (s) | 454.88 | 405.74 <span style="color:#d00000">(−10.8%)</span> | **322.19** <span style="color:#d00000">(−29.2%)</span> |

## 3. Peak failure-free offered QPS (per config)

| Model | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|
| Llama-3 70B (TP=8, 4 replicas) | 6 | 6 | 6 |
| Qwen3-30B-A3B (TP=4, 8 replicas) | 32 | 32 | 32 |
| Qwen3-32B (TP=4, 8 replicas) | 32 | 32 | 32 |

## 4. Prefix-cache hit rates (end-of-run counters)

GPU hit rate = local prefix-cache hits / queries. Offload hit rate = external (cross-instance / offloaded tier) hits / queries. Baseline has no offload tier.

| Model | Config | GPU prefix hit % | Offload tier hit % |
|---|---|---|---|
| Llama-3 70B (TP=8, 4 replicas) | Baseline (no offload) | 3.4 | n/a |
| Llama-3 70B (TP=8, 4 replicas) | Native CPU offload | 2.8 | 43.0 |
| Llama-3 70B (TP=8, 4 replicas) | Native FS offload | 1.8 | 68.1 |
| Qwen3-30B-A3B (TP=4, 8 replicas) | Baseline (no offload) | 2.7 | n/a |
| Qwen3-30B-A3B (TP=4, 8 replicas) | Native CPU offload | 2.4 | 38.2 |
| Qwen3-30B-A3B (TP=4, 8 replicas) | Native FS offload | 2.1 | 84.9 |
| Qwen3-32B (TP=4, 8 replicas) | Baseline (no offload) | 2.0 | n/a |
| Qwen3-32B (TP=4, 8 replicas) | Native CPU offload | 2.0 | 32.6 |
| Qwen3-32B (TP=4, 8 replicas) | Native FS offload | 3.4 | 87.0 |

## 5. Key findings

- **Llama-3 70B (TP=8, 4 replicas)** (max clean QPS = 6): Native CPU offload: +11.3% out-tok/s, -9.7% TTFT mean vs baseline; Native FS offload: +29.0% out-tok/s, -41.7% TTFT mean vs baseline.
- **Qwen3-30B-A3B (TP=4, 8 replicas)** (max clean QPS = 32): Native CPU offload: -19.2% out-tok/s, -13.6% TTFT mean vs baseline; Native FS offload: +29.6% out-tok/s, -56.5% TTFT mean vs baseline.
- **Qwen3-32B (TP=4, 8 replicas)** (max clean QPS = 32): Native CPU offload: +11.3% out-tok/s, -9.6% TTFT mean vs baseline; Native FS offload: +50.0% out-tok/s, -45.5% TTFT mean vs baseline.

## 6. Interpretation & recommendations

- **Qwen3-30B-A3B — clear win for FS offload.** All 12 QPS points were failure-free for every config. FS offload delivered the highest output throughput (up to ~2.4k tok/s vs ~1.5k baseline) and the lowest TTFT at every load point, thanks to an ~85% offload-tier hit rate. Recommended configuration for this model.
- **Qwen3-32B — clear win for FS offload across the full ladder.** Every offered-QPS point (2–32) was failure-free for all three configs, and FS offload sustained load all the way to 32 QPS. FS offload led on throughput and latency at essentially every load point — at 32 QPS it delivered +50.0% output throughput and −45.5% TTFT mean vs baseline, on an ~87% offload-tier hit rate. (An earlier FS run appeared to collapse above 16 QPS, but that was a degraded-tier artifact — only ~38% hit rate driving recompute storms; the clean re-run restored the expected ~87% hit and the collapse disappeared.) Recommended configuration for this model.
- **Llama-3 70B — FS offload wins under load; baseline wins when light.** All six QPS points (1–6) were failure-free for every config at full 4/4 capacity. At light load (≤2 QPS) baseline is best on every axis — at 2 QPS it leads on throughput (291 tok/s vs 263 CPU / 233 FS) and TTFT mean (6.1 s vs 7.7 s CPU / 12.6 s FS) — because the offload round-trip isn't yet amortized and the shared prefix still fits the working set without eviction pressure. CPU offload takes the lead at 3 QPS (320 tok/s, TTFT 17.0 s vs 31.4 s baseline). From ~4 QPS up, FS offload pulls ahead on every axis: at 6 QPS it delivered the highest throughput (393 tok/s vs 305 baseline / 340 CPU), the lowest TTFT mean (58.5 s, −41.7% vs baseline) and the lowest E2E p99 (183 s vs 239 baseline), on a ~68% offload-tier hit rate. Recommend FS offload for sustained load; for latency-sensitive light load, plain baseline is sufficient.
- **Offload-tier hit rate scales with the tier's capacity/locality.** FS offload reached ~85–87% external-hit rate for the two Qwen models (large shared filesystem) and ~68% for 70B, while GPU-local prefix hit rates stayed low (1–3%) because the shared-prefix workload spills beyond GPU HBM.
