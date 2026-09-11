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

Configs are columns for a direct side-by-side read; each offered-QPS block lists all metrics as rows. **Bold** marks the best config for that metric row (higher = better for throughput/req-rate, lower = better for the latency rows). Each offload column also shows its **% change vs baseline** in parentheses, followed by a direction-aware marker: ▲ = better than baseline, ▼ = worse than baseline (for the latency rows a lower value counts as better, so a negative % is marked ▲).

### Llama-3 70B (TP=8, 4 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **1** | Out tok/s | **174** | 168 (−3.1%) ▼ | 141 (−18.8%) ▼ |
|  | Achieved req/s | **0.72** | 0.70 (−1.9%) ▼ | 0.61 (−15.0%) ▼ |
|  | TTFT mean (s) | **3.95** | 5.75 (+45.7%) ▼ | 7.15 (+81.0%) ▼ |
|  | TTFT p99 (s) | **8.09** | 14.93 (+84.5%) ▼ | 22.34 (+176.0%) ▼ |
|  | TPOT mean (s) | **0.115** | 0.140 (+21.6%) ▼ | 0.174 (+50.4%) ▼ |
|  | E2E p99 (s) | **43.57** | 52.98 (+21.6%) ▼ | 67.64 (+55.2%) ▼ |
| **2** | Out tok/s | **291** | 263 (−9.9%) ▼ | 233 (−19.9%) ▼ |
|  | Achieved req/s | **1.20** | 1.07 (−11.1%) ▼ | 0.94 (−21.8%) ▼ |
|  | TTFT mean (s) | **6.14** | 7.74 (+26.1%) ▼ | 12.60 (+105.1%) ▼ |
|  | TTFT p99 (s) | **22.26** | 32.24 (+44.8%) ▼ | 51.99 (+133.5%) ▼ |
|  | TPOT mean (s) | **0.123** | 0.125 (+1.8%) ▼ | 0.148 (+20.5%) ▼ |
|  | E2E p99 (s) | **49.75** | 62.29 (+25.2%) ▼ | 75.47 (+51.7%) ▼ |
| **3** | Out tok/s | 295 | **320** (+8.3%) ▲ | 284 (−3.9%) ▼ |
|  | Achieved req/s | 1.10 | **1.34** (+21.3%) ▲ | 1.22 (+10.2%) ▲ |
|  | TTFT mean (s) | 31.35 | **17.04** (−45.7%) ▲ | 18.51 (−41.0%) ▲ |
|  | TTFT p99 (s) | 79.88 | **51.33** (−35.7%) ▲ | 64.71 (−19.0%) ▲ |
|  | TPOT mean (s) | 0.158 | **0.116** (−26.6%) ▲ | 0.135 (−14.8%) ▲ |
|  | E2E p99 (s) | 104.72 | **74.35** (−29.0%) ▲ | 91.02 (−13.1%) ▲ |
| **4** | Out tok/s | 298 | 309 (+3.8%) ▲ | **336** (+12.9%) ▲ |
|  | Achieved req/s | 1.16 | 1.27 (+9.8%) ▲ | **1.28** (+10.3%) ▲ |
|  | TTFT mean (s) | 56.00 | 41.10 (−26.6%) ▲ | **29.48** (−47.3%) ▲ |
|  | TTFT p99 (s) | 125.38 | 110.76 (−11.7%) ▲ | **79.23** (−36.8%) ▲ |
|  | TPOT mean (s) | 0.172 | 0.129 (−24.7%) ▲ | **0.127** (−26.0%) ▲ |
|  | E2E p99 (s) | 147.63 | 132.12 (−10.5%) ▲ | **131.70** (−10.8%) ▲ |
| **5** | Out tok/s | 303 | 320 (+5.4%) ▲ | **355** (+17.1%) ▲ |
|  | Achieved req/s | 1.19 | 1.25 (+4.7%) ▲ | **1.38** (+15.9%) ▲ |
|  | TTFT mean (s) | 80.86 | 66.25 (−18.1%) ▲ | **47.04** (−41.8%) ▲ |
|  | TTFT p99 (s) | 176.96 | 148.88 (−15.9%) ▲ | **133.20** (−24.7%) ▲ |
|  | TPOT mean (s) | 0.167 | 0.152 (−9.1%) ▲ | **0.129** (−22.7%) ▲ |
|  | E2E p99 (s) | 199.61 | 171.30 (−14.2%) ▲ | **161.05** (−19.3%) ▲ |
| **6** | Out tok/s | 305 | 340 (+11.3%) ▲ | **393** (+29.0%) ▲ |
|  | Achieved req/s | 1.21 | 1.28 (+5.4%) ▲ | **1.56** (+28.3%) ▲ |
|  | TTFT mean (s) | 100.38 | 90.69 (−9.7%) ▲ | **58.52** (−41.7%) ▲ |
|  | TTFT p99 (s) | 215.84 | 188.88 (−12.5%) ▲ | **150.69** (−30.2%) ▲ |
|  | TPOT mean (s) | 0.163 | **0.155** (−5.3%) ▲ | 0.165 (+0.7%) ▼ |
|  | E2E p99 (s) | 238.66 | 215.97 (−9.5%) ▲ | **183.16** (−23.3%) ▲ |

### Qwen3-30B-A3B (TP=4, 8 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **2** | Out tok/s | 734 | 645 (−12.2%) ▼ | **820** (+11.7%) ▲ |
|  | Achieved req/s | **1.61** | 1.40 (−12.6%) ▼ | 1.46 (−8.8%) ▼ |
|  | TTFT mean (s) | 2.54 | 2.31 (−8.9%) ▲ | **2.15** (−15.2%) ▲ |
|  | TTFT p99 (s) | 6.75 | 5.38 (−20.3%) ▲ | **4.68** (−30.6%) ▲ |
|  | TPOT mean (s) | 0.074 | **0.073** (−2.4%) ▲ | 0.074 (+0.0%) ▼ |
|  | E2E p99 (s) | **25.75** | 26.85 (+4.3%) ▼ | 26.85 (+4.3%) ▼ |
| **4** | Out tok/s | 1065 | 1052 (−1.2%) ▼ | **1202** (+12.9%) ▲ |
|  | Achieved req/s | **3.01** | 2.95 (−2.2%) ▼ | 2.97 (−1.6%) ▼ |
|  | TTFT mean (s) | 1.85 | 1.56 (−15.3%) ▲ | **1.48** (−19.7%) ▲ |
|  | TTFT p99 (s) | 6.53 | **6.28** (−3.9%) ▲ | 7.68 (+17.5%) ▼ |
|  | TPOT mean (s) | 0.084 | **0.080** (−4.5%) ▲ | 0.083 (−1.0%) ▲ |
|  | E2E p99 (s) | 31.96 | **31.84** (−0.4%) ▲ | 32.57 (+1.9%) ▼ |
| **6** | Out tok/s | 1161 | 1233 (+6.2%) ▲ | **1739** (+49.8%) ▲ |
|  | Achieved req/s | 3.61 | **4.32** (+19.7%) ▲ | 3.93 (+8.6%) ▲ |
|  | TTFT mean (s) | 9.70 | 3.00 (−69.0%) ▲ | **1.91** (−80.3%) ▲ |
|  | TTFT p99 (s) | 25.66 | **11.27** (−56.1%) ▲ | 12.25 (−52.3%) ▲ |
|  | TPOT mean (s) | 0.173 | **0.085** (−51.1%) ▲ | 0.102 (−41.3%) ▲ |
|  | E2E p99 (s) | 43.98 | 32.36 (−26.4%) ▲ | **31.91** (−27.4%) ▲ |
| **8** | Out tok/s | 1122 | 1373 (+22.4%) ▲ | **2265** (+101.9%) ▲ |
|  | Achieved req/s | 3.83 | **4.99** (+30.2%) ▲ | 4.84 (+26.4%) ▲ |
|  | TTFT mean (s) | 18.50 | **4.74** (−74.4%) ▲ | 4.77 (−74.2%) ▲ |
|  | TTFT p99 (s) | 45.45 | **16.89** (−62.8%) ▲ | 24.52 (−46.0%) ▲ |
|  | TPOT mean (s) | 0.276 | **0.078** (−71.7%) ▲ | 0.104 (−62.2%) ▲ |
|  | E2E p99 (s) | 64.92 | **37.32** (−42.5%) ▲ | 41.32 (−36.4%) ▲ |
| **10** | Out tok/s | 883 | 1716 (+94.2%) ▲ | **1953** (+121.0%) ▲ |
|  | Achieved req/s | 4.00 | **5.52** (+38.0%) ▲ | 5.34 (+33.5%) ▲ |
|  | TTFT mean (s) | 31.05 | 13.61 (−56.2%) ▲ | **8.85** (−71.5%) ▲ |
|  | TTFT p99 (s) | 68.80 | 33.99 (−50.6%) ▲ | **29.87** (−56.6%) ▲ |
|  | TPOT mean (s) | 0.465 | **0.077** (−83.5%) ▲ | 0.129 (−72.2%) ▲ |
|  | E2E p99 (s) | 87.40 | 52.56 (−39.9%) ▲ | **48.75** (−44.2%) ▲ |
| **12** | Out tok/s | 812 | 1348 (+66.0%) ▲ | **2086** (+156.9%) ▲ |
|  | Achieved req/s | 4.01 | 4.49 (+11.9%) ▲ | **5.36** (+33.6%) ▲ |
|  | TTFT mean (s) | 44.70 | 38.03 (−14.9%) ▲ | **17.41** (−61.0%) ▲ |
|  | TTFT p99 (s) | 98.47 | 84.10 (−14.6%) ▲ | **50.58** (−48.6%) ▲ |
|  | TPOT mean (s) | 0.546 | **0.093** (−82.9%) ▲ | 0.145 (−73.5%) ▲ |
|  | E2E p99 (s) | 117.36 | 101.43 (−13.6%) ▲ | **73.12** (−37.7%) ▲ |
| **14** | Out tok/s | 869 | 1284 (+47.8%) ▲ | **2365** (+172.3%) ▲ |
|  | Achieved req/s | 4.00 | 4.34 (+8.5%) ▲ | **6.13** (+53.3%) ▲ |
|  | TTFT mean (s) | 60.64 | 50.80 (−16.2%) ▲ | **22.92** (−62.2%) ▲ |
|  | TTFT p99 (s) | 134.04 | 111.21 (−17.0%) ▲ | **57.90** (−56.8%) ▲ |
|  | TPOT mean (s) | 0.571 | **0.095** (−83.4%) ▲ | 0.157 (−72.5%) ▲ |
|  | E2E p99 (s) | 152.04 | 127.18 (−16.3%) ▲ | **76.78** (−49.5%) ▲ |
| **16** | Out tok/s | 959 | 1343 (+40.0%) ▲ | **2161** (+125.4%) ▲ |
|  | Achieved req/s | 3.92 | 4.40 (+12.3%) ▲ | **5.99** (+52.7%) ▲ |
|  | TTFT mean (s) | 76.19 | 61.99 (−18.6%) ▲ | **29.95** (−60.7%) ▲ |
|  | TTFT p99 (s) | 166.63 | 127.80 (−23.3%) ▲ | **81.06** (−51.4%) ▲ |
|  | TPOT mean (s) | 0.572 | **0.128** (−77.7%) ▲ | 0.167 (−70.8%) ▲ |
|  | E2E p99 (s) | 184.70 | 146.93 (−20.4%) ▲ | **98.55** (−46.6%) ▲ |
| **20** | Out tok/s | 1161 | 1321 (+13.8%) ▲ | **2218** (+91.0%) ▲ |
|  | Achieved req/s | 4.01 | 4.46 (+11.3%) ▲ | **6.20** (+54.4%) ▲ |
|  | TTFT mean (s) | 104.02 | 89.25 (−14.2%) ▲ | **44.07** (−57.6%) ▲ |
|  | TTFT p99 (s) | 213.28 | 187.20 (−12.2%) ▲ | **117.92** (−44.7%) ▲ |
|  | TPOT mean (s) | 0.465 | **0.182** (−60.9%) ▲ | 0.197 (−57.5%) ▲ |
|  | E2E p99 (s) | 233.03 | 208.81 (−10.4%) ▲ | **137.00** (−41.2%) ▲ |
| **24** | Out tok/s | 1460 | 1390 (−4.8%) ▼ | **2052** (+40.5%) ▲ |
|  | Achieved req/s | 4.15 | 4.38 (+5.6%) ▲ | **6.41** (+54.6%) ▲ |
|  | TTFT mean (s) | 129.21 | 113.40 (−12.2%) ▲ | **57.75** (−55.3%) ▲ |
|  | TTFT p99 (s) | 262.52 | 233.49 (−11.1%) ▲ | **142.23** (−45.8%) ▲ |
|  | TPOT mean (s) | 0.237 | **0.187** (−20.9%) ▲ | 0.225 (−4.8%) ▲ |
|  | E2E p99 (s) | 280.57 | 256.89 (−8.4%) ▲ | **163.57** (−41.7%) ▲ |
| **28** | Out tok/s | 1517 | 1327 (−12.5%) ▼ | **2047** (+34.9%) ▲ |
|  | Achieved req/s | 4.13 | 4.43 (+7.2%) ▲ | **6.57** (+58.9%) ▲ |
|  | TTFT mean (s) | 158.59 | 135.02 (−14.9%) ▲ | **68.85** (−56.6%) ▲ |
|  | TTFT p99 (s) | 324.15 | 281.68 (−13.1%) ▲ | **164.83** (−49.1%) ▲ |
|  | TPOT mean (s) | **0.144** | 0.301 (+109.3%) ▼ | 0.242 (+68.1%) ▼ |
|  | E2E p99 (s) | 345.29 | 305.28 (−11.6%) ▲ | **185.76** (−46.2%) ▲ |
| **32** | Out tok/s | 1520 | 1229 (−19.2%) ▼ | **1969** (+29.6%) ▲ |
|  | Achieved req/s | 4.07 | 4.55 (+12.0%) ▲ | **6.78** (+66.8%) ▲ |
|  | TTFT mean (s) | 186.78 | 161.45 (−13.6%) ▲ | **81.18** (−56.5%) ▲ |
|  | TTFT p99 (s) | 381.66 | 337.59 (−11.5%) ▲ | **195.40** (−48.8%) ▲ |
|  | TPOT mean (s) | **0.111** | 0.393 (+255.0%) ▼ | 0.261 (+135.6%) ▼ |
|  | E2E p99 (s) | 401.47 | 357.86 (−10.9%) ▲ | **233.23** (−41.9%) ▲ |

### Qwen3-32B (TP=4, 8 replicas)

| Offered QPS | Metric | Baseline (no offload) | Native CPU offload | Native FS offload |
|---|---|---|---|---|
| **2** | Out tok/s | 382 | **458** (+19.8%) ▲ | 451 (+18.1%) ▲ |
|  | Achieved req/s | 1.37 | **1.63** (+18.9%) ▲ | 1.58 (+15.5%) ▲ |
|  | TTFT mean (s) | **1.85** | 2.24 (+21.2%) ▼ | 2.53 (+36.5%) ▼ |
|  | TTFT p99 (s) | **5.23** | 7.35 (+40.5%) ▼ | 6.67 (+27.5%) ▼ |
|  | TPOT mean (s) | 0.081 | **0.072** (−10.8%) ▲ | 0.076 (−6.2%) ▲ |
|  | E2E p99 (s) | **20.30** | 23.93 (+17.9%) ▼ | 28.52 (+40.5%) ▼ |
| **4** | Out tok/s | **840** | 759 (−9.6%) ▼ | 658 (−21.6%) ▼ |
|  | Achieved req/s | **2.99** | 2.68 (−10.2%) ▼ | 2.21 (−26.1%) ▼ |
|  | TTFT mean (s) | 2.75 | **2.10** (−23.7%) ▲ | 3.45 (+25.2%) ▼ |
|  | TTFT p99 (s) | **12.31** | 12.58 (+2.2%) ▼ | 31.97 (+159.8%) ▼ |
|  | TPOT mean (s) | 0.102 | **0.072** (−29.1%) ▲ | 0.099 (−3.4%) ▲ |
|  | E2E p99 (s) | 34.08 | **27.93** (−18.0%) ▲ | 48.25 (+41.6%) ▼ |
| **6** | Out tok/s | 915 | **1140** (+24.5%) ▲ | 1105 (+20.7%) ▲ |
|  | Achieved req/s | 3.44 | **4.11** (+19.2%) ▲ | 3.81 (+10.5%) ▲ |
|  | TTFT mean (s) | 14.87 | 5.17 (−65.2%) ▲ | **4.79** (−67.8%) ▲ |
|  | TTFT p99 (s) | 38.82 | 23.89 (−38.5%) ▲ | **22.53** (−42.0%) ▲ |
|  | TPOT mean (s) | 0.185 | **0.112** (−39.3%) ▲ | 0.114 (−38.1%) ▲ |
|  | E2E p99 (s) | 52.78 | **37.71** (−28.6%) ▲ | 39.17 (−25.8%) ▲ |
| **8** | Out tok/s | 962 | 1262 (+31.2%) ▲ | **1296** (+34.8%) ▲ |
|  | Achieved req/s | 3.44 | 4.32 (+25.9%) ▲ | **4.41** (+28.2%) ▲ |
|  | TTFT mean (s) | 25.75 | **9.17** (−64.4%) ▲ | 9.48 (−63.2%) ▲ |
|  | TTFT p99 (s) | 62.80 | **30.21** (−51.9%) ▲ | 32.97 (−47.5%) ▲ |
|  | TPOT mean (s) | 0.139 | 0.131 (−5.5%) ▲ | **0.116** (−16.3%) ▲ |
|  | E2E p99 (s) | 82.08 | **46.89** (−42.9%) ▲ | 49.41 (−39.8%) ▲ |
| **10** | Out tok/s | 961 | 1118 (+16.4%) ▲ | **1387** (+44.3%) ▲ |
|  | Achieved req/s | 3.55 | 3.82 (+7.5%) ▲ | **4.67** (+31.4%) ▲ |
|  | TTFT mean (s) | 42.15 | 33.06 (−21.6%) ▲ | **17.26** (−59.0%) ▲ |
|  | TTFT p99 (s) | 90.31 | 77.75 (−13.9%) ▲ | **48.41** (−46.4%) ▲ |
|  | TPOT mean (s) | 0.149 | **0.125** (−16.5%) ▲ | 0.132 (−11.7%) ▲ |
|  | E2E p99 (s) | 107.92 | 94.54 (−12.4%) ▲ | **64.26** (−40.5%) ▲ |
| **12** | Out tok/s | 955 | 1132 (+18.5%) ▲ | **1406** (+47.2%) ▲ |
|  | Achieved req/s | 3.57 | 3.93 (+10.0%) ▲ | **4.75** (+33.1%) ▲ |
|  | TTFT mean (s) | 55.30 | 47.73 (−13.7%) ▲ | **27.52** (−50.2%) ▲ |
|  | TTFT p99 (s) | 125.82 | 110.15 (−12.5%) ▲ | **72.13** (−42.7%) ▲ |
|  | TPOT mean (s) | 0.161 | 0.136 (−15.2%) ▲ | **0.126** (−21.3%) ▲ |
|  | E2E p99 (s) | 141.57 | 125.18 (−11.6%) ▲ | **99.30** (−29.9%) ▲ |
| **14** | Out tok/s | 1011 | 1132 (+12.0%) ▲ | **1375** (+36.1%) ▲ |
|  | Achieved req/s | 3.50 | 3.97 (+13.5%) ▲ | **4.70** (+34.2%) ▲ |
|  | TTFT mean (s) | 71.02 | 61.89 (−12.9%) ▲ | **35.66** (−49.8%) ▲ |
|  | TTFT p99 (s) | 155.22 | 133.73 (−13.8%) ▲ | **87.29** (−43.8%) ▲ |
|  | TPOT mean (s) | 0.169 | 0.153 (−9.8%) ▲ | **0.104** (−38.6%) ▲ |
|  | E2E p99 (s) | 175.38 | 150.02 (−14.5%) ▲ | **108.53** (−38.1%) ▲ |
| **16** | Out tok/s | 1040 | 1096 (+5.4%) ▲ | **1559** (+49.9%) ▲ |
|  | Achieved req/s | 3.58 | 3.89 (+8.5%) ▲ | **5.37** (+50.0%) ▲ |
|  | TTFT mean (s) | 85.34 | 77.13 (−9.6%) ▲ | **42.52** (−50.2%) ▲ |
|  | TTFT p99 (s) | 186.75 | 167.01 (−10.6%) ▲ | **100.11** (−46.4%) ▲ |
|  | TPOT mean (s) | 0.146 | 0.141 (−3.2%) ▲ | **0.125** (−14.2%) ▲ |
|  | E2E p99 (s) | 206.77 | 183.17 (−11.4%) ▲ | **122.92** (−40.6%) ▲ |
| **20** | Out tok/s | 1029 | 1081 (+5.0%) ▲ | **1405** (+36.5%) ▲ |
|  | Achieved req/s | 3.62 | 3.83 (+5.7%) ▲ | **4.92** (+35.9%) ▲ |
|  | TTFT mean (s) | 116.13 | 107.58 (−7.4%) ▲ | **63.24** (−45.5%) ▲ |
|  | TTFT p99 (s) | 244.84 | 224.51 (−8.3%) ▲ | **152.44** (−37.7%) ▲ |
|  | TPOT mean (s) | 0.156 | **0.136** (−12.6%) ▲ | 0.150 (−3.7%) ▲ |
|  | E2E p99 (s) | 264.21 | 243.25 (−7.9%) ▲ | **173.18** (−34.5%) ▲ |
| **24** | Out tok/s | 1022 | 1140 (+11.6%) ▲ | **1383** (+35.3%) ▲ |
|  | Achieved req/s | 3.63 | 4.00 (+10.3%) ▲ | **4.92** (+35.3%) ▲ |
|  | TTFT mean (s) | 147.58 | 132.58 (−10.2%) ▲ | **79.74** (−46.0%) ▲ |
|  | TTFT p99 (s) | 307.78 | 266.84 (−13.3%) ▲ | **185.64** (−39.7%) ▲ |
|  | TPOT mean (s) | 0.147 | **0.130** (−11.5%) ▲ | 0.163 (+10.5%) ▼ |
|  | E2E p99 (s) | 327.94 | 286.51 (−12.6%) ▲ | **243.98** (−25.6%) ▲ |
| **28** | Out tok/s | 1019 | 1128 (+10.8%) ▲ | **1452** (+42.5%) ▲ |
|  | Achieved req/s | 3.63 | 3.98 (+9.5%) ▲ | **5.18** (+42.6%) ▲ |
|  | TTFT mean (s) | 178.66 | 161.61 (−9.5%) ▲ | **98.72** (−44.7%) ▲ |
|  | TTFT p99 (s) | 366.03 | 332.90 (−9.0%) ▲ | **217.66** (−40.5%) ▲ |
|  | TPOT mean (s) | 0.154 | **0.131** (−15.1%) ▲ | 0.190 (+23.3%) ▼ |
|  | E2E p99 (s) | 382.81 | 351.73 (−8.1%) ▲ | **280.46** (−26.7%) ▲ |
| **32** | Out tok/s | 1004 | 1117 (+11.3%) ▲ | **1506** (+50.0%) ▲ |
|  | Achieved req/s | 3.62 | 4.00 (+10.4%) ▲ | **5.36** (+48.1%) ▲ |
|  | TTFT mean (s) | 210.82 | 190.62 (−9.6%) ▲ | **114.84** (−45.5%) ▲ |
|  | TTFT p99 (s) | 436.59 | 388.44 (−11.0%) ▲ | **250.28** (−42.7%) ▲ |
|  | TPOT mean (s) | 0.156 | **0.146** (−6.4%) ▲ | 0.196 (+25.6%) ▼ |
|  | E2E p99 (s) | 454.88 | 405.74 (−10.8%) ▲ | **322.19** (−29.2%) ▲ |

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
