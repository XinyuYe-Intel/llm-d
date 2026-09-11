# Tiered Prefix Cache benchmark on Intel Arc Pro B60 (XPU)

One-click benchmark harness that measures the effect of **KV-cache tiering**
(VRAM → CPU RAM → filesystem) on prefix-cache-heavy inference, running on the
Intel Arc Pro B60 (XPU) cluster with vLLM + llm-d + `llm-d-benchmark`.

Everything lives in [run-all-benchmarks.sh](run-all-benchmarks.sh): it deploys a
multi-replica vLLM model server, installs the llm-d router (EPP), drives the
guide's shared-prefix workload against it once per KV-tiering configuration, and
writes a consolidated comparison report.

> This README documents the whole setup **and every pitfall** hit while bringing
> the pipeline up on this cluster. If a run breaks, read
> [Pitfalls & fixes](#pitfalls--fixes) first — most failure modes are already
> handled by the script and explained there.

---

## 1. What it measures

For each feasible model the runner benchmarks three KV-cache configurations:

| Config       | KV path                     | Meaning |
|--------------|-----------------------------|---------|
| `baseline`   | VRAM only                   | Control — no offloading. |
| `native-cpu` | VRAM → CPU RAM              | vLLM `OffloadingConnector`, headline path. |
| `native-fs`  | VRAM → CPU RAM → filesystem | `TieringOffloadingSpec` with a node-local NVMe FS tier. |

The shared-prefix workload is sized (per replica) so the KV working set
(`num_groups × system_prompt_len`) exceeds what VRAM + CPU can hold, so only the
FS tier can cache the whole thing. Expected prefix-cache-hit ordering:
`baseline < native-cpu < native-fs`.

> LMCache is intentionally **excluded** — it has an XPU bug (cannot allocate an
> offload tensor > 16 GB).

---

## 2. Cluster assumptions (verified)

- **4 worker nodes × 8 × Intel Arc Pro B60** (24 GB GDDR6) = 32 XPUs, scheduled
  via **DRA** (`gpu.intel.com` driver, `ResourceClaimTemplate`).
- No Xe-Link between cards → every TP all-reduce **and** every KV VRAM↔host DMA
  rides PCIe. Each node = 4 PCIe roots × 2 GPUs split across 2 CPU sockets/NUMA
  nodes. The runner pins each TP group to a single socket (see
  `PIN_GPU_TOPOLOGY`).
- vLLM image `ghcr.io/llm-d/llm-d-xpu:v0.9.0` (vLLM v0.26.0), container name
  **`modelserver`**, TP via `multiproc_executor`, oneCCL/`xccl` backend.
- **No default `StorageClass` / no dynamic provisioner** — only static,
  no-provisioner local classes. PVCs must be backed by pre-created static PVs.
- Shared Hugging Face cache is an **NFS export** `10.112.228.229:/huggingface-cache`.
  It is mounted read-only into decode pods at `/cache/huggingface` and is also
  mounted locally on the host at `/srv/huggingface-cache` (writable).
- A corporate **HTTP proxy** (`proxy-ir.intel.com:912`) is required for external
  egress (github.io, ghcr.io, huggingface.co). Cluster IPs are in `NO_PROXY`.

---

## 3. Prerequisites

- `kubectl`, `helm`, `python3`, `git` on PATH; a working kubeconfig for the
  cluster.
- Namespace exists and has the HF-token secret:
  ```bash
  kubectl create ns llm-d-tiered-prefix-cache
  # see helpers/hf-token.md for the llm-d-hf-token secret
  ```
- `llm-d-benchmark` — auto-cloned + installed into its own venv if
  `llmdbenchmark` is not already on PATH (no `sudo` needed).
- Proxy env exported in the launching shell (the script keeps the proxy for
  external tools and surgically bypasses it for the in-cluster API — see
  [Pitfall 1](#pitfall-1--kubernetes-python-client-ignores-no_proxy)):
  ```bash
  export HTTP_PROXY=http://proxy-ir.intel.com:912
  export HTTPS_PROXY=http://proxy-ir.intel.com:912
  export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,.svc,.cluster.local,10.112.228.229
  ```

---

## 4. How to run

### 4a. Smoke test first (recommended)

Validates the **entire** pipeline (deploy → router → workload PV → tokenizer seed
→ harness → report) for every config using one tiny model that loads in seconds,
instead of ~17 min per 30B/70B cold load.

```bash
cd /path/to/llm-d
WS=./tpc-xpu-smoke-$(date +%Y%m%d-%H%M%S)
setsid nohup env SMOKE=true WORKSPACE="$WS" \
  bash guides/tiered-prefix-cache/benchmark-xpu/run-all-benchmarks.sh \
  > "$WS/run.log" 2>&1 &
echo "$WS" > /tmp/tpc_ws_path.txt   # optional: remember the workspace
```

Smoke knobs (all optional): `SMOKE_MODEL` (default `Qwen/Qwen3-0.6B`, **must be
in the NFS cache with weights** — see [Pitfall 6](#pitfall-6--nfs-cache-entry-was-tokenizer-only-no-weights)),
`SMOKE_TP` (2), `SMOKE_REPLICAS` (2), `SMOKE_CONFIGS`.

### 4b. Full production run

```bash
cd /path/to/llm-d
WS=./tpc-xpu-results-$(date +%Y%m%d-%H%M%S)
setsid nohup env WORKSPACE="$WS" \
  bash guides/tiered-prefix-cache/benchmark-xpu/run-all-benchmarks.sh \
  > "$WS/run.log" 2>&1 &
```

Default matrix: `MODELS=(qwen3-30b-a3b qwen3-32b llama3-70b)` ×
`CONFIGS=(baseline native-cpu native-fs)`.

**Always launch detached** (`setsid nohup … &`) — a single run is long, and the
router (helm release) is intentionally left running between runs.

### 4c. Useful overrides

| Env var | Default | Purpose |
|---|---|---|
| `NAMESPACE` | `llm-d-tiered-prefix-cache` | Target namespace. |
| `INCLUDE_FS` | `true` | Also run `native-fs`. |
| `ROLLOUT_TIMEOUT` | `7200s` | Rollout wait + `progressDeadlineSeconds` (large weights load slowly over NFS). |
| `PIN_GPU_TOPOLOGY` | `true` | Pin each TP<8 group to one CPU socket. |
| `FS_HOSTPATH` | `/mnt/data/tpc-kv-fs` | Node-local NVMe path backing the FS tier. |
| `LLMDBENCH_REPO` | `$HOME/llm-d-benchmark` | Reuse an existing checkout. |

---

## 5. What the runner does (stages)

1. **`bootstrap_benchmark`** — clone + venv-install `llm-d-benchmark` if needed.
2. **`preflight`** — check tools, namespace, HF-token secret; source `guides/env.sh`.
3. **`deploy_router`** — `helm install` the standalone EPP router once (reused
   across all models/configs).
4. **`ensure_workload_pv`** — create the static NFS PV backing the harness's
   RWX `workload-pvc` ([Pitfall 2](#pitfall-2--workload-pvc-stuck-pending)).
5. **`ensure_harness_tokenizers`** — seed each model's tokenizer files into the
   workload PVC's HF cache ([Pitfall 3](#pitfall-3--harness-cant-load-the-tokenizer-offline)).
6. **`gen_workload_profile`** — render per-model `inference-perf` workload YAML.
7. **`run_one`** per (model, config): render kustomize overlay → delete stale
   `ResourceClaimTemplate` → `kubectl apply -k` → wait rollout → run
   `llmdbenchmark … run --analyze` → tear down.
8. **`generate_report`** — aggregate every run's `summary_lifecycle_metrics.json`
   into `report.md` + `report.csv` ([Pitfall 5](#pitfall-5--consolidated-report-was-empty)).

---

## 6. Outputs

```
<WORKSPACE>/
├── run.log                         # full runner log
├── report.md                       # consolidated comparison (baseline vs offloading)
├── report.csv                      # machine-readable summary
└── <model>-<config>/               # e.g. smoke-baseline/, qwen3-32b-native-fs/
    └── <run-id>/results/<exp>_N/
        ├── *.png                   # latency_vs_qps, throughput_vs_qps, throughput_vs_latency
        ├── benchmark_report,_stage_N_lifecycle_metrics.json.yaml
        └── workspace.tar.zst       # PACKED: summary/stage/per_request JSON + config.yaml
```

> The canonical `summary_lifecycle_metrics.json` lives **inside**
> `workspace.tar.zst`, not on disk. Extract it with:
> ```bash
> tar --use-compress-program=unzstd -xOf workspace.tar.zst ./summary_lifecycle_metrics.json
> ```

---

## 7. Pitfalls & fixes

Every one of these was hit while bringing the pipeline up; all are now handled
automatically by the script (this section explains *why*).

### Pitfall 1 — Kubernetes Python client ignores `NO_PROXY`

- **Symptom:** `llmdbenchmark run` fails with
  `RuntimeError: Failed to scan cluster nodes … ProxyError 403` against
  `10.112.228.229:6443`.
- **Cause:** the k8s Python client reads `HTTPS_PROXY` into
  `Configuration.proxy` but **never populates `no_proxy`**, so
  `should_bypass_proxies()` tunnels in-cluster API calls through the corporate
  proxy → 403.
- **Fix:** a surgical `sitecustomize.py` shim in
  [k8s-proxy-fix/](k8s-proxy-fix/) that sets `Configuration.no_proxy` from the
  environment. The runner injects it via `PYTHONPATH` **only** for the
  `llmdbenchmark` call, so external tools (helm/skopeo) keep the proxy while the
  API host is bypassed. (The blunt alternative — stripping all proxy env — breaks
  helm repo/skopeo image lookups, so it was rejected.)

### Pitfall 2 — `workload-pvc` stuck `Pending`

- **Symptom:** harness step *"Prepare harness namespace"* creates an RWX
  `workload-pvc` that never binds; the run times out.
- **Cause:** the cluster has **no default `StorageClass`** / dynamic provisioner.
- **Fix:** `ensure_workload_pv()` pre-creates a **static NFS PV** (RWX, empty
  `storageClassName`, `Retain`, `claimRef` pre-bound to `workload-pvc`) backed by
  an NFS subdir. Idempotent and reused across configs.

### Pitfall 3 — Harness can't load the tokenizer offline

- **Symptom:** harness pod fails with
  `OSError: couldn't connect to huggingface.co … Tokenizer initialization failed`
  for **every** config.
- **Cause:** the harness pod template hardcodes `HF_HOME=/requests/.cache/huggingface`
  (the workload PVC), mounts **no** HF hub cache, and has no internet (proxy
  blocks HF). `AutoTokenizer` can't find the tokenizer.
- **Fix:** `ensure_harness_tokenizers()` seeds each model's **small non-weight
  files** (`tokenizer*.json`, `vocab.json`, `merges.txt`, `config.json`, …;
  weights excluded) — dereferenced (`cp -L`) — from the shared NFS hub cache into
  the workload PVC's HF cache subdir. The runner also injects
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` into the harness via
  `--envvarspod`. Keep `HARNESS_HF_HOME` empty so the seed path matches the
  template's `HF_HOME`.

### Pitfall 4 — Stale `ResourceClaimTemplate` is immutable

- **Symptom:** `kubectl apply -k` fails with
  `ResourceClaimTemplate … spec is immutable` after an interrupted run that used
  a different TP / pinning shape.
- **Cause:** `ResourceClaimTemplate.spec` is immutable; a leftover template can't
  be updated in place.
- **Fix:** `run_one()` does a **delete-before-apply** of the template
  (`xpu-vllm-<router-release>-intel-claim-template-decode`) before
  `kubectl apply -k`. Safe: already-running pods keep their bound
  `ResourceClaim`s; the template is only read at pod-creation time.

### Pitfall 5 — Consolidated report was empty

- **Symptom:** the full pipeline ran and every harness pod `Completed`, but
  `report.md` said *"No results found"* and `report.csv` rows were blank.
- **Cause:** the `inference-perf` harness does **not** leave
  `summary_lifecycle_metrics.json` on disk — it **packs it inside**
  `results/<exp>_N/workspace.tar.zst`. On disk you only get per-stage
  `benchmark_report,_stage_N_…json.yaml` (not the aggregated summary) + charts,
  so `generate_report()`'s glob found nothing.
- **Fix:** `newest_summary()` now falls back to extracting
  `./summary_lifecycle_metrics.json` from the newest `workspace.tar.zst`. Python
  3.12's `tarfile` has no zstd support, so it shells out to
  `tar --use-compress-program=unzstd -xOf …`.

### Pitfall 6 — NFS cache entry was tokenizer-only (no weights)

- **Symptom:** decode pod crash-loops with
  `RuntimeError: Cannot find any model weights with …/models--Qwen--Qwen3-0.6B/snapshots/<hash>`.
- **Cause:** the NFS hub-cache entry for `Qwen3-0.6B` was **tokenizer-only** — no
  `*.safetensors`. Decode pods mount the shared NFS at `/cache/huggingface` with
  **no subpath**, so they read the real hub cache; the weights simply weren't
  there. (Of the small models, only `Qwen3-8B` had full weights cached.)
- **Fix:** download the missing weights into the writable NFS export. The shared
  `hf` CLI wrapper pointed at another user's venv (`Permission denied`), so make a
  private one:
  ```bash
  python3 -m venv /tmp/hfdl && /tmp/hfdl/bin/pip install huggingface_hub
  HF_HOME=/srv/huggingface-cache /tmp/hfdl/bin/hf download \
    Qwen/Qwen3-0.6B --revision <snapshot-hash>
  chmod a+rX /srv/huggingface-cache/hub/models--Qwen--Qwen3-0.6B/.../model.safetensors
  ```
  (`/srv/huggingface-cache` is the same NFS export, mounted writable on the host.)

### Other gotchas

- **Editing the runner does not affect an already-running process** — bash parses
  functions at start. Kill and relaunch after any edit.
- **Cold model load is slow & NFS-bound** — ~17 min for a 30B/70B (weights stream
  over NFS at ~50 MB/s, then XPU `torch.compile` warmup). Workers go into D-state
  (IO wait) during load — normal. Hence `ROLLOUT_TIMEOUT`/`progressDeadlineSeconds`
  = 7200s.
- **Same-socket GPU pinning** — with no Xe-Link, spreading a TP group across
  sockets adds cross-NUMA PCIe hops. `PIN_GPU_TOPOLOGY=true` emits a
  `firstAvailable` claim with one CEL-restricted alternative per socket so the
  whole group lands on one socket (falls back to the other under contention).
- **Benign warnings** during setup (`WVA image tag could not be resolved`,
  `acceleratorType constraint dropped`, `customCommand override will not
  propagate`) are expected on this DRA cluster and do not fail the run.

---

## 8. Cleanup

```bash
# Router is left running by design; remove it when finished:
helm uninstall tiered-prefix-cache -n llm-d-tiered-prefix-cache

# The static workload PV persists (Retain) for reuse; delete if no longer needed:
kubectl delete pv b60bench-llm-d-tiered-prefix-cache-workload-pvc
```
