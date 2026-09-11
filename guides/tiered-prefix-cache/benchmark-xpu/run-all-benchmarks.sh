#!/usr/bin/env bash
#
# One-click tiered-prefix-cache benchmark runner for the Intel Arc Pro B60 (XPU) cluster.
#
# Cluster (verified): 4 worker nodes x 8 x Intel Arc Pro B60 (24 GB GDDR6 VRAM) = 32 XPUs.
#
# For each feasible model it deploys a multi-replica vLLM model server (sized to use all
# 32 XPUs) and drives the guide's shared-prefix workload against it, once per configuration:
#   - baseline   : VRAM only (no offloading)              -> control
#   - native-cpu : OffloadingConnector, VRAM -> CPU RAM    -> headline path
#   - native-fs  : TieringOffloadingSpec, VRAM -> CPU -> FS -> optional (needs RWX PVC)
#
# LMCache is intentionally excluded (XPU bug: cannot allocate an offload tensor > 16 GB).
#
# Workload parameters are sized to SHOWCASE the VRAM->CPU->FS tiering: per replica the KV
# working set is pushed past what VRAM+CPU can hold, so only the FS tier can cache all of it.
# In units of the in-VRAM KV cache "C" (per replica):
#   - working set ~= 3C  -> exceeds VRAM+CPU (=2C); baseline & native-cpu must evict/recompute
#   - CPU offload ~= 1C  -> native-cpu caches ~2/3 of the working set
#   - FS tier      = node-local NVMe (hostPath) -> native-fs caches ~all of the working set
#   Expected prefix-cache-hit ordering: baseline (~1/3) < native-cpu (~2/3) < native-fs (~all).
#   - load: CLOSED-LOOP fixed concurrency (inference-perf `type: concurrent`), NOT an open-loop
#     request-rate ladder. A fixed number of requests is kept in flight, so the client can never
#     enqueue faster than the servers drain -> latency stays bounded and NO request ever times out
#     (the previous open-loop Poisson ladder offered 4-32x the served rate and timed out ~80% of
#     requests at its top stage). The offload TIER (baseline/cpu/fs) is the single independent var.
#
# Methodology note: the workload is defined PER REPLICA and auto-scaled to cluster-wide totals
# by multiplying NUM_GROUPS and the in-flight CONCURRENCY by REPLICAS (the EPP pins each prefix
# group to a server). Change REPLICAS and the workload rescales automatically; just re-check the
# router EPP `lruCapacityPerServer` so prefix routing never dilutes a server's cache below its
# VRAM size (which would hide the offloading benefit). Concurrency stays far below each server's
# vLLM `--max-num-seq`, so even fully skewed prefix routing cannot fill a server's admission queue.
#
# Model weights load OFFLINE from the cluster's shared HF cache (inline NFS, no downloads);
# no HF token or network egress is required for the served models.
#
# All results are preserved for further analysis: each run's native harness output, local
# analysis, benchmark_report, and graphs are saved under WORKSPACE/<model>-<config>/, and a
# consolidated WORKSPACE/report.md (+ report.csv) comparing baseline vs offloading is written
# at the end.
#
# The llm-d-benchmark tool is bootstrapped automatically: if `llmdbenchmark` is not already
# on PATH, the script clones llm-d-benchmark into LLMDBENCH_REPO, creates a .venv, and
# pip-installs the CLI (no sudo required), then activates that venv for the rest of the run.
# Point LLMDBENCH_REPO at an existing checkout to reuse it.
#
# Usage:
#   export NAMESPACE=llm-d-tiered-prefix-cache        # your own namespace
#   export LLMDBENCH_REPO=/path/to/llm-d-benchmark      # optional; auto-cloned if absent
#   export INCLUDE_FS=false                             # true requires an RWX StorageClass
#   export WORKSPACE=/path/to/save/results              # optional; default ./tpc-xpu-results-<ts>
#   ./run-all-benchmarks.sh
#
set -euo pipefail

# ----------------------------------------------------------------------------------------
# 0. Configuration (override via environment)
# ----------------------------------------------------------------------------------------
NAMESPACE="${NAMESPACE:-llm-d-tiered-prefix-cache}"
INCLUDE_FS="${INCLUDE_FS:-true}"                       # also run the VRAM->CPU->FS path
ROUTER_RELEASE="${ROUTER_RELEASE:-tiered-prefix-cache}"
GATEWAY_CLASS="${GATEWAY_CLASS:-epponly}"               # standalone EPP mode
HARNESS="${HARNESS:-inference-perf}"
# Model servers load large weights (e.g. ~57 GiB for Qwen3-30B-A3B) OFFLINE over NFS and then
# run XPU torch.compile warmup, which can take well over the Deployment's DEFAULT 600s
# progressDeadlineSeconds. We therefore (a) wait this long on `kubectl rollout status`, and
# (b) stamp the SAME value into each generated Deployment's spec.progressDeadlineSeconds so it
# is not self-marked ProgressDeadlineExceeded mid-load. Accepts a bare number or a Ns suffix.
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10800s}"
PROGRESS_DEADLINE_SECONDS="${ROLLOUT_TIMEOUT%s}"
# Cold NFS loads of the largest model (131 GiB weights, TP=8, all 8 workers reading at once) can
# exceed the container image's DEFAULT startupProbe window (~60 min) and get restarted mid-load
# (kubelet: "failed startup probe, will be restarted"). Stamp a generous startupProbe into each
# generated Deployment so a cold load has ~3h before it is declared failed. Budget =
# initialDelaySeconds(15) + failureThreshold x periodSeconds. Success ends the probe immediately,
# so this only RAISES the ceiling and is harmless for models that load fast. Keep it <= the
# rollout/progress deadline above, or the Deployment is marked failed before the probe window ends.
STARTUP_PROBE_PERIOD="${STARTUP_PROBE_PERIOD:-30}"       # seconds between probe attempts
STARTUP_PROBE_FAILURES="${STARTUP_PROBE_FAILURES:-360}"  # x 30s -> ~3h max cold-load window

# Max wall-time llm-d-benchmark waits for the inference-perf HARNESS pod to reach a terminal
# phase before declaring the run failed ("Pods did not complete within Ns"). The harness default
# is 3600s (1h). A saturating rate ladder makes each over-ceiling stage drain a growing backlog
# (requests up to RATE_REQUEST_TIMEOUT), so total harness wall-time can exceed 1h -- e.g. the
# 12-stage 30b sweep needed ~66 min and was killed ~4 min short. Raise the ceiling here; it is a
# pure timeout so short runs are unaffected. Passed via --wait-timeout (env LLMDBENCH_WAIT_TIMEOUT).
HARNESS_WAIT_TIMEOUT="${HARNESS_WAIT_TIMEOUT:-10800}"    # seconds (3h); harness pod completion deadline

# Local Hugging Face cache: models are served OFFLINE from the cluster's shared NFS export
# (no downloads). Each pod mounts it at HF_MOUNT and sets HF_HOME there. See the cluster's
# /usr/local/share/doc/xpu-cluster-nfs-guide.txt. Flip HF_NFS_READONLY=false if HF must write
# lock files into the cache.
HF_NFS_SERVER="${HF_NFS_SERVER:-10.112.228.229}"
HF_NFS_PATH="${HF_NFS_PATH:-/huggingface-cache}"
HF_MOUNT="${HF_MOUNT:-/cache/huggingface}"
HF_NFS_READONLY="${HF_NFS_READONLY:-true}"

# Where model servers READ weights from. The shared NFS export is a single-server bandwidth
# bottleneck: every case reloads its weights, and a cold 70b load (132 GiB, TP=8, 8 workers
# reading at once) takes ~1h. Set HF_SOURCE=local to instead read from each worker's local NVMe
# (HF_LOCAL_PATH). Weights are staged ONCE from NFS onto every node's local disk by an
# idempotent DaemonSet (stage_models_local) before the deploy loop; subsequent loads hit local
# NVMe and are far faster. hub/ layout is preserved, so HF_HOME=HF_MOUNT is unchanged.
HF_SOURCE="${HF_SOURCE:-nfs}"                            # nfs | local
HF_LOCAL_PATH="${HF_LOCAL_PATH:-/mnt/data/huggingface-cache}"  # per-node NVMe HF cache (contains hub/)
# Image for the staging DaemonSet. Default = the model-server image, already present on every
# node (no extra pull). Only needs a shell + GNU cp (preserves the blobs/snapshots symlinks).
STAGE_IMAGE="${STAGE_IMAGE:-ghcr.io/llm-d/llm-d-xpu:v0.9.0}"
STAGE_NODES="${STAGE_NODES:-smc-18 smc-19 smc-20 smc-22}"      # worker nodes to stage onto
STAGE_KEEP="${STAGE_KEEP:-false}"                       # true = leave the staging DS running after

# Node-local scratch backing the FS tier (native-fs config). This cluster has NO RWX
# StorageClass (only static no-provisioner local classes), so the VRAM->CPU->FS "storage"
# tier is a per-node NVMe hostPath rather than a PVC. Each replica writes under its own
# POD_NAME subdir, and the native-fs pod runs as root so it can create that subdir. Point
# this at a large local disk (verified: /mnt/data is a ~3.4 TB NVMe with ~2 TB free).
FS_HOSTPATH="${FS_HOSTPATH:-/mnt/data/tpc-kv-fs}"

# --- GPU PCIe-topology-aware allocation (TP<8 only) -------------------------------------
# These are discrete Arc Pro B60 cards with NO Xe-Link: every TP all-reduce AND every KV
# VRAM<->host (CPU/FS tier) DMA rides PCIe. On each node the 8 XPUs sit on 4 PCIe roots x 2
# GPUs, and the 4 roots split across 2 CPU sockets / NUMA nodes (VERIFIED via the DRA
# ResourceSlice attributes; the NIC slices give the root->NUMA ground truth):
#   socket/NUMA 0 : root pci0000:26 (pciRoot "26")  + root pci0000:37 (pciRoot "37")
#   socket/NUMA 1 : root pci0000:a7 (pciRoot "a7")  + root pci0000:b7 (pciRoot "b7")
# The gpu.intel.com DRA driver exposes NO numaNode on GPUs, only `pciRoot`, so DRA cannot be
# asked for "same socket, either socket" via a single matchAttribute. Instead we express each
# socket as a `firstAvailable` alternative sub-request whose CEL selector restricts the claim
# to that socket's two roots: the allocator tries socket 0 first and, if its GPUs aren't all
# free on the chosen node, falls back to socket 1. Either way the whole TP group lands on ONE
# socket -> same-NUMA all-reduce + KV offload and no run-to-run placement variance, while still
# allowing two TP=4 replicas per node (one per socket) and staying schedulable under contention.
# (Confirmed real: a current unpinned TP=4 claim spans roots 37+a7+b7 = both sockets.)
# TP>=8 spans the whole node (both sockets) by definition -> no pinning is possible or needed.
# Set PIN_GPU_TOPOLOGY=false to fall back to the default random (unconstrained) allocation, or
# edit GPU_SOCKET_ROOTS (one socket per line, short `pciRoot` values) if the topology differs.
PIN_GPU_TOPOLOGY="${PIN_GPU_TOPOLOGY:-true}"
GPU_SOCKET_ROOTS="${GPU_SOCKET_ROOTS:-26 37
a7 b7}"

# Also propagate the offline HF env into the llm-d-benchmark HARNESS pod (it loads the
# tokenizer for the served model) via the tool's --envvarspod passthrough. NOTE: the upstream
# harness pod template exposes NO volume hook, so the NFS hub cannot be mounted into it --
# these vars only keep the harness offline; its tokenizer must already live in the harness
# HF_HOME (its own workload PVC). If the harness cannot load the tokenizer offline, either set
# HARNESS_HF_OFFLINE=false (it then fetches just the small tokenizer via the HF token), or set
# HARNESS_HF_HOME to a path where that workload PVC exposes the shared cache.
HARNESS_HF_OFFLINE="${HARNESS_HF_OFFLINE:-true}"
HARNESS_HF_HOME="${HARNESS_HF_HOME:-}"

REPO_ROOT="$(realpath "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)")"
GUIDE_DIR="${REPO_ROOT}/guides/tiered-prefix-cache"
WORKSPACE="${WORKSPACE:-${PWD}/tpc-xpu-results-$(date +%Y%m%d-%H%M%S)}"

# llm-d-benchmark checkout (contains workload/profiles + puts `llmdbenchmark` on PATH via its venv).
# Auto-cloned + installed by bootstrap_benchmark() if `llmdbenchmark` is not already on PATH.
LLMDBENCH_REPO="${LLMDBENCH_REPO:-${HOME}/llm-d-benchmark}"
LLMDBENCH_REPO_URL="${LLMDBENCH_REPO_URL:-https://github.com/llm-d/llm-d-benchmark.git}"
LLMDBENCH_BRANCH="${LLMDBENCH_BRANCH:-main}"
# Versions used by the no-sudo install fallback (upstream install.sh needs sudo for system
# packages; we install into the venv instead). Override if the pins drift.
BENCH_YQ_VERSION="${BENCH_YQ_VERSION:-v4.53.6}"
BENCH_PLANNER_REF="${BENCH_PLANNER_REF:-git+https://github.com/llm-d-incubation/llm-d-planner.git@v0.1.0}"

# Models to benchmark (feasible on this cluster). Order: light -> heavy.
# All weights are loaded OFFLINE from the local NFS cache (verified present in the hub).
# NOTE: Llama-3.1-70B-Instruct is not cached; the only local dense 70B is Meta-Llama-3-70B,
#       which is architecturally identical (80L, 8 KV heads, 128 head_dim -> 320 KiB/token),
#       so the sizing below is unchanged.
MODELS=(qwen3-30b-a3b qwen3-32b llama3-70b)
# Optional: restrict/reorder the model matrix (e.g. MODELS_OVERRIDE="qwen3-30b-a3b qwen3-32b" to
# validate just the two qwen models). Mirrors CONFIGS_OVERRIDE below.
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<< "${MODELS_OVERRIDE}"
fi

declare -A HF_MODEL=(
  [qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"
  [qwen3-32b]="Qwen/Qwen3-32B"
  [llama3-70b]="meta-llama/Meta-Llama-3-70B"
)
declare -A TP=(          [qwen3-30b-a3b]=4            [qwen3-32b]=4            [llama3-70b]=8 )
declare -A REPLICAS=(    [qwen3-30b-a3b]=8            [qwen3-32b]=8            [llama3-70b]=4 )
# Optional: override replica counts (e.g. for a small validation run). Accepts either a bare
# number applied to every model (REPLICAS_OVERRIDE=2) or space-separated per-model "model=count"
# pairs (REPLICAS_OVERRIDE="qwen3-30b-a3b=4 qwen3-32b=4 llama3-70b=2"). Per-replica sizing
# (working set, CPU tier, concurrency) is unchanged; only the cluster-wide NUM_GROUPS/CONC totals
# scale. Must run BEFORE the auto-scale derivations below.
if [[ -n "${REPLICAS_OVERRIDE:-}" ]]; then
  if [[ "${REPLICAS_OVERRIDE}" == *=* ]]; then
    for _kv in ${REPLICAS_OVERRIDE}; do REPLICAS["${_kv%%=*}"]="${_kv##*=}"; done
    unset _kv
  else
    for _m in "${MODELS[@]}"; do REPLICAS[$_m]="${REPLICAS_OVERRIDE}"; done
    unset _m
  fi
fi
#   VRAM KV cache "C" (tokens, MEASURED from vllm "GPU KV cache size" @ gpu-mem-util 0.85):
#                                30B-A3B 196,736   32B 50,944   70B ~72K (unmeasured est)
#   KV bytes/token:              30B-A3B 96 KiB  32B 256 KiB  70B 320 KiB
# To SHOWCASE the VRAM->CPU->FS tiering, size (per replica): working set ~3C. The CPU offload tier
# is DERIVED below as CPU_TIER_PCT%% (default 80%%) of the per-replica working set (in KV bytes), so
# native-cpu holds most of the working set in host RAM and native-fs's FS tier absorbs the remainder.
# NOTE: at 80%% the CPU tier alone ~= 0.8x working set, so VRAM+CPU covers essentially the whole
# working set -- native-cpu should now clearly beat baseline, but its margin over native-fs shrinks.
# CRITICAL retention rule: the KV of all IN-FLIGHT requests (CONC/replica x SYS_LEN) must stay
# well under C, or completed prefixes are evicted before they can be reused and the prefix-cache
# hit rate collapses for EVERY config (this is exactly what CONC_CLUSTER=128 did on 2026-09-04:
# 16 in-flight x 12000 = 192K ~= 98% of C, so native-cpu tied baseline and its TTFT got worse).
# Keep CONC_PER_REPLICA low; do NOT use CONC_CLUSTER for the tiering comparison. See guardrail below.
#   KV bytes/token (aggregate over TP workers; matches measured VRAM C): 30b 96 KiB, 32b 256 KiB, 70b 320 KiB
declare -A KV_BYTES_PER_TOKEN=( [qwen3-30b-a3b]=98304  [qwen3-32b]=262144  [llama3-70b]=327680 )
declare -A CPU_BYTES   # DERIVED below = CPU_TIER_PCT%% of per-replica working set (NUM_GROUPS_PER_REPLICA x SYS_LEN x KV bytes/token)
declare -A MEM=(         [qwen3-30b-a3b]=80Gi         [qwen3-32b]=80Gi         [llama3-70b]=100Gi )
# Workload (shared_prefix) shape, defined PER REPLICA. The cluster-wide NUM_GROUPS and CONCURRENCY
# are DERIVED below by multiplying these by REPLICAS. Per replica the working set
# (NUM_GROUPS x SYS_LEN) ~= 3C; the CPU tier is sized to 0.8x this working set (see CPU_BYTES).
declare -A NUM_GROUPS_PER_REPLICA=( [qwen3-30b-a3b]=58           [qwen3-32b]=56           [llama3-70b]=54 )
declare -A SYS_LEN=(                [qwen3-30b-a3b]=12000        [qwen3-32b]=3600         [llama3-70b]=4000 )
# Closed-loop in-flight concurrency PER REPLICA (cluster concurrency = this x REPLICAS). Kept well
# under the server's vLLM --max-num-seq (128) so requests are admitted immediately and never queue
# long enough to time out -- this is what structurally guarantees zero failed requests. It is a
# steady operating point, not a saturation sweep: the experiment varies the offload TIER, not load.
# Sized so live in-flight KV (this x SYS_LEN) stays ~<=0.5C of the MEASURED VRAM C above, leaving
# headroom to RETAIN completed prefixes for reuse: 30b 8x12000=96K (0.49C), 32b 6x3600=21.6K (0.42C).
declare -A CONC_PER_REPLICA=(       [qwen3-30b-a3b]=8            [qwen3-32b]=6            [llama3-70b]=8 )
PROMPTS_PER_GROUP=5
QUESTION_LEN=256
OUTPUT_LEN=256

# Auto-scale to cluster-wide totals: cluster NUM_GROUPS and in-flight CONCURRENCY = per-replica x REPLICAS.
declare -A NUM_GROUPS CONC
for _m in "${MODELS[@]}"; do
  _reps="${REPLICAS[$_m]}"
  NUM_GROUPS[$_m]=$(( NUM_GROUPS_PER_REPLICA[$_m] * _reps ))
  CONC[$_m]=$(( CONC_PER_REPLICA[$_m] * _reps ))
done
unset _m _reps

# Derive the CPU offload buffer per replica = CPU_TIER_PCT% (default 80) of the per-replica working-set
# KV bytes = (NUM_GROUPS_PER_REPLICA x SYS_LEN) x KV_BYTES_PER_TOKEN, rounded to a whole GiB.
CPU_TIER_PCT="${CPU_TIER_PCT:-80}"
_gib=1073741824
for _m in "${MODELS[@]}"; do
  _ws=$(( NUM_GROUPS_PER_REPLICA[$_m] * SYS_LEN[$_m] ))                    # reusable prefix tokens / replica
  _raw=$(( _ws * KV_BYTES_PER_TOKEN[$_m] / 100 * CPU_TIER_PCT ))          # target CPU bytes (/100 first to avoid overflow)
  CPU_BYTES[$_m]=$(( ( (_raw + _gib/2) / _gib ) * _gib ))                 # round to whole GiB
done
unset _m _ws _raw _gib

# Optional: pin every model's CLUSTER in-flight concurrency to a fixed value (overrides the
# per-replica x REPLICAS derivation above). This is a MAX-QPS PROBE ONLY: it saturates the batch
# but breaks prefix retention (live in-flight KV approaches VRAM C -> hit rate collapses for all
# configs), so do NOT use it for the tiering (baseline vs native-cpu vs native-fs) comparison.
if [[ -n "${CONC_CLUSTER:-}" ]]; then
  for _m in "${MODELS[@]}"; do CONC[$_m]="${CONC_CLUSTER}"; done
  unset _m
fi

# ---------------------------------------------------------------------------------------
# Load mode. Default `concurrent` = the closed-loop steady-state operating point used for the
# tiering (baseline vs native-cpu vs native-fs) comparison. `rate` = an OPEN-LOOP Poisson QPS
# sweep matching the published B60 benchmark-results tables (varying target rate; the per-stage
# lifecycle metrics give TTFT/E2E/throughput vs load).
#
# The ladder is defined PER REPLICA and multiplied by REPLICAS to get each stage's CLUSTER rate,
# exactly like NUM_GROUPS/CONC. The per-replica numbers are the viable single-replica baseline
# ladder: stage 1 sits just below one replica's saturation knee and the top stage ~3x it (the
# most that still drains losslessly). Calibrate on 1 replica first, e.g.:
#   LOAD_MODE=rate REPLICAS_OVERRIDE="qwen3-30b-a3b=1" CONFIGS_OVERRIDE=baseline MODELS_OVERRIDE=qwen3-30b-a3b ...
# and require every stage to report 0 failures with TTFT rising from stage 1 -> 4; then trim/raise
# the ladder and run all configs at full replicas (rates scale automatically).
#
# Guardrails learned from the earlier open-loop run that failed 100% at its 5-QPS floor:
#   - short 60s stages (NOT the old 600s warmup) so the backlog drains after each burst,
#   - request_timeout 1800s >> worst-case queue wait so nothing times out,
#   - top rate <= ~3x the MEASURED per-replica ceiling; never start above the ceiling.
LOAD_MODE="${LOAD_MODE:-concurrent}"                  # concurrent | rate
declare -A RATE_PER_REPLICA_QPS=(
  [qwen3-30b-a3b]="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0"                 # 12k prefix, TP=4; ceiling ~0.34 q/s/replica -> cluster 2/4/6/8
  [qwen3-32b]="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0"                     # 3.6k prefix, TP=4 -> cluster 4/6/8/10
  [llama3-70b]="0.25 0.5 0.75 1.0 1.25 1.5"                      # 4k prefix, TP=8; dense, heaviest per token -> cluster 2/4/6/8
)
RATE_STAGE_DURATION="${RATE_STAGE_DURATION:-60}"      # seconds per ladder stage
RATE_REQUEST_TIMEOUT="${RATE_REQUEST_TIMEOUT:-1800}"  # >> worst-case queue wait at the top rate

# CONFIGS_OVERRIDE (space-separated) restricts/reorders the configs, e.g. "baseline" to run
# only the control case. When unset, run baseline + native-cpu (+ native-fs if INCLUDE_FS).
if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
  read -r -a CONFIGS <<< "${CONFIGS_OVERRIDE}"
else
  CONFIGS=(baseline native-cpu)
  [[ "${INCLUDE_FS}" == "true" ]] && CONFIGS+=(native-fs)
fi

# Generated overlays must live inside the repo tree: kustomize rejects absolute
# `resources` paths, and relative `..` bases only resolve cleanly from here.
GEN_ROOT="${GUIDE_DIR}/benchmark-xpu/.gen"
rm -rf "${GEN_ROOT}"; mkdir -p "${GEN_ROOT}"
trap 'rm -rf "${GEN_ROOT}"' EXIT

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------------------
# 0b. SMOKE mode: validate the WHOLE pipeline cheaply with one tiny model first
# ----------------------------------------------------------------------------------------
# Set SMOKE=true to replace the production model matrix with a single small model that loads
# in seconds (vs ~17 min for the 30B/70B), exercising deploy -> router -> workload PV ->
# tokenizer seed -> harness -> report for EVERY config. Knobs (all optional):
#   SMOKE_MODEL   HF repo, must be in the NFS hub cache   (default Qwen/Qwen3-0.6B)
#   SMOKE_TP      tensor-parallel size                    (default 2)
#   SMOKE_REPLICAS number of decode replicas              (default 2)
#   SMOKE_CONFIGS space-separated config list             (default: baseline native-cpu[+native-fs])
if [[ "${SMOKE:-false}" == "true" ]]; then
  _sm="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
  _sk="smoke"
  MODELS=("${_sk}")
  HF_MODEL=(               [${_sk}]="${_sm}" )
  TP=(                     [${_sk}]="${SMOKE_TP:-2}" )
  REPLICAS=(               [${_sk}]="${SMOKE_REPLICAS:-2}" )
  CPU_BYTES=(              [${_sk}]=2147483648 )   # 2 GiB CPU offload buffer (native-cpu)
  MEM=(                    [${_sk}]=16Gi )
  NUM_GROUPS_PER_REPLICA=( [${_sk}]=4 )
  SYS_LEN=(                [${_sk}]=512 )
  CONC_PER_REPLICA=(       [${_sk}]="${SMOKE_CONC:-4}" )
  _reps="${REPLICAS[$_sk]}"
  NUM_GROUPS=( [${_sk}]=$(( NUM_GROUPS_PER_REPLICA[$_sk] * _reps )) )
  CONC=( [${_sk}]=$(( CONC_PER_REPLICA[$_sk] * _reps )) )
  [[ -n "${SMOKE_CONFIGS:-}" ]] && read -r -a CONFIGS <<< "${SMOKE_CONFIGS}"
  unset _sm _reps
  log "SMOKE mode: model=${HF_MODEL[$_sk]} TP=${TP[$_sk]} replicas=${REPLICAS[$_sk]} configs=(${CONFIGS[*]})"
  unset _sk
fi

# Prefix-retention guardrail: warn loudly if the live in-flight KV per replica (CONC/replica x
# SYS_LEN) approaches the MEASURED per-replica VRAM KV cache "C". Above ~60% of C there is no room
# to retain completed prefixes for reuse, so the prefix-cache hit rate collapses for every config
# (baseline and offload alike) -- exactly the CONC_CLUSTER=128 failure of 2026-09-04.
declare -A VRAM_KV_TOKENS=( [qwen3-30b-a3b]=196736 [qwen3-32b]=50944 [llama3-70b]=0 )
for _m in "${MODELS[@]}"; do
  _c="${VRAM_KV_TOKENS[$_m]:-0}"; (( _c == 0 )) && continue
  _reps="${REPLICAS[$_m]}"
  _live=$(( CONC[$_m] / _reps * SYS_LEN[$_m] ))   # in-flight prefix tokens per replica
  _pct=$(( _live * 100 / _c ))
  _cpugib=$(( CPU_BYTES[$_m] / 1073741824 ))       # derived CPU tier (GiB)
  _ws=$(( NUM_GROUPS_PER_REPLICA[$_m] * SYS_LEN[$_m] ))
  _cpupct=$(( CPU_BYTES[$_m] / KV_BYTES_PER_TOKEN[$_m] * 100 / _ws ))   # CPU tier as % of working set
  log "tier ${_m}: CPU offload ${_cpugib}GiB = ${_cpupct}% of working set (${_ws} tok/replica)"
  if (( _pct > 60 )); then
    log "WARN ${_m}: in-flight KV ${_live} tok = ${_pct}% of VRAM C (${_c}) -> prefix retention starved, hit rate will collapse; lower CONC_PER_REPLICA / drop CONC_CLUSTER."
  else
    log "retention ${_m}: in-flight KV ${_live} tok = ${_pct}% of VRAM C (${_c}) (ok, headroom to retain prefixes)"
  fi
done
unset _m _c _reps _live _pct _cpugib _ws _cpupct

# ----------------------------------------------------------------------------------------
# 1. Bootstrap + Preflight
# ----------------------------------------------------------------------------------------
# Ensure llm-d-benchmark is cloned + installed and its `llmdbenchmark` CLI is on PATH.
# Idempotent: reuses an existing checkout/venv; only clones/installs what is missing.
bootstrap_benchmark() {
  if command -v llmdbenchmark >/dev/null 2>&1 \
     && [[ -d "${LLMDBENCH_REPO}/workload/profiles/${HARNESS}" ]]; then
    log "llm-d-benchmark already available (llmdbenchmark on PATH); reusing ${LLMDBENCH_REPO}"
    return
  fi

  command -v git >/dev/null || fail "git not found (needed to clone llm-d-benchmark)"

  if [[ ! -d "${LLMDBENCH_REPO}/.git" ]]; then
    log "Cloning llm-d-benchmark (${LLMDBENCH_BRANCH}) into ${LLMDBENCH_REPO}"
    git clone --branch "${LLMDBENCH_BRANCH}" --depth 1 "${LLMDBENCH_REPO_URL}" "${LLMDBENCH_REPO}"
  else
    log "Reusing existing llm-d-benchmark checkout at ${LLMDBENCH_REPO}"
  fi

  if [[ ! -x "${LLMDBENCH_REPO}/.venv/bin/llmdbenchmark" ]]; then
    # Upstream install.sh needs sudo to install system packages, which is unavailable on many
    # shared clusters. Install into a self-contained venv instead (kubectl/helm are already
    # present; run-only mode does not need helmfile).
    command -v python3 >/dev/null || fail "python3 not found (needed to create the venv)"
    log "Creating virtualenv + installing llmdbenchmark (no sudo)"
    ( cd "${LLMDBENCH_REPO}"
      python3 -m venv .venv
      ./.venv/bin/python -m pip install --quiet --upgrade pip
      # llmd-benchmark-report is an in-repo package (not on PyPI); install it first so the
      # llmdbenchmark dependency on it resolves against the local checkout.
      ./.venv/bin/python -m pip install --quiet -e ./benchmark-report
      ./.venv/bin/python -m pip install --quiet -e .
      ./.venv/bin/python -m pip install --quiet "${BENCH_PLANNER_REF}" \
        || log "planner install failed (continuing; not required for run-only mode)"
    ) || fail "venv install failed in ${LLMDBENCH_REPO}"

    # yq is used for template rendering; provide it in the venv bin if the host lacks it.
    if ! command -v yq >/dev/null 2>&1 && [[ ! -x "${LLMDBENCH_REPO}/.venv/bin/yq" ]]; then
      local yq_arch=""
      case "$(uname -m)" in x86_64) yq_arch=amd64 ;; aarch64|arm64) yq_arch=arm64 ;; esac
      if [[ -n "${yq_arch}" ]] && command -v curl >/dev/null 2>&1; then
        log "Installing yq ${BENCH_YQ_VERSION} into venv (host yq missing, no sudo)"
        curl -fsSL "https://github.com/mikefarah/yq/releases/download/${BENCH_YQ_VERSION}/yq_linux_${yq_arch}" \
          -o "${LLMDBENCH_REPO}/.venv/bin/yq" && chmod +x "${LLMDBENCH_REPO}/.venv/bin/yq" \
          || log "yq download failed (continuing; install yq manually if rendering needs it)"
      fi
    fi
  fi

  # Activate the venv so `llmdbenchmark` is on PATH for the rest of this run.
  # shellcheck disable=SC1091
  source "${LLMDBENCH_REPO}/.venv/bin/activate"
  command -v llmdbenchmark >/dev/null \
    || fail "llmdbenchmark still not on PATH after install (check ${LLMDBENCH_REPO}/.venv)"
  log "llmdbenchmark ready: $(llmdbenchmark --version 2>/dev/null || echo 'version unknown')"
}

preflight() {
  command -v kubectl >/dev/null || fail "kubectl not found"
  command -v helm    >/dev/null || fail "helm not found"
  command -v llmdbenchmark >/dev/null || fail "llmdbenchmark not on PATH (bootstrap_benchmark should have installed it)"
  [[ -n "${LLMDBENCH_REPO}" && -d "${LLMDBENCH_REPO}/workload/profiles/${HARNESS}" ]] \
    || fail "LLMDBENCH_REPO=${LLMDBENCH_REPO} is missing workload/profiles/${HARNESS}"

  # shellcheck disable=SC1091
  source "${REPO_ROOT}/guides/env.sh"

  kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || fail "Namespace ${NAMESPACE} does not exist"
  kubectl get secret llm-d-hf-token -n "${NAMESPACE}" >/dev/null 2>&1 \
    || fail "Secret llm-d-hf-token missing in ${NAMESPACE} (see helpers/hf-token.md)"

  mkdir -p "${WORKSPACE}"
  log "Workspace: ${WORKSPACE}"
  log "Models: ${MODELS[*]}  |  Configs: ${CONFIGS[*]}  |  Namespace: ${NAMESPACE}"
}

# ----------------------------------------------------------------------------------------
# 2. Router (deployed once, reused across all models)
# ----------------------------------------------------------------------------------------
deploy_router() {
  if helm status "${ROUTER_RELEASE}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log "Router '${ROUTER_RELEASE}' already installed, reusing."
    return
  fi
  log "Installing router '${ROUTER_RELEASE}' (standalone EPP)"
  helm install "${ROUTER_RELEASE}" "${ROUTER_STANDALONE_CHART}" \
    -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
    -f "${GUIDE_DIR}/router/tiered-prefix-cache-cpu.values.yaml" \
    -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}"
}

# ----------------------------------------------------------------------------------------
# 2b. Workload PVC backing storage (static NFS PV, created once, reused across configs)
# ----------------------------------------------------------------------------------------
# The inference-perf harness creates a RWX `workload-pvc` (storageClassName "auto" ->
# omitted). This cluster has no dynamic provisioner / default StorageClass, so the PVC
# stays Pending forever. Mirror the cluster's established b60bench pattern: pre-create a
# static NFS PV (RWX, empty storageClassName) pre-bound to the namespace's workload-pvc.
WORKLOAD_PV_NAME="b60bench-${NAMESPACE}-workload-pvc"
WORKLOAD_NFS_SERVER="${WORKLOAD_NFS_SERVER:-10.112.228.229}"
WORKLOAD_NFS_EXPORT="${WORKLOAD_NFS_EXPORT:-/huggingface-cache}"
WORKLOAD_NFS_SUBDIR="${WORKLOAD_NFS_SUBDIR:-_b60bench-tpc-workload}"
ensure_workload_pv() {
  if kubectl get pv "${WORKLOAD_PV_NAME}" >/dev/null 2>&1; then
    log "Workload PV '${WORKLOAD_PV_NAME}' already present, reusing."
    return
  fi
  log "Creating workload PV '${WORKLOAD_PV_NAME}' (static NFS, RWX)"
  # Ensure the NFS subdirectory exists and is writable (harness rsyncs into it).
  kubectl run "wl-pv-mkdir-$$" --restart=Never --rm -i --image=busybox --timeout=120s \
    -n "${NAMESPACE}" \
    --overrides="{\"apiVersion\":\"v1\",\"spec\":{\"volumes\":[{\"name\":\"nfs\",\"nfs\":{\"server\":\"${WORKLOAD_NFS_SERVER}\",\"path\":\"${WORKLOAD_NFS_EXPORT}\"}}],\"containers\":[{\"name\":\"m\",\"image\":\"busybox\",\"command\":[\"sh\",\"-c\",\"mkdir -p /nfs/${WORKLOAD_NFS_SUBDIR} && chmod 0777 /nfs/${WORKLOAD_NFS_SUBDIR} && echo OK\"],\"volumeMounts\":[{\"name\":\"nfs\",\"mountPath\":\"/nfs\"}]}]}}" \
    >/dev/null 2>&1 || log "  (warning) NFS mkdir helper failed; assuming subdir exists"
  cat <<PVYAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${WORKLOAD_PV_NAME}
spec:
  accessModes: [ReadWriteOnce, ReadOnlyMany, ReadWriteMany]
  capacity:
    storage: 20Gi
  claimRef:
    apiVersion: v1
    kind: PersistentVolumeClaim
    name: workload-pvc
    namespace: ${NAMESPACE}
  nfs:
    path: ${WORKLOAD_NFS_EXPORT}/${WORKLOAD_NFS_SUBDIR}
    server: ${WORKLOAD_NFS_SERVER}
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  volumeMode: Filesystem
PVYAML
}

# ----------------------------------------------------------------------------------------
# 2c. Seed model tokenizers into the workload PVC's HF cache (offline harness support)
# ----------------------------------------------------------------------------------------
# The inference-perf harness pod loads each model's tokenizer via AutoTokenizer, but it
# mounts NO HuggingFace hub cache and has no internet (corporate proxy blocks HF). Its
# HF_HOME is hardcoded to <workloadPvc>/.cache/huggingface. Pre-seed the small tokenizer /
# config files (dereferenced, weights excluded) from the shared NFS hub cache into that
# path so HF_HUB_OFFLINE loads succeed. Same NFS export as the workload PVC, so the harness
# sees them at /requests/.cache/huggingface/hub. Idempotent (skips already-seeded models).
ensure_harness_tokenizers() {
  local model_dirs=""
  local m repo
  for m in "${MODELS[@]}"; do
    repo="${HF_MODEL[$m]}"
    [[ -n "${repo}" ]] || continue
    model_dirs+=" models--${repo//\//--}"
  done
  [[ -n "${model_dirs}" ]] || return 0
  log "Seeding harness tokenizers into workload PVC HF cache (models:${model_dirs})"
  local seed_pod="tpc-tok-seed-$$"
  cat <<TOKYAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${seed_pod}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  volumes:
  - name: nfs
    nfs:
      server: ${WORKLOAD_NFS_SERVER}
      path: ${WORKLOAD_NFS_EXPORT}
  containers:
  - name: seed
    image: busybox:1.36
    volumeMounts:
    - name: nfs
      mountPath: /nfs
    command:
    - sh
    - -c
    - |
      set -e
      DEST=/nfs/${WORKLOAD_NFS_SUBDIR}/.cache/huggingface/hub
      mkdir -p "\$DEST"
      for M in ${model_dirs}; do
        SRC=/nfs/hub/\$M
        if [ ! -d "\$SRC" ]; then echo "MISSING \$M (not in shared cache)"; continue; fi
        if find "\$DEST/\$M" -name tokenizer.json 2>/dev/null | grep -q .; then echo "SKIP \$M (already seeded)"; continue; fi
        echo "SEED \$M"
        mkdir -p "\$DEST/\$M"
        cp -r "\$SRC/refs" "\$DEST/\$M/" 2>/dev/null || true
        for snap in "\$SRC"/snapshots/*; do
          [ -d "\$snap" ] || continue
          h=\$(basename "\$snap")
          mkdir -p "\$DEST/\$M/snapshots/\$h"
          for f in "\$snap"/*; do
            b=\$(basename "\$f")
            case "\$b" in *.safetensors|*.bin|*.pt|*.pth|*.gguf|*.onnx|*.h5|*.msgpack) continue;; esac
            cp -L "\$f" "\$DEST/\$M/snapshots/\$h/\$b" 2>/dev/null || true
          done
        done
      done
      chmod -R a+rX "\$DEST" 2>/dev/null || true
      echo SEED_DONE
TOKYAML
  if kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/${seed_pod}" \
       -n "${NAMESPACE}" --timeout=180s >/dev/null 2>&1; then
    kubectl logs "${seed_pod}" -n "${NAMESPACE}" 2>/dev/null | sed 's/^/    | /'
    log "Tokenizer seeding complete."
  else
    kubectl logs "${seed_pod}" -n "${NAMESPACE}" 2>/dev/null | sed 's/^/    | /' || true
    log "  (warning) tokenizer seed pod did not report success; harness may fail to load tokenizers offline"
  fi
  kubectl delete pod "${seed_pod}" -n "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

# ----------------------------------------------------------------------------------------
# 3. Generate per-model workload profiles into the benchmark repo
# ----------------------------------------------------------------------------------------
gen_workload_profile() {
  local model="$1" out="$2"
  # Requests needed to touch every prefix group ~once (num_groups x prompts_per_group). Stage 0
  # warms the CPU/FS tiers with one such pass; stage 1 re-runs it at steady state and is the one
  # the report reads. The closed-loop `concurrent` load keeps exactly CONC requests in flight, so
  # the queue is self-limiting and no request can time out.
  local cover=$(( NUM_GROUPS[$model] * PROMPTS_PER_GROUP ))
  local conc="${CONC[$model]}"

  # Build the `load:` block for the selected LOAD_MODE. `concurrent` = closed-loop steady point
  # (the tiering comparison); `rate` = open-loop Poisson QPS sweep whose CLUSTER rate per stage is
  # the per-replica ladder x REPLICAS, matching the published B60 benchmark-results tables.
  local load_block
  if [[ "${LOAD_MODE}" == "rate" ]]; then
    local reps="${REPLICAS[$model]}"
    local -a _pr; read -r -a _pr <<< "${RATE_PER_REPLICA_QPS[$model]}"
    local _stages="" _r _clu
    for _r in "${_pr[@]}"; do
      _clu=$(awk -v a="${_r}" -v r="${reps}" 'BEGIN{printf "%g", a*r}')
      _stages+="    - { rate: ${_clu}, duration: ${RATE_STAGE_DURATION} }"$'\n'
    done
    load_block="load:
  type: poisson
  interval: 60.0
  request_timeout: ${RATE_REQUEST_TIMEOUT}.0
  stages:
${_stages%$'\n'}"
  else
    load_block="load:
  type: concurrent
  request_timeout: 900.0
  stages:
    - { concurrency_level: ${conc}, num_requests: ${cover} }
    - { concurrency_level: ${conc}, num_requests: ${cover} }"
  fi

  cat > "${out}" <<EOF
# Auto-generated tiered-prefix-cache XPU workload for ${HF_MODEL[$model]}  (LOAD_MODE=${LOAD_MODE})
# num_groups=${NUM_GROUPS[$model]} system_prompt_len=${SYS_LEN[$model]} -> working set ~3x in-VRAM
# KV cache per replica (> VRAM+CPU, < VRAM+CPU+FS). concurrent: closed-loop conc=${conc} (cluster).
# rate: open-loop Poisson, cluster QPS/stage = per-replica ladder x ${REPLICAS[$model]} replicas.
${load_block}
api:
  type: completion
  streaming: true
server:
  type: vllm
  model_name: REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL
  base_url: REPLACE_ENV_LLMDBENCH_HARNESS_STACK_ENDPOINT_URL
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL
data:
  type: shared_prefix
  shared_prefix:
    num_groups: ${NUM_GROUPS[$model]}
    num_prompts_per_group: ${PROMPTS_PER_GROUP}
    system_prompt_len: ${SYS_LEN[$model]}
    question_len: ${QUESTION_LEN}
    output_len: ${OUTPUT_LEN}
    enable_multi_turn_chat: false
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
storage:
  local_storage:
    path: /workspace
EOF
}

# ----------------------------------------------------------------------------------------
# 4. Render a kustomize overlay for one (model, config)
# ----------------------------------------------------------------------------------------
render_overlay() {
  local model="$1" config="$2" dir="$3"
  local tp="${TP[$model]}" mdl="${HF_MODEL[$model]}" mem="${MEM[$model]}"
  local reps="${REPLICAS[$model]}" bytes="${CPU_BYTES[$model]}"
  mkdir -p "${dir}"

  # Connector JSON differs per config. Base paths are RELATIVE to the overlay dir
  # (${GEN_ROOT}/<tag>/), which sits at guides/tiered-prefix-cache/benchmark-xpu/.gen/<tag>/.
  local connector_json="" base_overlay
  case "${config}" in
    baseline)
      base_overlay="../../../modelserver/xpu/vllm/base"
      ;;
    native-cpu)
      base_overlay="../../../modelserver/xpu/vllm/base"
      connector_json="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${bytes}}}"
      ;;
    native-fs)
      # No RWX StorageClass on this cluster -> back the FS tier with a per-node NVMe hostPath
      # (wired into the plain base below), not the upstream fs/base PVC. root_dir is the in-pod
      # mountPath; each replica writes under its own POD_NAME subdir (see patch-deploy.yaml).
      base_overlay="../../../modelserver/xpu/vllm/base"
      connector_json="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":${bytes},\"block_size\":256,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"/mnt/files-storage\",\"n_read_threads\":16,\"n_write_threads\":16}]}}"
      ;;
  esac

  # Assemble the full vLLM args block (14-space indented, backslash-continued).
  # Built here so command substitution never strips a trailing newline mid-command.
  local args_block
  args_block="              exec vllm serve \\"$'\n'
  args_block+="                ${mdl} \\"$'\n'
  args_block+="                --dtype bfloat16 \\"$'\n'
  args_block+="                --block-size 64 \\"$'\n'
  args_block+="                --tensor-parallel-size=${tp} \\"$'\n'
  args_block+="                --gpu-memory-utilization 0.85 \\"$'\n'
  if [[ -n "${connector_json}" ]]; then
    args_block+="                --kv-transfer-config \\"$'\n'
    args_block+="                '${connector_json}' \\"$'\n'
  fi
  args_block+="                --max-num-seq 128 \\"$'\n'
  args_block+="                --uvicorn-log-level warning"

  # FS tier (native-fs only): node-local NVMe via hostPath, isolated per replica by POD_NAME,
  # and run as root so the connector can write its DirectoryOrCreate subdir (hostPath volumes
  # are not chowned by fsGroup; matches the upstream gpu/native/fs reference's runAsUser: 0).
  # All four snippets are empty for the baseline/native-cpu configs.
  local sec_ctx="" fs_env="" fs_mount="" fs_volume=""
  if [[ "${config}" == "native-fs" ]]; then
    sec_ctx=$'      securityContext:\n        runAsUser: 0\n'
    fs_env=$'            - name: POD_NAME\n              valueFrom:\n                fieldRef:\n                  fieldPath: metadata.name\n'
    fs_mount=$'            - name: files-storage\n              mountPath: /mnt/files-storage\n              subPathExpr: $(POD_NAME)\n'
    fs_volume=$'\n        - name: files-storage\n          hostPath:\n            path: '"${FS_HOSTPATH}"$'\n            type: DirectoryOrCreate'
  fi

  # hf-cache volume source: shared NFS (default) or per-node local NVMe (HF_SOURCE=local). For
  # local, weights must already be staged under ${HF_LOCAL_PATH}/hub by stage_models_local().
  local hf_volume
  if [[ "${HF_SOURCE}" == "local" ]]; then
    hf_volume=$'        - name: hf-cache\n          hostPath:\n            path: '"${HF_LOCAL_PATH}"$'\n            type: DirectoryOrCreate'
  else
    hf_volume=$'        - name: hf-cache\n          nfs:\n            server: '"${HF_NFS_SERVER}"$'\n            path: '"${HF_NFS_PATH}"$'\n            readOnly: '"${HF_NFS_READONLY}"
  fi

  cat > "${dir}/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ${NAMESPACE}
resources:
  - ${base_overlay}
patches:
  - target: { kind: Deployment, name: xpu-vllm-decode }
    path: patch-deploy.yaml
  - target: { kind: ResourceClaimTemplate }
    path: patch-claim.yaml
EOF

  cat > "${dir}/patch-deploy.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xpu-vllm-decode
spec:
  replicas: ${reps}
  progressDeadlineSeconds: ${PROGRESS_DEADLINE_SECONDS}
  template:
    spec:
${sec_ctx}      containers:
        - name: modelserver
          command: ["/bin/bash", "-c"]
          args:
            - |
${args_block}
          env:
            - name: HF_HOME
              value: ${HF_MOUNT}
            - name: HF_HUB_OFFLINE
              value: "1"
            - name: TRANSFORMERS_OFFLINE
              value: "1"
${fs_env}          volumeMounts:
            - name: hf-cache
              mountPath: ${HF_MOUNT}
${fs_mount}          resources:
            requests: { cpu: '8', memory: ${mem} }
            limits:   { cpu: '8', memory: ${mem} }
          startupProbe:
            httpGet:
              path: /v1/models
              port: modelserver
            initialDelaySeconds: 15
            periodSeconds: ${STARTUP_PROBE_PERIOD}
            timeoutSeconds: 5
            failureThreshold: ${STARTUP_PROBE_FAILURES}
      volumes:
${hf_volume}${fs_volume}
EOF

  # Topology-aware GPU pinning (see PIN_GPU_TOPOLOGY notes above): for TP<8 emit the request as
  # a `firstAvailable` list with one alternative per socket, each restricted (CEL) to that
  # socket's PCIe roots, so the whole TP group lands on a single socket (same-NUMA all-reduce +
  # KV offload, no placement variance) while the allocator can still fall back to the other
  # socket. TP>=8 already spans the whole node, so the request stays a plain `exactly` count.
  local claim_request
  if [[ "${PIN_GPU_TOPOLOGY}" == "true" && "${tp}" -lt 8 ]]; then
    claim_request=$'          firstAvailable:'
    local _sidx=0 _roots _r _roots_cel
    while IFS= read -r _roots; do
      [[ -z "${_roots// }" ]] && continue
      _roots_cel=""
      for _r in ${_roots}; do _roots_cel+=", \"${_r}\""; done
      _roots_cel="${_roots_cel#, }"
      claim_request+=$'\n            - name: socket'"${_sidx}"$'\n              deviceClassName: gpu.intel.com\n              allocationMode: ExactCount\n              count: '"${tp}"$'\n              selectors:\n                - cel:\n                    expression: device.attributes["gpu.intel.com"].pciRoot in ['"${_roots_cel}"$']'
      _sidx=$((_sidx + 1))
    done <<< "${GPU_SOCKET_ROOTS}"
  else
    claim_request=$'          exactly:\n            deviceClassName: gpu.intel.com\n            allocationMode: ExactCount\n            count: '"${tp}"
  fi

  cat > "${dir}/patch-claim.yaml" <<EOF
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: xpu-vllm-tiered-prefix-cache-intel-claim-template-decode
spec:
  spec:
    devices:
      requests:
        - name: gpu
${claim_request}
EOF
}

# ----------------------------------------------------------------------------------------
# Scrape vLLM prefix-cache counters from every decode pod into the run's workspace, so the
# report can show the hit rate per config (proves baseline actually spills / tiers get reuse).
# ----------------------------------------------------------------------------------------
capture_prefix_cache() {
  local tag="$1"
  local outf="${WORKSPACE}/${tag}/prefix_cache_metrics.txt"
  mkdir -p "${WORKSPACE}/${tag}"
  : > "${outf}"
  # Scrape each decode pod's /metrics directly by pod IP from this control node (reachable
  # via the cluster network); kubectl-exec scraping proved unreliable. vLLM exposes token-level
  # counters vllm:prefix_cache_{hits,queries}_total (hit rate = hits/queries).
  local pod ip
  while read -r pod ip; do
    [[ -z "${ip}" ]] && continue
    echo "### pod=${pod} ip=${ip}" >> "${outf}"
    curl -sS -m 10 "http://${ip}:8000/metrics" 2>/dev/null \
      | grep -E 'prefix_cache_(hits|queries)_total' >> "${outf}" || true
  done < <(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/role=decode \
             -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}' 2>/dev/null)
  log "Captured prefix-cache metrics for ${tag} -> ${outf}"
}

# ----------------------------------------------------------------------------------------
# Save each decode pod's vLLM log (current + previous crash, if any) into the run workspace
# so failures (xccl init hangs, OOM, preemption storms) can be diagnosed offline.
# ----------------------------------------------------------------------------------------
capture_decode_logs() {
  local tag="$1"
  local dir="${WORKSPACE}/${tag}/vllm-logs"
  mkdir -p "${dir}"
  local pod
  for pod in $(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/role=decode \
                 -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    kubectl logs "${pod}" -n "${NAMESPACE}" -c modelserver --tail=-1 \
      > "${dir}/${pod}.log" 2>&1 || true
    kubectl logs "${pod}" -n "${NAMESPACE}" -c modelserver --previous --tail=-1 2>/dev/null \
      > "${dir}/${pod}.previous.log" || rm -f "${dir}/${pod}.previous.log"
  done
  log "Captured decode vLLM logs for ${tag} -> ${dir}"
}

# ----------------------------------------------------------------------------------------
# 5. Run one (model, config): deploy -> wait -> benchmark -> teardown
# ----------------------------------------------------------------------------------------
# Ensure the router's endpoint-picker (EPP) is scaled up and ready. Without a running EPP the
# gateway has an empty datastore and rejects every request with 503 "no endpoint candidates",
# which silently ruins a whole case. The EPP deployment has no owner/HPA, so a plain scale sticks.
ensure_epp_ready() {
  local dep="${ROUTER_RELEASE}-epp" want="${EPP_REPLICAS:-1}" cur
  cur="$(kubectl -n "${NAMESPACE}" get deploy "${dep}" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "")"
  if [[ "${cur}" != "${want}" ]]; then
    log "EPP ${dep} replicas=${cur:-none}; scaling to ${want}."
    kubectl -n "${NAMESPACE}" scale deploy "${dep}" --replicas="${want}" >/dev/null 2>&1 || true
  fi
  kubectl -n "${NAMESPACE}" rollout status deploy "${dep}" --timeout=180s >/dev/null 2>&1 \
    || log "WARNING: EPP ${dep} not ready after 180s."
}

# Poll the gateway until it actually serves the expected model (EPP datastore populated), so we
# never launch a benchmark against a dead endpoint. Returns non-zero if it never becomes ready.
wait_endpoint_ready() {
  local endpoint="$1" model="$2" tries="${ENDPOINT_READY_TRIES:-30}" i resp
  for (( i=1; i<=tries; i++ )); do
    resp="$(curl -sS -m 10 "${endpoint}/v1/models" 2>/dev/null || true)"
    if [[ "${resp}" == *"\"${model}\""* ]]; then
      log "Endpoint ready (serving ${model}) after ${i} check(s)."
      return 0
    fi
    log "Endpoint not ready (attempt ${i}/${tries}): $(printf '%s' "${resp}" | head -c 140)"
    sleep 10
  done
  return 1
}

# ----------------------------------------------------------------------------------------
# Wipe the offloaded KV-cache files written by a native-fs run. The FS tier is a per-node
# NVMe hostPath (${FS_HOSTPATH}); each replica writes under its own POD_NAME subdir. After a
# native-fs case tears down we clear the tier on EVERY node so the next native-fs run starts
# from a cold, empty tier (no stale blocks leaking across runs, no unbounded disk growth). A
# one-shot DaemonSet clears ${FS_HOSTPATH}/* on each node and signals completion via a
# readiness marker (same wait pattern as stage_models_local). Best-effort: warns but never
# aborts the run on failure.
# ----------------------------------------------------------------------------------------
cleanup_fs_tier() {
  local node_vals="" n
  for n in ${STAGE_NODES}; do node_vals+=$'\n                      - '"${n}"; done

  local ds_yaml="${GEN_ROOT}/fs-cleanup-ds.yaml"
  mkdir -p "${GEN_ROOT}"
  cat > "${ds_yaml}" <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fs-cleanup
  namespace: ${NAMESPACE}
  labels: { app: fs-cleanup }
spec:
  selector:
    matchLabels: { app: fs-cleanup }
  template:
    metadata:
      labels: { app: fs-cleanup }
    spec:
      securityContext:
        runAsUser: 0
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:${node_vals}
      terminationGracePeriodSeconds: 5
      containers:
        - name: cleanup
          image: ${STAGE_IMAGE}
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -uo pipefail
              echo "[fs-cleanup] node=\$(hostname) FS tier BEFORE:"; du -sh /fs 2>/dev/null || true
              rm -rf /fs/* /fs/.[!.]* /fs/..?* 2>/dev/null || true
              echo "[fs-cleanup] node=\$(hostname) FS tier AFTER:";  du -sh /fs 2>/dev/null || true
              touch /state/done
              echo "[fs-cleanup] DONE on \$(hostname)"
              exec sleep infinity
          readinessProbe:
            exec:
              command: ["/bin/sh", "-c", "test -f /state/done"]
            initialDelaySeconds: 2
            periodSeconds: 3
            timeoutSeconds: 5
            failureThreshold: 100000
          resources:
            requests: { cpu: '1', memory: '256Mi' }
            limits:   { cpu: '2', memory: '512Mi' }
          volumeMounts:
            - { name: fs-tier, mountPath: /fs }
            - { name: state,   mountPath: /state }
      volumes:
        - name: fs-tier
          hostPath:
            path: ${FS_HOSTPATH}
            type: DirectoryOrCreate
        - name: state
          emptyDir: {}
EOF

  log "Wiping native-fs offload tier (${FS_HOSTPATH}) on nodes: ${STAGE_NODES}"
  if ! kubectl apply -f "${ds_yaml}" >/dev/null 2>&1; then
    log "WARNING: failed to apply fs-cleanup DaemonSet; FS tier NOT wiped."
    return 0
  fi
  if kubectl -n "${NAMESPACE}" rollout status ds/fs-cleanup --timeout="${FS_CLEANUP_TIMEOUT:-300s}"; then
    log "native-fs offload tier wiped on all nodes."
  else
    log "WARNING: fs-cleanup DaemonSet not Ready in time; recent logs:"
    kubectl -n "${NAMESPACE}" logs -l app=fs-cleanup --tail=20 --prefix 2>/dev/null || true
  fi
  kubectl -n "${NAMESPACE}" delete ds/fs-cleanup --wait=false >/dev/null 2>&1 || true
}

run_one() {
  local model="$1" config="$2"
  local tag="${model}-${config}"
  local overlay="${GEN_ROOT}/${tag}"
  local profile="tpc_xpu_${model}.yaml"

  log "=============================================================="
  log "RUN  ${tag}  (TP=${TP[$model]}, replicas=${REPLICAS[$model]})"
  log "=============================================================="

  render_overlay "${model}" "${config}" "${overlay}"

  log "Deploying model server..."
  # ResourceClaimTemplate.spec is immutable: a leftover template from an interrupted run with
  # a different TP or pinning shape makes `kubectl apply` fail ("field is immutable"). Delete
  # it first so apply recreates it with the current shape. Safe: already-running pods keep
  # their bound ResourceClaims; the template is only consulted at pod-creation time.
  kubectl delete resourceclaimtemplate \
    "xpu-vllm-${ROUTER_RELEASE}-intel-claim-template-decode" \
    -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
  kubectl apply -k "${overlay}"
  if ! kubectl rollout status deploy/xpu-vllm-decode -n "${NAMESPACE}" --timeout="${ROLLOUT_TIMEOUT}"; then
    log "Rollout FAILED for ${tag}; capturing diagnostics and skipping."
    kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/role=decode -o wide || true
    capture_decode_logs "${tag}" || true
    kubectl delete -k "${overlay}" --ignore-not-found || true
    return 1
  fi

  local endpoint
  endpoint="http://$(kubectl get service "${ROUTER_RELEASE}-epp" -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')"
  log "Endpoint: ${endpoint}"

  # Gate: the EPP must be up and the gateway must actually serve this model before we benchmark,
  # otherwise the whole case fills with 503 "no endpoint candidates" failures.
  ensure_epp_ready
  if ! wait_endpoint_ready "${endpoint}" "${HF_MODEL[$model]}"; then
    log "ERROR: endpoint never served ${HF_MODEL[$model]} for ${tag}; capturing logs and skipping."
    capture_decode_logs "${tag}" || true
    kubectl delete -k "${overlay}" --ignore-not-found || true
    kubectl wait --for=delete pod -n "${NAMESPACE}" -l llm-d.ai/role=decode --timeout=300s 2>/dev/null || true
    return 1
  fi

  log "Benchmarking with profile ${profile}..."
  # Offline/cache env for the harness pod (propagated by name via --envvarspod; values are
  # read from this launching shell's environment by llm-d-benchmark).
  local -a bench_env=() hf_pass=()
  if [[ "${HARNESS_HF_OFFLINE}" == "true" ]]; then
    bench_env+=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
    hf_pass+=(HF_HUB_OFFLINE TRANSFORMERS_OFFLINE)
    if [[ -n "${HARNESS_HF_HOME}" ]]; then
      bench_env+=("HF_HOME=${HARNESS_HF_HOME}")
      hf_pass+=(HF_HOME)
    fi
  fi
  local -a envvarspod_arg=()
  (( ${#hf_pass[@]} )) && envvarspod_arg=(--envvarspod "$(IFS=,; printf '%s' "${hf_pass[*]}")")

  # Keep the proxy for external tools (helm repo add against *.github.io, skopeo
  # against ghcr.io), but make the Kubernetes Python client honor NO_PROXY for the
  # in-cluster API host. The k8s client reads HTTPS_PROXY into Configuration.proxy
  # but ignores NO_PROXY (leaves no_proxy=None), so API calls get tunnelled through
  # https_proxy and 403. The sitecustomize shim on PYTHONPATH populates no_proxy
  # from the environment so should_bypass_proxies() bypasses the API host.
  local proxy_fix_dir="${GUIDE_DIR}/benchmark-xpu/k8s-proxy-fix"
  ( cd "${LLMDBENCH_REPO}" \
    && env "PYTHONPATH=${proxy_fix_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
           "${bench_env[@]}" \
    llmdbenchmark \
      --workspace "${WORKSPACE}/${tag}" \
      --spec      guides/tiered-prefix-cache \
      run \
      --endpoint-url  "${endpoint}" \
      --gateway-class "${GATEWAY_CLASS}" \
      --model         "${HF_MODEL[$model]}" \
      --namespace     "${NAMESPACE}" \
      --harness       "${HARNESS}" \
      --workload      "${profile}" \
      --wait-timeout  "${HARNESS_WAIT_TIMEOUT}" \
      "${envvarspod_arg[@]}" \
      --analyze ) || log "Benchmark reported an error for ${tag} (continuing)."

  # Snapshot prefix-cache hit counters from the decode pods before we tear them down.
  capture_prefix_cache "${tag}" || log "Prefix-cache capture failed for ${tag} (non-fatal)."

  # Save the decode vLLM logs before teardown for offline debugging.
  capture_decode_logs "${tag}" || log "vLLM log capture failed for ${tag} (non-fatal)."

  log "Tearing down model server for ${tag}..."
  kubectl delete -k "${overlay}" --ignore-not-found
  kubectl wait --for=delete pod -n "${NAMESPACE}" -l llm-d.ai/role=decode --timeout=300s 2>/dev/null || true

  # native-fs writes KV blocks to a per-node NVMe hostPath tier. Now that the decode pods (the
  # only writers) are gone, wipe that tier on every node so the next native-fs run starts cold
  # and the tier does not accumulate stale blocks / grow without bound across runs.
  if [[ "${config}" == "native-fs" ]]; then
    cleanup_fs_tier
  fi
}

# ----------------------------------------------------------------------------------------
# 6. Consolidated report across all (model, config) runs
# ----------------------------------------------------------------------------------------
# Each llmdbenchmark run has already saved its full native results + local analysis under
# ${WORKSPACE}/<model>-<config>/ (harness summary/per-request JSON, benchmark_report, graphs).
# This step just aggregates every run's summary_lifecycle_metrics.json into a single
# report.md (comparison tables + change-vs-baseline) and report.csv for further analysis.
generate_report() {
  log "Generating consolidated report from ${WORKSPACE} ..."
  if ! command -v python3 >/dev/null 2>&1; then
    log "python3 not found; skipping report generation (raw results are under ${WORKSPACE})."
    return 0
  fi

  local hfids=()
  local m
  for m in "${MODELS[@]}"; do hfids+=("${HF_MODEL[$m]}"); done

  TPC_WORKSPACE="${WORKSPACE}" \
  TPC_MODELS="$(IFS=,; printf '%s' "${MODELS[*]}")" \
  TPC_CONFIGS="$(IFS=,; printf '%s' "${CONFIGS[*]}")" \
  TPC_HFMODELS="$(IFS=,; printf '%s' "${hfids[*]}")" \
  python3 - <<'PYEOF'
import csv, glob, json, os, re, subprocess

ws      = os.environ["TPC_WORKSPACE"]
models  = [x for x in os.environ["TPC_MODELS"].split(",") if x]
configs = [x for x in os.environ["TPC_CONFIGS"].split(",") if x]
hfids   = os.environ["TPC_HFMODELS"].split(",")
hf      = dict(zip(models, hfids))


def get(d, *keys, scale=1.0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur * scale if isinstance(cur, (int, float)) else cur


def _tar_extract_json(p, member):
    try:
        out = subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xOf", p, member],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def prefix_hit_rate(tag):
    # Sum vLLM prefix-cache counters scraped from all decode pods for this run
    # (prefix_cache_metrics.txt written by capture_prefix_cache before teardown).
    path = os.path.join(ws, tag, "prefix_cache_metrics.txt")
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return None
    hits = queries = 0.0
    found = False
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.partition(" ")
        base = name.split("{", 1)[0]
        try:
            v = float(val.split()[0])
        except (ValueError, IndexError):
            continue
        if base.endswith("gpu_prefix_cache_hits_total") or base.endswith("prefix_cache_hits_total"):
            hits += v
            found = True
        elif base.endswith("gpu_prefix_cache_queries_total") or base.endswith("prefix_cache_queries_total"):
            queries += v
            found = True
    if found and queries > 0:
        return hits / queries * 100.0
    return None


def stage_summaries(tag):
    # The inference-perf harness packs its metrics INSIDE results/<exp>_N/workspace.tar.zst
    # rather than leaving them on disk. Python 3.12's tarfile has no zstd support, so shell out to
    # `tar --use-compress-program=unzstd`. Return {stage_index: data} for ALL per-stage files so
    # the report can show both the tier-cold first stage and the warm final (steady-state) stage.
    root = os.path.join(ws, tag)
    tars = sorted(glob.glob(os.path.join(root, "**", "workspace.tar.zst"), recursive=True))
    for p in reversed(tars):
        try:
            members = subprocess.run(
                ["tar", "--use-compress-program=unzstd", "-tf", p],
                capture_output=True, check=True, text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            members = []
        stages = {}
        for m in members:
            mm = re.match(r"stage_(\d+)_lifecycle_metrics\.json$", m.rsplit("/", 1)[-1])
            if mm:
                d = _tar_extract_json(p, m)
                if d is not None:
                    stages[int(mm.group(1))] = d
        if stages:
            return stages
        # Fall back to the aggregate summary if no per-stage files are present.
        for member in ("./summary_lifecycle_metrics.json", "summary_lifecycle_metrics.json"):
            d = _tar_extract_json(p, member)
            if d is not None:
                return {0: d}
    # Last resort: a plain summary on disk (older harness layout).
    for p in reversed(sorted(glob.glob(
            os.path.join(root, "**", "summary_lifecycle_metrics.json"), recursive=True))):
        try:
            with open(p, encoding="utf-8") as f:
                return {0: json.load(f)}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def newest_summary(tag):
    # Warm/steady-state stage = highest stage index (the final measured pass).
    st = stage_summaries(tag)
    return st[max(st)] if st else None


def cold_warm(stages):
    # cold = lowest stage index (caches tier-cold), warm = highest (steady state).
    if not stages:
        return None, None
    return stages[min(stages)], stages[max(stages)]


# (label, key-path, scale, precision)  -- schema matches inference-perf summary_lifecycle_metrics.json
ROWS = [
    ("Duration (s)",              ("benchmark_time_seconds",),                                1,    0),
    ("Total requests",           ("load_summary", "count"),                                  1,    0),
    ("Successes",                ("successes", "count"),                                     1,    0),
    ("Failures",                 ("failures", "count"),                                      1,    0),
    ("Avg prompt len (tok)",     ("successes", "prompt_len", "mean"),                        1,    1),
    ("Avg output len (tok)",     ("successes", "output_len", "mean"),                        1,    1),
    ("Throughput (req/s)",       ("successes", "throughput", "requests_per_sec"),            1,    2),
    ("Throughput input (tok/s)", ("successes", "throughput", "input_tokens_per_sec"),        1,    0),
    ("Throughput output (tok/s)",("successes", "throughput", "output_tokens_per_sec"),       1,    0),
    ("Latency mean (ms)",        ("successes", "latency", "request_latency", "mean"),        1000, 1),
    ("Latency p99 (ms)",         ("successes", "latency", "request_latency", "p99"),         1000, 1),
    ("TTFT mean (ms)",           ("successes", "latency", "time_to_first_token", "mean"),    1000, 1),
    ("TTFT p99 (ms)",            ("successes", "latency", "time_to_first_token", "p99"),     1000, 1),
    ("TPOT mean (ms)",           ("successes", "latency", "time_per_output_token", "mean"),  1000, 1),
    ("TPOT p99 (ms)",            ("successes", "latency", "time_per_output_token", "p99"),   1000, 1),
    ("ITL mean (ms)",            ("successes", "latency", "inter_token_latency", "mean"),    1000, 1),
    ("ITL p99 (ms)",             ("successes", "latency", "inter_token_latency", "p99"),     1000, 1),
]

# Headline metrics for the change-vs-baseline block (higher_is_better flags the arrow).
DELTA_ROWS = [
    ("Throughput output (tok/s)", ("successes", "throughput", "output_tokens_per_sec"),      1,    True),
    ("TTFT p99 (ms)",             ("successes", "latency", "time_to_first_token", "p99"),    1000, False),
    ("TPOT mean (ms)",            ("successes", "latency", "time_per_output_token", "mean"), 1000, False),
    ("Latency p99 (ms)",          ("successes", "latency", "request_latency", "p99"),        1000, False),
]


def fmt(v, prec):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


stages_data = {(m, c): stage_summaries(f"{m}-{c}") for m in models for c in configs}
data = {k: (v[max(v)] if v else None) for k, v in stages_data.items()}

md = [
    "# Tiered Prefix Cache Benchmark Report",
    "",
    f"- Workspace: `{ws}`",
    f"- Configs compared: {', '.join(configs)}",
    "- `baseline` = VRAM-only KV cache (control); `native-cpu` = VRAM->CPU offload; "
    "`native-fs` = VRAM->CPU->FS tiering.",
    "- Per-run native results, benchmark_report, and graphs are kept under "
    "`<workspace>/<model>-<config>/`.",
    "- Metrics are the final steady-state stage (an earlier warmup stage fills the CPU/FS tiers); "
    "the closed-loop fixed-concurrency load keeps a bounded number of requests in flight so no "
    "request times out (failures should be 0 everywhere).",
    "- The per-model _warmup effect_ table below reports the tier-cold first stage vs the warm "
    "final stage: TTFT is a warm/steady-state number (not a cold first-touch latency), and the "
    "cold->warm drop is the prefix-cache benefit each tier delivers.",
    "",
]

for m in models:
    md += [f"## {m}  (`{hf.get(m, m)}`)", ""]
    missing = [c for c in configs if data[(m, c)] is None]
    if len(missing) == len(configs):
        md += ["_No results found for this model._", ""]
        continue
    md.append("| Metric | " + " | ".join(configs) + " |")
    md.append("|---|" + "---:|" * len(configs))
    for label, keys, scale, prec in ROWS:
        vals = [get(data[(m, c)], *keys, scale=scale) for c in configs]
        md.append(f"| {label} | " + " | ".join(fmt(v, prec) for v in vals) + " |")

    if "baseline" in configs and data[(m, "baseline")] is not None:
        base = data[(m, "baseline")]
        md += ["", "_Change vs baseline (arrow = direction of improvement):_", ""]
        md.append("| Metric | " + " | ".join(configs) + " |")
        md.append("|---|" + "---:|" * len(configs))
        for label, keys, scale, hib in DELTA_ROWS:
            bv = get(base, *keys, scale=scale)
            cells = []
            for c in configs:
                if c == "baseline":
                    cells.append("—")
                    continue
                cv = get(data[(m, c)], *keys, scale=scale)
                if not bv or cv is None:
                    cells.append("n/a")
                    continue
                pct = (cv - bv) / bv * 100.0
                improved = pct > 0 if hib else pct < 0
                arrow = "✅" if improved else "🔻"
                cells.append(f"{pct:+.1f}% {arrow}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")

    # Prefix-cache warmup effect: cold (first stage) vs warm (final stage). TTFT is the metric
    # most sensitive to prefix caching (a hit skips the prefill of the cached prefix).
    if any(len(stages_data[(m, c)]) >= 2 for c in configs):
        md += ["", "_Prefix-cache warmup effect (cold first stage -> warm final stage):_", ""]
        md.append("| Metric | " + " | ".join(configs) + " |")
        md.append("|---|" + "---:|" * len(configs))
        CW_ROWS = [
            ("TTFT mean (ms)",            ("successes", "latency", "time_to_first_token", "mean"), 1000, 1, False),
            ("TTFT p99 (ms)",             ("successes", "latency", "time_to_first_token", "p99"),  1000, 1, False),
            ("Throughput output (tok/s)", ("successes", "throughput", "output_tokens_per_sec"),    1,    0, True),
        ]
        for label, keys, scale, prec, hib in CW_ROWS:
            cold_cells, warm_cells, delta_cells = [], [], []
            for c in configs:
                cold, warm = cold_warm(stages_data[(m, c)])
                cv = get(cold, *keys, scale=scale) if cold is not None else None
                wv = get(warm, *keys, scale=scale) if warm is not None else None
                cold_cells.append(fmt(cv, prec))
                warm_cells.append(fmt(wv, prec))
                if cv and wv is not None and cold is not warm:
                    pct = (wv - cv) / cv * 100.0
                    improved = pct > 0 if hib else pct < 0
                    delta_cells.append(f"{pct:+.1f}% {'✅' if improved else '🔻'}")
                else:
                    delta_cells.append("—")
            md.append(f"| {label} · cold | " + " | ".join(cold_cells) + " |")
            md.append(f"| {label} · warm | " + " | ".join(warm_cells) + " |")
            md.append(f"| {label} · Δ warm vs cold | " + " | ".join(delta_cells) + " |")

    # Prefix-cache hit rate scraped from the decode pods' /metrics at end of each run.
    # A clearly lower baseline hit rate confirms the working set actually spilled VRAM
    # (so the CPU/FS tiers had something to serve); similar rates would mean no spill.
    hr = {c: prefix_hit_rate(f"{m}-{c}") for c in configs}
    if any(v is not None for v in hr.values()):
        md += ["", "_Prefix-cache hit rate (decode pods, whole run):_", ""]
        md.append("| Metric | " + " | ".join(configs) + " |")
        md.append("|---|" + "---:|" * len(configs))
        md.append("| Prefix cache hit rate (%) | "
                  + " | ".join(fmt(hr[c], 1) for c in configs) + " |")

    if missing:
        md += ["", f"_Missing configs: {', '.join(missing)}._"]
    md.append("")

report_md = os.path.join(ws, "report.md")
with open(report_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

report_csv = os.path.join(ws, "report.csv")
with open(report_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "config", "stage_role", "stage_index"] + [r[0] for r in ROWS])
    for m in models:
        for c in configs:
            st = stages_data[(m, c)]
            if not st:
                w.writerow([m, c, "warm", ""] + [""] * len(ROWS))
                continue
            lo, hi = min(st), max(st)
            entries = [("warm", hi, st[hi])]
            if lo != hi:
                entries.insert(0, ("cold", lo, st[lo]))
            for role, idx, s in entries:
                row = [m, c, role, idx]
                for _, keys, scale, _p in ROWS:
                    v = get(s, *keys, scale=scale)
                    row.append("" if v is None else v)
                w.writerow(row)

print(report_md)
print(report_csv)
PYEOF
}

# ----------------------------------------------------------------------------------------
# Stage the run's model weights from shared NFS onto every worker's local NVMe (HF_LOCAL_PATH),
# so HF_SOURCE=local deploys read from fast local disk instead of the single shared NFS server.
# One DaemonSet pod per node copies each model's hub dir into a temp ".incomplete-<dir>" and
# atomically renames it into place, guarded by a ".done-<dir>" marker so re-runs skip staged
# models. One pod per node => no two pods ever touch the same snapshot. No-op unless HF_SOURCE=local.
# ----------------------------------------------------------------------------------------
stage_models_local() {
  [[ "${HF_SOURCE}" == "local" ]] || return 0

  local -a model_dirs=()
  local m repo
  for m in "${MODELS[@]}"; do
    repo="${HF_MODEL[$m]}"
    model_dirs+=("models--${repo//\//--}")
  done
  local dirs_str="${model_dirs[*]}"
  log "Staging ${#model_dirs[@]} model(s) onto node-local ${HF_LOCAL_PATH}/hub (nodes: ${STAGE_NODES})"
  log "  cache dirs: ${dirs_str}"

  local node_vals="" n
  for n in ${STAGE_NODES}; do node_vals+=$'\n                      - '"${n}"; done

  local ds_yaml="${GEN_ROOT}/hf-stage-ds.yaml"
  mkdir -p "${GEN_ROOT}"
  cat > "${ds_yaml}" <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: hf-stage
  namespace: ${NAMESPACE}
  labels: { app: hf-stage }
spec:
  selector:
    matchLabels: { app: hf-stage }
  template:
    metadata:
      labels: { app: hf-stage }
    spec:
      securityContext:
        runAsUser: 0
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:${node_vals}
      terminationGracePeriodSeconds: 5
      containers:
        - name: stage
          image: ${STAGE_IMAGE}
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -uo pipefail
              SRC=/nfs/hub
              DST=/local/hub
              mkdir -p "\$DST"
              echo "[stage] node=\$(hostname) local free space:"; df -h /local | tail -1
              rc=0
              for M in ${dirs_str}; do
                MARK="\$DST/.done-\$M"
                if [ -f "\$MARK" ] && [ -d "\$DST/\$M" ]; then echo "[stage] SKIP \$M (already staged)"; continue; fi
                if [ ! -d "\$SRC/\$M" ]; then echo "[stage] ERROR source missing: \$SRC/\$M"; rc=1; continue; fi
                echo "[stage] COPY \$M ..."; t0=\$(date +%s)
                TMP="\$DST/.incomplete-\$M"
                rm -rf "\$TMP"; mkdir -p "\$TMP"
                if cp -dR "\$SRC/\$M/." "\$TMP/"; then
                  chmod -R a+rX "\$TMP" 2>/dev/null || true
                  rm -rf "\$DST/\$M"
                  mv "\$TMP" "\$DST/\$M"
                  touch "\$MARK"
                  echo "[stage] DONE \$M in \$(( \$(date +%s) - t0 ))s"
                else
                  echo "[stage] FAILED \$M"; rm -rf "\$TMP"; rc=1
                fi
              done
              if [ "\$rc" -eq 0 ]; then touch "\$DST/.all-done"; echo "[stage] ALL STAGED on \$(hostname)"; else rm -f "\$DST/.all-done"; echo "[stage] staging had errors on \$(hostname)"; fi
              exec sleep infinity
          readinessProbe:
            exec:
              command: ["/bin/sh", "-c", "test -f /local/hub/.all-done"]
            initialDelaySeconds: 10
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 100000
          resources:
            requests: { cpu: '2', memory: '2Gi' }
            limits:   { cpu: '4', memory: '4Gi' }
          volumeMounts:
            - { name: nfs-cache, mountPath: /nfs, readOnly: true }
            - { name: local-cache, mountPath: /local }
      volumes:
        - name: nfs-cache
          nfs:
            server: ${HF_NFS_SERVER}
            path: ${HF_NFS_PATH}
            readOnly: true
        - name: local-cache
          hostPath:
            path: ${HF_LOCAL_PATH}
            type: DirectoryOrCreate
EOF

  kubectl apply -f "${ds_yaml}" >/dev/null || fail "failed to apply staging DaemonSet"
  local n_nodes; n_nodes=$(wc -w <<< "${STAGE_NODES}")
  log "Waiting for ${n_nodes} node(s) to finish staging (one-time bulk copy, ~251 GiB/node)..."
  if ! kubectl -n "${NAMESPACE}" rollout status ds/hf-stage --timeout="${STAGE_TIMEOUT:-7200s}"; then
    log "Staging DaemonSet not Ready in time; recent pod logs:"
    kubectl -n "${NAMESPACE}" logs -l app=hf-stage --tail=30 --prefix 2>/dev/null || true
    fail "model staging to local NVMe failed"
  fi
  log "All nodes staged to ${HF_LOCAL_PATH}/hub."
  if [[ "${STAGE_KEEP}" != "true" ]]; then
    kubectl -n "${NAMESPACE}" delete ds/hf-stage --wait=false >/dev/null 2>&1 || true
    log "Staging DaemonSet removed (local files persist on each node)."
  fi
}

# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
main() {
  bootstrap_benchmark
  preflight
  deploy_router
  ensure_workload_pv
  ensure_harness_tokenizers
  stage_models_local

  log "Generating workload profiles into ${LLMDBENCH_REPO}/workload/profiles/${HARNESS}/"
  for m in "${MODELS[@]}"; do
    gen_workload_profile "${m}" "${LLMDBENCH_REPO}/workload/profiles/${HARNESS}/tpc_xpu_${m}.yaml"
  done

  for m in "${MODELS[@]}"; do
    for c in "${CONFIGS[@]}"; do
      run_one "${m}" "${c}" || true
    done
  done

  generate_report || log "Report generation failed (raw per-run results are still under ${WORKSPACE})."

  log "All runs complete."
  log "  Raw results + per-run analysis : ${WORKSPACE}/<model>-<config>/"
  log "  Consolidated report            : ${WORKSPACE}/report.md"
  log "  Machine-readable summary       : ${WORKSPACE}/report.csv"
  log "Router '${ROUTER_RELEASE}' left running. Remove with: helm uninstall ${ROUTER_RELEASE} -n ${NAMESPACE}"
}

main "$@"
