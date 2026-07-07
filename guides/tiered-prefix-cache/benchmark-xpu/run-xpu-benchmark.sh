#!/usr/bin/env bash
# Automated tiered-prefix-cache benchmark for Intel XPU (DRA-based deploy).
#
# Deploys the HBM-only baseline and the CPU-offload model server in turn, drives
# each with the same shared-prefix workload via the llmdbenchmark CLI, and
# compares the resulting throughput/latency.
#
# Both XPU model server overlays use Dynamic Resource Allocation (DRA): they
# request Intel GPUs through a ResourceClaimTemplate (deviceClassName
# gpu.intel.com) instead of the gpu.intel.com/xe device-plugin resource.
#
# Topology: Qwen3-32B served TP4 on a SINGLE decode replica pinned to the 4
# socket-0 B60s (XPU 0-3 = PCI 2b/2f/3c/40) via the ResourceClaimTemplate CEL
# selector, with ZE_AFFINITY_MASK 0-3 over those 4 isolated render nodes. Both
# the baseline and the offload variant use the SAME single-replica 4-GPU
# topology so the comparison is apples-to-apples. The shared-prefix workload
# (workload-xpu-single.yaml) is sized so the replica's KV working set overruns
# its HBM KV budget (so the HBM-only baseline must recompute evicted prefixes
# while the CPU-offload variant keeps them resident in host RAM). The offered
# QPS is kept gentle so the offload benefit shows up as lower TTFT / higher
# throughput, not as a difference in success rate.
#
# The default offload variant is the LMCache connector
# (modelserver/xpu/vllm/lmcache-connector/cpu/base); override OFFLOAD_OVERLAY /
# OFFLOAD_LABEL to benchmark the built-in vLLM OffloadingConnector
# (modelserver/xpu/vllm/native/cpu/base) instead.
#
# Usage:
#   ./run-xpu-benchmark.sh
#
# Override defaults via env vars, e.g.:
#   NAMESPACE=tiered-prefix-cache-cpu XPU_IMAGE=vllm-xpu-env-lm-cache:latest ./run-xpu-benchmark.sh
# -E (errtrace) makes the ERR trap fire for failures inside functions too, so an
# unexpected error anywhere collects diagnostics and cleans up before exiting.
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(realpath "$(git rev-parse --show-toplevel)")}"
NAMESPACE="${NAMESPACE:-tiered-prefix-cache-cpu}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
XPU_IMAGE_NAME="${XPU_IMAGE_NAME:-vllm-xpu-env-lm-cache}"   # leave empty to keep kustomization default
XPU_IMAGE_TAG="${XPU_IMAGE_TAG:-latest}"
GUIDE_DIR="${REPO_ROOT}/guides/tiered-prefix-cache"
# Base dir containing the benchmark spec templates/scenarios (config/...).
BENCH_BASE_DIR="${BENCH_BASE_DIR:-${REPO_ROOT}/llm-d-benchmark}"
# Benchmark spec name (config/scenarios/<SPEC>.yaml and --spec).
SPEC="${SPEC:-guides/tiered-prefix-cache}"
# llmdbenchmark resolves --workload as a bare profile name under
# <base-dir>/workload/profiles/<harness>/.
HARNESS="${HARNESS:-inference-perf}"
# The shipped guide_tiered-prefix-cache_1.yaml profile (250 prefix groups x 5
# prompts, 16K-token shared prefix, 8 stages ramping rate 5->40) targets the
# published 16-GPU CUDA results and overwhelms this single-replica XPU TP4
# engine -- every request after the warmup stage fails. We therefore default to
# a DEDICATED XPU profile name that does NOT ship with llm-d-benchmark, so
# install_workload_profile() always installs our XPU-tuned
# workload-xpu-single.yaml (60 groups x 5, 3K-token prefix, gentle rate
# 0.5->1.0 on the one engine) as the profile template instead of silently
# falling back to the heavy shipped guide profile. Override
# WORKLOAD_PROFILE_NAME=guide_tiered-prefix-cache_1.yaml to reproduce the
# upstream CUDA workload.
WORKLOAD_PROFILE_NAME="${WORKLOAD_PROFILE_NAME:-tiered-prefix-cache-xpu.yaml}"
# Source of the custom XPU workload, installed under WORKLOAD_PROFILE_NAME as a
# .yaml.in profile template (the dedicated name above never collides with a
# shipped profile, so this is always installed and rendered).
WORKLOAD="${WORKLOAD:-${GUIDE_DIR}/benchmark-xpu/workload-xpu-single.yaml}"
# Harness image the benchmark deploys in-cluster. The kind node has no registry
# egress, so we pull it on the host and side-load it into the node.
HARNESS_IMAGE="${HARNESS_IMAGE:-ghcr.io/llm-d/llm-d-benchmark:v0.7.0}"
KIND_CLUSTER="${KIND_CLUSTER:-kind}"
WORKSPACE="${WORKSPACE:-${GUIDE_DIR}/benchmark-xpu/results-$(date +%Y%m%d-%H%M%S)}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"                # seconds to wait for pod ready
# Corporate HTTP(S) proxy the in-cluster harness pod uses to fetch the model
# tokenizer from huggingface.co. The kind node has no direct egress, but the
# proxy is reachable from pods. Defaults to the host proxy; set empty to disable.
HARNESS_HF_PROXY="${HARNESS_HF_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}"

BASELINE_OVERLAY="${BASELINE_OVERLAY:-${GUIDE_DIR}/modelserver/xpu/vllm/base}"
# Default offload variant is the LMCache connector. Override OFFLOAD_OVERLAY to
# ${GUIDE_DIR}/modelserver/xpu/vllm/native/cpu/base (with OFFLOAD_LABEL=offloading)
# to benchmark the built-in vLLM OffloadingConnector instead.
OFFLOAD_OVERLAY="${OFFLOAD_OVERLAY:-${GUIDE_DIR}/modelserver/xpu/vllm/lmcache-connector/cpu/base}"
# Results-subdir / log labels for each variant. Override OFFLOAD_LABEL (e.g. to
# "offloading") when pointing OFFLOAD_OVERLAY at a different connector overlay.
BASELINE_LABEL="${BASELINE_LABEL:-baseline}"
OFFLOAD_LABEL="${OFFLOAD_LABEL:-lmcache}"

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bench]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[bench]\033[0m %s\n' "$*" >&2; }

# Track the overlay/label currently deployed so an unexpected exit can dump
# diagnostics and tear it down, leaving the cluster clean for the next run.
CURRENT_OVERLAY=""
CURRENT_LABEL=""

# Collect actionable diagnostics (pod status, scheduling events, recent vLLM
# logs) into the workspace and echo a one-line pod summary. Safe to call anytime.
diagnose() {  # $1 = label (optional)
  set +e
  local label dir pods p
  label="${1:-${CURRENT_LABEL:-unknown}}"
  dir="${WORKSPACE:-/tmp}/${label}/diagnostics"
  mkdir -p "${dir}" 2>/dev/null
  warn "Collecting diagnostics for '${label}' -> ${dir}"
  {
    echo "### kubectl get pods -o wide"
    kubectl get pods -n "${NAMESPACE}" -o wide 2>&1
    echo; echo "### decode deploy/replicaset"
    kubectl get deploy,rs -n "${NAMESPACE}" -l llm-d.ai/role=decode -o wide 2>&1
    echo; echo "### describe decode pods"
    kubectl describe pod -n "${NAMESPACE}" -l llm-d.ai/role=decode 2>&1
    echo; echo "### recent namespace events"
    kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp 2>&1 | tail -40
    echo; echo "### resourceclaims"
    kubectl get resourceclaim -n "${NAMESPACE}" 2>&1
  } > "${dir}/cluster-state.txt" 2>&1
  pods="$(kubectl get pod -n "${NAMESPACE}" -l llm-d.ai/role=decode \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)"
  for p in ${pods}; do
    kubectl logs -n "${NAMESPACE}" "${p}" -c modelserver --tail=100 \
      > "${dir}/${p}.tail.log" 2>&1
    # Preserve the crash traceback from the previous container instance if the
    # modelserver restarted (e.g. XPU UR_RESULT_ERROR_DEVICE_LOST GPU hang).
    local restarts
    restarts="$(kubectl get pod -n "${NAMESPACE}" "${p}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="modelserver")].restartCount}' 2>/dev/null || echo 0)"
    if [[ "${restarts:-0}" -gt 0 ]]; then
      kubectl logs -n "${NAMESPACE}" "${p}" -c modelserver --previous \
        > "${dir}/${p}.crash.log" 2>&1
    fi
  done
  kubectl get pods -n "${NAMESPACE}" -o wide 2>/dev/null | sed 's/^/[bench][diag] /'
  set -e
}

on_err() {
  local ec=$?
  set +e
  err "FAILED (exit ${ec}). Gathering diagnostics and cleaning up ..."
  diagnose "${CURRENT_LABEL:-error}"
  if [[ -n "${CURRENT_OVERLAY}" ]]; then
    warn "Tearing down current overlay ${CURRENT_OVERLAY} to leave the cluster clean."
    kubectl delete -n "${NAMESPACE}" -k "${CURRENT_OVERLAY}" --ignore-not-found >/dev/null 2>&1
    kubectl wait -n "${NAMESPACE}" --for=delete pod -l llm-d.ai/role=decode --timeout=120s >/dev/null 2>&1
  fi
  exit "${ec}"
}
trap on_err ERR

# ---------------------------------------------------------------------------
# Pre-flight: fail fast (with a clear message) on the things that silently broke
# earlier runs -- an unreachable cluster or missing model weights.
# ---------------------------------------------------------------------------
preflight() {
  kubectl get ns >/dev/null 2>&1 || {
    err "kubectl cannot reach the cluster (context: $(kubectl config current-context 2>/dev/null || echo none)). Fix your kubeconfig and retry."
    exit 1
  }
  local model_dir="/localdisk/HF_CACHE/${MODEL##*/}"
  [[ -d "${model_dir}" ]] || warn "Model dir ${model_dir} not found on this host (decode mounts it via hostPath); ensure the kind node has the weights or pods will fail to load."
}

# ---------------------------------------------------------------------------
# 0. Clean up stale deployments / orphaned GPU claims so all 8 B60s are free
# ---------------------------------------------------------------------------
cleanup() {
  log "Cleaning up prior deployments + orphaned GPU claims ..."
  kubectl delete -k "${BASELINE_OVERLAY}" -n "${NAMESPACE}" --ignore-not-found --grace-period=0 --force 2>/dev/null || true
  kubectl delete -k "${OFFLOAD_OVERLAY}"  -n "${NAMESPACE}" --ignore-not-found --grace-period=0 --force 2>/dev/null || true
  # Drop leftover decode pods/replicasets in our namespace.
  kubectl delete pod,rs -n "${NAMESPACE}" -l llm-d.ai/role=decode --ignore-not-found --grace-period=0 --force 2>/dev/null || true
  # Orphan bench-* namespaces can hold ResourceClaims that keep the B60s reserved.
  # for ns in $(kubectl get ns -o name 2>/dev/null | sed 's|namespace/||' | grep '^bench-'); do
  #   log "Deleting orphan bench namespace ${ns}"
  #   kubectl delete ns "${ns}" --ignore-not-found --grace-period=0 --force 2>/dev/null || true
  # done
  # Cluster-wide: clear any orphan ResourceClaims/templates still reserving devices.
  # kubectl get resourceclaim -A --no-headers 2>/dev/null | awk '$1 ~ /^bench-/ {print $1, $2}' | \
  #   while read -r ns name; do kubectl delete resourceclaim "${name}" -n "${ns}" --ignore-not-found --grace-period=0 --force 2>/dev/null || true; done
  # log "Cleanup done."
}

# ---------------------------------------------------------------------------
# 0. Ensure namespace + router (EPP) exist
# ---------------------------------------------------------------------------
ensure_namespace() {
  kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || {
    log "Creating namespace ${NAMESPACE}"
    kubectl create namespace "${NAMESPACE}"
  }
  # HF token secret (decode pods mount HF_TOKEN from secret 'llm-d-hf-token').
  if ! kubectl get secret llm-d-hf-token -n "${NAMESPACE}" >/dev/null 2>&1; then
    if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
      log "Creating llm-d-hf-token secret from ~/.cache/huggingface/token"
      kubectl create secret generic llm-d-hf-token -n "${NAMESPACE}" \
        --from-literal=HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
    else
      warn "No HF token found at ~/.cache/huggingface/token; decode pods may fail to pull gated models"
    fi
  fi
}

ensure_router() {
  if kubectl get svc tiered-prefix-cache-epp -n "${NAMESPACE}" >/dev/null 2>&1; then
    log "Router/EPP already present in ${NAMESPACE}"
    return
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/guides/env.sh"
  if helm install tiered-prefix-cache "${ROUTER_STANDALONE_CHART}" \
       -f "${GUIDE_DIR}/../recipes/router/base.values.yaml" \
       -f "${GUIDE_DIR}/router/tiered-prefix-cache-cpu.values.yaml" \
       -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}" 2>/dev/null; then
    log "Deployed router via helm chart ${ROUTER_CHART_VERSION}"
    return
  fi
  # Chart registry unreachable — apply the bundled, self-contained manifest.
  log "Chart unavailable; deploying bundled router manifest into ${NAMESPACE}"
  sed "s/__NAMESPACE__/${NAMESPACE}/g" "${GUIDE_DIR}/benchmark-xpu/router.yaml" \
    | kubectl apply -n "${NAMESPACE}" -f -
  kubectl rollout status deploy/tiered-prefix-cache-epp -n "${NAMESPACE}" --timeout=120s
}


# ---------------------------------------------------------------------------
# 1. Install/locate the llmdbenchmark CLI
# ---------------------------------------------------------------------------
ensure_bench_cli() {
  if command -v llmdbenchmark >/dev/null 2>&1; then
    log "llmdbenchmark found: $(command -v llmdbenchmark)"
    return
  fi
  log "Installing llmdbenchmark CLI under ${REPO_ROOT}/llm-d-benchmark ..."
  if [[ ! -d "${REPO_ROOT}/llm-d-benchmark" ]]; then
    curl -sSL https://raw.githubusercontent.com/llm-d/llm-d-benchmark/main/install.sh | bash
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/llm-d-benchmark/.venv/bin/activate"
  command -v llmdbenchmark >/dev/null 2>&1 || { warn "llmdbenchmark not on PATH after install"; exit 1; }
}

# ---------------------------------------------------------------------------
# 1a2. Work around a kubernetes-client bug that breaks no_proxy.
#      In kubernetes>=31 the Configuration.__init__ loads `proxy` from the
#      HTTPS_PROXY/HTTP_PROXY env vars, then loads `no_proxy` from NO_PROXY --
#      but a stray `self.no_proxy = None` AFTER that block clobbers the value
#      back to None. The net effect: when a proxy is set in the environment the
#      Python kube client proxies EVERY request (including the kind API server
#      at 127.0.0.1:<port>) and no_proxy can never take effect -> the corporate
#      proxy answers 403 and every render/deploy cluster call fails.
#      We need the proxy SET (so the harness pod can reach huggingface.co via
#      `-g`, and so image-tag resolution stays fast), so we repair the bug by
#      deleting the erroneous reset line. Idempotent and safe to re-run.
# ---------------------------------------------------------------------------
fix_kube_no_proxy_bug() {
  local cfg
  cfg="$(python3 -c 'import kubernetes.client.configuration as c; print(c.__file__)' 2>/dev/null)" || {
    warn "kubernetes python client not importable; skipping no_proxy fix"
    return
  }
  python3 - "${cfg}" <<'PY'
import io, re, sys
path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
# Remove the stray `self.no_proxy = None` that sits between the `Proxy URL`
# attribute docstring and the `bypass proxy ...` docstring, which clobbers the
# value just loaded from the NO_PROXY/no_proxy environment variables.
pat = re.compile(
    r'(\"\"\"Proxy URL\s*\"\"\"\s*\n)\s*self\.no_proxy = None\n(\s*\"\"\"bypass proxy)',
)
new, n = pat.subn(r'\1\2', src)
if n:
    io.open(path, "w", encoding="utf-8").write(new)
    print(f"patched kubernetes no_proxy bug ({n} occurrence) in {path}")
else:
    print("kubernetes no_proxy bug already patched (or not present)")
PY
}

# ---------------------------------------------------------------------------
# 1b. Install our workload as a profile under the benchmark profiles dir.
#     llmdbenchmark resolves --workload as a bare filename under
#     <base-dir>/workload/profiles/<harness>/ and renders the REPLACE_ENV_*
#     tokens, so the source must be installed there as a .yaml.in template.
# ---------------------------------------------------------------------------
install_workload_profile() {
  local dest_dir="${BENCH_BASE_DIR}/workload/profiles/${HARNESS}"
  [[ -d "${dest_dir}" ]] || { warn "Profiles dir not found: ${dest_dir}"; exit 1; }
  # When a custom XPU workload source exists, ALWAYS (re)install it under the
  # profile name. This guarantees edits to workload-xpu.yaml take effect and we
  # never silently fall back to a stale prior copy or the heavy shipped guide
  # profile. We install as a .yaml.in template so the REPLACE_ENV_* tokens are
  # substituted with the runtime endpoint URL/model at render time.
  if [[ -f "${WORKLOAD}" ]]; then
    log "Installing custom workload profile ${WORKLOAD_PROFILE_NAME} -> ${dest_dir}"
    # Drop any pre-rendered/plain copy so only our fresh template is used.
    rm -f "${dest_dir}/${WORKLOAD_PROFILE_NAME}"
    cp "${WORKLOAD}" "${dest_dir}/${WORKLOAD_PROFILE_NAME}.in"
    return
  fi
  # No custom source available -- fall back to a profile shipped with the repo.
  if [[ -f "${dest_dir}/${WORKLOAD_PROFILE_NAME}" || -f "${dest_dir}/${WORKLOAD_PROFILE_NAME}.in" ]]; then
    log "Using shipped workload profile ${WORKLOAD_PROFILE_NAME}"
    return
  fi
  warn "Workload profile ${WORKLOAD_PROFILE_NAME} not found and no custom WORKLOAD source at ${WORKLOAD}"
  exit 1
}

# ---------------------------------------------------------------------------
# 1b2. Patch the scenario for this single-node kind cluster:
#      (a) Force the workload/model PVCs to ReadWriteOnce -- kind's local-path
#          provisioner only supports RWO/RWOP, but the upstream default is
#          ReadWriteMany which never binds and stalls the harness_namespace step.
#      (b) Pin the infra/modelservice chart versions to "skip" -- they are not
#          deployed here (we use kustomize + an existing endpoint), and leaving
#          them "auto" makes every render hang ~2 min each on a registry lookup.
#      Idempotent.
# ---------------------------------------------------------------------------
patch_scenario() {
  local scenario="${BENCH_BASE_DIR}/config/scenarios/${SPEC}.yaml"
  [[ -f "${scenario}" ]] || { warn "Scenario file not found: ${scenario}"; return; }
  log "Patching scenario for kind (RWO PVCs + skip chart resolution)"
  python3 - "${scenario}" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path) as fh:
    doc = yaml.safe_load(fh) or {}
# The scenario file is a list of scenario items under the top-level
# "scenario" key; overrides live on each item, NOT at the top level.
items = doc.get("scenario") or []
for item in items:
    storage = item.setdefault("storage", {})
    for key in ("workloadPvc", "modelPvc"):
        pvc = storage.setdefault(key, {})
        pvc["accessModes"] = ["ReadWriteOnce"]
    storage["modelPvc"].setdefault("size", "1Ti")
    charts = item.setdefault("chartVersions", {})
    charts["llmDInfra"] = "skip"
    charts["llmDModelservice"] = "skip"
with open(path, "w") as fh:
    yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)
PY
}

# ---------------------------------------------------------------------------# 1c. Side-load the harness image into the kind node (no registry egress there).
# ---------------------------------------------------------------------------
ensure_harness_image() {
  local ref="${HARNESS_IMAGE}"
  if docker exec "${KIND_CLUSTER}-control-plane" ctr -n k8s.io images ls 2>/dev/null \
       | awk '{print $1}' | grep -qx "docker.io/library/${ref##*/}" 2>/dev/null; then
    log "Harness image already present in kind node"
    return
  fi
  if docker exec "${KIND_CLUSTER}-control-plane" ctr -n k8s.io images ls -q 2>/dev/null \
       | grep -qx "${ref}"; then
    log "Harness image ${ref} already present in kind node"
    return
  fi
  log "Loading harness image ${ref} into kind node ${KIND_CLUSTER}"
  docker image inspect "${ref}" >/dev/null 2>&1 || {
    log "Pulling ${ref} on host ..."
    docker pull "${ref}" || { warn "Failed to pull ${ref}; harness pod may ImagePullBackOff"; return; }
  }
  kind load docker-image "${ref}" --name "${KIND_CLUSTER}" || \
    warn "kind load failed for ${ref}; harness pod may ImagePullBackOff"
}

# ---------------------------------------------------------------------------
# 2. Set the XPU image override (optional)
# ---------------------------------------------------------------------------
set_image() {
  [[ -z "${XPU_IMAGE_NAME}" ]] && return
  log "Pinning XPU image to ${XPU_IMAGE_NAME}:${XPU_IMAGE_TAG}"
  ( cd "${BASELINE_OVERLAY}" && \
    kubectl kustomize . >/dev/null 2>&1 && \
    cd "${BASELINE_OVERLAY}" && \
    kustomize edit set image "ghcr.io/llm-d/llm-d-xpu=${XPU_IMAGE_NAME}:${XPU_IMAGE_TAG}" ) 2>/dev/null || \
    warn "kustomize edit skipped (image already overridden in kustomization.yaml)"
}

# Side-load the local-only XPU model-server image into the kind node. decode
# pods use imagePullPolicy IfNotPresent and this image has no registry to pull
# from, so it MUST already be in the node's containerd or the pod ImagePullBackOffs.
ensure_xpu_image() {
  [[ -z "${XPU_IMAGE_NAME}" ]] && return 0
  local ref="${XPU_IMAGE_NAME}:${XPU_IMAGE_TAG}"
  if docker exec "${KIND_CLUSTER}-control-plane" ctr -n k8s.io images ls -q 2>/dev/null \
       | grep -qE "(^|/)${XPU_IMAGE_NAME}:${XPU_IMAGE_TAG}$"; then
    log "XPU image ${ref} already present in kind node"
    return 0
  fi
  if ! docker image inspect "${ref}" >/dev/null 2>&1; then
    warn "XPU image ${ref} not in kind node and not on host; build/tag it before running or decode pods will ImagePullBackOff."
    return 0
  fi
  log "Loading XPU image ${ref} into kind node ${KIND_CLUSTER} (large image, may take a while) ..."
  kind load docker-image "${ref}" --name "${KIND_CLUSTER}" \
    || warn "kind load failed for ${ref}; decode pods may ImagePullBackOff"
}

# ---------------------------------------------------------------------------
# 3. Deploy / teardown helpers
# ---------------------------------------------------------------------------
deploy() {  # $1 = overlay dir
  CURRENT_OVERLAY="$1"
  log "Deploying $1"
  kubectl apply -n "${NAMESPACE}" -k "$1"
  wait_rollout
  # A Ready decode pod is necessary but NOT sufficient: the EPP discovers the
  # new pod via its InferencePool informer and only starts routing to it after
  # a successful metrics scrape, which lags pod-Ready by tens of seconds. If we
  # launch the benchmark immediately, its verify_model step curls the EPP before
  # routing is live and times out (60s window) -> the whole run aborts. Gate on
  # the EPP actually serving /v1/models end-to-end before returning.
  wait_epp_ready
}

# Wait for the decode Deployment to reach its desired available replicas, but
# FAIL FAST (instead of blocking the full WAIT_TIMEOUT) when a pod lands in a
# state it will never recover from on its own: image pull/container errors, or a
# pod that stays Pending because DRA cannot allocate the 8 GPUs (e.g. a socket
# still wedged). Earlier overnight runs burned the whole timeout on one stuck
# pod with no explanation; this surfaces the root cause in seconds.
wait_rollout() {
  local dep desired start now avail bad pending
  dep="$(kubectl get deploy -n "${NAMESPACE}" -l llm-d.ai/role=decode \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -z "${dep}" ]] && { warn "No decode deployment found to wait on"; return 0; }
  desired="$(kubectl get deploy -n "${NAMESPACE}" "${dep}" -o jsonpath='{.spec.replicas}' 2>/dev/null)"
  [[ -z "${desired}" ]] && desired=1
  log "Waiting for decode rollout (deploy/${dep}, ${desired} replica(s), timeout ${WAIT_TIMEOUT}s) ..."
  start=$(date +%s)
  while :; do
    avail="$(kubectl get deploy -n "${NAMESPACE}" "${dep}" -o jsonpath='{.status.availableReplicas}' 2>/dev/null)"
    [[ -z "${avail}" ]] && avail=0
    if [[ "${avail}" -ge "${desired}" ]]; then
      log "Decode rollout complete (${avail}/${desired} available)."
      return 0
    fi
    # Terminal container states -- no point waiting them out.
    bad="$(kubectl get pod -n "${NAMESPACE}" -l llm-d.ai/role=decode \
      -o jsonpath='{range .items[*]}{.metadata.name}{"="}{range .status.containerStatuses[*]}{.state.waiting.reason}{","}{end}{"\n"}{end}' 2>/dev/null \
      | grep -Ei "ImagePullBackOff|ErrImagePull|InvalidImageName|CrashLoopBackOff|CreateContainerError|RunContainerError|CreateContainerConfigError" || true)"
    if [[ -n "${bad}" ]]; then
      err "Decode pod in a terminal state:"; printf '%s\n' "${bad}" >&2
      return 1
    fi
    now=$(date +%s)
    # After a grace window, a still-Pending pod means the scheduler/DRA cannot
    # place it -- fail fast with the scheduling events instead of hanging.
    if [[ $(( now - start )) -ge 90 ]]; then
      pending="$(kubectl get pod -n "${NAMESPACE}" -l llm-d.ai/role=decode \
        --field-selector=status.phase=Pending -o name 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${pending}" -gt 0 ]]; then
        err "Decode pod(s) still Pending after 90s (likely DRA cannot allocate GPUs). Recent scheduling events:"
        kubectl get event -n "${NAMESPACE}" --field-selector reason=FailedScheduling 2>/dev/null | tail -5 >&2 || true
        return 1
      fi
    fi
    if [[ $(( now - start )) -ge ${WAIT_TIMEOUT} ]]; then
      err "Decode rollout did not complete within ${WAIT_TIMEOUT}s (${avail}/${desired} available)."
      return 1
    fi
    sleep 5
  done
}
# Poll the EPP-routed model endpoint from inside the cluster until it serves the
# model list (i.e. the EPP has a healthy decode endpoint) or we hit the timeout.
# We probe by exec-ing into the already-Ready decode pod (which has cluster
# networking + curl) instead of spawning a fresh curl pod: a new pod adds image
# pull/scheduling lag and the nested sh -c quoting proved fragile (it silently
# failed and the gate always "timed out"). The decode pod is the ideal vantage
# point -- it is on the cluster network and already running.
wait_epp_ready() {
  local ip timeout dpod end
  ip="$(kubectl get service tiered-prefix-cache-epp -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')"
  # CPU-offload decode pods initialize the OffloadingConnector + 100GiB host KV
  # buffer and take noticeably longer to load than the HBM-only baseline, so we
  # allow a generous default window.
  timeout="${EPP_READY_TIMEOUT:-600}"
  dpod="$(kubectl get pod -n "${NAMESPACE}" -l llm-d.ai/role=decode \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -z "${dpod}" ]]; then
    warn "No Running decode pod found to probe EPP readiness; skipping gate."
    return
  fi
  log "Waiting for EPP to route to a ready decode endpoint (http://${ip}:80 via ${dpod}, timeout ${timeout}s) ..."
  end=$(( $(date +%s) + timeout ))
  while [[ $(date +%s) -lt ${end} ]]; do
    if kubectl exec -n "${NAMESPACE}" "${dpod}" -- \
         sh -c "curl -sf -m 5 http://${ip}:80/v1/models 2>/dev/null" 2>/dev/null \
         | grep -q '"object"'; then
      log "EPP is routing to the decode endpoint."
      return
    fi
    sleep 5
  done
  warn "EPP did not become ready within ${timeout}s; benchmark verify_model may fail."
}
# Dump each decode pod's vLLM (modelserver) container log into the results dir
# once the engine is ready, and surface the KV-cache sizing lines inline. Must
# run BEFORE teardown, while the pods still exist.
capture_vllm_logs() {  # $1 = label
  local dir pods p
  dir="${WORKSPACE}/$1/vllm-logs"
  mkdir -p "${dir}"
  pods="$(kubectl get pod -n "${NAMESPACE}" -l llm-d.ai/role=decode \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
  if [[ -z "${pods}" ]]; then
    warn "No decode pods found; skipping vLLM log capture."
    return
  fi
  for p in ${pods}; do
    log "Capturing vLLM log: ${p} -> ${dir}/${p}.log"
    kubectl logs -n "${NAMESPACE}" "${p}" -c modelserver \
      > "${dir}/${p}.log" 2>&1 || warn "Failed to capture logs for ${p}"
    # An XPU UR_RESULT_ERROR_DEVICE_LOST GPU hang can kill the EngineCore
    # subprocess WITHOUT the container (PID 1 / APIServer) exiting -- so the
    # crash traceback is in the CURRENT log and restartCount stays 0 (the pod
    # just starts returning 503 ServiceUnavailable). Always scan the current
    # log for the crash signature, and additionally capture --previous whenever
    # a prior container instance exists (harmless if there is none).
    local crash_re="DEVICE_LOST|level_zero backend failed|Engine reset|EngineCore.*(died|failed|encountered an issue)|EngineCore.*shut|RuntimeError|Segmentation|SIGSEGV|SIGABRT|Aborted|CUDA error|out of memory|OOM-kill|FATAL|Traceback "
    if grep -qiE "${crash_re}" "${dir}/${p}.log" 2>/dev/null; then
      warn "Decode pod ${p}: crash signature found in current log -- see ${dir}/${p}.log"
      grep -hiE "${crash_re}" "${dir}/${p}.log" 2>/dev/null | tail -20 | sed 's/^/[bench][crash] /' || true
    fi
    local restarts
    restarts="$(kubectl get pod -n "${NAMESPACE}" "${p}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="modelserver")].restartCount}' 2>/dev/null || echo 0)"
    if [[ "${restarts:-0}" -gt 0 ]]; then
      warn "Decode pod ${p} modelserver restarted ${restarts}x -- capturing crash log (--previous)"
      if kubectl logs -n "${NAMESPACE}" "${p}" -c modelserver --previous \
           > "${dir}/${p}.crash.log" 2>&1; then
        grep -hiE "${crash_re}" \
          "${dir}/${p}.crash.log" 2>/dev/null | tail -20 | sed 's/^/[bench][crash] /' || true
      else
        warn "Failed to capture crash log for ${p}"
      fi
    fi
  done
  # Echo the KV cache sizing / concurrency lines for at-a-glance verification.
  grep -hiE "Available KV cache memory|GPU KV cache size|Maximum concurrency" \
    "${dir}"/*.log 2>/dev/null | sed 's/^/[bench][vllm] /' || true
}
teardown() {  # $1 = overlay dir
  log "Tearing down $1"
  kubectl delete -n "${NAMESPACE}" -k "$1" --ignore-not-found >/dev/null 2>&1 || true
  # Block until the decode pods are fully gone so their DRA ResourceClaims
  # release the 8 GPUs before the next variant tries to schedule -- otherwise
  # the offload deploy races the terminating baseline pods and can't be placed.
  log "Waiting for decode pods to terminate (release GPUs) ..."
  kubectl wait -n "${NAMESPACE}" --for=delete pod -l llm-d.ai/role=decode --timeout=180s >/dev/null 2>&1 || true
  CURRENT_OVERLAY=""
}

# ---------------------------------------------------------------------------
# 4. Run one benchmark stage
# ---------------------------------------------------------------------------
run_bench() {  # $1 = label
  local ws="${WORKSPACE}/$1"
  mkdir -p "${ws}"
  local ip
  ip="$(kubectl get service tiered-prefix-cache-epp -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}')"
  log "Benchmarking '$1' against http://${ip} -> ${ws}"
  # Proxy strategy: the benchmark CLIENT and the in-cluster harness pod have
  # OPPOSITE needs. Cluster traffic (kind API at 127.0.0.1, EPP clusterIP, and
  # in-cluster services) must stay DIRECT -- the corporate proxy 403s it. But
  # the harness pod must reach huggingface.co (the kind node has no direct
  # egress) THROUGH the proxy to download the model tokenizer. We satisfy both
  # by SETTING the proxy with a comprehensive no_proxy (cluster API + EPP IP +
  # cluster CIDRs + .svc), so the client only proxies genuinely external hosts
  # (image registries, huggingface.co), and by propagating those same vars into
  # the harness pod via `-g/--envvarspod`, which copies the named vars from this
  # process's environment into the pod. (Note: the scenario's harness.extraEnvVars
  # is rejected by the strict config schema -- `-g` is the supported mechanism.)
  #
  # CRITICAL: the kubernetes Python client (used for render-time node scans and
  # deploy-time reachability checks) honors no_proxy via requests'
  # should_bypass_proxies, which matches the API host *including its port*. The
  # kubeconfig server is https://127.0.0.1:<port>, so a bare "127.0.0.1" entry
  # does NOT match and the call gets proxied (-> 403 Forbidden). We must add the
  # exact host:port of the API server (and localhost:port) to no_proxy.
  local api_hp
  api_hp="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' \
            | sed -E 's#^https?://##')"
  local api_port="${api_hp##*:}"
  local np="127.0.0.1,127.0.0.1:${api_port},localhost,localhost:${api_port},${api_hp},${ip},10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local,.cluster.local"
  local envvarspod=()
  if [[ -n "${HARNESS_HF_PROXY}" ]]; then
    export http_proxy="${HARNESS_HF_PROXY}" https_proxy="${HARNESS_HF_PROXY}"
    export HTTP_PROXY="${HARNESS_HF_PROXY}" HTTPS_PROXY="${HARNESS_HF_PROXY}"
    envvarspod=(-g "http_proxy,https_proxy,HTTP_PROXY,HTTPS_PROXY,no_proxy,NO_PROXY")
  else
    warn "No HARNESS_HF_PROXY set; harness tokenizer download may fail offline"
    export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY=""
  fi
  export no_proxy="${np}" NO_PROXY="${np}"
  llmdbenchmark \
    --workspace    "${ws}" \
    --spec         "${SPEC}" \
    run \
    --base-dir     "${BENCH_BASE_DIR}" \
    --endpoint-url "http://${ip}" \
    --gateway-class epponly \
    --model        "${MODEL}" \
    --namespace    "${NAMESPACE}" \
    --harness      "${HARNESS}" \
    --workload     "${WORKLOAD_PROFILE_NAME}" \
    "${envvarspod[@]}" \
    --analyze
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
preflight
ensure_bench_cli
fix_kube_no_proxy_bug
install_workload_profile
patch_scenario
ensure_harness_image
cleanup
set_image
ensure_xpu_image
ensure_namespace
ensure_router
mkdir -p "${WORKSPACE}"

log "=== Baseline: VRAM-only ==="
if [[ "${SKIP_BASELINE:-0}" == "1" ]]; then
  log "SKIP_BASELINE=1 -> skipping VRAM-only baseline variant"
else
  CURRENT_LABEL="${BASELINE_LABEL}"
  deploy "${BASELINE_OVERLAY}"
  capture_vllm_logs "${BASELINE_LABEL}"
  run_bench "${BASELINE_LABEL}"
  teardown "${BASELINE_OVERLAY}"
fi

log "=== Offloading: CPU RAM (${OFFLOAD_LABEL}) ==="
if [[ "${SKIP_OFFLOAD:-0}" == "1" ]]; then
  log "SKIP_OFFLOAD=1 -> skipping CPU-offload variant"
else
  CURRENT_LABEL="${OFFLOAD_LABEL}"
  deploy "${OFFLOAD_OVERLAY}"
  capture_vllm_logs "${OFFLOAD_LABEL}"
  run_bench "${OFFLOAD_LABEL}"
  teardown "${OFFLOAD_OVERLAY}"
fi

log "Done. Results: ${WORKSPACE}"
log "Compare ${BASELINE_LABEL}/ vs ${OFFLOAD_LABEL}/ summaries to assess CPU-offload gains."
