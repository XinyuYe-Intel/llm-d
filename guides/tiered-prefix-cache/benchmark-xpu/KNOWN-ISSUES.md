# Known issues (XPU tiered-prefix-cache benchmark)

## 1. vLLM OffloadingConnector store-job assertion crash (native-fs) — OPEN

**Status:** documented, fix deferred. Affects the `native-fs` config (VRAM→CPU→FS tiering)
under high KV-offload store pressure. Do NOT trust `native-fs` numbers from any run where
this fires; the model server crashes mid-run.

**Symptom:** decode pods repeatedly crash and restart (many `*.previous.log` files + tiny
stub logs in `<workspace>/<model>-<config>/vllm-logs/`); requests fail in bursts; per-stage
throughput collapses to ~0 while the load ladder is still climbing.

**Root cause:** an `AssertionError` in vLLM's KV-offload scheduler, thrown every scheduler
step while building the offload store jobs, kills the EngineCore → `EngineDeadError` → the
container restarts, then crashes again under the same load:

```
vllm/v1/core/sched/scheduler.py:1174  schedule
 -> _build_kv_connector_meta
 -> vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:157  build_connector_meta
 -> vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1157  build_connector_meta
 -> vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:975   _build_store_jobs
      assert len(offload_keys) == len(offload_block_ids)
    AssertionError
```

**Scope (rate run `tpc-xpu-rate-local-32b-70b-20260908-083707`)** — restarted pods /
`EngineDeadError` count / offload-assert hits:

| case | restarts | EngineDeadError | offload-assert |
|---|---:|---:|---:|
| qwen3-32b baseline | 0 | 0 | 0 |
| qwen3-32b native-cpu | 0 | 0 | 0 |
| **qwen3-32b native-fs** | **6** | **142** | **6** |
| llama3-70b native-fs | 1 | 10 | 0 (scrolled out of captured tail) |
| all others | 0 | 0 | 0 |

The crash is **exclusive to native-fs**: `baseline` has no offload connector, and
`native-cpu` uses `OffloadingConnector` with a CPU-only tier and never trips it. Only the
FS secondary tier (`TieringOffloadingSpec`, `secondary_tiers[fs]`) hits the buggy store path.

**Trigger conditions:** high offload **store** churn. Amplified here by a tiny GPU KV cache
(qwen3-32b = 50,944 tok/replica on 24 GB cards) + `--max-num-seq 128` + long prompts +
the open-loop rate ladder overdriving arrival rate past the service ceiling, so blocks are
constantly evicted/offloaded. `vllm:kv_offload_lookup_sync_delay_seconds_count` spiked from
~150 to ~1600 immediately before the crash.

**Environment:** image `ghcr.io/llm-d/llm-d-xpu:v0.9.0`, vLLM v0.26.0.

**Mitigations / TODO:**
- Reproduce on a small model and file upstream against `offloading/scheduler.py:975`.
- Until fixed: run the tiering comparison in closed-loop `concurrent` mode (bounded in-flight
  requests keep offload churn out of the buggy regime), and/or lower `--max-num-seq` /
  raise `--gpu-memory-utilization` to reduce KV oversubscription.

## 2. Open-loop `rate` ladder must stay near the measured service ceiling

An open-loop Poisson ladder whose stages sit **above** the server's drain rate builds an
unbounded backlog: every over-ceiling stage's tail requests wait up to `RATE_REQUEST_TIMEOUT`
(1800s), so the harness wall-time grows without bound and `llmdbenchmark` fails the case with
`Pods did not complete within <wait-timeout>s`.

**Measured cluster ceilings (this cluster):**
- llama3-70b (TP=8, 4 replicas): **~0.7 req/s cluster (~0.175 q/s per replica).**
- qwen3-32b (TP=4, 8 replicas): saturates well below the top of its ladder too.

The `llama3-70b` baseline and native-cpu cases in the rate run above timed out at the 3h cap
for exactly this reason: their whole ladder (cluster 1–12 q/s) was above the 0.7 q/s ceiling.
The `llama3-70b` ladder in `run-all-benchmarks.sh` has been re-centered on the measured
ceiling (per-replica 0.05–0.25 → cluster 0.2–1.0 q/s). Keep every ladder's top stage
≤ ~1.5× the measured ceiling so backlog stays bounded and cases finish in minutes.
