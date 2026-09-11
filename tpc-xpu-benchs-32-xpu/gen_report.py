#!/usr/bin/env python3
"""Generate a comparative benchmark report for the 32-XPU tiered-prefix-cache runs.

For each model, only offered-QPS points that are FAILURE-FREE across all three
configurations (baseline, native-cpu, native-fs) are used for the head-to-head
comparison, per the requirement to compare on clean load points only.
"""
import glob
import os
import re
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["llama3-70b", "qwen3-30b-a3b", "qwen3-32b"]
CONFIGS = ["baseline", "native-cpu", "native-fs"]
CFG_PRETTY = {
    "baseline": "Baseline (no offload)",
    "native-cpu": "Native CPU offload",
    "native-fs": "Native FS offload",
}
MODEL_PRETTY = {
    "llama3-70b": "Llama-3 70B (TP=8, 4 replicas)",
    "qwen3-30b-a3b": "Qwen3-30B-A3B (TP=4, 8 replicas)",
    "qwen3-32b": "Qwen3-32B (TP=4, 8 replicas)",
}

# Per-model deployment + workload context (from run-all-benchmarks.sh and the run
# artifacts). kv_tok = KV bytes/token; vram_c = measured in-VRAM KV cache in tokens
# per replica at gpu-mem-util 0.85 (0 = not measured).
MODEL_META = {
    "qwen3-30b-a3b": dict(hf="Qwen/Qwen3-30B-A3B", tp=4, reps=8, prefix=12000,
                          inlen=12256, groups=464, kv_kib=96, vram_c=196736),
    "qwen3-32b": dict(hf="Qwen/Qwen3-32B", tp=4, reps=8, prefix=3600,
                      inlen=3856, groups=448, kv_kib=256, vram_c=50944),
    "llama3-70b": dict(hf="meta-llama/Meta-Llama-3-70B", tp=8, reps=4, prefix=4000,
                       inlen=4255, groups=216, kv_kib=320, vram_c=0),
}


def rfind(o, key):
    """Recursively find first value for `key`."""
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = rfind(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = rfind(v, key)
            if r is not None:
                return r
    return None


def load_case(model, config):
    """Return dict: offered_qps -> metrics for every v0.2 stage file."""
    d = os.path.join(BASE, f"{model}-{config}")
    pat = os.path.join(d, "*", "results", "*",
                       "benchmark_report_v0.2,_stage_*_lifecycle_metrics.json.yaml")
    stages = {}
    for f in sorted(glob.glob(pat)):
        doc = yaml.safe_load(open(f))
        agg = doc["results"]["request_performance"]["aggregate"]
        req = agg.get("requests", {})
        thr = agg.get("throughput", {})
        lat = agg.get("latency", {})

        def g(block, name, stat):
            return block.get(name, {}).get(stat)

        qps = float(rfind(doc, "rate_qps"))
        stages[qps] = {
            "failures": int(req.get("failures", 0)),
            "total": int(req.get("total", 0)),
            "out_tok_s": g(thr, "output_token_rate", "mean"),
            "in_tok_s": g(thr, "input_token_rate", "mean"),
            "req_s": g(thr, "request_rate", "mean"),
            "ttft_mean": g(lat, "time_to_first_token", "mean"),
            "ttft_p99": g(lat, "time_to_first_token", "p99"),
            "tpot_mean": g(lat, "time_per_output_token", "mean"),
            "tpot_p99": g(lat, "time_per_output_token", "p99"),
            "e2e_mean": g(lat, "request_latency", "mean"),
            "e2e_p99": g(lat, "request_latency", "p99"),
        }
    return stages


def load_prefix(model, config):
    """Aggregate prefix-cache counters across pods; return hit rates."""
    fp = os.path.join(BASE, f"{model}-{config}", "prefix_cache_metrics.txt")
    sums = {"q": 0.0, "h": 0.0, "eq": 0.0, "eh": 0.0}
    keymap = {
        "vllm:prefix_cache_queries_total": "q",
        "vllm:prefix_cache_hits_total": "h",
        "vllm:external_prefix_cache_queries_total": "eq",
        "vllm:external_prefix_cache_hits_total": "eh",
    }
    if os.path.exists(fp):
        for line in open(fp):
            line = line.strip()
            if line.startswith("#") or line.startswith("###") or not line:
                continue
            m = re.match(r"(vllm:[a-z_]+)\{.*\}\s+([0-9.eE+]+)", line)
            if not m:
                continue
            k = keymap.get(m.group(1))
            if k:
                sums[k] += float(m.group(2))
    gpu = (sums["h"] / sums["q"] * 100) if sums["q"] else None
    ext = (sums["eh"] / sums["eq"] * 100) if sums["eq"] else None
    return gpu, ext, sums


def fmt(x, nd=1):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    data = {m: {c: load_case(m, c) for c in CONFIGS} for m in MODELS}
    prefix = {m: {c: load_prefix(m, c) for c in CONFIGS} for m in MODELS}

    out = []
    W = out.append
    W("# Tiered Prefix Cache — 32-XPU Benchmark Report\n")
    W("Comparison of KV-cache offloading strategies on a 32× Intel Arc Pro B60 "
      "cluster (vLLM v0.26.0, `llm-d-xpu:v0.9.0`).\n")
    W("**Configurations compared**\n")
    W("- **Baseline** — GPU-only prefix cache, no offloading.")
    W("- **Native CPU offload** — KV blocks offloaded to host RAM (CPU tier).")
    W("- **Native FS offload** — KV blocks offloaded to a shared filesystem tier.\n")
    W("**Methodology.** Each model was driven through a Poisson request ladder of "
      "increasing offered QPS. For the head-to-head tables below, **only offered-QPS "
      "points that recorded zero failed requests in all three configurations are "
      "used** — any QPS at which any config dropped requests is excluded so the "
      "comparison is apples-to-apples on stable load. Throughput and token rates are "
      "cluster-aggregate; latencies are per-request aggregate percentiles.\n")

    # Experimental setup / related context
    W("## 0. Experimental setup & context\n")
    W("**Cluster.** 4 worker nodes × 8 Intel Arc Pro B60 (24 GB GDDR6) = **32 XPUs**. "
      "The B60s are discrete PCIe cards with **no Xe-Link**, so every tensor-parallel "
      "all-reduce and every KV VRAM↔host/FS transfer rides PCIe — relevant to the "
      "offload-tier latencies below.\n")
    W("**Software.** vLLM v0.26.0, image `ghcr.io/llm-d/llm-d-xpu:v0.9.0`, served via "
      "llm-d with a standalone endpoint-picker (EPP) router doing prefix-aware routing. "
      "vLLM args common to all runs: `--dtype bfloat16 --block-size 64 "
      "--gpu-memory-utilization 0.85 --max-num-seq 128`. Each model is sized to fill all "
      "32 XPUs (TP × replicas = 32).\n")
    W("**Offload tiers under test.**")
    W("- **Baseline** — VRAM-only KV cache (control), no offloading.")
    W("- **Native CPU offload** — `OffloadingConnector`, VRAM → host RAM; the CPU tier is "
      "sized to ~80% of each replica's reusable-prefix working set.")
    W("- **Native FS offload** — `TieringOffloadingSpec`, VRAM → CPU → node-local NVMe "
      "filesystem (hostPath); the FS tier is large enough to hold ~the whole working set.")
    W("\nExpected prefix-cache-hit ordering by design: baseline < native-cpu < native-fs. "
      "The workload deliberately sizes each replica's reusable-prefix working set to "
      "~3× the in-VRAM KV cache, so baseline must evict/recompute while the offload tiers "
      "can retain more of it.\n")
    W("**Per-model deployment & workload.** Shared-prefix workload (`shared_prefix`): "
      "each request = a fixed shared system prefix + a 256-token question, generating a "
      "256-token answer; 5 prompts/group, 5 users/group. Open-loop Poisson load, 60 s per "
      "stage, per-request timeout 1800 s. Cluster offered QPS per stage = per-replica "
      "ladder × replicas.\n")
    W("| Model | HF repo | TP | Replicas | Prefix len (tok) | Input len (tok) | "
      "Output len | Prefix groups | KV B/token | Meas. VRAM KV/replica (tok) |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        mm = MODEL_META[m]
        vram = f"{mm['vram_c']:,}" if mm["vram_c"] else "n/a"
        W(f"| {MODEL_PRETTY[m].split(' (')[0]} | `{mm['hf']}` | {mm['tp']} | "
          f"{mm['reps']} | {mm['prefix']:,} | {mm['inlen']:,} | 256 | {mm['groups']} | "
          f"{mm['kv_kib']} KiB | {vram} |")
    W("")
    W("**Offered-QPS ladders (cluster).** Llama-3 70B: 1, 2, 3, 4, 5, 6. "
      "Qwen3-30B-A3B & Qwen3-32B: 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32.\n")
    W("**Metric definitions.** *Out tok/s* = cluster output-token throughput; "
      "*Achieved req/s* = measured completed requests/s (vs offered QPS); *TTFT* = "
      "time-to-first-token; *TPOT* = time-per-output-token; *E2E* = end-to-end request "
      "latency. Latencies are aggregate over all requests in the stage.\n")

    # Stability overview
    W("## 1. Stability overview (failed requests per offered QPS)\n")
    for m in MODELS:
        W(f"### {MODEL_PRETTY[m]}\n")
        all_qps = sorted(set().union(*[set(data[m][c]) for c in CONFIGS]))
        header = "| Offered QPS | " + " | ".join(CFG_PRETTY[c] for c in CONFIGS) + " |"
        W(header)
        W("|" + "---|" * (len(CONFIGS) + 1))
        for q in all_qps:
            cells = []
            for c in CONFIGS:
                s = data[m][c].get(q)
                if s is None:
                    cells.append("—")
                elif s["failures"] == 0:
                    cells.append(f"OK ({s['total']})")
                else:
                    cells.append(f"**{s['failures']} failed** / {s['total']}")
            W(f"| {q:g} | " + " | ".join(cells) + " |")
        W("")

    # Clean QPS comparison per model.
    # Layout: the three configs sit side-by-side as columns so they compare at a
    # glance; each offered-QPS block lists every metric as a row. The best config
    # per metric row is **bold** (higher is better for throughput/req-rate, lower
    # for the latency metrics).
    #   (label, key, decimals, "max"|"min" = which direction wins)
    METRICS_2 = [
        ("Out tok/s", "out_tok_s", 0, "max"),
        ("Achieved req/s", "req_s", 2, "max"),
        ("TTFT mean (s)", "ttft_mean", 2, "min"),
        ("TTFT p99 (s)", "ttft_p99", 2, "min"),
        ("TPOT mean (s)", "tpot_mean", 3, "min"),
        ("E2E p99 (s)", "e2e_p99", 2, "min"),
    ]
    W("## 2. Head-to-head at failure-free QPS points\n")
    W("Configs are columns for a direct side-by-side read; each offered-QPS block "
      "lists all metrics as rows. **Bold** marks the best config for that metric "
      "row (higher = better for throughput/req-rate, lower = better for the latency "
      "rows). Each offload column also shows its **% change vs baseline** in "
      "parentheses — for the throughput/req-rate rows positive is better, and for "
      "the latency rows negative is better. The percentage is coloured "
      "<span style=\"color:#d00000\">red when it is an improvement over baseline</span> "
      "and <span style=\"color:#008000\">green when it is worse</span>.\n")
    clean_sets = {}
    for m in MODELS:
        common = set(data[m][CONFIGS[0]])
        for c in CONFIGS[1:]:
            common &= set(data[m][c])
        clean = sorted(q for q in common
                       if all(data[m][c][q]["failures"] == 0 for c in CONFIGS))
        clean_sets[m] = clean

        W(f"### {MODEL_PRETTY[m]}\n")
        excluded = sorted(q for q in common if q not in clean)
        if excluded:
            W(f"_Excluded (failures in at least one config): "
              f"{', '.join(f'{q:g}' for q in excluded)} QPS._\n")
        if not clean:
            W("_No offered QPS was failure-free across all three configs — no "
              "comparison table._\n")
            continue

        W("| Offered QPS | Metric | " + " | ".join(CFG_PRETTY[c] for c in CONFIGS)
          + " |")
        W("|---|---|" + "---|" * len(CONFIGS))
        base_cfg = CONFIGS[0]
        for q in clean:
            for i, (label, key, nd, direction) in enumerate(METRICS_2):
                vals = {c: data[m][c][q][key] for c in CONFIGS}
                present = {c: v for c, v in vals.items() if v is not None}
                best = None
                if len(present) > 1:
                    best = (max if direction == "max" else min)(
                        present, key=lambda c: present[c])
                base_val = vals[base_cfg]
                cells = []
                for c in CONFIGS:
                    txt = fmt(vals[c], nd)
                    if c == best:
                        txt = f"**{txt}**"
                    if (c != base_cfg and vals[c] is not None
                            and base_val not in (None, 0)):
                        d = (vals[c] - base_val) / base_val * 100.0
                        sign = "+" if d >= 0 else "\u2212"
                        pct = f"({sign}{abs(d):.1f}%)"
                        improved = (d > 0) if direction == "max" else (d < 0)
                        if d != 0:
                            color = "#d00000" if improved else "#008000"
                            pct = f'<span style="color:{color}">{pct}</span>'
                        txt += f" {pct}"
                    cells.append(txt)
                qcol = f"**{q:g}**" if i == 0 else ""
                W(f"| {qcol} | {label} | " + " | ".join(cells) + " |")
        W("")

    # Peak stable QPS
    W("## 3. Peak failure-free offered QPS (per config)\n")
    W("| Model | " + " | ".join(CFG_PRETTY[c] for c in CONFIGS) + " |")
    W("|" + "---|" * (len(CONFIGS) + 1))
    for m in MODELS:
        cells = []
        for c in CONFIGS:
            clean_c = [q for q, s in data[m][c].items() if s["failures"] == 0]
            cells.append(f"{max(clean_c):g}" if clean_c else "—")
        W(f"| {MODEL_PRETTY[m]} | " + " | ".join(cells) + " |")
    W("")

    # Prefix cache hit rates
    W("## 4. Prefix-cache hit rates (end-of-run counters)\n")
    W("GPU hit rate = local prefix-cache hits / queries. Offload hit rate = "
      "external (cross-instance / offloaded tier) hits / queries. Baseline has no "
      "offload tier.\n")
    W("| Model | Config | GPU prefix hit % | Offload tier hit % |")
    W("|---|---|---|---|")
    for m in MODELS:
        for c in CONFIGS:
            gpu, ext, _ = prefix[m][c]
            W(f"| {MODEL_PRETTY[m]} | {CFG_PRETTY[c]} | {fmt(gpu,1)} | "
              f"{fmt(ext,1)} |")
    W("")

    # Key findings (data-driven deltas at max clean QPS)
    W("## 5. Key findings\n")
    for m in MODELS:
        clean = clean_sets[m]
        if not clean:
            W(f"- **{MODEL_PRETTY[m]}**: no common failure-free QPS across configs.")
            continue
        q = clean[-1]
        base = data[m]["baseline"][q]
        rows = []
        for c in ["native-cpu", "native-fs"]:
            s = data[m][c][q]
            d_tok = (s["out_tok_s"] - base["out_tok_s"]) / base["out_tok_s"] * 100
            d_ttft = (s["ttft_mean"] - base["ttft_mean"]) / base["ttft_mean"] * 100
            rows.append(f"{CFG_PRETTY[c]}: {d_tok:+.1f}% out-tok/s, "
                        f"{d_ttft:+.1f}% TTFT mean vs baseline")
        W(f"- **{MODEL_PRETTY[m]}** (max clean QPS = {q:g}): " + "; ".join(rows) + ".")
    W("")

    W("## 6. Interpretation & recommendations\n")
    W("- **Qwen3-30B-A3B — clear win for FS offload.** All 12 QPS points were "
      "failure-free for every config. FS offload delivered the highest output "
      "throughput (up to ~2.4k tok/s vs ~1.5k baseline) and the lowest TTFT at "
      "every load point, thanks to an ~85% offload-tier hit rate. Recommended "
      "configuration for this model.")
    W("- **Qwen3-32B — clear win for FS offload across the full ladder.** Every "
      "offered-QPS point (2–32) was failure-free for all three configs, and FS "
      "offload sustained load all the way to 32 QPS. FS offload led on throughput "
      "and latency at essentially every load point — at 32 QPS it delivered +50.0% "
      "output throughput and −45.5% TTFT mean vs baseline, on an ~87% offload-tier "
      "hit rate. (An earlier FS run appeared to collapse above 16 QPS, but that was "
      "a degraded-tier artifact — only ~38% hit rate driving recompute storms; the "
      "clean re-run restored the expected ~87% hit and the collapse disappeared.) "
      "Recommended configuration for this model.")
    W("- **Llama-3 70B — FS offload wins under load; baseline wins when light.** All "
      "six QPS points (1–6) were failure-free for every config at full 4/4 capacity. At "
      "light load (≤2 QPS) baseline is best on every axis — at 2 QPS it leads on "
      "throughput (291 tok/s vs 263 CPU / 233 FS) and TTFT mean (6.1 s vs 7.7 s CPU / "
      "12.6 s FS) — because the offload round-trip isn't yet amortized and the shared "
      "prefix still fits the working set without eviction pressure. CPU offload takes the "
      "lead at 3 QPS (320 tok/s, TTFT 17.0 s vs 31.4 s baseline). From ~4 QPS up, FS "
      "offload pulls ahead on every axis: at 6 QPS it delivered the highest throughput "
      "(393 tok/s vs 305 baseline / 340 CPU), the lowest TTFT mean (58.5 s, −41.7% vs "
      "baseline) and the lowest E2E p99 (183 s vs 239 baseline), on a ~68% offload-tier "
      "hit rate. Recommend FS offload for sustained load; for latency-sensitive light "
      "load, plain baseline is sufficient.")
    W("- **Offload-tier hit rate scales with the tier's capacity/locality.** FS "
      "offload reached ~85–87% external-hit rate for the two Qwen models "
      "(large shared filesystem) and ~68% for 70B, while GPU-local prefix hit rates "
      "stayed low (1–3%) because the shared-prefix workload spills beyond GPU HBM.")
    W("")

    report = "\n".join(out)
    with open(os.path.join(BASE, "report.md"), "w") as fh:
        fh.write(report)
    print(report)
    print("\n=== clean QPS sets ===")
    for m in MODELS:
        print(m, clean_sets[m])


if __name__ == "__main__":
    main()
