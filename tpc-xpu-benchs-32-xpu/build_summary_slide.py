#!/usr/bin/env python3
"""Consolidate the 32-XPU tiered-prefix-cache benchmark into a single slide.

Produces:
  benchmark_summary.png   - 2x3 comparison figure (throughput + TTFT vs QPS)
  benchmark_summary.pptx  - one 16:9 slide: chart + concise conclusions

Reuses the data loaders from gen_report.py so the numbers stay in sync with
report.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gen_report as gr

BASE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(BASE, "benchmark_summary.png")
PPTX = os.path.join(BASE, "benchmark_summary.pptx")

# baseline / cpu / fs -> colour + marker
STYLE = {
    "baseline":   dict(color="#7f7f7f", marker="o", label="Baseline"),
    "native-cpu": dict(color="#1f77b4", marker="s", label="Native CPU offload"),
    "native-fs":  dict(color="#2ca02c", marker="^", label="Native FS offload"),
}
MODEL_SHORT = {
    "llama3-70b": "Llama-3 70B\n(TP=8 x4)",
    "qwen3-30b-a3b": "Qwen3-30B-A3B\n(TP=4 x8)",
    "qwen3-32b": "Qwen3-32B\n(TP=4 x8)",
}


def series(case, key):
    qs = sorted(case)
    xs, ys = [], []
    for q in qs:
        v = case[q].get(key)
        if v is not None and case[q]["failures"] == 0:
            xs.append(q)
            ys.append(v)
    return xs, ys


def build_figure(data):
    fig, axes = plt.subplots(2, len(gr.MODELS), figsize=(13.2, 6.0), dpi=200)
    rows = [("out_tok_s", "Output throughput (tok/s)", False),
            ("ttft_mean", "TTFT mean (s)", True)]
    for r, (key, ylabel, logy) in enumerate(rows):
        for c, m in enumerate(gr.MODELS):
            ax = axes[r][c]
            for cfg in gr.CONFIGS:
                xs, ys = series(data[m][cfg], key)
                if not xs:
                    continue
                st = STYLE[cfg]
                ax.plot(xs, ys, marker=st["marker"], color=st["color"],
                        label=st["label"], linewidth=1.8, markersize=4.5)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if r == 0:
                ax.set_title(MODEL_SHORT[m], fontsize=11, fontweight="bold")
            if r == len(rows) - 1:
                ax.set_xlabel("Offered QPS (cluster)", fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)


def pct(a, b):
    return (a - b) / b * 100.0 if b else 0.0


def peak_delta(case_fs, case_base):
    """FS vs baseline out-tok/s and ttft % at the top shared failure-free QPS."""
    common = sorted(set(case_fs) & set(case_base),
                    key=float)
    common = [q for q in common
              if case_fs[q]["failures"] == 0 and case_base[q]["failures"] == 0]
    if not common:
        return None
    q = common[-1]
    dt = pct(case_fs[q]["out_tok_s"], case_base[q]["out_tok_s"])
    dl = pct(case_fs[q]["ttft_mean"], case_base[q]["ttft_mean"])
    return q, dt, dl


def build_conclusions(data, prefix):
    lines = []
    for m in gr.MODELS:
        d = peak_delta(data[m]["native-fs"], data[m]["baseline"])
        _, ext_fs, _ = prefix[m]["native-fs"]
        if d:
            q, dt, dl = d
            lines.append((m, q, dt, dl, ext_fs))
    return lines


def add_slide(lines):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Title
    tb = slide.shapes.add_textbox(Inches(0.35), Inches(0.18),
                                  Inches(12.6), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ("Tiered Prefix-Cache KV-Offloading on 32x Intel Arc Pro B60 - "
              "Baseline vs CPU vs FS")
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1a, 0x1a, 0x1a)
    sub = tf.add_paragraph()
    rs = sub.add_run()
    rs.text = ("vLLM v0.26.0 - shared-prefix workload, Poisson load ladder - "
               "cluster-aggregate throughput / per-request latency")
    rs.font.size = Pt(11)
    rs.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # Chart
    slide.shapes.add_picture(PNG, Inches(0.3), Inches(1.0), width=Inches(8.75))

    # Conclusions panel (right)
    box = slide.shapes.add_textbox(Inches(9.2), Inches(1.05),
                                   Inches(3.95), Inches(6.2))
    tf = box.text_frame
    tf.word_wrap = True

    def head(text):
        p = tf.add_paragraph()
        rr = p.add_run()
        rr.text = text
        rr.font.size = Pt(13)
        rr.font.bold = True
        rr.font.color.rgb = RGBColor(0x15, 0x3a, 0x6b)
        p.space_before = Pt(6)

    def bullet(text, color=RGBColor(0x22, 0x22, 0x22)):
        p = tf.add_paragraph()
        rr = p.add_run()
        rr.text = "- " + text
        rr.font.size = Pt(10.5)
        rr.font.color.rgb = color
        p.space_after = Pt(2)

    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = "Key findings"
    r0.font.size = Pt(15)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(0x0f, 0x0f, 0x0f)

    dmap = {m: (q, dt, dl, ext) for (m, q, dt, dl, ext) in lines}

    q, dt, dl, ext = dmap["qwen3-30b-a3b"]
    head("Qwen3-30B-A3B - FS wins")
    bullet(f"FS best across the full 2-32 ladder; +{dt:.0f}% tok/s and "
           f"{dl:.0f}% TTFT vs baseline at {q:g} QPS, ~{ext:.0f}% tier-hit.")

    q, dt, dl, ext = dmap["qwen3-32b"]
    head("Qwen3-32B - FS wins")
    bullet(f"FS best across the full 2-32 ladder; +{dt:.0f}% tok/s and "
           f"{dl:.0f}% TTFT vs baseline at {q:g} QPS, ~{ext:.0f}% tier-hit.")

    q, dt, dl, ext = dmap["llama3-70b"]
    head("Llama-3 70B - load-dependent")
    bullet("Baseline best at light load (<=2 QPS): offload round-trip not "
           "yet amortized.")
    bullet(f"FS wins from ~4 QPS up; +{dt:.0f}% tok/s and {dl:.0f}% TTFT vs "
           f"baseline at {q:g} QPS, ~{ext:.0f}% tier-hit.")

    head("Bottom line")
    bullet("Native FS offload is the recommended tier for sustained load on "
           "all three models.", RGBColor(0x18, 0x60, 0x18))
    bullet("Benefit scales with offered load and prefix-reuse working set; "
           "PCIe-bound (no Xe-Link), so it pays off once recompute pressure "
           "exceeds transfer cost.")

    prs.save(PPTX)


def main():
    data = {m: {c: gr.load_case(m, c) for c in gr.CONFIGS} for m in gr.MODELS}
    prefix = {m: {c: gr.load_prefix(m, c) for c in gr.CONFIGS} for m in gr.MODELS}
    build_figure(data)
    lines = build_conclusions(data, prefix)
    add_slide(lines)
    print("wrote", PNG)
    print("wrote", PPTX)
    for m, q, dt, dl, ext in lines:
        print(f"  {m}: @QPS {q:g}  FS vs base  out+{dt:.1f}%  ttft{dl:.1f}%  "
              f"ext-hit {ext:.0f}%")


if __name__ == "__main__":
    main()
