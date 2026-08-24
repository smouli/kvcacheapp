#!/usr/bin/env python3
"""Merge benchmark CSVs into a hostable HTML report (model + serving + hardware)."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "benchmarks" / "sample_results" / "demo_runs.csv"
SAMPLE_MODAL = ROOT / "benchmarks" / "sample_results" / "modal_runs.csv"
DATA_JSON = ROOT / "public" / "benchmark-data.json"
REPORTS = ROOT / "benchmarks" / "reports"

F_IN = "input_tokens_target"
F_OUT = "output_tokens_target"
F_TTFT = "ttft_ms_p50"
F_TOKS = "output_tok_s_p50"
F_CACHED = "cached_prompt_tokens"
F_KV_GQA = "kv_gib_modeled_gqa"
F_KV_DENSE = "kv_gib_modeled_dense"
F_PEAK = "peak_gpu_gib"

STYLE = r"""
:root {
  --bg: #e9eef3;
  --ink: #0b1220;
  --muted: #5b6577;
  --panel: rgba(255,255,255,.78);
  --panel-solid: #f7fafc;
  --line: #c9d4e0;
  --teal: #0d9488;
  --teal-deep: #0f766e;
  --sky: #0369a1;
  --amber: #d97706;
  --ok-bg: #ccfbf1;
  --ok-ink: #115e59;
  --todo-bg: #e8edf3;
  --todo-ink: #64748b;
  --display: "Syne", system-ui, sans-serif;
  --sans: "DM Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  font-family: var(--sans);
  background:
    linear-gradient(180deg, #dfe8f2 0%, transparent 42%),
    repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(11,18,32,.04) 31px, rgba(11,18,32,.04) 32px),
    repeating-linear-gradient(90deg, transparent, transparent 31px, rgba(11,18,32,.04) 31px, rgba(11,18,32,.04) 32px),
    var(--bg);
  min-height: 100vh;
}
a { color: var(--teal-deep); text-decoration-thickness: 1px; text-underline-offset: 3px; }
a:hover { color: var(--sky); }

.topnav {
  position: sticky; top: 0; z-index: 20;
  backdrop-filter: blur(12px);
  background: rgba(233,238,243,.82);
  border-bottom: 1px solid var(--line);
}
.topnav-inner {
  max-width: 1120px; margin: 0 auto;
  padding: .7rem 1.25rem;
  display: flex; flex-wrap: wrap; gap: .75rem 1.25rem;
  align-items: center; justify-content: space-between;
}
.brand-mark {
  font-family: var(--display); font-weight: 800; font-size: 1.05rem;
  letter-spacing: -.02em; color: var(--ink); text-decoration: none;
}
.brand-mark span { color: var(--teal); }
.nav-links { display: flex; flex-wrap: wrap; gap: .35rem .85rem; }
.nav-links a {
  font-size: .82rem; font-weight: 600; color: var(--muted); text-decoration: none;
}
.nav-links a:hover { color: var(--ink); }

.wrap { max-width: 1120px; margin: 0 auto; padding: 0 1.25rem 4rem; }

.hero {
  padding: 2.75rem 0 1.5rem;
  animation: rise .7s ease both;
}
.hero-brand {
  font-family: var(--display);
  font-weight: 800;
  font-size: clamp(3rem, 9vw, 5.25rem);
  line-height: .92;
  letter-spacing: -.045em;
  margin: 0 0 .85rem;
  max-width: 12ch;
}
.hero-brand em {
  font-style: normal;
  background: linear-gradient(120deg, var(--teal), var(--sky));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub {
  font-size: clamp(1.05rem, 2.2vw, 1.25rem);
  color: var(--muted); max-width: 38rem; line-height: 1.5; margin: 0 0 1.5rem;
}
.hero-cta { display: flex; flex-wrap: wrap; gap: .6rem; }
.btn {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .65rem 1rem; border-radius: 8px; font-weight: 650; font-size: .9rem;
  text-decoration: none; border: 1px solid transparent;
}
.btn-primary { background: var(--ink); color: #fff; }
.btn-primary:hover { background: #1e293b; color: #fff; }
.btn-ghost { background: var(--panel); color: var(--ink); border-color: var(--line); }
.btn-ghost:hover { border-color: var(--teal); color: var(--teal-deep); }

.stack-strip {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem;
  margin: 1.75rem 0 0;
  animation: rise .8s .08s ease both;
}
@media (max-width: 720px) { .stack-strip { grid-template-columns: 1fr; } }
.stack-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 1rem 1.05rem; backdrop-filter: blur(8px);
}
.stack-card .step {
  font-family: var(--mono); font-size: .7rem; color: var(--teal-deep);
  letter-spacing: .08em; text-transform: uppercase; font-weight: 500;
}
.stack-card h3 {
  font-family: var(--display); font-size: 1.15rem; margin: .35rem 0 .4rem;
  letter-spacing: -.02em;
}
.stack-card p { margin: 0; font-size: .88rem; color: var(--muted); line-height: 1.45; }

.kpi {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: .75rem; margin: 2rem 0 0;
  animation: rise .85s .12s ease both;
}
.stat {
  background: var(--panel-solid); border: 1px solid var(--line); border-radius: 12px;
  padding: .9rem 1rem;
}
.stat-k {
  display: block; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); font-weight: 600;
}
.stat-v {
  display: block; font-family: var(--display); font-weight: 800;
  font-size: clamp(1.55rem, 3vw, 2rem); letter-spacing: -.03em; margin-top: .25rem;
  color: var(--teal-deep);
}
.stat-s { display: block; font-size: .82rem; color: var(--muted); margin-top: .2rem; }

section {
  margin-top: 3rem;
  animation: rise .6s ease both;
}
section h2 {
  font-family: var(--display); font-weight: 700; font-size: clamp(1.45rem, 3vw, 1.85rem);
  letter-spacing: -.03em; margin: 0 0 .35rem;
}
.lead { color: var(--muted); font-size: 1rem; line-height: 1.5; max-width: 46rem; margin: 0 0 1rem; }
.muted { color: var(--muted); }
h3 {
  font-size: .92rem; margin: 1.15rem 0 .45rem; font-weight: 650; color: var(--ink);
}

.cov-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: .7rem; margin-top: .85rem;
}
.cov {
  border: 1px solid var(--line); border-radius: 10px; padding: .85rem .9rem;
  background: var(--panel-solid);
}
.cov.ok { border-color: #99f6e4; background: linear-gradient(160deg, #f0fdfa, #fff); }
.cov.todo { opacity: .78; }
.cov-top { display: flex; justify-content: space-between; gap: .5rem; align-items: start; }
.detail { font-size: .82rem; color: var(--muted); margin: .35rem 0 0; }

.badge {
  font-size: .62rem; letter-spacing: .08em; text-transform: uppercase;
  padding: .18rem .42rem; border-radius: 6px; font-weight: 700; white-space: nowrap;
}
.badge.ok { background: var(--ok-bg); color: var(--ok-ink); }
.badge.todo { background: var(--todo-bg); color: var(--todo-ink); }

.pill {
  display: inline-block; font-size: .78rem; padding: .28rem .55rem; margin: .15rem .2rem .15rem 0;
  border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--ink);
}

.charts {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0 1.5rem;
}
@media (max-width: 860px) { .charts { grid-template-columns: 1fr; } }
.chart {
  margin: 0; background: var(--panel-solid); border: 1px solid var(--line);
  border-radius: 14px; padding: 1rem 1rem .75rem; overflow: hidden;
}
.chart figcaption { display: flex; flex-direction: column; gap: .15rem; margin-bottom: .35rem; }
.chart figcaption strong { font-family: var(--display); font-size: 1.05rem; }
.chart-leg { display: flex; flex-wrap: wrap; gap: .55rem; margin-bottom: .35rem; }
.leg { display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem; color: var(--muted); }
.leg i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.chart svg { width: 100%; height: auto; display: block; }
.chart .g { stroke: #d5dee8; stroke-width: 1; }
.chart .tick { fill: var(--muted); font-size: 10px; font-family: var(--mono); }
.chart .axis { fill: var(--muted); font-size: 11px; font-family: var(--sans); }
.chart .line {
  stroke-dasharray: 1200; stroke-dashoffset: 1200;
  animation: draw 1.2s ease forwards .2s;
}
.bar-anim { transform-origin: bottom; animation: grow .7s ease both; }

.bar-row {
  display: grid; grid-template-columns: minmax(9rem,15rem) 1fr minmax(12rem,22rem);
  gap: .7rem; align-items: center; margin: .42rem 0;
}
.lbl { font-size: .72rem; color: var(--muted); font-family: var(--mono); }
.track {
  background: #d9e2ec; height: 9px; border-radius: 999px; overflow: hidden;
}
.fill {
  height: 100%; width: 0; border-radius: 999px; background: var(--c, var(--teal));
  animation: fillbar .9s cubic-bezier(.2,.8,.2,1) forwards;
}
.fill { --w: 0%; }
@keyframes fillbar { to { width: var(--w); } }
.val { font-size: .72rem; color: #475569; font-family: var(--mono); }

.filters {
  display: flex; flex-wrap: wrap; gap: .45rem; margin: .75rem 0 1rem;
}
.filters button {
  font-family: var(--sans); font-size: .8rem; font-weight: 600;
  padding: .4rem .7rem; border-radius: 8px; border: 1px solid var(--line);
  background: #fff; color: var(--muted); cursor: pointer;
}
.filters button.active, .filters button:hover {
  border-color: var(--teal); color: var(--teal-deep); background: #f0fdfa;
}

.table-wrap {
  overflow-x: auto; border: 1px solid var(--line); border-radius: 12px;
  background: var(--panel-solid);
}
table {
  width: 100%; border-collapse: collapse; font-size: .76rem;
}
th, td {
  padding: .5rem .55rem; text-align: left; border-bottom: 1px solid #e8eef4; white-space: nowrap;
}
th {
  background: #eef3f8; font-size: .64rem; letter-spacing: .07em;
  text-transform: uppercase; color: var(--muted); position: sticky; top: 0;
}
tr:hover td { background: #f4f8fb; }
.mono { font-family: var(--mono); font-size: .92em; }

pre.mono {
  background: #0b1220; color: #d6e4f0; border-radius: 12px; padding: 1.1rem 1.2rem;
  overflow: auto; font-size: .78rem; line-height: 1.5; border: 1px solid #1e293b;
}
pre.mono .c { color: #5eead4; }

footer {
  margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
  font-size: .78rem; color: var(--muted); display: flex; flex-wrap: wrap;
  gap: .5rem 1.25rem; justify-content: space-between;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: none; }
}
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes grow {
  from { transform: scaleY(0); }
  to { transform: scaleY(1); }
}

@media (max-width: 800px) {
  .bar-row { grid-template-columns: 1fr; gap: .2rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important; transition: none !important;
  }
  .fill { width: var(--w) !important; }
}

.hw-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:.75rem; margin:1rem 0; }
.hw-card { background:var(--panel-solid); border:1px solid var(--line); border-radius:12px; padding:1rem; }
.hw-card .sku { font-family:var(--display); font-weight:800; font-size:1.2rem; letter-spacing:-.02em; }
.hw-card .name { font-size:.82rem; color:var(--muted); margin:.2rem 0 .75rem; }
.hw-metrics { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; }
.hw-metrics div { background:#eef3f8; border-radius:8px; padding:.55rem .65rem; }
.hw-metrics span { display:block; font-size:.65rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.hw-metrics b { font-family:var(--mono); font-size:1.05rem; color:var(--teal-deep); }
"""


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(normalize_row(row))
    return rows


def normalize_row(row: dict) -> dict:
    out = dict(row)
    aliases = {
        F_IN: ("input_tokens_target", "input_tokens"),
        F_OUT: ("output_tokens_target", "output_tokens"),
        F_TTFT: ("ttft_ms_p50", "ttft_ms"),
        F_TOKS: ("output_tok_s_p50", "tok_s_p50"),
        F_CACHED: ("cached_prompt_tokens", "cached_tokens"),
        F_KV_GQA: ("kv_gib_modeled_gqa",),
        F_KV_DENSE: ("kv_gib_modeled_dense",),
        F_PEAK: ("peak_gpu_gib",),
        "memcpy_gbps": ("memcpy_gbps", "memcpy_bandwidth_gbps"),
        "matmul_tflops_bf16": ("matmul_tflops_bf16", "matmul_tflops_bf16"),
        "matmul_tflops_fp16": ("matmul_tflops_fp16",),
        "nccl_busbw_gbps": ("nccl_busbw_gbps", "nccl_bus_bw_gbps"),
        "nccl_algbw_gbps": ("nccl_algbw_gbps",),
        "gpu_sku": ("gpu_sku",),
        "gpu_name": ("gpu_name",),
        "memory_gib": ("memory_gib",),
        "run_id": ("run_id",),
    }
    for canon, alts in aliases.items():
        if out.get(canon) not in (None, ""):
            continue
        for a in alts:
            if row.get(a) not in (None, ""):
                out[canon] = row[a]
                break

    wl = (out.get("workload") or "single").strip()
    if wl in ("warm_prefix", "warm"):
        out["workload"] = "warm_prefix"
    elif wl in ("cold_prefix", "cold"):
        out["workload"] = "cold_prefix"
    elif wl in ("hardware_probe", "hw_probe"):
        out["workload"] = "hardware_probe"
    elif wl in ("nccl_allreduce", "nccl"):
        out["workload"] = "nccl_allreduce"
    else:
        out["workload"] = wl or "single"

    out["stack_layer"] = infer_stack_layer(out)
    # Prefer gpu label for hardware rows
    if out["stack_layer"] == "hardware":
        out["model_short"] = (
            out.get("gpu_sku")
            or short_model(out.get("gpu_name") or out.get("model") or "GPU")
        )
    else:
        out["model_short"] = short_model(out.get("model") or "")
    out["live"] = is_live(out)
    tp = out.get("tensor_parallel")
    if tp in (None, ""):
        eng = out.get("engine") or ""
        m = re.search(r"TP(\d+)", str(eng), re.I)
        if m:
            tp = m.group(1)
        else:
            tp = 1
    try:
        out["tensor_parallel"] = int(float(tp))
    except (TypeError, ValueError):
        out["tensor_parallel"] = 1
    out["engine_family"] = engine_family_from_row(out)
    if not out.get("gpu_sku") and out["stack_layer"] == "serving":
        tp_n = out["tensor_parallel"]
        out["gpu_sku"] = f"A100:{tp_n}" if tp_n > 1 else "A100"
    return out


def short_model(m: str) -> str:
    m = m.split("/")[-1]
    return re.sub(r"-Instruct$", "", m, flags=re.I) or "?"


def is_live(row: dict) -> bool:
    notes = (row.get("notes") or "").lower()
    provider = (row.get("provider") or "").lower()
    if any(x in notes for x in ("sample", "demo", "placeholder", "illustrative")):
        return False
    return provider == "modal"


def engine_family_from_row(r: dict) -> str:
    fam = (r.get("engine_family") or "").strip().lower()
    if fam in ("vllm", "sglang"):
        return fam
    eng = (r.get("engine") or "").lower()
    if "sglang" in eng:
        return "sglang"
    if "vllm" in eng:
        return "vllm"
    return "other"


def infer_stack_layer(row: dict) -> str:
    explicit = (row.get("stack_layer") or "").strip()
    if explicit in ("model", "serving", "hardware"):
        return explicit
    layer = (row.get("layer") or "").strip()
    provider = (row.get("provider") or "").strip()
    engine = (row.get("engine") or "").lower()
    wl = (row.get("workload") or "").strip()
    if layer in ("gpu", "nccl") or wl in ("hardware_probe", "nccl_allreduce"):
        return "hardware"
    if "vllm" in engine or "sglang" in engine:
        return "serving"
    if layer in ("inference", "model") or (provider == "modal" and engine_family_from_row(row) == "other"):
        return "model"
    return "serving"


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def fmt_toks(raw) -> str:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return "—"
    if v <= 0 or v > 50_000:
        return "—"
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def fmt_num(raw, digits: int = 3) -> str:
    if raw in (None, ""):
        return "—"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return html.escape(str(raw))
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.{digits}g}"


def bar(value: float, max_v: float, color: str) -> str:
    w = 0.0 if max_v <= 0 else min(100.0, 100.0 * value / max_v)
    return (
        f'<div class="track"><div class="fill" style="--w:{w:.1f}%;--c:{color}"></div></div>'
    )


def coverage_section(rows: list[dict]) -> str:
    items = [
        ("Hardware · Memcpy / GEMM", "Modal GPU bandwidth + TFLOPS",
         any(r["stack_layer"] == "hardware" and r.get("workload") == "hardware_probe" and r["live"] for r in rows)),
        ("Hardware · NCCL all-reduce", "Multi-GPU bus bandwidth",
         any(r["stack_layer"] == "hardware" and r.get("workload") == "nccl_allreduce" and r["live"] for r in rows)),
        ("Model · KV math", "GQA vs dense GiB vs S", True),
        ("Model · Prefill TTFT", "Modal transformers, batch=1",
         any(r["stack_layer"] == "model" and r["live"] for r in rows)),
        ("Model · Decode tok/s", "Isolated generate path",
         any(r["stack_layer"] == "model" and fnum(r, F_TOKS) > 0 for r in rows)),
        ("Serving · vLLM batching", "Modal A100 continuous batch",
         any(engine_family_from_row(r) == "vllm" and r["live"] for r in rows)),
        ("Serving · SGLang batching", "Same matrix as vLLM on Modal",
         any(engine_family_from_row(r) == "sglang" and r["live"] for r in rows)),
        ("Serving · Prefix cache", "cold vs warm_prefix",
         any(r.get("workload") == "warm_prefix" and r["live"] for r in rows)),
        ("Serving · Concurrency", "conc ≥ 4",
         any(fnum(r, "concurrency") >= 4 and r["live"] for r in rows)),
        ("Serving · Tensor parallel", "vLLM TP=2 · A100×2 NCCL",
         any(fnum(r, "tensor_parallel") >= 2 and r["live"] for r in rows)),
    ]
    cards = []
    for title, detail, ok in items:
        cls = "ok" if ok else "todo"
        cards.append(
            f'<article class="cov {cls}"><div class="cov-top">'
            f"<strong>{html.escape(title)}</strong>"
            f'<span class="badge {cls}">{"live" if ok else "todo"}</span></div>'
            f'<p class="detail">{html.escape(detail)}</p></article>'
        )
    return '<div class="cov-grid">' + "".join(cards) + "</div>"


def insights_section(rows: list[dict]) -> str:
    model = [r for r in rows if r["stack_layer"] == "model" and r["live"]]
    serving = [r for r in rows if r["stack_layer"] == "serving" and r["live"]]
    cards: list[str] = []

    hw = [r for r in rows if r["stack_layer"] == "hardware" and r["live"] and r.get("workload") == "hardware_probe"]
    if hw:
        best = max(hw, key=lambda r: fnum(r, "memcpy_gbps"))
        cards.append(_insight(
            "Peak memcpy",
            f"{fnum(best, 'memcpy_gbps'):.0f} GB/s",
            str(best.get("gpu_sku") or best.get("gpu_name") or "GPU"),
        ))
        best_t = max(hw, key=lambda r: fnum(r, "matmul_tflops_bf16"))
        cards.append(_insight(
            "Peak BF16 GEMM",
            f"{fnum(best_t, 'matmul_tflops_bf16'):.0f} TFLOPS",
            str(best_t.get("gpu_sku") or "GPU"),
        ))

    nccl_rows = [
        r
        for r in rows
        if r["stack_layer"] == "hardware"
        and r["live"]
        and r.get("workload") == "nccl_allreduce"
        and fnum(r, "nccl_busbw_gbps") > 0
    ]
    if nccl_rows:
        best_n = max(nccl_rows, key=lambda r: fnum(r, "nccl_busbw_gbps"))
        cards.append(
            _insight(
                "NCCL busbw (Modal)",
                f"{fnum(best_n, 'nccl_busbw_gbps'):.0f} GB/s",
                f"CUDA all-reduce · world {best_n.get('nccl_world_size') or 2}",
            )
        )

    by_s: dict[float, list[float]] = defaultdict(list)
    for r in model:
        s = fnum(r, F_IN)
        if s:
            by_s[s].append(fnum(r, F_TTFT))
    if len(by_s) >= 2:
        s_lo, s_hi = min(by_s), max(by_s)
        t_lo = sum(by_s[s_lo]) / len(by_s[s_lo])
        t_hi = sum(by_s[s_hi]) / len(by_s[s_hi])
        if t_lo > 0:
            cards.append(_insight("Prefill ∝ context", f"{t_hi / t_lo:.1f}×", f"Model TTFT {s_lo:.0f}→{s_hi:.0f} tok"))

    ratios: list[float] = []
    for w in serving:
        if w.get("workload") != "warm_prefix":
            continue
        for c in serving:
            if (
                c.get("workload") == "cold_prefix"
                and c.get("provider") == w.get("provider")
                and c.get(F_IN) == w.get(F_IN)
            ):
                wt, ct = fnum(w, F_TTFT), fnum(c, F_TTFT)
                if wt > 0:
                    ratios.append(ct / wt)
    if ratios:
        cards.append(_insight("Prefix cache win", f"{sum(ratios)/len(ratios):.1f}×", "Cold → warm TTFT (matched)"))

    cards.append(_insight("GQA vs dense KV", "7×", "Qwen2.5-7B · n_kv=4"))
    cards.append(_insight("Live measurements", str(sum(1 for r in rows if r["live"])), f"of {len(rows)} rows"))
    return "".join(cards)


def _insight(title: str, big: str, sub: str) -> str:
    return (
        f'<div class="stat"><span class="stat-k">{html.escape(title)}</span>'
        f'<span class="stat-v">{html.escape(big)}</span>'
        f'<span class="stat-s">{html.escape(sub)}</span></div>'
    )


def bars_section(rows: list[dict], color: str, label_fn) -> str:
    if not rows:
        return '<p class="muted">No rows yet.</p>'
    max_v = max((fnum(r, F_TTFT) for r in rows), default=1.0) or 1.0
    parts = []
    for r in sorted(rows, key=lambda x: (fnum(x, F_IN), fnum(x, "concurrency"), x.get("workload") or "")):
        v = fnum(r, F_TTFT)
        toks = fmt_toks(r.get(F_TOKS))
        kv = fmt_num(r.get(F_KV_GQA))
        cached = r.get(F_CACHED) or "0"
        peak = r.get(F_PEAK)
        peak_s = f" · peak {fmt_num(peak)} GiB" if peak not in (None, "") else ""
        parts.append(
            f'<div class="bar-row"><span class="lbl">{html.escape(label_fn(r))}</span>'
            f"{bar(v, max_v, color)}"
            f'<span class="val">{v:.0f} ms · {html.escape(toks)} tok/s · KV {html.escape(kv)} · '
            f"cache {html.escape(str(cached))}{html.escape(peak_s)}</span></div>"
        )
    return "\n".join(parts)


def hardware_section(rows: list[dict]) -> str:
    hw = [r for r in rows if r["stack_layer"] == "hardware" and r["live"]]
    if not hw:
        return (
            '<p class="muted">No hardware probes yet. Run '
            "<code class=\"mono\">npm run benchmark:modal:hardware</code>.</p>"
        )
    probes = [r for r in hw if r.get("workload") == "hardware_probe"]
    nccl = [r for r in hw if r.get("workload") == "nccl_allreduce"]
    parts: list[str] = ['<div class="hw-grid">']
    for r in sorted(probes, key=lambda x: fnum(x, "memcpy_gbps"), reverse=True):
        parts.append(
            f'<article class="hw-card"><div class="sku">{html.escape(str(r.get("gpu_sku") or "?"))}</div>'
            f'<div class="name">{html.escape(str(r.get("gpu_name") or r["model_short"]))} · '
            f'{html.escape(str(r.get("memory_gib") or "—"))} GiB · CC '
            f'{html.escape(str(r.get("compute_capability") or "—"))}</div>'
            f'<div class="hw-metrics">'
            f'<div><span>Memcpy</span><b>{html.escape(fmt_num(r.get("memcpy_gbps")))} GB/s</b></div>'
            f'<div><span>BF16 GEMM</span><b>{html.escape(fmt_num(r.get("matmul_tflops_bf16")))} TFLOPS</b></div>'
            f'<div><span>FP16 GEMM</span><b>{html.escape(fmt_num(r.get("matmul_tflops_fp16")))} TFLOPS</b></div>'
            f'<div><span>SMs</span><b>{html.escape(str(r.get("sm_count") or "—"))}</b></div>'
            f"</div></article>"
        )
    for r in nccl:
        parts.append(
            f'<article class="hw-card"><div class="sku">{html.escape(str(r.get("gpu_sku") or "NCCL"))}</div>'
            f'<div class="name">NCCL all-reduce · world {html.escape(str(r.get("nccl_world_size") or "—"))}</div>'
            f'<div class="hw-metrics">'
            f'<div><span>Bus BW</span><b>{html.escape(fmt_num(r.get("nccl_busbw_gbps")))} GB/s</b></div>'
            f'<div><span>Alg BW</span><b>{html.escape(fmt_num(r.get("nccl_algbw_gbps")))} GB/s</b></div>'
            f"</div></article>"
        )
    parts.append("</div>")

    if probes:
        max_bw = max((fnum(r, "memcpy_gbps") for r in probes), default=1.0) or 1.0
        parts.append("<h3>Memcpy bandwidth</h3>")
        for r in sorted(probes, key=lambda x: fnum(x, "memcpy_gbps")):
            v = fnum(r, "memcpy_gbps")
            parts.append(
                f'<div class="bar-row"><span class="lbl">{html.escape(str(r.get("gpu_sku") or "?"))}</span>'
                f'{bar(v, max_bw, "var(--sky)")}'
                f'<span class="val">{v:.0f} GB/s</span></div>'
            )
        max_tf = max((fnum(r, "matmul_tflops_bf16") for r in probes), default=1.0) or 1.0
        parts.append("<h3>BF16 GEMM</h3>")
        for r in sorted(probes, key=lambda x: fnum(x, "matmul_tflops_bf16")):
            v = fnum(r, "matmul_tflops_bf16")
            parts.append(
                f'<div class="bar-row"><span class="lbl">{html.escape(str(r.get("gpu_sku") or "?"))}</span>'
                f'{bar(v, max_tf, "var(--teal)")}'
                f'<span class="val">{v:.1f} TFLOPS</span></div>'
            )
    return "\n".join(parts)


def table_section(rows: list[dict]) -> str:
    body = []
    for r in sorted(
        rows,
        key=lambda x: (
            0 if x["stack_layer"] == "model" else 1,
            x.get("provider") or "",
            x["model_short"],
            fnum(x, F_IN),
            fnum(x, "concurrency"),
            x.get("workload") or "",
        ),
    ):
        badge = (
            '<span class="badge ok">live</span>'
            if r["live"]
            else '<span class="badge todo">sample</span>'
        )
        layer = r.get("stack_layer", "")
        provider = r.get("provider") or ""
        body.append(
            f'<tr data-layer="{html.escape(layer)}" data-provider="{html.escape(provider)}" '
            f'data-live="{"1" if r["live"] else "0"}">'
            f"<td>{badge}</td>"
            f"<td>{html.escape(layer)}</td>"
            f"<td>{html.escape(provider)}</td>"
            f'<td class="mono">{html.escape(r["model_short"])}</td>'
            f'<td class="mono">{html.escape(str(r.get("engine") or "")[:28])}</td>'
            f'<td class="mono">{html.escape(str(r.get(F_IN) or ""))}</td>'
            f'<td class="mono">{html.escape(str(r.get(F_OUT) or ""))}</td>'
            f'<td class="mono">{html.escape(str(r.get("concurrency") or "1"))}</td>'
            f"<td>{html.escape(r.get('workload') or '')}</td>"
            f'<td class="mono">{fnum(r, F_TTFT):.0f}</td>'
            f'<td class="mono">{html.escape(fmt_toks(r.get(F_TOKS)))}</td>'
            f'<td class="mono">{html.escape(str(r.get(F_CACHED) or "0"))}</td>'
            f'<td class="mono">{html.escape(fmt_num(r.get(F_KV_GQA)))}</td>'
            f'<td class="mono">{html.escape(fmt_num(r.get(F_KV_DENSE)))}</td>'
            f'<td class="mono">{html.escape(fmt_num(r.get(F_PEAK)))}</td>'
            "</tr>"
        )
    return "\n".join(body)


def pills(counter: Counter) -> str:
    return " ".join(
        f'<span class="pill">{html.escape(str(k))} · {v}</span>'
        for k, v in counter.most_common()
    )


def render_html(rows: list[dict]) -> str:
    model_rows = [r for r in rows if r["stack_layer"] == "model"]
    serving_rows = [r for r in rows if r["stack_layer"] == "serving"]
    hardware_rows = [r for r in rows if r["stack_layer"] == "hardware"]
    live_model = [r for r in model_rows if r["live"]] or model_rows
    live_serving = [r for r in serving_rows if r["live"]] or serving_rows
    inference_rows = [r for r in rows if r["stack_layer"] != "hardware"]
    n_live = sum(1 for r in rows if r["live"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    providers = sorted({(r.get("provider") or "?") for r in rows})

    def model_label(r: dict) -> str:
        return f"{r['model_short'][:18]} · S={r.get(F_IN)}"

    def serving_label(r: dict) -> str:
        return (
            f"{(r.get('provider') or '?')[:8]} · {r['model_short'][:12]} · "
            f"S={r.get(F_IN)} · c={r.get('concurrency') or 1} · {r.get('workload') or 'single'}"
        )

    provider_btns = "".join(
        f'<button type="button" data-filter="provider:{html.escape(p)}">{html.escape(p)}</button>'
        for p in providers
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KV Cache · Benchmark Report</title>
<meta name="description" content="Modal-only inference discovery: CUDA, NCCL, model prefill, vLLM serving with TP=2."/>
<meta property="og:title" content="KV Cache · Benchmark Report"/>
<meta property="og:description" content="{n_live} live runs across hardware, model, and serving layers."/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap" rel="stylesheet"/>
<style>
{STYLE}
.bar-row {{
  display: grid; grid-template-columns: minmax(9rem,15rem) 1fr minmax(10rem,22rem);
  gap: .7rem; align-items: center; margin: .42rem 0;
}}
.lbl {{ font-size: .72rem; color: var(--muted); font-family: var(--mono); }}
.track {{ background: #d9e2ec; height: 9px; border-radius: 999px; overflow: hidden; }}
.fill {{
  height: 100%; width: 0; border-radius: 999px; background: var(--c, var(--teal));
  animation: fillbar .9s cubic-bezier(.2,.8,.2,1) forwards;
}}
.fill {{ --w: 0%; --c: var(--teal); }}
@keyframes fillbar {{ to {{ width: var(--w); }} }}
.val {{ font-size: .72rem; color: #475569; font-family: var(--mono); }}
@media (max-width: 800px) {{ .bar-row {{ grid-template-columns: 1fr; gap: .2rem; }} }}
@media (prefers-reduced-motion: reduce) {{
  .fill {{ width: var(--w) !important; animation: none !important; }}
}}
</style>
</head>
<body>
  <nav class="topnav">
    <div class="topnav-inner">
      <a class="brand-mark" href="/">KV <span>Cache</span></a>
      <div class="nav-links">
        <a href="#coverage">Coverage</a>
        <a href="#insights">Insights</a>
        <a href="#hardware">Hardware</a>
        <a href="#model">Model</a>
        <a href="#serving">Serving</a>
        <a href="#matrix">Matrix</a>
        <a href="/">Explorer</a>
      </div>
    </div>
  </nav>

  <main class="wrap">
    <header class="hero">
      <h1 class="hero-brand">KV <em>Cache</em></h1>
      <p class="hero-sub">
        Three layers on Modal: raw GPU silicon, isolated model math, and serving SLOs —
        CUDA / NCCL hardware floors, model-layer KV &amp; prefill, and vLLM serving
        (TP=1 and TP=2) — all on Modal.
      </p>
      <div class="hero-cta">
        <a class="btn btn-primary" href="#hardware">Hardware probes</a>
        <a class="btn btn-ghost" href="#matrix">Full matrix</a>
      </div>

      <div class="stack-strip">
        <article class="stack-card">
          <div class="step">01 · Hardware</div>
          <h3>GPU microbench</h3>
          <p>Memcpy GB/s, GEMM TFLOPS, optional NCCL busbw — the floor under tok/s.</p>
        </article>
        <article class="stack-card">
          <div class="step">02 · Model</div>
          <h3>KV + prefill</h3>
          <p>Batch=1 transformers for GQA KV GiB and TTFT ≈ prefill.</p>
        </article>
        <article class="stack-card">
          <div class="step">03 · Serving</div>
          <h3>vLLM + API</h3>
          <p>Concurrency, prefix cache, TP=2 NCCL inference on Modal A100.</p>
        </article>
      </div>

      <div class="kpi">
        <div class="stat"><span class="stat-k">Total runs</span><span class="stat-v">{len(rows)}</span></div>
        <div class="stat"><span class="stat-k">Live</span><span class="stat-v">{n_live}</span></div>
        <div class="stat"><span class="stat-k">Hardware</span><span class="stat-v">{len(hardware_rows)}</span></div>
        <div class="stat"><span class="stat-k">Model</span><span class="stat-v">{len(model_rows)}</span></div>
        <div class="stat"><span class="stat-k">Serving</span><span class="stat-v">{len(serving_rows)}</span></div>
        <div class="stat"><span class="stat-k">GQA savings</span><span class="stat-v">7×</span></div>
      </div>
    </header>

    <section id="coverage">
      <h2>Coverage</h2>
      <p class="lead">What the harness can measure — green means live data is on this page.</p>
      {coverage_section(rows)}
    </section>

    <section id="insights">
      <h2>Headline insights</h2>
      <p class="lead">Hardware floors + model/serving story from this matrix.</p>
      <div class="kpi">{insights_section(rows)}</div>
    </section>

    <section id="hardware">
      <h2>Hardware layer</h2>
      <p class="lead">
        CUDA microbench on Modal GPUs (device memcpy + GEMM) and multi-GPU
        <strong>NCCL all-reduce</strong> on A100:2 — the floor under decode tok/s and tensor-parallel cost.
      </p>
      {hardware_section(rows)}
    </section>

    <section id="inventory">
      <h2>Inventory</h2>
      <h3>Providers</h3>
      <div>{pills(Counter((r.get("provider") or "?") for r in rows))}</div>
      <h3>Models / GPUs</h3>
      <div>{pills(Counter(r["model_short"] for r in rows))}</div>
      <h3>Workloads</h3>
      <div>{pills(Counter((r.get("workload") or "?") for r in rows))}</div>
    </section>

    <section id="model">
      <h2>Model layer</h2>
      <p class="lead">Architecture &amp; KV on Modal — batch=1, no HTTP. TTFT ≈ prefill cost.</p>
      {bars_section(live_model, "var(--sky)", model_label)}
    </section>

    <section id="serving">
      <h2>Serving layer</h2>
      <p class="lead">vLLM on Modal A100 (TP=1) and A100×2 (TP=2). Prefix cache + concurrency.</p>
      {bars_section(live_serving, "var(--teal)", serving_label)}
    </section>

    <section id="matrix">
      <h2>Inference results matrix</h2>
      <p class="lead">{len(inference_rows)} rows · hardware probes shown above</p>
      <div class="filters" id="filters" role="group" aria-label="Filter results">
        <button type="button" class="active" data-filter="all">All</button>
        <button type="button" data-filter="layer:model">Model</button>
        <button type="button" data-filter="layer:serving">Serving</button>
        <button type="button" data-filter="live:1">Live only</button>
        {provider_btns}
      </div>
      <div class="table-wrap">
        <table id="results">
          <thead>
            <tr>
              <th>Status</th><th>Stack</th><th>Provider</th><th>Model</th><th>Engine</th>
              <th>In</th><th>Out</th><th>Conc</th><th>Workload</th>
              <th>TTFT p50</th><th>Tok/s</th><th>Cached</th>
              <th>KV GQA</th><th>KV dense</th><th>Peak GPU</th>
            </tr>
          </thead>
          <tbody>
            {table_section(inference_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section id="reproduce">
      <h2>Reproduce</h2>
      <pre class="mono"><span class="c"># Full stack (hardware+NCCL → model → vLLM → API → report)</span>
npm run benchmark:full

<span class="c"># Or piece by piece — CUDA memcpy/GEMM + NCCL on Modal</span>
npm run benchmark:modal:hardware:nccl

<span class="c"># Model layer (transformers)</span>
npm run benchmark:modal

<span class="c"># Serving layer (vLLM + prefix cache)</span>
npm run benchmark:modal:vllm

<span class="c"># Serving — vLLM TP=2 (32B · A100×2 · NCCL)</span>
npm run benchmark:modal:vllm:tp2

<span class="c"># Rebuild this page</span>
npm run report && npm run build</pre>
    </section>

    <footer>
      <span>Generated {generated}</span>
      <span>Hardware → model → serving · Modal only</span>
      <a href="/">← KV explorer</a>
    </footer>
  </main>

<script>
(function () {{
  const buttons = document.querySelectorAll("#filters button");
  const rows = document.querySelectorAll("#results tbody tr");
  function apply(filter) {{
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
    rows.forEach((tr) => {{
      let show = true;
      if (filter === "all") show = true;
      else if (filter.startsWith("layer:")) show = tr.dataset.layer === filter.slice(6);
      else if (filter.startsWith("provider:")) show = tr.dataset.provider === filter.slice(9);
      else if (filter === "live:1") show = tr.dataset.live === "1";
      tr.style.display = show ? "" : "none";
    }});
  }}
  buttons.forEach((b) => b.addEventListener("click", () => apply(b.dataset.filter)));
}})();
</script>
</body>
</html>
"""



def _pick_row(rows: list[dict], **want) -> dict | None:
    """Best-effort match on live rows; prefers exact workload/concurrency."""
    live = [r for r in rows if r.get("live")]
    scored: list[tuple[int, dict]] = []
    for r in live:
        score = 0
        for k, v in want.items():
            rv = r.get(k)
            if rv is None:
                continue
            if k in (F_IN, F_OUT, "concurrency"):
                if abs(fnum(r, k) - float(v)) < 0.5:
                    score += 3
                elif abs(fnum(r, k) - float(v)) <= max(float(v) * 0.25, 512):
                    score += 1
            elif str(rv).lower() == str(v).lower():
                score += 4
        if score:
            scored.append((score, r))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def serving_compare_key(r: dict) -> tuple:
    return (
        r.get("model_short") or short_model(r.get("model") or ""),
        fnum(r, F_IN),
        fnum(r, F_OUT),
        fnum(r, "concurrency"),
        r.get("workload") or "",
        int(r.get("tensor_parallel") or 1),
        r.get("gpu_sku") or "",
    )


def build_concurrency_panels(rows: list[dict]) -> list[dict]:
    """TTFT and tok/s vs concurrency — engines overlaid per model / S / GPU / TP."""
    serving = [
        r
        for r in rows
        if r.get("stack_layer") == "serving"
        and r.get("live")
        and (r.get("provider") or "") == "modal"
    ]
    panels: dict[tuple, dict[str, dict[float, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"ttft": [], "toks": []}))
    )
    for r in serving:
        wl = r.get("workload") or "single"
        if wl != "single":
            continue
        conc = fnum(r, "concurrency")
        if conc <= 0:
            continue
        eng = engine_family_from_row(r)
        if eng not in ("vllm", "sglang"):
            continue
        tp = int(r.get("tensor_parallel") or 1)
        gpu = r.get("gpu_sku") or ("A100:2" if tp >= 2 else "A100")
        panel_key = (
            r.get("model_short") or short_model(r.get("model") or ""),
            fnum(r, F_IN),
            wl,
            gpu,
            tp,
        )
        panels[panel_key][eng][conc]["ttft"].append(fnum(r, F_TTFT))
        panels[panel_key][eng][conc]["toks"].append(fnum(r, F_TOKS))

    out: list[dict] = []
    for panel_key, by_eng in sorted(panels.items()):
        model, inp, wl, gpu, tp = panel_key
        ttft_series: list[dict] = []
        toks_series: list[dict] = []
        for eng in sorted(by_eng):
            by_conc = by_eng[eng]
            pts_ttft: list[dict] = []
            pts_toks: list[dict] = []
            for c in sorted(by_conc):
                ttfts = [t for t in by_conc[c]["ttft"] if t > 0]
                toks = [t for t in by_conc[c]["toks"] if t > 0]
                if not ttfts:
                    continue
                pts_ttft.append({"x": c, "y": round(sum(ttfts) / len(ttfts), 2)})
                if toks:
                    pts_toks.append({"x": c, "y": round(sum(toks) / len(toks), 2)})
            label = "vLLM" if eng == "vllm" else "SGLang"
            if len(pts_ttft) >= 2:
                ttft_series.append({"name": label, "points": pts_ttft})
            if len(pts_toks) >= 2:
                toks_series.append({"name": label, "points": pts_toks})
        if not ttft_series and not toks_series:
            continue
        fabric = "single GPU · no NCCL" if tp <= 1 else f"TP={tp} · {gpu} · NCCL tensor parallel"
        out.append(
            {
                "id": f"{model}-s{int(inp)}-{gpu}-tp{tp}",
                "model": model,
                "input_tokens": inp,
                "workload": wl,
                "gpu_sku": gpu,
                "tensor_parallel": tp,
                "fabric_note": fabric,
                "ttft_series": ttft_series,
                "toks_series": toks_series,
            }
        )
    return out


def build_engine_compare(rows: list[dict]) -> list[dict]:
    """Paired vLLM vs SGLang rows — identical model / shape / GPU / TP."""
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("stack_layer") != "serving" or not r.get("live"):
            continue
        fam = engine_family_from_row(r)
        if fam not in ("vllm", "sglang"):
            continue
        by_key[serving_compare_key(r)][fam] = r

    pairs: list[dict] = []
    for key, engines in sorted(by_key.items()):
        if "vllm" not in engines or "sglang" not in engines:
            continue
        v, s = engines["vllm"], engines["sglang"]
        vt, st = fnum(v, F_TTFT), fnum(s, F_TTFT)
        vo, so = fnum(v, F_TOKS), fnum(s, F_TOKS)
        pairs.append({
            "model": v.get("model_short") or key[0],
            "gpu_sku": v.get("gpu_sku") or "",
            "tensor_parallel": int(v.get("tensor_parallel") or 1),
            "input_tokens": fnum(v, F_IN),
            "output_tokens": fnum(v, F_OUT),
            "concurrency": fnum(v, "concurrency"),
            "workload": v.get("workload") or "single",
            "vllm": {"ttft_ms": vt, "tok_s": vo, "run_id": v.get("run_id")},
            "sglang": {"ttft_ms": st, "tok_s": so, "run_id": s.get("run_id")},
            "ttft_delta_ms": round(vt - st, 2),
            "tok_s_delta": round(so - vo, 2),
            "ttft_winner": "vllm" if vt < st else ("sglang" if st < vt else "tie"),
            "tok_s_winner": "sglang" if so > vo else ("vllm" if vo > so else "tie"),
        })
    return pairs


def build_discovery(rows: list[dict]) -> dict:
    """Customer-discovery narrative — what an SA learns before quoting capacity."""
    model = [r for r in rows if r.get("stack_layer") == "model" and r.get("live")]
    serving = [r for r in rows if r.get("stack_layer") == "serving" and r.get("live")]
    hardware = [r for r in rows if r.get("stack_layer") == "hardware" and r.get("live")]

    def disc(label: str, value: str, detail: str, anchor: str = "") -> dict:
        return {"label": label, "value": value, "detail": detail, "anchor": anchor}

    questions = [
        {
            "id": "input",
            "question": "Typical input length (tokens)?",
            "why": "Drives KV memory and prefill (TTFT). Grows ~linearly with context.",
        },
        {
            "id": "output",
            "question": "Typical output length (tokens)?",
            "why": "Decode time and batch slot occupancy under continuous batching.",
        },
        {
            "id": "cache",
            "question": "How often is the prompt prefix reused?",
            "why": "High cache hit → warm prefix TTFT; low hit → size for cold prefill.",
        },
        {
            "id": "concurrency",
            "question": "Peak concurrent requests?",
            "why": "Sets batch depth, queue delay, and GPU count for serving SLOs.",
        },
        {
            "id": "latency",
            "question": "TTFT / end-to-end latency SLO?",
            "why": "Separates model-layer prefill from HTTP + queue overhead in serving.",
        },
    ]

    scenarios: list[dict] = []

    # 1 — AI-native: repeated system prompt, high cache, moderate concurrency
    warm = _pick_row(
        serving,
        provider="modal",
        workload="warm_prefix",
        **{F_IN: 4096, "concurrency": 4},
    ) or _pick_row(serving, provider="modal", workload="warm_prefix", **{F_IN: 4096})
    cold = _pick_row(
        serving,
        provider="modal",
        workload="cold_prefix",
        **{F_IN: 4096},
    )
    conc8 = _pick_row(
        serving,
        provider="modal",
        workload="single",
        **{F_IN: 4096, "concurrency": 8},
    )
    if warm or cold:
        warm_ttft = fnum(warm, F_TTFT) if warm else 0
        cold_ttft = fnum(cold, F_TTFT) if cold else 0
        cache_win = f"{cold_ttft / warm_ttft:.1f}×" if warm_ttft > 0 and cold_ttft > 0 else "—"
        discoveries = [
            disc(
                "Cold TTFT (first request)",
                f"{cold_ttft:.0f} ms" if cold_ttft else "—",
                "No prefix cached — full prefill cost.",
                "#serving",
            ),
            disc(
                "Warm TTFT (cached prefix)",
                f"{warm_ttft:.0f} ms" if warm_ttft else "—",
                f"~{fnum(warm, F_CACHED):.0f} prompt tokens cached on repeat.",
                "#serving",
            ),
            disc(
                "Prefix cache win",
                cache_win,
                "Same workload shape — cache hit is the biggest TTFT lever.",
                "#serving",
            ),
        ]
        if conc8:
            discoveries.append(
                disc(
                    "Decode @ conc=8",
                    f"{fnum(conc8, F_TOKS):.0f} tok/s",
                    f"TTFT ~{fnum(conc8, F_TTFT):.0f} ms with 8 parallel requests.",
                    "#serving",
                )
            )
        scenarios.append(
            {
                "id": "cached-chat",
                "title": "AI-native app · repeated system prompt",
                "subtitle": "Cursor-style: same instructions every request, many users",
                "profile": {
                    "input_tokens": 4096,
                    "output_tokens": 128,
                    "concurrency": 8,
                    "cache": "high",
                },
                "discoveries": discoveries,
                "recommendation": (
                    "Size for warm-prefix TTFT and peak concurrency, not first-request cold prefill. "
                    "Enable prefix caching in vLLM; expect a material TTFT drop once prompts stabilize."
                ),
            }
        )

    # 2 — Batch docs: cold prefix, fixed shape, latency SLO
    model_4k = _pick_row(model, **{F_IN: 4096}) or _pick_row(model, **{F_IN: 8192})
    kv_row = _pick_row(model, **{F_IN: 4096}) or model_4k
    if model_4k:
        s = fnum(model_4k, F_IN)
        ttft = fnum(model_4k, F_TTFT)
        kv = fnum(kv_row, F_KV_GQA) if kv_row else 0
        discoveries = [
            disc(
                "Model-layer TTFT (prefill)",
                f"{ttft:.0f} ms",
                f"Batch=1, no HTTP — pure prefill at S={s:.0f}.",
                "#model",
            ),
            disc(
                "KV cache (GQA)",
                f"{kv:.2f} GiB" if kv else "—",
                "Memory floor for this context length on one replica.",
                "/",
            ),
            disc(
                "GQA vs dense KV",
                "7× smaller",
                "Qwen2.5-7B with n_kv=4 — planning with dense math over-provisions VRAM.",
                "/",
            ),
        ]
        if cold:
            discoveries.append(
                disc(
                    "Serving TTFT (cold, vLLM)",
                    f"{fnum(cold, F_TTFT):.0f} ms",
                    "Adds engine + batching overhead on top of model prefill.",
                    "#serving",
                )
            )
        scenarios.append(
            {
                "id": "batch-docs",
                "title": "Batch document processing · cold prefix",
                "subtitle": "Fixed PDF shape, 2s latency target, low cache reuse",
                "profile": {
                    "input_tokens": int(s),
                    "output_tokens": 128,
                    "concurrency": 1,
                    "cache": "low",
                },
                "discoveries": discoveries,
                "recommendation": (
                    "Profile model-layer TTFT first (prefill bound). If SLO is tight, "
                    "add GPUs / tensor parallel before tuning batch size. Do not quote from API TTFT alone."
                ),
            }
        )

    # 3 — Capacity / hardware floor
    best_probe = max(
        (r for r in hardware if r.get("workload") == "hardware_probe"),
        key=lambda r: fnum(r, "memcpy_gbps"),
        default=None,
    )
    nccl = next(
        (r for r in hardware if r.get("workload") == "nccl_allreduce" and fnum(r, "nccl_busbw_gbps") > 0),
        None,
    )
    if best_probe or nccl:
        discoveries = []
        if best_probe:
            discoveries.append(
                disc(
                    "Peak device memcpy",
                    f"{fnum(best_probe, 'memcpy_gbps'):.0f} GB/s",
                    f"{best_probe.get('gpu_sku')} — decode/KV read bandwidth ceiling.",
                    "#hardware",
                )
            )
            discoveries.append(
                disc(
                    "BF16 GEMM",
                    f"{fnum(best_probe, 'matmul_tflops_bf16'):.0f} TFLOPS",
                    "Prefill / large matmul throughput on same SKU.",
                    "#hardware",
                )
            )
        if nccl:
            discoveries.append(
                disc(
                    "NCCL all-reduce busbw",
                    f"{fnum(nccl, 'nccl_busbw_gbps'):.0f} GB/s",
                    f"A100×{nccl.get('nccl_world_size') or 2} — TP scaling cost for 70B+.",
                    "#hardware",
                )
            )
        scenarios.append(
            {
                "id": "capacity-plan",
                "title": "Capacity planning · before you quote tokens",
                "subtitle": "Hardware floor for multi-GPU and memory-bound decode",
                "profile": {
                    "input_tokens": 8192,
                    "output_tokens": 256,
                    "concurrency": 4,
                    "cache": "mixed",
                },
                "discoveries": discoveries,
                "recommendation": (
                    "Use hardware memcpy + NCCL to sanity-check TP configs. "
                    "Pair with model-layer KV GiB before committing dedicated GPU contracts."
                ),
            }
        )

    return {
        "purpose": {
            "headline": "Profile inference workloads before quoting GPU capacity",
            "subhead": (
                "The same discovery questions we'd ask before a dedicated inference deal — "
                "input shape, cache behavior, concurrency, and latency — backed by live benches "
                "on hardware, model, and serving layers — all on Modal (CUDA, NCCL, vLLM)."
            ),
            "audience": "Solutions architects · AI-native inference on owned GPU capacity",
        },
        "questions": questions,
        "scenarios": scenarios,
    }


def build_payload(rows: list[dict]) -> dict:
    """Structured JSON consumed by the React report page."""
    hardware = [r for r in rows if r["stack_layer"] == "hardware"]
    model = [r for r in rows if r["stack_layer"] == "model"]
    serving = [r for r in rows if r["stack_layer"] == "serving"]
    live = [r for r in rows if r["live"]]

    def num(r, k):
        return fnum(r, k)

    insights = []
    probes = [r for r in hardware if r["live"] and r.get("workload") == "hardware_probe"]
    if probes:
        best = max(probes, key=lambda r: num(r, "memcpy_gbps"))
        insights.append({
            "label": "Peak memcpy",
            "value": f"{num(best, 'memcpy_gbps'):.0f} GB/s",
            "sub": str(best.get("gpu_sku") or best.get("gpu_name") or "GPU"),
        })
        best_t = max(probes, key=lambda r: num(r, "matmul_tflops_bf16"))
        insights.append({
            "label": "Peak BF16 GEMM",
            "value": f"{num(best_t, 'matmul_tflops_bf16'):.0f} TFLOPS",
            "sub": str(best_t.get("gpu_sku") or "GPU"),
        })
    nccl = [r for r in hardware if r["live"] and r.get("workload") == "nccl_allreduce" and num(r, "nccl_busbw_gbps") > 0]
    if nccl:
        best_n = max(nccl, key=lambda r: num(r, "nccl_busbw_gbps"))
        insights.append({
            "label": "NCCL busbw",
            "value": f"{num(best_n, 'nccl_busbw_gbps'):.0f} GB/s",
            "sub": f"Modal A100×{best_n.get('nccl_world_size') or 2}",
        })

    # model prefill scaling
    by_s: dict[float, list[float]] = defaultdict(list)
    for r in model:
        if not r["live"]:
            continue
        s = num(r, F_IN)
        if s:
            by_s[s].append(num(r, F_TTFT))
    if len(by_s) >= 2:
        s_lo, s_hi = min(by_s), max(by_s)
        t_lo = sum(by_s[s_lo]) / len(by_s[s_lo])
        t_hi = sum(by_s[s_hi]) / len(by_s[s_hi])
        if t_lo > 0:
            insights.append({
                "label": "Prefill ∝ context",
                "value": f"{t_hi / t_lo:.1f}×",
                "sub": f"TTFT {s_lo:.0f}→{s_hi:.0f} tok",
            })

    ratios = []
    live_serving = [r for r in serving if r["live"]]
    for w in live_serving:
        if w.get("workload") != "warm_prefix":
            continue
        for c in live_serving:
            if (
                c.get("workload") == "cold_prefix"
                and c.get("provider") == w.get("provider")
                and c.get(F_IN) == w.get(F_IN)
            ):
                wt = num(w, F_TTFT)
                if wt > 0:
                    ratios.append(num(c, F_TTFT) / wt)
    if ratios:
        insights.append({
            "label": "Prefix cache win",
            "value": f"{sum(ratios)/len(ratios):.1f}×",
            "sub": "Cold → warm TTFT",
        })
    insights.append({"label": "GQA vs dense KV", "value": "7×", "sub": "Qwen2.5-7B · n_kv=4"})
    insights.append({"label": "Runs", "value": str(len(rows)), "sub": "total benchmark rows"})

    # TTFT series by model for chart
    series_map: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in model:
        if not r["live"]:
            continue
        s, t = num(r, F_IN), num(r, F_TTFT)
        if s and t:
            series_map[r["model_short"]][s].append(t)
    model_series = [
        {
            "name": name,
            "points": [{"x": s, "y": sum(vs)/len(vs)} for s, vs in sorted(by.items())],
        }
        for name, by in sorted(series_map.items())
    ]

    # cache compare for modal serving
    cache_by: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in live_serving:
        if (r.get("provider") or "") != "modal":
            continue
        if r.get("workload") not in ("cold_prefix", "warm_prefix"):
            continue
        cache_by[num(r, F_IN)][r["workload"]].append(num(r, F_TTFT))
    cache_compare = []
    for s in sorted(cache_by):
        cold = cache_by[s].get("cold_prefix") or []
        warm = cache_by[s].get("warm_prefix") or []
        if cold and warm:
            cache_compare.append({
                "tokens": s,
                "cold": sum(cold)/len(cold),
                "warm": sum(warm)/len(warm),
            })

    concurrency_panels = build_concurrency_panels(rows)

    coverage = [
        {"id": "hw-memcpy", "title": "Hardware · Memcpy / GEMM", "detail": "Modal CUDA bandwidth + TFLOPS", "live": bool(probes)},
        {"id": "hw-nccl", "title": "Hardware · NCCL", "detail": "Multi-GPU all-reduce busbw", "live": bool(nccl)},
        {"id": "model-kv", "title": "Model · KV math", "detail": "GQA vs dense GiB", "live": True},
        {"id": "model-ttft", "title": "Model · Prefill TTFT", "detail": "transformers batch=1", "live": any(r["live"] for r in model)},
        {"id": "serve-vllm", "title": "Serving · vLLM", "detail": "Modal A100 continuous batch", "live": any(engine_family_from_row(r) == "vllm" and r["live"] for r in serving)},
        {"id": "serve-sglang", "title": "Serving · SGLang", "detail": "Same matrix as vLLM on Modal", "live": any(engine_family_from_row(r) == "sglang" and r["live"] for r in serving)},
        {"id": "serve-compare", "title": "Serving · Engine A/B", "detail": "Paired vLLM vs SGLang rows", "live": bool(build_engine_compare(rows))},
        {"id": "serve-cache", "title": "Serving · Prefix cache", "detail": "cold vs warm_prefix", "live": any(r.get("workload") == "warm_prefix" and r["live"] for r in serving)},
        {"id": "serve-conc", "title": "Serving · Concurrency curves", "detail": "TTFT & tok/s vs conc", "live": bool(concurrency_panels)},
        {"id": "serve-tp", "title": "Serving · Tensor parallel", "detail": "vLLM TP=2 · A100×2 NCCL", "live": any(num(r, "tensor_parallel") >= 2 and r["live"] and engine_family_from_row(r) == "vllm" for r in serving)},
    ]

    def slim(r: dict) -> dict:
        keys = [
            "run_id", "stack_layer", "layer", "provider", "model", "model_short", "engine",
            "engine_family", "live", "workload", "concurrency", F_IN, F_OUT, F_TTFT, F_TOKS, F_CACHED,
            F_KV_GQA, F_KV_DENSE, F_PEAK,
            "gpu_sku", "gpu_name", "memory_gib", "compute_capability", "sm_count",
            "memcpy_gbps", "matmul_tflops_bf16", "matmul_tflops_fp16",
            "nccl_busbw_gbps", "nccl_algbw_gbps", "nccl_world_size", "tensor_parallel",
            "notes", "timestamp",
        ]
        return {k: r.get(k) for k in keys if k in r or r.get(k) is not None}

    live_serving_fams = sorted({
        engine_family_from_row(r)
        for r in serving
        if r["live"] and engine_family_from_row(r) in ("vllm", "sglang")
    })
    gpu_skus = sorted({
        r.get("gpu_sku") or ""
        for r in serving
        if r["live"] and r.get("gpu_sku")
    })

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "brand": "KV Cache",
        "summary": {
            "total": len(rows),
            "live": len(live),
            "hardware": len(hardware),
            "model": len(model),
            "serving": len(serving),
            "providers": sorted({(r.get("provider") or "?") for r in rows}),
            "engines": live_serving_fams,
            "gpu_skus": gpu_skus,
        },
        "filters": {
            "engines": live_serving_fams,
            "gpu_skus": gpu_skus,
        },
        "discovery": build_discovery(rows),
        "insights": insights,
        "coverage": coverage,
        "hardware": [slim(r) for r in hardware],
        "model_series": model_series,
        "cache_compare": cache_compare,
        "concurrency_panels": concurrency_panels,
        "engine_compare": build_engine_compare(rows),
        "inventory": {
            "providers": dict(Counter((r.get("provider") or "?") for r in rows)),
            "models": dict(Counter(r["model_short"] for r in rows)),
            "workloads": dict(Counter((r.get("workload") or "?") for r in rows)),
            "engines": dict(Counter(
                engine_family_from_row(r)
                for r in serving
                if r["live"] and engine_family_from_row(r) in ("vllm", "sglang")
            )),
            "gpu_skus": dict(Counter(
                r.get("gpu_sku") or "?"
                for r in serving
                if r["live"]
            )),
        },
        "rows": [slim(r) for r in rows],
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", type=Path, default=[])
    ap.add_argument("--use-sample", action="store_true", default=True)
    args = ap.parse_args()

    paths = list(args.inputs) or [
        ROOT / "benchmarks" / "results" / "hardware_runs.csv",
        ROOT / "benchmarks" / "results" / "modal_runs.csv",
        ROOT / "benchmarks" / "results" / "modal_serving_runs.csv",
        ROOT / "benchmarks" / "results" / "modal_vllm_runs.csv",
        SAMPLE_MODAL,
    ]

    rows = load_rows(paths)
    rows = [r for r in rows if (r.get("provider") or "").lower() == "modal"]
    seen_ids: set[str] = set()
    seen_serving: set[tuple] = set()
    deduped: list[dict] = []
    for r in rows:
        rid = r.get("run_id") or ""
        if rid and rid in seen_ids:
            continue
        if r.get("stack_layer") == "serving":
            sk = (
                engine_family_from_row(r),
                r.get("model") or "",
                str(r.get(F_IN)),
                str(r.get(F_OUT)),
                str(r.get("concurrency")),
                r.get("workload") or "",
                str(r.get("tensor_parallel") or ""),
                str(r.get("gpu_sku") or ""),
            )
            if sk in seen_serving:
                continue
            seen_serving.add(sk)
        if rid:
            seen_ids.add(rid)
        deduped.append(r)
    rows = deduped
    if not rows and args.use_sample:
        rows = load_rows([SAMPLE_MODAL])
        rows = [r for r in rows if (r.get("provider") or "").lower() == "modal"]

    payload = build_payload(rows)
    doc = render_html(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Static HTML fallback (also used if someone opens the file directly)
    (REPORTS / "latest.html").write_text(doc, encoding="utf-8")
    # React report reads this JSON
    DATA_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Keep a lightweight redirect so old /benchmark-report.html bookmarks hit the React app after build
    # Dev: Vite serves root benchmark-report.html; prod: dist/benchmark-report.html
    print(
        f"Report data → {DATA_JSON} ({payload['summary']['total']} runs, "
        f"{payload['summary']['live']} live, "
        f"{payload['summary']['hardware']} hardware)"
    )


if __name__ == "__main__":
    main()
