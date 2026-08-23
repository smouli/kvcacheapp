#!/usr/bin/env python3
"""Merge benchmark CSVs and regenerate the HTML executive report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "benchmarks" / "sample_results" / "demo_runs.csv"
SAMPLE_MODAL = ROOT / "benchmarks" / "sample_results" / "modal_runs.csv"
PUBLIC = ROOT / "public" / "benchmark-report.html"
REPORTS = ROOT / "benchmarks" / "reports"


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["stack_layer"] = infer_stack_layer(row)
                rows.append(row)
    return rows


def infer_stack_layer(row: dict) -> str:
    explicit = (row.get("stack_layer") or "").strip()
    if explicit in ("model", "serving"):
        return explicit
    layer = (row.get("layer") or "").strip()
    provider = (row.get("provider") or "").strip()
    if layer == "inference" or provider == "modal":
        return "model"
    return "serving"


def bar(value: float, max_v: float, color: str = "#6366f1") -> str:
    w = 0 if max_v <= 0 else min(100, 100 * value / max_v)
    return f'<div class="bar" style="width:{w:.1f}%;background:{color}"></div>'


def render_html(rows: list[dict]) -> str:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_phase[r.get("phase", "?")].append(r)

    warm = [r for r in rows if r.get("workload") == "warm_prefix"]
    cold = [r for r in rows if r.get("workload") == "cold_prefix"]
    speedup = ""
    if warm and cold:
        w = float(warm[0].get("ttft_ms_p50") or 0)
        c = float(cold[0].get("ttft_ms_p50") or 0)
        if w > 0:
            speedup = f"{c / w:.1f}× TTFT reduction (serving, warm vs cold @ 10k)"

    model_rows = [r for r in rows if r.get("stack_layer") == "model"]
    serving_rows = [r for r in rows if r.get("stack_layer") == "serving"]
    max_model_ttft = max(
        (float(r.get("ttft_ms_p50") or 0) for r in model_rows), default=1
    )
    max_serving_ttft = max(
        (float(r.get("ttft_ms_p50") or 0) for r in serving_rows), default=1
    )

    def ttft_bars_for(subset: list[dict], max_ttft: float, color: str) -> str:
        out = ""
        for r in sorted(subset, key=lambda x: float(x.get("input_tokens_target") or 0)):
            inp = r.get("input_tokens_target", "")
            ttft = float(r.get("ttft_ms_p50") or 0)
            kv = r.get("kv_gib_modeled_gqa", "")
            peak = r.get("peak_gpu_gib", "")
            peak_s = f" · peak {peak} GiB" if peak else ""
            out += f"""<div class="row">
          <span class="lbl">S≈{html.escape(str(inp))}</span>
          {bar(ttft, max_ttft, color)}
          <span class="val">{ttft:.0f} ms · KV {html.escape(str(kv))} GiB{html.escape(peak_s)}</span>
        </div>"""
        return out or "<p class=\"lead\">No model-layer runs yet — run <code>npm run benchmark:modal</code>.</p>"

    model_bars = ttft_bars_for(model_rows, max_model_ttft, "#6366f1")
    serving_bars = ttft_bars_for(serving_rows, max_serving_ttft, "#0d9488")

    rows_html = ""
    for r in sorted(
        rows,
        key=lambda x: (x.get("stack_layer", ""), x.get("phase", ""), x.get("input_tokens_target", "")),
    ):
        ttft = float(r.get("ttft_ms_p50") or 0)
        rows_html += f"""<tr>
          <td>{html.escape(r.get('stack_layer',''))}</td>
          <td>{html.escape(r.get('phase',''))}</td>
          <td>{html.escape(r.get('layer',''))}</td>
          <td>{html.escape(r.get('provider',''))}</td>
          <td class="mono">{html.escape(str(r.get('input_tokens_target','')))}</td>
          <td>{html.escape(r.get('workload',''))}</td>
          <td class="mono">{ttft:.0f}</td>
          <td class="mono">{html.escape(str(r.get('output_tok_s_p50','')))}</td>
          <td class="mono">{html.escape(str(r.get('cached_prompt_tokens','')))}</td>
          <td class="mono">{html.escape(str(r.get('kv_gib_modeled_gqa','')))}</td>
        </tr>"""

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_model = len(model_rows)
    n_serving = len(serving_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>KV Cache Inference Benchmark Report</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; color: #0f172a; background: #f8fafc; }}
    body {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
    h1 {{ font-size: 1.75rem; letter-spacing: -0.02em; }}
    .lead {{ color: #475569; line-height: 1.55; max-width: 42rem; }}
    .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.1rem; }}
    .card b {{ display: block; font-size: 1.5rem; color: #4f46e5; }}
    .card span {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
    section {{ margin-top: 2rem; }}
    h2 {{ font-size: 1.1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: #fff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }}
    th, td {{ padding: 0.55rem 0.65rem; text-align: left; border-bottom: 1px solid #f1f5f9; }}
    th {{ background: #f1f5f9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .mono {{ font-family: ui-monospace, monospace; font-size: 0.92em; }}
    .bar-wrap {{ background: #e2e8f0; border-radius: 999px; height: 10px; flex: 1; overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 999px; }}
    .row {{ display: flex; align-items: center; gap: 0.75rem; margin: 0.5rem 0; }}
    .lbl {{ width: 4.5rem; font-size: 0.8rem; color: #64748b; }}
    .val {{ font-size: 0.78rem; color: #475569; white-space: nowrap; }}
    a {{ color: #4f46e5; }}
    footer {{ margin-top: 2.5rem; font-size: 0.78rem; color: #94a3b8; }}
  </style>
</head>
<body>
  <p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.12em;color:#4f46e5;font-weight:600;">KV Cache App · Inference benchmark</p>
  <h1>Executive report: memory model ↔ measured latency</h1>
  <p class="lead">Two benchmark tiers: <strong>model layer</strong> (isolated GPU, KV math, prefill/decode) vs <strong>serving layer</strong> (HTTP, concurrency, prefix cache). Links GQA KV sizing to measured TTFT. {html.escape(speedup)}. See <code>docs/LAYERS.md</code>.</p>

  <div class="kpi">
    <div class="card"><span>Total runs</span><b>{len(rows)}</b></div>
    <div class="card"><span>Model layer</span><b>{n_model}</b></div>
    <div class="card"><span>Serving layer</span><b>{n_serving}</b></div>
    <div class="card"><span>GQA vs dense</span><b>7×</b></div>
  </div>

  <section>
    <h2>Model layer — architecture &amp; KV (batch=1, no API)</h2>
    <p class="lead">Isolated forward/generate: prefill TTFT, decode tok/s, modeled KV GiB, peak GPU memory. <strong>Modal</strong> or single-process transformers/vLLM.</p>
    {model_bars}
  </section>

  <section>
    <h2>Serving layer — what clients see (API / vLLM under load)</h2>
    <p class="lead">HTTP endpoints, concurrency, prefix-cache hits (<code>cached_prompt_tokens</code>), queueing. vLLM+llmperf, Fireworks, DO Dedicated.</p>
    {serving_bars}
  </section>

  <section>
    <h2>Full results matrix</h2>
    <table>
      <thead><tr>
        <th>Stack</th><th>Phase</th><th>Layer</th><th>Provider</th><th>Input</th><th>Workload</th>
        <th>TTFT p50</th><th>Tok/s</th><th>Cached</th><th>KV GiB</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </section>

  <section>
    <h2>How to reproduce</h2>
    <pre class="mono" style="background:#fff;border:1px solid #e2e8f0;padding:1rem;border-radius:8px;overflow:auto;font-size:0.8rem;">
# API (no GPU)
pip install -r benchmarks/requirements.txt
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
python benchmarks/scripts/run_api_benchmark.py

# Engine (DigitalOcean GPU droplet)
bash benchmarks/scripts/run_engine_benchmark.sh

# Regenerate this page
python benchmarks/scripts/generate_report.py
    </pre>
  </section>

  <p><a href="/">← Back to KV explorer</a></p>
  <footer>Generated {generated} · Sample data included for demo; replace with live runs via scripts.</footer>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", type=Path, default=[])
    ap.add_argument("--use-sample", action="store_true", default=True)
    args = ap.parse_args()

    paths = list(args.inputs)
    if not paths:
        paths = [
            ROOT / "benchmarks" / "results" / "modal_runs.csv",
            ROOT / "benchmarks" / "results" / "api_runs.csv",
            ROOT / "benchmarks" / "results" / "engine_runs.csv",
            SAMPLE_MODAL,
            SAMPLE,
        ]

    rows = load_rows(paths)
    # Dedupe by run_id (results/ may duplicate sample_results/)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        rid = r.get("run_id") or ""
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        deduped.append(r)
    rows = deduped
    if not rows and args.use_sample:
        rows = load_rows([SAMPLE_MODAL, SAMPLE])

    doc = render_html(rows)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(doc, encoding="utf-8")
    (REPORTS / "latest.html").write_text(doc, encoding="utf-8")
    (REPORTS / "latest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Report → {PUBLIC} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
