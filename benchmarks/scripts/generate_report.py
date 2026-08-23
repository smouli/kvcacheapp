#!/usr/bin/env python3
"""Merge benchmark CSVs into a polished HTML coverage + results report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
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
                rows.append(normalize_row(row))
    return rows


def normalize_row(row: dict) -> dict:
    out = dict(row)
    # Common alternate names → canonical
    remap = {
        "input_tokens_target": ("input_tokens_target",),
        "output_tokens_target": ("output_tokens_target",),
        "ttft_ms_p50": ("ttft_ms_p50",),
        "output_tok_s_p50": ("output_tok_s_p50",),
        "cached_prompt_tokens": ("cached_prompt_tokens",),
        "kv_gib_modeled_gqa": ("kv_gib_modeled_gqa",),
        "kv_gib_modeled_dense": ("kv_gib_modeled_dense",),
        "peak_gpu_gib": ("peak_gpu_gib",),
        "run_id": ("run_id",),
    }
    for canon, alts in remap.items():
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
    else:
        out["workload"] = wl or "single"

    out["stack_layer"] = infer_stack_layer(out)
    out["model_short"] = short_model(out.get("model") or "")
    out["live"] = is_live(out)
    return out


def short_model(m: str) -> str:
    m = m.split("/")[-1]
    return re.sub(r"-Instruct$", "", m, flags=re.I) or "?"


def is_live(row: dict) -> bool:
    notes = (row.get("notes") or "").lower()
    provider = (row.get("provider") or "").lower()
    if any(x in notes for x in ("sample", "demo", "placeholder", "illustrative")):
        return False
    if provider == "digitalocean_gpu":
        return "live" in notes or "llmperf" in notes
    if provider in ("modal", "fireworks"):
        return True
    return "live" in notes


def infer_stack_layer(row: dict) -> str:
    explicit = (row.get("stack_layer") or "").strip()
    if explicit in ("model", "serving"):
        return explicit
    layer = (row.get("layer") or "").strip()
    provider = (row.get("provider") or "").strip()
    engine = (row.get("engine") or "").lower()
    if "vllm" in engine:
        return "serving"
    if layer == "inference" or (provider == "modal" and "vllm" not in engine):
        return "model"
    return "serving"


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def bar(value: float, max_v: float, color: str) -> str:
    w = 0.0 if max_v <= 0 else min(100.0, 100.0 * value / max_v)
    return (
        f'<div class="track"><div class="fill" style="width:{w:.1f}%;'
        f'background:{color}"></div></div>'
    )


def coverage_section(rows: list[dict]) -> str:
    items = [
        ("Model · KV math", "GQA vs dense GiB vs S", True),
        (
            "Model · Prefill TTFT",
            "Modal transformers, batch=1",
            any(r["stack_layer"] == "model" and r["live"] for r in rows),
        ),
        (
            "Model · Decode tok/s",
            "Isolated generate path",
            any(
                r["stack_layer"] == "model" and fnum(r, "output_tok_s_p50") > 0
                for r in rows
            ),
        ),
        (
            "Serving · vLLM batching",
            "Modal A100 continuous batch",
            any("vllm" in (r.get("engine") or "").lower() and r["live"] for r in rows),
        ),
        (
            "Serving · Prefix cache",
            "cold vs warm_prefix",
            any(r.get("workload") == "warm_prefix" and r["live"] for r in rows),
        ),
        (
            "Serving · Concurrency",
            "conc ≥ 4",
            any(fnum(r, "concurrency") >= 4 and r["live"] for r in rows),
        ),
        (
            "Serving · Hosted API",
            "Fireworks OpenAI-compatible",
            any((r.get("provider") or "") == "fireworks" and r["live"] for r in rows),
        ),
        (
            "Serving · DO GPU droplet",
            "vLLM on DigitalOcean (optional)",
            any(
                (r.get("provider") or "") == "digitalocean_gpu" and r["live"]
                for r in rows
            ),
        ),
    ]
    cards = []
    for title, detail, ok in items:
        cls = "ok" if ok else "todo"
        cards.append(
            f'<div class="cov {cls}"><div class="cov-top"><b>{html.escape(title)}</b>'
            f'<span class="badge {cls}">{"live" if ok else "todo"}</span></div>'
            f'<div class="detail">{html.escape(detail)}</div></div>'
        )
    return '<div class="cov-grid">' + "".join(cards) + "</div>"


def insights_section(rows: list[dict]) -> str:
    model = [r for r in rows if r["stack_layer"] == "model" and r["live"]]
    serving = [r for r in rows if r["stack_layer"] == "serving" and r["live"]]
    cards: list[str] = []

    by_s: dict[float, list[float]] = defaultdict(list)
    for r in model:
        s = fnum(r, "input_tokens_target")
        if s:
            by_s[s].append(fnum(r, "ttft_ms_p50"))
    if len(by_s) >= 2:
        s_lo, s_hi = min(by_s), max(by_s)
        t_lo = sum(by_s[s_lo]) / len(by_s[s_lo])
        t_hi = sum(by_s[s_hi]) / len(by_s[s_hi])
        if t_lo > 0:
            cards.append(
                _insight(
                    "Prefill ∝ context",
                    f"{t_hi / t_lo:.1f}×",
                    f"Model TTFT {s_lo:.0f}→{s_hi:.0f} tok",
                )
            )

    ratios: list[float] = []
    for w in serving:
        if w.get("workload") != "warm_prefix":
            continue
        for c in serving:
            if (
                c.get("workload") == "cold_prefix"
                and c.get("provider") == w.get("provider")
                and c.get("input_tokens_target") == w.get("input_tokens_target")
            ):
                wt, ct = fnum(w, "ttft_ms_p50"), fnum(c, "ttft_ms_p50")
                if wt > 0:
                    ratios.append(ct / wt)
    if ratios:
        cards.append(
            _insight(
                "Prefix cache win",
                f"{sum(ratios) / len(ratios):.1f}×",
                "Cold → warm TTFT (matched)",
            )
        )

    cards.append(_insight("GQA vs dense KV", "7×", "Qwen2.5-7B n_kv=4"))
    cards.append(
        _insight(
            "Live measurements",
            str(sum(1 for r in rows if r["live"])),
            f"of {len(rows)} rows",
        )
    )
    return "".join(cards)


def _insight(title: str, big: str, sub: str) -> str:
    return (
        f'<div class="card insight"><span>{html.escape(title)}</span>'
        f"<b>{html.escape(big)}</b><small>{html.escape(sub)}</small></div>"
    )


def bars_section(rows: list[dict], color: str, label_fn) -> str:
    if not rows:
        return '<p class="muted">No rows yet.</p>'
    max_v = max((fnum(r, "ttft_ms_p50") for r in rows), default=1.0) or 1.0
    parts = []
    for r in sorted(
        rows,
        key=lambda x: (
            fnum(x, "input_tokens_target"),
            fnum(x, "concurrency"),
            x.get("workload") or "",
        ),
    ):
        v = fnum(r, "ttft_ms_p50")
        toks = r.get("output_tok_s_p50") or "—"
        kv = r.get("kv_gib_modeled_gqa") or "—"
        cached = r.get("cached_prompt_tokens") or "0"
        peak = r.get("peak_gpu_gib")
        peak_s = f" · peak {peak} GiB" if peak not in (None, "") else ""
        parts.append(
            f'<div class="row"><span class="lbl">{html.escape(label_fn(r))}</span>'
            f"{bar(v, max_v, color)}"
            f'<span class="val">{v:.0f} ms · {html.escape(str(toks))} tok/s · '
            f"KV {html.escape(str(kv))} · cache {html.escape(str(cached))}"
            f"{html.escape(peak_s)}</span></div>"
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
            fnum(x, "input_tokens_target"),
            fnum(x, "concurrency"),
            x.get("workload") or "",
        ),
    ):
        badge = (
            '<span class="badge ok">live</span>'
            if r["live"]
            else '<span class="badge todo">sample</span>'
        )
        body.append(
            "<tr>"
            f"<td>{badge}</td>"
            f"<td>{html.escape(r.get('stack_layer',''))}</td>"
            f"<td>{html.escape(r.get('provider') or '')}</td>"
            f"<td class=\"mono\">{html.escape(r['model_short'])}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('engine') or '')[:28])}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('input_tokens_target') or ''))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('output_tokens_target') or ''))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('concurrency') or '1'))}</td>"
            f"<td>{html.escape(r.get('workload') or '')}</td>"
            f"<td class=\"mono\">{fnum(r,'ttft_ms_p50'):.0f}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('output_tok_s_p50') or '—'))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('cached_prompt_tokens') or '0'))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('kv_gib_modeled_gqa') or '—'))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('kv_gib_modeled_dense') or '—'))}</td>"
            f"<td class=\"mono\">{html.escape(str(r.get('peak_gpu_gib') or '—'))}</td>"
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
    live_model = [r for r in model_rows if r["live"]] or model_rows
    live_serving = [r for r in serving_rows if r["live"]] or serving_rows
    n_live = sum(1 for r in rows if r["live"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def model_label(r: dict) -> str:
        return f"{r['model_short'][:16]} · S={r.get('input_tokens_target')}"

    def serving_label(r: dict) -> str:
        return (
            f"{(r.get('provider') or '?')[:7]} · {r['model_short'][:10]} · "
            f"S={r.get('input_tokens_target')} · c={r.get('concurrency') or 1} · "
            f"{r.get('workload') or 'single'}"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KV Cache · Benchmark Report</title>
<style>
:root {{
  --bg:#f4f1ea; --ink:#1c1917; --muted:#78716c; --card:#fffcf7;
  --line:#e7e0d5; --serve:#0f766e; --accent:#b45309;
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  color:var(--ink); background:var(--bg);
}}
* {{ box-sizing:border-box; }}
body {{
  max-width:1120px; margin:0 auto; padding:2.5rem 1.25rem 4rem;
  background:
    radial-gradient(1200px 500px at 10% -10%, #fde68a55, transparent 60%),
    radial-gradient(900px 400px at 100% 0%, #a7f3d055, transparent 55%),
    var(--bg);
}}
.eyebrow {{
  font-family:ui-sans-serif,system-ui,sans-serif; font-size:.72rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
  font-weight:700; margin:0 0 .6rem;
}}
h1 {{
  font-size:clamp(2rem,4vw,2.75rem); line-height:1.1; margin:0 0 .75rem;
  letter-spacing:-.03em; max-width:18ch;
}}
.lead {{ color:var(--muted); font-size:1.05rem; line-height:1.55; max-width:46rem; }}
.muted {{ color:var(--muted); }}
.sans {{ font-family:ui-sans-serif,system-ui,sans-serif; }}
.kpi {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:.85rem; margin:1.75rem 0;
}}
.card {{
  background:var(--card); border:1px solid var(--line); border-radius:4px;
  padding:1rem 1.05rem;
}}
.card span {{
  display:block; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
}}
.card b {{ display:block; font-size:1.65rem; margin-top:.35rem; letter-spacing:-.02em; }}
.card.insight small {{ display:block; margin-top:.35rem; color:var(--muted); font-size:.85rem; }}
.card.insight b {{ color:var(--accent); }}
section {{ margin-top:2.75rem; }}
h2 {{ font-size:1.45rem; margin:0 0 .4rem; letter-spacing:-.02em; }}
h3 {{ font-size:.95rem; margin:1.2rem 0 .45rem; font-family:ui-sans-serif,system-ui,sans-serif; }}
.pill {{
  display:inline-block; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:.75rem; padding:.25rem .55rem; margin:.15rem .2rem .15rem 0;
  border:1px solid var(--line); border-radius:999px; background:#fff;
}}
.cov-grid {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:.75rem; margin-top:1rem;
}}
.cov {{
  border:1px solid var(--line); border-radius:4px; padding:.85rem .9rem;
  background:var(--card); font-family:ui-sans-serif,system-ui,sans-serif;
}}
.cov.ok {{ border-color:#99f6e4; background:#f0fdfa; }}
.cov.todo {{ opacity:.72; }}
.cov-top {{ display:flex; justify-content:space-between; gap:.5rem; align-items:start; }}
.detail {{ font-size:.82rem; color:var(--muted); margin-top:.35rem; }}
.badge {{
  font-size:.65rem; letter-spacing:.08em; text-transform:uppercase;
  padding:.15rem .4rem; border-radius:999px; font-weight:700;
  font-family:ui-sans-serif,system-ui,sans-serif;
}}
.badge.ok {{ background:#ccfbf1; color:#0f766e; }}
.badge.todo {{ background:#f5f5f4; color:#78716c; }}
.row {{
  display:grid; grid-template-columns:minmax(10rem,16rem) 1fr minmax(14rem,22rem);
  gap:.75rem; align-items:center; margin:.45rem 0;
  font-family:ui-sans-serif,system-ui,sans-serif;
}}
.lbl {{ font-size:.72rem; color:var(--muted); }}
.track {{ background:#e7e5e4; height:10px; border-radius:999px; overflow:hidden; }}
.fill {{ height:100%; border-radius:999px; }}
.val {{ font-size:.72rem; color:#57534e; }}
.table-wrap {{
  overflow-x:auto; border:1px solid var(--line); border-radius:4px; background:var(--card);
}}
table {{
  width:100%; border-collapse:collapse; font-size:.78rem;
  font-family:ui-sans-serif,system-ui,sans-serif;
}}
th, td {{
  padding:.5rem .55rem; text-align:left; border-bottom:1px solid #f5f5f4; white-space:nowrap;
}}
th {{
  background:#fafaf9; font-size:.65rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted);
}}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.92em; }}
pre.mono {{
  background:var(--card); border:1px solid var(--line); padding:1rem;
  border-radius:4px; overflow:auto; font-size:.78rem; line-height:1.45;
}}
a {{ color:var(--serve); }}
footer {{
  margin-top:3rem; font-size:.78rem; color:#a8a29e;
  font-family:ui-sans-serif,system-ui,sans-serif;
}}
@media (max-width:800px) {{ .row {{ grid-template-columns:1fr; gap:.25rem; }} }}
</style>
</head>
<body>
  <p class="eyebrow">KV Cache App · inference benchmark suite</p>
  <h1>What we measure, and what the numbers say</h1>
  <p class="lead">
    Two tiers: <strong>model layer</strong> (isolated GPU — KV math, prefill, decode)
    and <strong>serving layer</strong> (vLLM / API — concurrency, prefix cache, client TTFT).
    Live runs on Modal + Fireworks; DO GPU droplet remains optional.
  </p>

  <div class="kpi">
    <div class="card"><span>Total runs</span><b>{len(rows)}</b></div>
    <div class="card"><span>Live runs</span><b>{n_live}</b></div>
    <div class="card"><span>Model layer</span><b>{len(model_rows)}</b></div>
    <div class="card"><span>Serving layer</span><b>{len(serving_rows)}</b></div>
    <div class="card"><span>Providers</span><b>{len(Counter((r.get('provider') or '?') for r in rows))}</b></div>
    <div class="card"><span>GQA savings</span><b>7×</b></div>
  </div>

  <section>
    <h2>Coverage</h2>
    <p class="lead muted">Green = live data on hand. Grey = harness ready, not run yet here.</p>
    {coverage_section(rows)}
  </section>

  <section>
    <h2>Headline insights</h2>
    <div class="kpi">{insights_section(rows)}</div>
  </section>

  <section>
    <h2>Inventory</h2>
    <h3>Providers</h3>
    <div>{pills(Counter((r.get('provider') or '?') for r in rows))}</div>
    <h3>Models</h3>
    <div>{pills(Counter(r['model_short'] for r in rows))}</div>
    <h3>Workloads</h3>
    <div>{pills(Counter((r.get('workload') or '?') for r in rows))}</div>
  </section>

  <section>
    <h2>Model layer — architecture &amp; KV</h2>
    <p class="lead muted">Batch=1, no HTTP. TTFT ≈ prefill. Links latency to modeled GQA KV GiB.</p>
    {bars_section(live_model, "#4f46e5", model_label)}
  </section>

  <section>
    <h2>Serving layer — clients &amp; continuous batching</h2>
    <p class="lead muted">vLLM on Modal, Fireworks API, optional DO GPU. Watch warm_prefix vs cold_prefix and concurrency.</p>
    {bars_section(live_serving, "#0f766e", serving_label)}
  </section>

  <section>
    <h2>Full results matrix</h2>
    <p class="lead muted">{len(rows)} rows · stack → provider → model → S → concurrency</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Status</th><th>Stack</th><th>Provider</th><th>Model</th><th>Engine</th>
            <th>In</th><th>Out</th><th>Conc</th><th>Workload</th>
            <th>TTFT p50</th><th>Tok/s</th><th>Cached</th>
            <th>KV GQA</th><th>KV dense</th><th>Peak GPU</th>
          </tr>
        </thead>
        <tbody>
          {table_section(rows)}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>How to grow the matrix</h2>
    <pre class="mono"># Model layer (isolated GPU)
npm run benchmark:modal

# Serving layer on Modal (vLLM + prefix cache + concurrency)
npm run benchmark:modal:vllm

# Hosted API (Fireworks)
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
npm run benchmark:api

# Regenerate this page
npm run report && npm run build</pre>
  </section>

  <p class="sans"><a href="/">← KV explorer</a></p>
  <footer>Generated {generated} · Prefer live Modal/Fireworks rows over sample DO placeholders.</footer>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", type=Path, default=[])
    ap.add_argument("--use-sample", action="store_true", default=True)
    args = ap.parse_args()

    paths = list(args.inputs) or [
        ROOT / "benchmarks" / "results" / "modal_runs.csv",
        ROOT / "benchmarks" / "results" / "modal_vllm_runs.csv",
        ROOT / "benchmarks" / "results" / "api_runs.csv",
        ROOT / "benchmarks" / "results" / "engine_runs.csv",
        SAMPLE_MODAL,
        SAMPLE,
    ]

    rows = load_rows(paths)
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
    print(
        f"Report → {PUBLIC} ({len(rows)} runs, "
        f"{sum(1 for r in rows if r.get('live'))} live)"
    )


if __name__ == "__main__":
    main()
