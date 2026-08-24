"""
Shared workload matrix, metrics, and CSV merge for Modal serving benches.

Used by modal_vllm.py and modal_sglang.py so engine A/B runs identical cases.
"""

from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CANDIDATES = [
    Path(__file__).resolve().parent / "config" / "catalog.yaml",
    Path("/pkg/config/catalog.yaml"),
]
SERVING_CSV = Path(__file__).resolve().parent / "results" / "modal_serving_runs.csv"
LEGACY_VLLM_CSV = Path(__file__).resolve().parent / "results" / "modal_vllm_runs.csv"


def catalog_path() -> Path:
    for p in CATALOG_CANDIDATES:
        if p.exists():
            return p
    return CATALOG_CANDIDATES[0]


CATALOG_PATH = catalog_path()

# phase, layer, input, output, concurrency, workload
Case = tuple[str, str, int, int, int, str]

KV_ARCH: dict[str, tuple[int, int, int]] = {
    "0.5B": (24, 2, 896),
    "7B": (28, 4, 3584),
    "32B": (64, 8, 5120),
    "9B": (40, 8, 4096),
    "glm9": (40, 8, 4096),
}

DEFAULT_MODEL_TP1 = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_TP2 = "Qwen/Qwen2.5-32B-Instruct"


def load_catalog() -> dict[str, Any]:
    path = catalog_path()
    if yaml is None or not path.exists():
        return {"models": {}, "workload_profiles": {}}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def catalog_entry(key: str | None = None, model_id: str = "", tp: int = 1) -> dict[str, Any]:
    cat = load_catalog()
    models: dict = cat.get("models") or {}
    if key and key in models:
        ent = dict(models[key])
        ent["_key"] = key
        return ent
    if model_id:
        for k, ent in models.items():
            if ent.get("id") == model_id and int(ent.get("tp") or 1) == tp:
                out = dict(ent)
                out["_key"] = k
                return out
    # defaults from catalog when present
    key = "qwen32b" if tp >= 2 else "qwen7b"
    if key in models:
        ent = dict(models[key])
        ent["_key"] = key
        return ent
    if tp >= 2:
        return {
            "_key": "qwen32b",
            "id": DEFAULT_MODEL_TP2,
            "tp": 2,
            "gpu": "A100:2",
            "matrix": "tp_smoke",
            "trust_remote_code": False,
        }
    return {
        "_key": "qwen7b",
        "id": DEFAULT_MODEL_TP1,
        "tp": 1,
        "gpu": "A100",
        "matrix": "legacy",
        "trust_remote_code": False,
    }


def profile_name(entry: dict, quick: bool, tp: int) -> str:
    if quick:
        return "discovery"
    return str(entry.get("matrix") or ("tp_smoke" if tp >= 2 else "full"))


def build_cases_from_profile(profile: dict) -> list[Case]:
    cases: list[Case] = []
    ins = profile.get("input_tokens") or []
    outs = profile.get("output_tokens") or [128]
    concs = profile.get("concurrency") or [1]
    workloads = profile.get("workloads") or ["single"]
    for inp in ins:
        for out in outs:
            for conc in concs:
                if "single" in workloads:
                    cases.append(("A", "engine", int(inp), int(out), int(conc), "single"))
            if int(inp) >= 4096:
                if "cold_prefix" in workloads:
                    cases.append(("B", "kv_cache", int(inp), int(out), 1, "cold_prefix"))
                if "warm_prefix" in workloads:
                    cases.append(("B", "kv_cache", int(inp), int(out), 1, "warm_prefix"))
                    if int(conc) >= 4 or 4 in concs:
                        cases.append(("B", "kv_cache", int(inp), int(out), 4, "warm_prefix"))
    return cases


def build_cases(*, quick: bool = False, tp: int = 1, profile: str | None = None) -> list[Case]:
    """Build benchmark cases — quick uses a fixed smoke grid; else catalog profile."""
    if quick:
        return [
            ("A", "engine", 1024, 64, 1, "single"),
            ("B", "kv_cache", 4096, 64, 1, "cold_prefix"),
            ("B", "kv_cache", 4096, 64, 1, "warm_prefix"),
            ("A", "engine", 1024, 64, 4, "single"),
        ]

    # Parity grid — matches live vLLM rows (not the expanded catalog "full" cartesian)
    use_legacy = profile == "legacy"
    if not use_legacy:
        cat = load_catalog()
        profiles: dict = cat.get("workload_profiles") or {}
        prof = profiles.get(profile or ("tp_smoke" if tp >= 2 else "full"))
        if prof:
            cases = build_cases_from_profile(prof)
            if cases:
                return cases

    # fallback: legacy hardcoded grids
    if tp >= 2:
        cases = []
        for inp, out in [(1024, 128), (4096, 128), (8192, 128)]:
            for conc in (1, 4, 8):
                cases.append(("A", "engine", inp, out, conc, "single"))
            if inp >= 4096:
                cases.append(("B", "kv_cache", inp, out, 1, "cold_prefix"))
                cases.append(("B", "kv_cache", inp, out, 1, "warm_prefix"))
                cases.append(("B", "kv_cache", inp, out, 4, "warm_prefix"))
        return cases

    cases = []
    for inp, out in [
        (512, 64),
        (1024, 128),
        (2048, 128),
        (4096, 128),
        (8192, 128),
        (10240, 128),
    ]:
        for conc in (1, 4, 8):
            cases.append(("A", "engine", inp, out, conc, "single"))
        if inp >= 4096:
            cases.append(("B", "kv_cache", inp, out, 1, "cold_prefix"))
            cases.append(("B", "kv_cache", inp, out, 1, "warm_prefix"))
            cases.append(("B", "kv_cache", inp, out, 4, "warm_prefix"))
    cases.append(("A", "engine", 10240, 512, 1, "single"))
    cases.append(("A", "engine", 10240, 512, 8, "single"))
    return cases


def make_prompt(n_tokens: int, seed: str = "shared") -> str:
    chunk = f"[{seed}] The quick brown fox jumps over the lazy dog. Context. "
    text = chunk
    while len(text) < n_tokens * 4:
        text += chunk
    return text[: n_tokens * 4]


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def kv_dims(model_id: str) -> tuple[int, int, int]:
    mid = model_id.lower()
    if "32b" in mid:
        return KV_ARCH["32B"]
    if "0.5b" in mid:
        return KV_ARCH["0.5B"]
    if "9b" in mid or "glm-4-9" in mid:
        return KV_ARCH["9B"]
    return KV_ARCH["7B"]


def kv_gib(model_id: str, seq: int) -> tuple[float, float]:
    layers, n_kv, hidden = kv_dims(model_id)
    gqa = 2 * layers * seq * n_kv * 128 * 2 / (1024**3)
    dense = 2 * layers * seq * hidden * 2 / (1024**3)
    return round(gqa, 4), round(dense, 4)


def gpu_sku_for(tp: int, entry_gpu: str | None = None) -> str:
    if entry_gpu:
        return entry_gpu
    return f"A100:{tp}" if tp > 1 else "A100"


def prompts_for_case(
    workload: str, inp: int, conc: int, *, prime_fn=None
) -> tuple[list[str], int]:
    """Build prompts; optional prime_fn(prompt) warms prefix cache before batch."""
    params_out = max(conc, 3) if workload != "single" else max(conc, 1)
    cached_est = 0

    if workload == "cold_prefix":
        prompts = [make_prompt(inp, seed=f"cold-{i}") for i in range(max(conc, 3))]
    elif workload == "warm_prefix":
        shared = make_prompt(max(inp - 64, 256), seed="warm-shared")
        prime = shared + "\nUser: priming"
        if prime_fn:
            prime_fn(prime)
        prompts = [
            shared + f"\nUser: follow-up question number {i}."
            for i in range(max(conc, 3))
        ]
        cached_est = max(inp - 32, 0)
    else:
        prompts = [make_prompt(inp, seed=f"single-{i}") for i in range(max(conc, 1))]

    batch = prompts[:params_out]
    return batch, cached_est


def approx_metrics(
    *,
    wall_s: float,
    n_outputs: int,
    prompt_tokens: int,
    completion_tokens: int,
    inp_fallback: int,
) -> dict[str, float]:
    n = max(n_outputs, 1)
    pt = prompt_tokens or inp_fallback
    ct = max(completion_tokens, 1)
    total = max(pt + ct, 1)
    ttft = (wall_s * 1000) * (pt / total) / n
    tok_s = ct / max(wall_s / n, 1e-6)
    return {
        "ttft_ms_p50": ttft,
        "ttft_ms_p95": ttft * 1.08,
        "output_tok_s_p50": tok_s,
        "tpot_ms_p50": 1000.0 / max(tok_s, 1e-6),
        "prompt_tokens": float(pt),
        "completion_tokens": float(ct),
    }


def make_serving_row(
    *,
    engine_family: str,
    engine_label: str,
    model_id: str,
    case: Case,
    metrics: dict[str, Any],
    tp: int,
    gpu_sku: str,
    cached: int,
    batch_n: int,
    wall_s: float,
    extra_notes: str = "",
) -> dict[str, Any]:
    phase, layer, inp, out, conc, workload = case
    seq = int(metrics.get("prompt_tokens") or inp)
    kv_gqa, kv_dense = kv_gib(model_id, seq)
    prefix = "modal-vllm" if engine_family == "vllm" else "modal-sglang"
    notes = (
        f"Modal {engine_family} serving; TP={tp}; gpu={gpu_sku}; "
        f"batch={batch_n}; wall_s={wall_s:.2f}"
    )
    if extra_notes:
        notes += f"; {extra_notes}"

    return {
        "run_id": f"{prefix}-{uuid.uuid4().hex[:8]}",
        "phase": phase,
        "stack_layer": "serving",
        "layer": layer,
        "provider": "modal",
        "model": model_id,
        "engine": engine_label,
        "engine_family": engine_family,
        "input_tokens_target": inp,
        "output_tokens_target": out,
        "concurrency": conc,
        "workload": workload,
        "cache_mode": "on" if workload == "warm_prefix" else "off",
        "session_affinity": "local" if workload == "warm_prefix" else "",
        "ttft_ms_p50": round(float(metrics["ttft_ms_p50"]), 2),
        "ttft_ms_p95": round(float(metrics.get("ttft_ms_p95", metrics["ttft_ms_p50"])), 2),
        "tpot_ms_p50": round(float(metrics.get("tpot_ms_p50", 0)), 2),
        "output_tok_s_p50": round(float(metrics["output_tok_s_p50"]), 2),
        "prompt_tokens": int(metrics.get("prompt_tokens") or inp),
        "completion_tokens": int(metrics.get("completion_tokens") or out),
        "cached_prompt_tokens": cached,
        "kv_gib_modeled_gqa": kv_gqa,
        "kv_gib_modeled_dense": kv_dense,
        "tensor_parallel": tp,
        "gpu_sku": gpu_sku,
        "nccl_world_size": tp if tp > 1 else "",
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def row_merge_key(r: dict) -> tuple:
    return (
        r.get("engine_family") or engine_family_from_row(r),
        r.get("model", ""),
        str(r.get("input_tokens_target", "")),
        str(r.get("output_tokens_target", "")),
        str(r.get("concurrency", "")),
        r.get("workload", ""),
        str(r.get("tensor_parallel") or ""),
        str(r.get("gpu_sku") or ""),
    )


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


def merge_serving_csv(rows: list[dict], out_csv: Path | None = None) -> Path:
    out = out_csv or SERVING_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    # migrate legacy vllm-only file once
    if LEGACY_VLLM_CSV.exists() and LEGACY_VLLM_CSV != out:
        with LEGACY_VLLM_CSV.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("engine_family"):
                    r["engine_family"] = "vllm"
                existing.append(r)

    by = {row_merge_key(r): r for r in existing}
    for r in rows:
        if not r.get("engine_family"):
            r["engine_family"] = engine_family_from_row(r)
        by[row_merge_key(r)] = r
    merged = list(by.values())
    fields: list[str] = []
    for r in merged:
        for k in r:
            if k not in fields:
                fields.append(k)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    print(f"Wrote {len(rows)} new / {len(merged)} total → {out}")
    return out


def serving_csv_paths() -> list[Path]:
    paths = [SERVING_CSV, LEGACY_VLLM_CSV]
    return [p for p in paths if p.exists()]
