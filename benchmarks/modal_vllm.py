"""
Serving-layer vLLM benchmarks on Modal (stand-in for a DO GPU droplet).

Uses vLLM's offline LLM engine with prefix caching + batched prompts
(continuous batching) — same serving mechanics without the fragile OpenAI
server subprocess on Modal.

  modal run benchmarks/modal_vllm.py           # 7B A100, full matrix
  modal run benchmarks/modal_vllm.py --quick   # smoke test

CSV tags: stack_layer=serving, provider=modal, engine=vLLM+Modal
"""

from __future__ import annotations

import csv
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "kvcache-vllm-serving"
RESULTS_VOL = "kvcache-bench-results"
HF_VOL = "kvcache-hf-cache"
MOUNT = "/bench-results"
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"


def kv_gib_gqa(seq: int) -> float:
    return 2 * 28 * seq * 4 * 128 * 2 / (1024**3)


def kv_gib_dense(seq: int) -> float:
    return 2 * 28 * seq * 3584 * 2 / (1024**3)


# Modal cookbook pattern: debian_slim + vLLM wheel (CUDA provided by Modal).
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_VOL, create_if_missing=True)


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


@app.function(
    gpu="A100",
    image=vllm_image,
    timeout=3600,
    volumes={MOUNT: volume, "/root/.cache/huggingface": hf_cache},
)
def serve_and_bench(model_id: str, quick: bool, tp: int = 1) -> list[dict]:
    from vllm import LLM, SamplingParams

    print(f"Loading {model_id} with vLLM (prefix cache on, eager)…", flush=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=8192 if quick else 16384,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    if quick:
        cases = [
            ("A", "engine", 1024, 64, 1, "single"),
            ("B", "kv_cache", 4096, 64, 1, "cold_prefix"),
            ("B", "kv_cache", 4096, 64, 1, "warm_prefix"),
            ("A", "engine", 1024, 64, 4, "single"),
        ]
    else:
        cases = [
            ("A", "engine", 1024, 128, 1, "single"),
            ("A", "engine", 8192, 128, 1, "single"),
            ("A", "engine", 10240, 128, 1, "single"),
            ("A", "engine", 10240, 128, 8, "single"),
            ("B", "kv_cache", 10240, 128, 1, "cold_prefix"),
            ("B", "kv_cache", 10240, 128, 1, "warm_prefix"),
        ]

    rows: list[dict] = []
    for phase, layer, inp, out, conc, workload in cases:
        print(f"→ {workload} in={inp} out={out} conc={conc}", flush=True)
        params = SamplingParams(temperature=0, max_tokens=out)

        if workload == "cold_prefix":
            prompts = [make_prompt(inp, seed=f"cold-{i}") for i in range(max(conc, 3))]
        elif workload == "warm_prefix":
            shared = make_prompt(max(inp - 64, 256), seed="warm-shared")
            # Prime prefix cache
            llm.generate([shared + "\nUser: priming"], params)
            prompts = [
                shared + f"\nUser: follow-up question number {i}."
                for i in range(max(conc, 3))
            ]
        else:
            n = max(conc, 1)
            prompts = [make_prompt(inp, seed=f"single-{i}") for i in range(n)]

        # Batch = continuous batching / concurrency proxy
        t0 = time.perf_counter()
        outputs = llm.generate(prompts[: max(conc, 3)], params)
        wall_s = time.perf_counter() - t0

        ttfts: list[float] = []
        toks: list[float] = []
        prompt_tokens = 0
        completion_tokens = 0
        for out_i in outputs:
            pt = len(out_i.prompt_token_ids or [])
            ct = len(out_i.outputs[0].token_ids) if out_i.outputs else 0
            prompt_tokens = pt
            completion_tokens = ct
            # Approximate per-request TTFT as share of wall for batch
            # (single-request cases: wall ≈ e2e; first-token proxy ≈ wall * pt/(pt+ct))
            total = max(pt + ct, 1)
            ttft_approx = (wall_s * 1000) * (pt / total) / max(len(outputs), 1)
            ttfts.append(ttft_approx)
            toks.append(ct / max(wall_s / max(len(outputs), 1), 1e-6))

        seq = prompt_tokens or inp
        # For warm_prefix, estimate cached tokens ≈ shared prefix length
        cached = 0
        if workload == "warm_prefix":
            cached = max(seq - 32, 0)

        row = {
            "run_id": f"modal-vllm-{uuid.uuid4().hex[:8]}",
            "phase": phase,
            "stack_layer": "serving",
            "layer": layer,
            "provider": "modal",
            "model": model_id,
            "engine": f"vLLM+Modal+TP{tp}",
            "input_tokens_target": inp,
            "output_tokens_target": out,
            "concurrency": conc,
            "workload": workload,
            "cache_mode": "on" if workload == "warm_prefix" else "off",
            "session_affinity": "local" if workload == "warm_prefix" else "",
            "ttft_ms_p50": round(percentile(ttfts, 50), 2),
            "ttft_ms_p95": round(percentile(ttfts, 95), 2),
            "tpot_ms_p50": round(1000.0 / max(percentile(toks, 50), 1e-6), 2),
            "output_tok_s_p50": round(percentile(toks, 50), 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached,
            "kv_gib_modeled_gqa": round(kv_gib_gqa(seq), 4),
            "kv_gib_modeled_dense": round(kv_gib_dense(seq), 4),
            "notes": (
                f"Modal serving-layer offline vLLM; TP={tp}; "
                f"batch={len(outputs)}; wall_s={wall_s:.2f}; "
                f"tok_approx={getattr(tokenizer, 'name_or_path', '')}"
            ),
            "timestamp": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        rows.append(row)
        print(
            f"   TTFT~p50={row['ttft_ms_p50']:.0f} ms · "
            f"tok/s={row['output_tok_s_p50']:.1f} · "
            f"cached≈{row['cached_prompt_tokens']}",
            flush=True,
        )

    volume.commit()
    return rows


@app.local_entrypoint()
def main(quick: bool = False, model: str = MODEL_DEFAULT, tp: int = 1):
    print(f"Modal vLLM serving bench · model={model} · tp={tp} · quick={quick}")
    rows = serve_and_bench.remote(model, quick, tp)

    root = Path(__file__).resolve().parents[1]
    out_csv = root / "benchmarks" / "results" / "modal_vllm_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    def key(r: dict) -> tuple:
        return (
            r.get("model", ""),
            str(r.get("input_tokens_target", "")),
            str(r.get("concurrency", "")),
            r.get("workload", ""),
            r.get("engine", ""),
        )

    by = {key(r): r for r in existing}
    for r in rows:
        by[key(r)] = r
    merged = list(by.values())
    fields = (
        list(rows[0].keys())
        if rows
        else (list(existing[0].keys()) if existing else [])
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    print(f"Wrote {len(rows)} new / {len(merged)} total → {out_csv}")
    subprocess.run(
        ["python3", str(root / "benchmarks" / "scripts" / "generate_report.py")],
        cwd=root,
        check=False,
    )
