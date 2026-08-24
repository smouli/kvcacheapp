"""
Serving-layer SGLang benchmarks on Modal — same matrix as modal_vllm.py.

Uses the official lmsysorg/sglang CUDA image (FlashInfer / flash-attn prebuilt)
so engine A/B vs vLLM is fair.

  modal run benchmarks/modal_sglang.py --profile qwen7b
  modal run benchmarks/modal_sglang.py --quick
  modal run benchmarks/modal_sglang.py --profile qwen32b --tp 2

CSV: benchmarks/results/modal_serving_runs.csv (engine_family=sglang)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import modal

from modal_bench_lib import (
    catalog_entry,
    gpu_sku_for,
    merge_serving_csv,
    profile_name,
)

APP_NAME = "kvcache-sglang-serving"
RESULTS_VOL = "kvcache-bench-results"
HF_VOL = "kvcache-hf-cache"
MOUNT = "/bench-results"
BENCH_DIR = Path(__file__).resolve().parent

# Official SGLang runtime — CUDA 12.4 stack matches Modal A100 + vLLM flashinfer cu124
SGLANG_DOCKER = "lmsysorg/sglang:v0.4.7-cu124"

sglang_image = (
    modal.Image.from_dockerfile(str(BENCH_DIR / "Dockerfile.sglang"))
    .add_local_file(
        str(BENCH_DIR / "modal_bench_lib.py"),
        remote_path="/pkg/modal_bench_lib.py",
    )
    .add_local_file(
        str(BENCH_DIR / "config" / "catalog.yaml"),
        remote_path="/pkg/config/catalog.yaml",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_VOL, create_if_missing=True)


def _remote_bench_lib():
    import sys

    if "/pkg" not in sys.path:
        sys.path.insert(0, "/pkg")
    from modal_bench_lib import (  # noqa: WPS433
        approx_metrics,
        build_cases,
        make_serving_row,
        prompts_for_case,
    )

    return build_cases, make_serving_row, prompts_for_case, approx_metrics


def _parse_sglang_output(outputs, inp: int, out: int) -> tuple[int, int]:
    """Best-effort token counts from SGLang generate() payloads."""
    prompt_tokens = 0
    completion_tokens = 0
    if not outputs:
        return inp, out
    item = outputs[0]
    if isinstance(item, dict):
        meta = item.get("meta_info") or item.get("meta") or {}
        prompt_tokens = int(
            meta.get("prompt_tokens")
            or meta.get("prompt_token_len")
            or meta.get("input_token_logprobs_len")
            or inp
        )
        completion_tokens = int(
            meta.get("completion_tokens")
            or meta.get("completion_token_len")
            or meta.get("finish_reason", {}).get("length")
            or out
        )
        text = item.get("text") or ""
        if not completion_tokens and text:
            completion_tokens = max(len(text.split()) * 2, 1)
    return prompt_tokens or inp, completion_tokens or out


def _run_bench(
    model_id: str,
    quick: bool,
    tp: int,
    profile: str,
    gpu_sku: str,
    trust_remote_code: bool,
) -> list[dict]:
    import asyncio
    import os

    import uvloop
    from sglang import Engine

    build_cases, make_serving_row, prompts_for_case, approx_metrics = _remote_bench_lib()

    if tp > 1:
        os.environ.setdefault("NCCL_DEBUG", "WARN")

    max_len = 8192 if quick else (12288 if tp >= 2 else 16384)
    print(
        f"Loading {model_id} · SGLang · TP={tp} · gpu={gpu_sku} · "
        f"max_len={max_len} · image={SGLANG_DOCKER}",
        flush=True,
    )

    # flashinfer/fa from official image — disable CUDA graphs so more VRAM stays
    # available for KV (graph capture left ~2.5 GiB and hung 4k prefix batches)
    engine = Engine(
        model_path=model_id,
        tp_size=tp,
        trust_remote_code=trust_remote_code,
        context_length=max_len,
        mem_fraction_static=0.85,
        disable_cuda_graph=True,
    )

    cases = build_cases(quick=quick, tp=tp, profile=profile)
    rows: list[dict] = []

    # One event loop for the whole bench — recreating per call hangs SGLang Engine
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)

    def run_batch(prompts: list[str], max_tokens: int) -> tuple[list, float]:
        sampling = {"temperature": 0.0, "max_new_tokens": max_tokens}
        t0 = time.perf_counter()
        outputs = engine.generate(prompts, sampling)
        return outputs, time.perf_counter() - t0

    def prime(prompt: str) -> None:
        run_batch([prompt], 8)

    # Warm JIT / scheduler so timed cases aren't cold-start dominated
    print("→ warmup generate", flush=True)
    run_batch(["warmup"], 8)

    for case in cases:
        phase, layer, inp, out, conc, workload = case
        print(f"→ SGLang {workload} in={inp} out={out} conc={conc} tp={tp}", flush=True)
        batch, cached_est = prompts_for_case(workload, inp, conc, prime_fn=prime)
        outputs, wall_s = run_batch(batch, out)

        pt, ct = _parse_sglang_output(outputs, inp, out)
        m = approx_metrics(
            wall_s=wall_s,
            n_outputs=len(batch),
            prompt_tokens=pt,
            completion_tokens=ct,
            inp_fallback=inp,
        )
        row = make_serving_row(
            engine_family="sglang",
            engine_label=f"SGLang+Modal+TP{tp}",
            model_id=model_id,
            case=case,
            metrics=m,
            tp=tp,
            gpu_sku=gpu_sku,
            cached=cached_est,
            batch_n=len(batch),
            wall_s=wall_s,
        )
        rows.append(row)
        print(
            f"   TTFT~{row['ttft_ms_p50']:.0f} ms · "
            f"tok/s={row['output_tok_s_p50']:.1f} · cached≈{cached_est}",
            flush=True,
        )

    engine.shutdown()
    volume.commit()
    return rows


@app.function(
    gpu="A100",
    image=sglang_image,
    timeout=3600,
    volumes={MOUNT: volume, "/root/.cache/huggingface": hf_cache},
)
def serve_and_bench_tp1(
    model_id: str,
    quick: bool,
    profile: str,
    gpu_sku: str,
    trust_remote_code: bool,
) -> list[dict]:
    return _run_bench(model_id, quick, 1, profile, gpu_sku, trust_remote_code)


@app.function(
    gpu="A100:2",
    image=sglang_image,
    timeout=5400,
    volumes={MOUNT: volume, "/root/.cache/huggingface": hf_cache},
)
def serve_and_bench_tp2(
    model_id: str,
    quick: bool,
    profile: str,
    gpu_sku: str,
    trust_remote_code: bool,
) -> list[dict]:
    return _run_bench(model_id, quick, 2, profile, gpu_sku, trust_remote_code)


@app.local_entrypoint()
def main(
    quick: bool = False,
    model: str = "",
    tp: int = 0,
    profile: str = "",
):
    tp_arg = int(tp) if tp else 0
    ent = catalog_entry(profile or None, model, tp_arg or 1)
    model_id = model or ent["id"]
    tp = tp_arg or int(ent.get("tp") or 1)
    gpu_sku = gpu_sku_for(tp, ent.get("gpu"))
    prof = profile_name(ent, quick, tp) if not quick else "discovery"
    trust = bool(ent.get("trust_remote_code"))

    print(f"Modal SGLang · model={model_id} · tp={tp} · gpu={gpu_sku} · profile={prof}")
    fn = serve_and_bench_tp2 if tp >= 2 else serve_and_bench_tp1
    rows = fn.remote(model_id, quick, prof, gpu_sku, trust)

    root = Path(__file__).resolve().parents[1]
    merge_serving_csv(rows)
    subprocess.run(
        [sys.executable, str(root / "benchmarks" / "scripts" / "generate_report.py")],
        cwd=root,
        check=False,
    )
