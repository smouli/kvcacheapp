"""
Serving-layer vLLM benchmarks on Modal.

  modal run benchmarks/modal_vllm.py --profile qwen7b
  modal run benchmarks/modal_vllm.py --quick
  modal run benchmarks/modal_vllm.py --profile qwen32b --tp 2

CSV: benchmarks/results/modal_serving_runs.csv (engine_family=vllm)
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

APP_NAME = "kvcache-vllm-serving"
RESULTS_VOL = "kvcache-bench-results"
HF_VOL = "kvcache-hf-cache"
MOUNT = "/bench-results"
BENCH_DIR = Path(__file__).resolve().parent

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        "pyyaml>=6.0",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
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


def _run_bench(model_id: str, quick: bool, tp: int, profile: str, gpu_sku: str) -> list[dict]:
    import os

    from vllm import LLM, SamplingParams

    build_cases, make_serving_row, prompts_for_case, approx_metrics = _remote_bench_lib()

    if tp > 1:
        os.environ.setdefault("NCCL_DEBUG", "WARN")

    max_len = 8192 if quick else (12288 if tp >= 2 else 16384)
    print(f"Loading {model_id} · vLLM · TP={tp} · gpu={gpu_sku} · max_len={max_len}", flush=True)

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )

    cases = build_cases(quick=quick, tp=tp, profile=profile)
    rows: list[dict] = []

    def prime(prompt: str) -> None:
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8))

    for case in cases:
        phase, layer, inp, out, conc, workload = case
        print(f"→ vLLM {workload} in={inp} out={out} conc={conc} tp={tp}", flush=True)
        params = SamplingParams(temperature=0, max_tokens=out)
        batch, cached_est = prompts_for_case(workload, inp, conc, prime_fn=prime)

        t0 = time.perf_counter()
        outputs = llm.generate(batch, params)
        wall_s = time.perf_counter() - t0

        prompt_tokens = 0
        completion_tokens = 0
        for out_i in outputs:
            prompt_tokens = len(out_i.prompt_token_ids or [])
            completion_tokens = len(out_i.outputs[0].token_ids) if out_i.outputs else 0

        m = approx_metrics(
            wall_s=wall_s,
            n_outputs=len(outputs),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            inp_fallback=inp,
        )
        row = make_serving_row(
            engine_family="vllm",
            engine_label=f"vLLM+Modal+TP{tp}",
            model_id=model_id,
            case=case,
            metrics=m,
            tp=tp,
            gpu_sku=gpu_sku,
            cached=cached_est,
            batch_n=len(outputs),
            wall_s=wall_s,
        )
        rows.append(row)
        print(
            f"   TTFT~{row['ttft_ms_p50']:.0f} ms · "
            f"tok/s={row['output_tok_s_p50']:.1f} · cached≈{cached_est}",
            flush=True,
        )

    volume.commit()
    return rows


@app.function(
    gpu="A100",
    image=vllm_image,
    timeout=3600,
    volumes={MOUNT: volume, "/root/.cache/huggingface": hf_cache},
)
def serve_and_bench_tp1(
    model_id: str, quick: bool, profile: str, gpu_sku: str
) -> list[dict]:
    return _run_bench(model_id, quick, tp=1, profile=profile, gpu_sku=gpu_sku)


@app.function(
    gpu="A100:2",
    image=vllm_image,
    timeout=5400,
    volumes={MOUNT: volume, "/root/.cache/huggingface": hf_cache},
)
def serve_and_bench_tp2(
    model_id: str, quick: bool, profile: str, gpu_sku: str
) -> list[dict]:
    return _run_bench(model_id, quick, tp=2, profile=profile, gpu_sku=gpu_sku)


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

    print(f"Modal vLLM · model={model_id} · tp={tp} · gpu={gpu_sku} · profile={prof}")
    fn = serve_and_bench_tp2 if tp >= 2 else serve_and_bench_tp1
    rows = fn.remote(model_id, quick, prof, gpu_sku)

    root = Path(__file__).resolve().parents[1]
    merge_serving_csv(rows)
    subprocess.run(
        [sys.executable, str(root / "benchmarks" / "scripts" / "generate_report.py")],
        cwd=root,
        check=False,
    )
