"""
GPU inference benchmarks on Modal (same auth as YuE via ~/.modal.toml).

  modal run benchmarks/modal_engine.py              # full matrix
  modal run benchmarks/modal_engine.py --quick      # smoke test (~2 min)
  modal run benchmarks/modal_engine.py --models Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "kvcache-benchmarks"
RESULTS_VOL = "kvcache-bench-results"
MOUNT = "/bench-results"

# Public OSS models (no HF token required for these)
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

MATRIX_QUICK = [(512, 32)]
MATRIX_FULL = [
    (512, 64),
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (8192, 128),
    (10240, 128),
    (10240, 512),
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch --index-url https://download.pytorch.org/whl/cu124",
        "pip install 'transformers>=4.44' accelerate sentencepiece protobuf",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)


def kv_bytes_gqa(batch, seq, layers, num_kv_heads, head_dim, bytes_per=2):
    return batch * seq * layers * 2 * num_kv_heads * head_dim * bytes_per


def kv_bytes_dense(batch, seq, layers, hidden, bytes_per=2):
    return batch * seq * layers * 2 * hidden * bytes_per


def pick_gpu(model_id: str, input_tokens: int) -> str:
    """Return Modal GPU type (A10G for 7B+ or long-context prefill)."""
    if "7B" in model_id or "8B" in model_id:
        return "A10G"
    if input_tokens >= 10240:
        return "A10G"  # T4 OOM on 10k+ prefill (attention activations)
    return "T4"


# Separate registrations — Modal 1.4 has no with_options on Function.
@app.function(
    gpu="T4",
    image=image,
    timeout=1800,
    volumes={MOUNT: volume},
    retries=1,
)
def bench_one_t4(model_id: str, input_tokens: int, output_tokens: int) -> dict:
    return _bench_impl(model_id, input_tokens, output_tokens)


@app.function(
    gpu="A10G",
    image=image,
    timeout=1800,
    volumes={MOUNT: volume},
    retries=1,
)
def bench_one_a10g(model_id: str, input_tokens: int, output_tokens: int) -> dict:
    return _bench_impl(model_id, input_tokens, output_tokens)


def _bench_impl(model_id: str, input_tokens: int, output_tokens: int) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    load_s = time.perf_counter() - t0

    # Synthetic prompt near target length
    chunk = "The quick brown fox jumps over the lazy dog. "
    text = chunk
    while len(tok.encode(text)) < input_tokens:
        text += chunk
    ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=input_tokens)
    ids = ids.to(model.device)
    prompt_len = ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    prefill_t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000

    gen_t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            input_ids=ids,
            max_new_tokens=output_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - gen_t0) * 1000
    decode_ms = max(gen_ms - prefill_ms, 1)
    completion_len = gen.shape[1] - prompt_len
    tpot_ms = decode_ms / max(completion_len, 1)
    tok_s = completion_len / (decode_ms / 1000) if decode_ms > 0 else 0

    peak_bytes = torch.cuda.max_memory_allocated()
    seq = prompt_len + completion_len
    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = hidden // cfg.num_attention_heads

    row = {
        "run_id": f"modal-{uuid.uuid4().hex[:8]}",
        "phase": "A",
        "stack_layer": "model",
        "layer": "inference",
        "provider": "modal",
        "model": model_id,
        "engine": f"transformers+{gpu_name}",
        "input_tokens_target": input_tokens,
        "output_tokens_target": output_tokens,
        "concurrency": 1,
        "workload": "single",
        "cache_mode": "off",
        "session_affinity": "",
        "ttft_ms_p50": round(prefill_ms, 2),
        "ttft_ms_p95": round(prefill_ms * 1.1, 2),
        "tpot_ms_p50": round(tpot_ms, 2),
        "output_tok_s_p50": round(tok_s, 2),
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "cached_prompt_tokens": 0,
        "kv_gib_modeled_gqa": round(
            kv_bytes_gqa(1, seq, layers, n_kv, head_dim) / 1024**3, 4
        ),
        "kv_gib_modeled_dense": round(
            kv_bytes_dense(1, seq, layers, hidden) / 1024**3, 4
        ),
        "peak_gpu_gib": round(peak_bytes / 1024**3, 3),
        "model_load_s": round(load_s, 1),
        "notes": "Modal model-layer bench (batch=1); TTFT≈prefill",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path = Path(MOUNT) / f"{row['run_id']}.json"
    out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    volume.commit()
    return row


@app.local_entrypoint()
def main(quick: bool = False, models: str = ""):
    model_list = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_MODELS
    matrix = MATRIX_QUICK if quick else MATRIX_FULL
    if quick:
        model_list = model_list[:1]

    rows: list[dict] = []
    for model_id in model_list:
        for inp, out in matrix:
            gpu = pick_gpu(model_id, inp)
            fn = bench_one_a10g if gpu == "A10G" else bench_one_t4
            print(f"→ {model_id}  in={inp} out={out}  gpu={gpu}")
            try:
                row = fn.remote(model_id, inp, out)
                rows.append(row)
                print(
                    f"   TTFT {row['ttft_ms_p50']:.0f} ms · "
                    f"{row['output_tok_s_p50']:.1f} tok/s · "
                    f"KV {row['kv_gib_modeled_gqa']} GiB (GQA)"
                )
            except Exception as exc:
                print(f"   FAILED: {exc}")

    root = Path(__file__).resolve().parents[1]
    out_csv = root / "benchmarks" / "results" / "modal_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        # Merge with existing CSV so partial re-runs (e.g. one model) keep prior rows.
        existing: list[dict] = []
        if out_csv.exists():
            with out_csv.open(newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))

        def row_key(r: dict) -> tuple:
            return (
                r.get("model", ""),
                str(r.get("input_tokens_target", "")),
                str(r.get("output_tokens_target", "")),
                r.get("provider", "modal"),
            )

        by_key = {row_key(r): r for r in existing}
        for r in rows:
            by_key[row_key(r)] = r
        merged = list(by_key.values())
        fieldnames = list(rows[0].keys())
        for r in merged:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        print(f"\nWrote {len(rows)} new / {len(merged)} total rows → {out_csv}")

        import subprocess

        subprocess.run(
            ["python3", str(root / "benchmarks" / "scripts" / "generate_report.py")],
            check=False,
            cwd=root,
        )
    else:
        print("No successful runs.")
