#!/usr/bin/env python3
"""OpenAI-compatible API benchmark (Phase B/C). Set OPENAI_API_KEY + OPENAI_BASE_URL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "benchmarks" / "prompts"
RESULTS = ROOT / "benchmarks" / "results"
SAMPLE = ROOT / "benchmarks" / "sample_results" / "demo_runs.csv"

# GQA KV for Qwen2.5-7B @ bf16: 2*28*S*4*128*2 bytes
def kv_gib_gqa(seq: int, batch: int = 1) -> float:
    b = 2 * 28 * seq * 4 * 128 * 2 * batch
    return b / (1024**3)


def kv_gib_dense(seq: int, batch: int = 1) -> float:
    b = 2 * 28 * seq * 3584 * 2 * batch
    return b / (1024**3)


def load_prefix_tokens(target: int) -> str:
    path = PROMPTS / "prefix_10k.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else "Context block. " * 500
    # Rough expansion to ~target tokens (4 chars ≈ 1 token heuristic)
    while len(text) < target * 4:
        text += text[: min(2000, len(text))]
    return text[: target * 4]


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_chat(
    client: httpx.Client,
    model: str,
    messages: list[dict],
    session: str | None,
    max_tokens: int,
) -> dict:
    headers = {}
    if session:
        headers["x-session-affinity"] = session
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached = 0

    with client.stream(
        "POST",
        "/chat/completions",
        json=body,
        headers=headers,
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if ttft is None and data.get("choices"):
                delta = data["choices"][0].get("delta", {})
                if delta.get("content"):
                    ttft = (time.perf_counter() - t0) * 1000
            if data.get("choices"):
                c = data["choices"][0].get("delta", {}).get("content") or ""
                chunks.append(c)
            usage = data.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
        cached_hdr = resp.headers.get("fireworks-cached-prompt-tokens")
        if cached_hdr:
            cached = int(cached_hdr)
        if prompt_tokens == 0:
            prompt_tokens = int(resp.headers.get("fireworks-prompt-tokens", 0) or 0)

    t1 = time.perf_counter()
    if ttft is None:
        ttft = (t1 - t0) * 1000
    decode_s = max(t1 - t0 - ttft / 1000, 1e-6)
    out_tok = max(completion_tokens, len("".join(chunks)) // 4, 1)
    tpot = (decode_s * 1000) / out_tok
    tok_s = out_tok / decode_s if decode_s > 0 else 0
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "output_tok_s": tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out_tok,
        "cached_prompt_tokens": cached,
    }


def benchmark_cell(
    client: httpx.Client,
    model: str,
    input_t: int,
    output_t: int,
    workload: str,
    reps: int,
) -> dict:
    session = f"bench-{uuid.uuid4().hex[:8]}"
    prefix = load_prefix_tokens(max(input_t - 50, 512))
    ttfts: list[float] = []
    tpots: list[float] = []
    tokss: list[float] = []
    last: dict = {}

    for i in range(reps):
        if workload == "cold_prefix":
            user = f"Unique question {uuid.uuid4()}: summarize in one sentence."
            messages = [
                {"role": "system", "content": prefix + f"\n\nSalt:{uuid.uuid4()}"},
                {"role": "user", "content": user},
            ]
            sess = None
        elif workload == "warm_prefix":
            user = f"Follow-up {i}: one sentence answer."
            messages = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": user},
            ]
            sess = session
        else:
            messages = [
                {"role": "user", "content": prefix[: input_t * 4] + "\n\nSay hi briefly."}
            ]
            sess = None

        last = run_chat(client, model, messages, sess, output_t)
        ttfts.append(last["ttft_ms"])
        tpots.append(last["tpot_ms"])
        tokss.append(last["output_tok_s"])

    seq = last.get("prompt_tokens") or input_t
    return {
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "tpot_ms_p50": percentile(tpots, 50),
        "output_tok_s_p50": percentile(tokss, 50),
        "prompt_tokens": last.get("prompt_tokens", input_t),
        "completion_tokens": last.get("completion_tokens", output_t),
        "cached_prompt_tokens": last.get("cached_prompt_tokens", 0),
        "kv_gib_modeled_gqa": round(kv_gib_gqa(seq), 4),
        "kv_gib_modeled_dense": round(kv_gib_dense(seq), 4),
    }


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="API inference benchmark")
    p.add_argument("--dry-run", action="store_true", help="Copy sample CSV only")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", "accounts/fireworks/models/qwen2p5-7b-instruct"))
    p.add_argument("--provider", default="fireworks")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", type=Path, default=RESULTS / "api_runs.csv")
    args = p.parse_args()

    if args.dry_run:
        import shutil
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SAMPLE, args.out)
        print(f"Dry run: wrote sample data to {args.out}")
        return

    if httpx is None:
        raise SystemExit("Install deps: pip install -r benchmarks/requirements.txt")

    base = os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY or FIREWORKS_API_KEY")

    matrix = [
        ("C", "api", 1024, 128, "single"),
        ("C", "api", 10240, 128, "single"),
        ("B", "kv_cache", 10240, 128, "cold_prefix"),
        ("B", "kv_cache", 10240, 128, "warm_prefix"),
    ]

    with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}) as client:
        for phase, layer, inp, out, workload in matrix:
            print(f"Running {phase} {workload} in={inp} out={out}...")
            m = benchmark_cell(client, args.model, inp, out, workload, args.reps)
            row = {
                "run_id": f"api-{uuid.uuid4().hex[:8]}",
                "phase": phase,
                "stack_layer": "serving",
                "layer": layer,
                "provider": args.provider,
                "model": args.model,
                "engine": "openai-compatible-api",
                "input_tokens_target": inp,
                "output_tokens_target": out,
                "concurrency": 1,
                "workload": workload,
                "cache_mode": "on" if workload == "warm_prefix" else "unknown",
                "session_affinity": "yes" if workload == "warm_prefix" else "",
                **m,
                "notes": "live api run",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            write_row(args.out, row)
            print(f"  TTFT p50={m['ttft_ms_p50']:.0f}ms cached={m['cached_prompt_tokens']}")

    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
