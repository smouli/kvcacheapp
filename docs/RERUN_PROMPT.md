# Rerun prompt (paste to another LLM)

Copy everything below the line into another agent/LLM that has access to this repo.

---

## Task

You are working in the **`kvcacheapp`** repo (React/Vite KV-cache explorer + inference benchmark harness).

**Goal:** Re-run the full live benchmark stack we already built, regenerate the hostable HTML report, and rebuild the static site — same order and meaning as a complete refresh of results.

Do **not** invent new benchmarks. Use the existing scripts. Do **not** compare model-layer TTFT 1:1 with API TTFT (different stacks).

## Prerequisites

1. Working directory: repo root (`kvcacheapp`).
2. Node deps: `npm ci` (or `npm install`).
3. Modal CLI authenticated (`modal` on PATH; `~/.modal.toml` or equivalent).
4. Optional Fireworks API for serving-provider path:
   ```bash
   export OPENAI_API_KEY=…   # or FIREWORKS_API_KEY if the script expects that
   export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
   ```
   Use a model ID that exists on the account (e.g. `accounts/fireworks/models/qwen3p7-plus` if Qwen2.5-7B is not deployed).
5. Optional DigitalOcean GPU droplet is **not** required for a full Modal refresh.

## Architecture (do not collapse these)

| Layer | Question | How we run it |
|-------|----------|----------------|
| **Hardware** | Memcpy GB/s, GEMM TFLOPS, NCCL busbw | `benchmarks/modal_hardware.py` |
| **Model** | Batch=1 KV GiB, prefill≈TTFT, decode tok/s | `benchmarks/modal_engine.py` |
| **Serving** | Concurrency, prefix cache, client TTFT | `benchmarks/modal_vllm.py` + optional API script |

Docs: `docs/LAYERS.md`, `docs/GPU_STACK.md`, `docs/COVERAGE.md`.

## Exact run order

Execute sequentially (later steps overwrite/merge CSVs and refresh the report). Prefer **full** matrices unless the user asks for quick.

```bash
# 0) deps
npm ci

# Prefer one-shot full refresh (hardware+NCCL → model → vLLM → API → report → build)
npm run benchmark:full

# Or step-by-step:
# 1) HARDWARE — CUDA memcpy/GEMM on T4/A10G/A100 + NCCL on A100:2 (yes, NCCL runs on Modal)
npm run benchmark:modal:hardware:nccl
# 2) MODEL layer — transformers on Modal (T4 / A10G)
npm run benchmark:modal
# 3) SERVING layer — vLLM on Modal A100 (prefix cache + concurrency)
npm run benchmark:modal:vllm
# 4) SERVING — hosted API (Fireworks), if key is set
npm run benchmark:api
# 5) Report + static site
npm run report && npm run build
```

**NCCL / CUDA on Modal:** Yes. `benchmarks/modal_hardware.py` runs CUDA D2D memcpy + GEMM on single GPUs and `torch.distributed` NCCL all-reduce on `gpu="A100:2"`. No local NVIDIA GPU required.

## Expected artifacts

| Path | Contents |
|------|----------|
| `benchmarks/results/hardware_runs.csv` | Hardware probes |
| `benchmarks/results/modal_runs.csv` | Model-layer Modal runs |
| `benchmarks/results/modal_vllm_runs.csv` | Serving vLLM Modal runs |
| `benchmarks/results/api_runs.csv` | Fireworks/API runs |
| `public/benchmark-report.html` | Hostable report (also copied into `dist/` on build) |
| `benchmarks/reports/latest.html` / `latest.json` | Report mirrors |

Open/report URL path: **`/benchmark-report.html`**.

## Acceptance checks

After runs finish:

1. Report shows **hardware**, **model**, and **serving** coverage as live where data exists.
2. Hardware rows include memcpy + BF16/FP16 TFLOPS for T4/A10G/A100; NCCL row if `--with-nccl` was used.
3. Model rows: `stack_layer=model`, provider `modal`, batch=1.
4. Serving rows: vLLM concurrency + `cold_prefix` / `warm_prefix` where applicable; API rows tagged live if Fireworks succeeded.
5. Do **not** treat sample/placeholder DigitalOcean rows as live unless notes say live.
6. `npm run build` succeeds; `dist/benchmark-report.html` exists.

## Rules for the agent

- Prefer existing npm scripts over ad-hoc commands.
- If a Modal job fails (OOM, image, flashinfer), fix the **smallest** thing consistent with prior patterns in this repo (e.g. route long context to A10G; offline vLLM instead of broken `vllm serve`).
- Merge CSVs on re-run; do not wipe unrelated result files.
- Never commit secrets (API keys). Rotate any key pasted into chat.
- Do not force-push or change git config.
- When done, summarize: commands run, row counts per CSV, headline numbers (peak memcpy, peak BF16 TFLOPS, prefix-cache win, live run count), and path to the report.

## Optional (only if user asks)

- Deploy `dist/` to DigitalOcean App Platform.
- Live DO GPU droplet vLLM (scripts under `benchmarks/scripts/setup_gpu_stack.sh`, `serve_vllm.sh`).
- SWE-bench / τ-bench quality track (separate from latency/KV; do not mix into TTFT CSVs).
