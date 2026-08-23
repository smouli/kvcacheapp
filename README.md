# KV Cache Explorer + Inference Benchmark Harness

Interactive **KV cache sizing** (dense vs **GQA-accurate**) and a **reproducible benchmark pipeline** for engine, prefix-cache, and API layers — designed to demo on **DigitalOcean GPU droplets** and compare with provider APIs (Fireworks-style).

## Live demo (local)

```bash
npm install
npm run dev          # http://localhost:5174
npm run report       # regenerates public/benchmark-report.html
npm run build && npm run preview
```

Open **Benchmark report** from the app header or `/benchmark-report.html`.

## What impresses in a 5-minute review

1. **GQA vs dense KV** — Qwen2.5-7B shows **7×** memory difference; chart plots both curves vs GPU budget.
2. **Executive HTML report** — TTFT vs context length, warm vs cold prefix, modeled KV GiB linked to each run.
3. **Production runbook** — vLLM on DO GPU + llmperf matrix + API benchmark script → CSV → Excel.
4. **One-command deploy** — Docker/nginx for static app; GPU benchmark script for droplet.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐
│  React KV Explorer  │     │  benchmarks/scripts/      │
│  (GQA + dense math) │     │  run_api_benchmark.py     │
└─────────────────────┘     │  run_engine_benchmark.sh  │
                            │  generate_report.py       │
                            └────────────┬─────────────┘
                                         ▼
                            benchmark-report.html + CSV → Excel
```

## Benchmark phases

| Phase | Layer | Where | Tool |
|-------|--------|-------|------|
| **A** | Engine | DO GPU droplet | vLLM + [llmperf](https://github.com/run-ai/llmperf) |
| **B** | KV / prefix cache | GPU or API | cold vs warm prefix, `x-session-affinity` |
| **C** | API | Laptop / cron | OpenAI-compatible (Fireworks) |
| **D** | Quality | Optional | lm-eval-harness (separate) |

See `docs/BENCHMARKING.md`, `docs/LAYERS.md`, `docs/GPU_STACK.md`, and the full path in **`docs/CHECKLIST.md`**.

## Quick benchmark commands

```bash
# Python deps
pip install -r benchmarks/requirements.txt

# Demo report (sample data — no API key)
python benchmarks/scripts/run_api_benchmark.py --dry-run
npm run report

# Live API (Fireworks)
export OPENAI_API_KEY=fw_...
export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
python benchmarks/scripts/run_api_benchmark.py
python benchmarks/scripts/generate_report.py

# Engine (on GPU droplet after vLLM is up)
bash benchmarks/scripts/run_engine_benchmark.sh
```

Results append to `benchmarks/results/*.csv` — import into Excel using `benchmarks/schema.csv` headers.

## KV formulas

**GQA (deployed):** `KV = B × S × L × 2 × n_kv × d × bytes`  
**Dense (upper bound):** `KV = B × S × L × 2 × H × bytes`

Qwen2.5-7B: `n_kv=4`, `d=128`, `L=28`, `H=3584` → **7×** ratio.

## Deploy on DigitalOcean

- **Static UI:** App Platform or Droplet + Docker (`Dockerfile`)
- **GPU stack:** `docs/GPU_STACK.md` — CUDA, **NCCL** (multi-GPU TP), vLLM, **FP8/AWQ** quant
- **Benchmarks:** [GPU Droplets](https://docs.digitalocean.com/products/droplets/how-to/gpu/) + `docs/DEPLOY_DIGITALOCEAN.md`

```bash
# On a DO GPU droplet
bash benchmarks/scripts/setup_gpu_stack.sh
bash benchmarks/scripts/serve_vllm.sh qwen32b_tp2_bf16   # NCCL tensor parallel
bash benchmarks/scripts/run_engine_benchmark.sh
```

## Project structure

```
src/                 React explorer + log-log KV chart
benchmarks/          Matrix, scripts, sample results, reports
public/              benchmark-report.html (generated)
docs/                DO deploy + benchmarking guide
```

## License

MIT — demo / education project.
