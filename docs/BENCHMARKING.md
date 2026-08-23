# Benchmarking guide

See **`docs/LAYERS.md`** for the full split between **model layer** (isolated GPU, KV math, batch=1) and **serving layer** (HTTP, concurrency, prefix cache).

## CSV schema

All runs use columns in `benchmarks/schema.csv`. Import merged CSVs into Excel for pivots:

- **`stack_layer`** — `model` vs `serving` (primary pivot)
- **TTFT vs input_tokens_target** — prefill / context cost
- **output_tok_s vs concurrency** — serving saturation
- **cached_prompt_tokens** — prefix cache hits (serving only)
- **kv_gib_modeled_gqa** — ties latency rows to the explorer math

## Model layer (Phase A — isolated GPU)

**Question:** How much KV and compute does this architecture need at sequence length S?

```bash
npm run benchmark:modal:quick   # Modal + transformers, batch=1
npm run benchmark:modal         # full matrix
```

CSV: `stack_layer=model`, `layer=inference`, `provider=modal`.

Do **not** compare Modal TTFT 1:1 with API numbers (no network/queue).

## Serving layer — engine (Phase A on DigitalOcean GPU)

1. Create an **H100 GPU droplet** (Ubuntu 22.04+).
2. Install CUDA driver, Python 3.11, vLLM:
   ```bash
   pip install vllm
   vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 32768
   ```
3. On same droplet or bastion:
   ```bash
   bash benchmarks/scripts/run_engine_benchmark.sh
   ```

Matrix defaults: input `1024 / 8192 / 10240`, output `128 / 512`, concurrency `1 / 8`.

CSV: `stack_layer=serving`, `layer=engine`.

## Serving layer — KV / prefix cache (Phase B)

**GPU:** enable vLLM prefix caching; rerun warm/cold workloads from `run_engine_benchmark.sh` (extend with `warm_prefix` cases).

**API (Fireworks):**

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
python benchmarks/scripts/run_api_benchmark.py
```

Read `fireworks-cached-prompt-tokens` response header; use stable `x-session-affinity`.

## Serving layer — API monitoring (Phase C)

- Prompt shapes: **1024** and **10240** input tokens
- Cron every 3–6 hours; store median TTFT over 72h in Excel
- Region note: AA uses GCP us-central1; your numbers will differ by RTT

## Phase D — Quality (optional)

Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on a separate tab — do not mix with perf rows.

## Regenerate HTML report

```bash
python benchmarks/scripts/generate_report.py
# or
npm run report
```

Outputs:

- `public/benchmark-report.html` (shipped with `npm run build`)
- `benchmarks/reports/latest.html`
- `benchmarks/reports/latest.json`
