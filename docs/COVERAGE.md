# What we benchmark

Two tiers (see also `docs/LAYERS.md`):

## Model layer — architecture & GPU math

| What | Tool | Status |
|------|------|--------|
| GQA vs dense KV GiB vs sequence length | Explorer UI + modeled columns | Live |
| Prefill TTFT (batch=1) | Modal + transformers | Live |
| Decode tok/s | Modal + transformers | Live |
| Peak GPU GiB | Modal | Live |
| Models | Qwen2.5-0.5B, Qwen2.5-7B | Live |
| Sequence lengths | 512 → 10k | Expanding |

**Command:** `npm run benchmark:modal`

## Serving layer — what clients feel

| What | Tool | Status |
|------|------|--------|
| vLLM continuous batching | Modal A100 + vLLM | Live |
| Prefix cache cold vs warm | Modal vLLM / Fireworks | Live |
| Concurrency (1 / 4 / 8+) | Modal vLLM | Live |
| Hosted API TTFT | Fireworks | Live |
| DO GPU droplet vLLM | DigitalOcean | Optional (sample rows only today) |

**Commands:**
```bash
npm run benchmark:modal:vllm   # dense serving matrix on Modal
npm run benchmark:api          # Fireworks
```

## Not in this harness (quality / agents)

| Benchmark | Measures | Notes |
|-----------|----------|-------|
| SWE-bench | Coding agent issue resolve % | Quality track |
| τ-bench (TauBench) | Tool-using agent task success | Quality track |

Those belong in a separate eval lane — do not mix with TTFT / KV CSVs.

## Report

```bash
npm run report   # → public/benchmark-report.html
npm run dev      # open /benchmark-report.html
```
