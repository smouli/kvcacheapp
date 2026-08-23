# Benchmark stack: model layer vs serving layer

Inference benchmarks are often mixed together. This project separates **two tiers** so Excel rows and the HTML report stay interpretable.

```
┌─────────────────────────────────────────────────────────────┐
│  SERVING LAYER  — what callers / users experience           │
│  HTTP API · queue · concurrency · routing · prefix cache   │
│  Tools: llmperf, Fireworks, DO Dedicated/Serverless         │
└───────────────────────────┬─────────────────────────────────┘
                            │ sits on top of
┌───────────────────────────▼─────────────────────────────────┐
│  MODEL LAYER  — what the weights + KV cache do on GPU       │
│  prefill · decode · GQA KV bytes · peak VRAM · batch=1      │
│  Tools: Modal + transformers, isolated vLLM (no load)       │
└─────────────────────────────────────────────────────────────┘
```

## Model layer

**Question:** For this **architecture** (e.g. Qwen2.5-7B GQA) and sequence length **S**, how much **KV memory** and **compute** does inference need?

| Property | Typical setup |
|----------|----------------|
| **Isolation** | Single process, **batch = 1**, **concurrency = 1** |
| **No** | HTTP, queue, router, network RTT, multi-tenant |
| **KV focus** | Modeled GQA/dense GiB, optional **peak GPU GiB** |
| **Latency focus** | Prefill time (TTFT proxy), TPOT / decode tok/s |

**What you learn**

- KV sizing matches the explorer (`2 × L × S × n_kv × d`)
- TTFT grows with **input length** (prefill)
- Decode tok/s is mostly **decode-bound** (KV read bandwidth)

**How we run it here**

```bash
npm run benchmark:modal:quick   # Modal GPU + transformers
npm run benchmark:modal         # matrix across models / S
```

CSV: `stack_layer=model`, `layer=inference`, `provider=modal`.

---

## Serving layer

**Question:** Under **real serving** (API or engine with batching), what **latency and throughput** do clients see — including **prefix cache**, **concurrency**, and **queueing**?

| Property | Typical setup |
|----------|----------------|
| **Interface** | OpenAI-compatible **HTTP** or vLLM server + llmperf |
| **Load** | **Concurrency** 1 → 32+, continuous batching |
| **KV focus** | **cached_prompt_tokens**, warm vs cold prefix, optional `kv_blocks_fraction` (DO/Fireworks metrics) |
| **Latency focus** | TTFT p50/p95 **under load**, system tok/s, E2E response |

**What you learn**

- Same model can look **much faster** on repeat prefixes (serving cache ≠ model math)
- High concurrency **inflates TTFT** (queue) even if model layer prefill is unchanged
- Provider API numbers **include network**; not comparable 1:1 to Modal model runs

**How we run it here**

```bash
# API / provider (Fireworks, DO Dedicated endpoint)
npm run benchmark:api

# Engine on GPU droplet (vLLM + llmperf)
npm run benchmark:engine
```

CSV: `stack_layer=serving`, `layer` = `api` | `engine` | `kv_cache`.

---

## Side-by-side

| | **Model layer** | **Serving layer** |
|--|-----------------|-------------------|
| **Unit of work** | One forward / generate path | Many requests, endpoints |
| **KV cache** | Bytes formula + GPU peak | Prefix hits, block utilization, eviction |
| **TTFT** | Mostly **prefill** | Prefill + **queue** + **network** |
| **Concurrency** | Fixed at 1 | Swept (1, 8, 32…) |
| **Compare to LMCache sizing** | Yes (GQA formula) | Indirect (cached tokens, TTFT drop) |
| **Compare to Artificial Analysis** | No | Yes (API-style) |

---

## Phases (this repo) mapped to stack

| Phase | Stack | What |
|-------|--------|------|
| **A** | Model *or* Serving | Model = Modal; Serving = vLLM+llmperf on droplet |
| **B** | Serving | Prefix cache cold/warm, multi-turn |
| **C** | Serving | Provider API monitoring (AA-like shapes) |
| **D** | — | Quality (MMLU etc.) — not latency |

---

## Excel / report columns

- **`stack_layer`**: `model` \| `serving` — use this for pivots
- **`layer`**: finer grain — `inference`, `engine`, `api`, `kv_cache`
- **`kv_gib_modeled_gqa`**: meaningful at **model** layer; still logged at serving for linkage
- **`cached_prompt_tokens`**: **serving only**

Do **not** compare Modal model TTFT directly to Fireworks API TTFT without noting network + queue. **Do** compare both to the same **KV GiB** column for a given **S**.
