# Inference customer discovery

**Live page:** `/benchmark-report.html` → **Start discovery**

## Purpose

Before quoting tokens or dedicated GPU capacity, profile the customer's workload shape. This harness runs the same discovery on three layers:

1. **Hardware** — memcpy, GEMM, NCCL (capacity / TP sanity check)
2. **Model** — KV GiB, prefill TTFT, decode tok/s (batch=1, no HTTP)
3. **Serving** — vLLM under load, prefix cache, concurrency, API TTFT

## Five questions (ask every AI-native deal)

| Question | Why it matters |
|----------|----------------|
| Typical **input length** (tokens)? | KV memory + prefill time |
| Typical **output length**? | Decode occupancy in continuous batching |
| **Prefix reuse** (cache hit rate)? | Warm vs cold TTFT can differ 2×+ |
| Peak **concurrency**? | Queue delay, batch depth, GPU count |
| **TTFT / latency SLO**? | Model-layer vs serving-layer numbers are not interchangeable |

## Scenarios on the report

| Scenario | Profile | What you discover |
|----------|---------|-------------------|
| AI-native · repeated system prompt | S≈4k, cache=high, conc=8 | Prefix cache win, warm TTFT, decode under load |
| Batch docs · cold prefix | S≈4–8k, cache=low, conc=1 | Model prefill, KV GiB, serving overhead |
| Capacity planning | Mixed | Memcpy / GEMM / NCCL floors before TP quotes |

## Message to send

> Hi Diogo — following up on our conversation about profiling customer traffic before sizing inference. I put together a short discovery page that walks through the five questions and three workload scenarios, backed by live hardware → model → serving numbers: **[YOUR URL]/benchmark-report.html** — start at **Discover**.

Replace `YOUR URL` with your deployed App Platform or GitHub Pages URL after `npm run report && npm run build`.
