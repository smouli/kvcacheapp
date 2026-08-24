# Benchmark stack: hardware → model → serving

Inference numbers are ambiguous without the GPU underneath. This project separates **three tiers**.

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
└───────────────────────────┬─────────────────────────────────┘
                            │ sits on top of
┌───────────────────────────▼─────────────────────────────────┐
│  HARDWARE LAYER — raw silicon capability                    │
│  memcpy GB/s · GEMM TFLOPS · NCCL busbw · VRAM · SM count  │
│  Tools: Modal GPU microbench (PyTorch)                      │
└─────────────────────────────────────────────────────────────┘
```

## Hardware layer

**Question:** What is this GPU *actually* delivering for bandwidth, compute, and multi-GPU collectives?

| Probe | Why it matters |
|-------|----------------|
| Device→device memcpy GB/s | Decode is often memory-bound (KV reads) |
| BF16 / FP16 GEMM TFLOPS | Prefill / large matmuls |
| NCCL all-reduce busbw | Tensor-parallel scale-out cost |
| VRAM / SM count / CC | Capacity and architecture class |

```bash
npm run benchmark:modal:hardware          # T4 + A10G + A100
npm run benchmark:modal:hardware:quick    # T4 only
npm run benchmark:modal:hardware:nccl     # + A100:2 NCCL
```

CSV: `stack_layer=hardware` → `benchmarks/results/hardware_runs.csv`.

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

```bash
npm run benchmark:modal:quick
npm run benchmark:modal
```

## Serving layer

**Question:** Under **real serving** (API or engine with batching), what do clients see — including **prefix cache** and **concurrency**?

```bash
npm run benchmark:modal:vllm
npm run benchmark:modal:vllm:tp2   # 32B · NCCL · A100×2
```

## Side-by-side

| | **Hardware** | **Model** | **Serving** |
|--|--------------|-----------|-------------|
| **Question** | How fast is the GPU? | How expensive is this arch/S? | What do clients see? |
| **Batch** | n/a | 1 | many |
| **Network** | no | no | yes (API) / local (engine) |
| **Prefix cache** | n/a | off | on (warm vs cold) |
