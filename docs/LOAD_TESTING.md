# How load / concurrency is measured

**There is no Locust script in this repo today.**

## What we run on Modal (primary path)

| Script | Mechanism | HTTP? | Concurrency |
|--------|-----------|-------|-------------|
| `modal_engine.py` | HuggingFace `model.generate`, batch=1 | No | Always 1 |
| `modal_vllm.py` | vLLM offline `LLM.generate(prompts[])` | No | N prompts in one batch (1/4/8) |
| `modal_hardware.py` | PyTorch memcpy/GEMM; NCCL all-reduce | No | N/A |

**Concurrency in `modal_vllm.py`** = pass multiple prompts into one `llm.generate()` call. vLLM continuous-batching handles them on **one GPU** (or TP group). This is **not** Locust and **not** separate HTTP clients — it's an in-process batch proxy for production load shape.

**NCCL / TP** = `--tensor-parallel-size N` splits **one model** across N GPUs on the same Modal container (`A100:2`, `H100:8`, etc.). Separate from concurrency.

## Optional HTTP load (not wired to Modal yet)

| Tool | Where | Use |
|------|-------|-----|
| **llmperf** | `benchmarks/scripts/run_engine_benchmark.sh` | Ray-based; hits OpenAI `/v1/chat/completions`; sweeps in/out/conc |
| **httpx loop** | `run_api_benchmark.py` (legacy) | Sequential API calls |

To benchmark **SGLang on Modal** like a real server:

1. Start `sglang.launch_server` in a Modal function (`H100:8`, `--tp 8`).
2. Point **llmperf** or a thin httpx driver at the ephemeral URL.
3. Same matrix as vLLM (in/out/conc/workload).

Locust is possible but redundant if llmperf already sweeps token shapes; add Locust only if you need custom user scenarios (ramp-up, mixed endpoints).

## Planned: engine comparison

```
Same matrix × {vLLM offline, SGLang offline, SGLang HTTP + llmperf}
Same models from benchmarks/config/catalog.yaml
```

See `benchmarks/config/catalog.yaml` for model × TP × GPU targets (GLM, Kimi, MiniMax, Muse).
