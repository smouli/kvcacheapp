# GPU stack: CUDA, NCCL, vLLM, quantization

How this repo maps to production LLM serving skills (multi-GPU, INT4/FP8, DO droplets).

## Stack diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Serving layer (HTTP, prefix cache, concurrency)             │
│  vLLM OpenAI API · llmperf · run_engine_benchmark.sh         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  vLLM runtime                                                │
│  continuous batching · PagedAttention · prefix caching       │
│  tensor parallel (TP) · pipeline parallel · quantization     │
└────────────────────────────┬─────────────────────────────────┘
                             │ TP > 1
┌────────────────────────────▼─────────────────────────────────┐
│  NCCL (GPU ↔ GPU collectives)                                │
│  all-reduce for sharded weights · KV across ranks            │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  CUDA / cuDNN / driver                                       │
│  bf16/fp16/fp8 kernels · FlashAttention · custom ops         │
└──────────────────────────────────────────────────────────────┘
```

You do **not** call NCCL directly when using vLLM — `--tensor-parallel-size N` spins up worker processes that use NCCL under the hood. You **do** verify NCCL when debugging multi-GPU hangs or slow TP.

---

## Quick start (DigitalOcean GPU droplet)

```bash
git clone <this-repo> && cd kvcacheapp

# 1) Driver + CUDA (DO GPU image usually has this)
nvidia-smi

# 2) Install PyTorch, vLLM, benchmark deps
bash benchmarks/scripts/setup_gpu_stack.sh
source ~/venv-kvcache/bin/activate

# 3) Pick a profile from benchmarks/config/models.yaml
bash benchmarks/scripts/serve_vllm.sh qwen7b_bf16          # 1× GPU
bash benchmarks/scripts/serve_vllm.sh qwen32b_tp2_bf16     # 2× GPU, NCCL
bash benchmarks/scripts/serve_vllm.sh qwen72b_awq_int4       # INT4, 1× H100

# 4) Run serving benchmarks
export BENCH_MODEL='Qwen/Qwen2.5-7B-Instruct'
export BENCH_PROFILE='qwen7b_bf16'
bash benchmarks/scripts/run_engine_benchmark.sh
```

---

## Model profiles

| Profile | Model | GPUs | Quant | ~VRAM | Use when |
|---------|-------|------|-------|-------|----------|
| `qwen7b_bf16` | Qwen2.5-7B | 1 | bf16 | 16 GiB | Baseline, fits A10G/H100 |
| `qwen7b_fp8` | Qwen2.5-7B | 1 | **FP8** | 10 GiB | H100 weight compression |
| `qwen32b_tp2_bf16` | Qwen2.5-32B | **TP=2** | bf16 | 40+ GiB each | NCCL demo on 2× GPU |
| `qwen72b_tp2_bf16` | Qwen2.5-72B | **TP=2** | bf16 | 2× H100 80GB | Full-precision 72B |
| `qwen72b_awq_int4` | Qwen2.5-72B-AWQ | 1 | **INT4 AWQ** | ~48 GiB | Large model, one GPU |
| `qwen72b_awq_int4_tp2` | Qwen2.5-72B-AWQ | **TP=2** | INT4 | ~24 GiB each | Long context headroom |

Edit or add profiles in `benchmarks/config/models.yaml`.

---

## Quantization (job-relevant)

| Method | Bits | vLLM flag | Typical hardware | Tradeoff |
|--------|------|-----------|------------------|----------|
| **bf16/fp16** | 16 | `--dtype bfloat16` | All modern GPUs | Best quality; most VRAM |
| **FP8** | 8 | `--quantization fp8` | **H100**, L40S | ~2× weight savings; slight quality loss |
| **AWQ INT4** | 4 | `--quantization awq` + AWQ checkpoint | H100, A100 | ~4× weights; needs AWQ model on HF |
| **GPTQ INT4** | 4 | `--quantization gptq` | Same | Similar to AWQ; different calib |

**KV cache** is separate from weight quant: vLLM can use `--kv-cache-dtype fp8` on supported GPUs to shrink KV footprint (directly ties to this app's explorer math).

---

## NCCL verification

```bash
source ~/venv-kvcache/bin/activate
python3 benchmarks/scripts/verify_nccl.py

# Multi-GPU only (2+ visible devices):
NCCL_DEBUG=WARN torchrun --nproc_per_node=2 benchmarks/scripts/nccl_smoke.py
```

Common cloud fixes (already set in `serve_vllm.sh` for TP > 1):

```bash
export NCCL_IB_DISABLE=1    # no InfiniBand on most DO VMs
export NCCL_DEBUG=WARN      # surface collective errors
```

---

## Mapping to job requirements

| Requirement | Where in this repo |
|-------------|-------------------|
| **PyTorch / Hugging Face** | `modal_engine.py` (model layer), vLLM serves HF weights |
| **Deploy / fine-tune LLMs** | `serve_vllm.sh` profiles, `docs/DEPLOY_DIGITALOCEAN.md` |
| **Linux / GPU optimization** | `setup_gpu_stack.sh`, `verify_nccl.py`, CUDA wheels |
| **Distributed systems** | Tensor parallel = NCCL multi-process; notes on K8s below |
| **vLLM** | Primary serving engine for Phase A/B benchmarks |
| **INT4 / INT8 / FP8** | AWQ + FP8 profiles in `models.yaml` |
| **Object storage** | HF Hub model download; production: DO Spaces for weights/artifacts |
| **NFS / K8s** | Pattern below for DOKS + GPU node pools |

---

## Production pattern (Kubernetes + NFS + object storage)

Not fully automated in this demo repo, but the architecture you'd describe in an interview:

1. **Object storage (DO Spaces)** — store fine-tuned weights, LoRA adapters, benchmark CSVs.
2. **NFS / shared volume** — mount read-only model weights on GPU nodes (avoid re-pull per pod).
3. **Kubernetes (DOKS)** — GPU node pool; one vLLM Deployment per model profile:
   ```yaml
   # sketch — vLLM pod
   args:
     - serve
     - Qwen/Qwen2.5-72B-Instruct-AWQ
     - --tensor-parallel-size
     - "2"
     - --quantization
     - awq
   resources:
     limits:
       nvidia.com/gpu: 2
   ```
4. **NCCL** — set `NCCL_IB_DISABLE=1` unless RDMA; use headless Service for TP worker discovery (vLLM handles this when launched via their entrypoint).

---

## Modal vs DO GPU

| | Modal (`benchmark:modal`) | DO GPU (`serve_vllm.sh`) |
|--|---------------------------|---------------------------|
| Purpose | **Model layer** — KV math, prefill TTFT | **Serving layer** — vLLM, TP, quant, load |
| Multi-GPU | Single GPU per function today | Full TP / NCCL via vLLM |
| Quantization | bf16/fp16 only (transformers) | FP8, AWQ, GPTQ via vLLM |
| Cost | Pay per second | Droplet hourly |

Use **Modal** for cheap architecture/KV experiments; use **DO GPU + vLLM** for the manager-facing serving story with NCCL and quantization.
