# Deploy on DigitalOcean

End-to-end demo: **static KV explorer + benchmark report** on App Platform or a Droplet, **vLLM benchmarks** on a **GPU Droplet**.

## 1. Static app (App Platform or Droplet)

### Option A — App Platform (recommended for demo URL)

1. Push this repo to GitHub.
2. DigitalOcean → **Apps** → Create → GitHub repo.
3. Build command: `npm ci && npm run report && npm run build`
4. Output directory: `dist`
5. HTTP port: `8080` if using Docker; for static site use DO static component pointing at `dist`.

### Option B — Docker on a Droplet

```bash
docker build -t kvcacheapp .
docker run -d -p 8080:80 --name kvcache kvcacheapp
```

Open `http://<droplet-ip>:8080` and `/benchmark-report.html`.

## 2. GPU droplet for Phase A benchmarks

### Bootstrap CUDA + NCCL + vLLM

```bash
git clone <repo> && cd kvcacheapp
bash benchmarks/scripts/setup_gpu_stack.sh
source ~/venv-kvcache/bin/activate
python3 benchmarks/scripts/verify_nccl.py
```

See **`docs/GPU_STACK.md`** for tensor parallel (NCCL), FP8/AWQ quantization, and K8s notes.

### Serve a model profile

Profiles live in `benchmarks/config/models.yaml`:

```bash
# 7B bf16 — 1 GPU
bash benchmarks/scripts/serve_vllm.sh qwen7b_bf16

# 32B bf16 — 2 GPUs, NCCL tensor parallel
bash benchmarks/scripts/serve_vllm.sh qwen32b_tp2_bf16

# 72B INT4 AWQ — fits single H100
bash benchmarks/scripts/serve_vllm.sh qwen72b_awq_int4
```

### Run benchmarks

```bash
export BENCH_MODEL='Qwen/Qwen2.5-7B-Instruct'
export BENCH_PROFILE='qwen7b_bf16'
bash benchmarks/scripts/run_engine_benchmark.sh
```

Legacy one-liner (manual vLLM):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-prefix-caching
```

## 3. Firewall

- App droplet: allow **80/443** public.
- GPU droplet: restrict **8000** to bastion IP only (do not expose vLLM publicly without auth).

## 4. Cost-aware demo flow for managers

| Step | Time | Show |
|------|------|------|
| Open explorer | 2 min | GQA 54 MiB vs dense 383 MiB @ S=1000 |
| Open benchmark report | 3 min | TTFT vs 10k context, 4.7× warm prefix |
| Mention DO GPU | 1 min | Same matrix runnable on droplet via `run_engine_benchmark.sh` |

## 5. Environment variables

| Var | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | Fireworks or compatible API |
| `OPENAI_BASE_URL` | e.g. `https://api.fireworks.ai/inference/v1` |
| `BENCH_MODEL` | HF or Fireworks model id |
| `BENCH_PROFILE` | vLLM profile from `benchmarks/config/models.yaml` |
| `BENCH_OUT` | CSV output path |

## 6. Troubleshooting

- **Port 5173 busy:** dev server uses **5174** (`vite.config.js`).
- **vLLM OOM:** lower `--max-model-len` or batch in matrix.
- **Empty report:** run `npm run report` or `python benchmarks/scripts/generate_report.py`.
