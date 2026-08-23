# Full demo checklist (do everything)

Ordered path from laptop → public URL → GPU serving numbers.

## Status snapshot

| Piece | Status |
|-------|--------|
| KV explorer + GQA math | Done |
| Model vs serving layers | Done (`docs/LAYERS.md`) |
| Modal model-layer benches | Mostly done (7B complete; 0.5B @ 10k backfill in progress) |
| GPU stack scripts (NCCL/vLLM/quant) | Done (`docs/GPU_STACK.md`) |
| Live Fireworks API | Needs `OPENAI_API_KEY` |
| DO App Platform URL | Needs GitHub push + DO app |
| DO GPU serving benches | Needs GPU droplet |

---

## Track A — Laptop (now)

```bash
cd ~/projects/kvcacheapp

# A1. Fill missing Modal rows (0.5B @ 10k → A10G)
npm run benchmark:modal -- --models Qwen/Qwen2.5-0.5B-Instruct

# A2. Refresh report + production build
npm run report && npm run build
npm run preview   # http://localhost:4173

# A3. Live API (optional but good)
export OPENAI_API_KEY=fw_...          # Fireworks
export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
export BENCH_MODEL=accounts/fireworks/models/qwen2p5-7b-instruct
npm run benchmark:api
npm run report && npm run build
```

---

## Track B — Public static demo (DO App Platform)

1. Create GitHub repo and push this project.
2. DigitalOcean → **Apps** → Create App → GitHub.
3. Settings:
   - Build: `npm ci && npm run report && npm run build`
   - Output: `dist`
   - Or use Dockerfile (nginx on port 80 / map 8080).
4. Open `https://<app>.ondigitalocean.app/` and `/benchmark-report.html`.

Docker alternative on a cheap Droplet:

```bash
docker build -t kvcacheapp .
docker run -d -p 8080:80 --name kvcache kvcacheapp
```

---

## Track C — DO GPU droplet (serving layer + NCCL)

### C1. Create droplet
- Product: **GPU Droplet**
- Prefer: **1× H100 80GB** (or 2× if you want TP=2 demo)
- Image: Ubuntu + NVIDIA/CUDA-ready GPU image
- SSH key on

### C2. Bootstrap on the droplet

```bash
git clone <YOUR_REPO_URL> kvcacheapp && cd kvcacheapp
bash benchmarks/scripts/setup_gpu_stack.sh
source ~/venv-kvcache/bin/activate
python3 benchmarks/scripts/verify_nccl.py
# if 2+ GPUs:
NCCL_DEBUG=WARN torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
  benchmarks/scripts/nccl_smoke.py
```

### C3. Serve + benchmark

```bash
# Terminal 1 — serve (pick profile)
bash benchmarks/scripts/serve_vllm.sh qwen7b_bf16
# later: qwen7b_fp8 | qwen32b_tp2_bf16 | qwen72b_awq_int4

# Terminal 2 — load test
export BENCH_MODEL='Qwen/Qwen2.5-7B-Instruct'
export BENCH_PROFILE='qwen7b_bf16'
export OPENAI_BASE_URL='http://127.0.0.1:8000/v1'
bash benchmarks/scripts/run_engine_benchmark.sh
```

### C4. Pull results to laptop

```bash
# from laptop
scp root@<GPU_IP>:~/kvcacheapp/benchmarks/results/engine_runs.csv \
  benchmarks/results/
npm run report && npm run build
# redeploy App Platform or rebuild Docker image
```

### C5. Quant / multi-GPU extras (if time)

| Profile | Needs |
|---------|-------|
| `qwen7b_fp8` | H100 |
| `qwen32b_tp2_bf16` | 2 GPUs (NCCL) |
| `qwen72b_awq_int4` | 1× H100 80GB |

---

## Pitch order (5–7 min)

1. Explorer: GQA vs dense (7×) @ S=1000  
2. Report → **Model layer**: Modal TTFT vs S (7B numbers)  
3. Report → **Serving layer**: warm prefix / concurrency story  
4. `docs/GPU_STACK.md`: vLLM + NCCL TP + AWQ/FP8 on DO GPU  
5. Live URL of the app  

---

## Env vars cheat sheet

| Var | Where |
|-----|--------|
| `OPENAI_API_KEY` | Fireworks / OpenAI-compatible |
| `OPENAI_BASE_URL` | Provider or `http://127.0.0.1:8000/v1` |
| `BENCH_MODEL` | HF or provider model id |
| `BENCH_PROFILE` | Name from `benchmarks/config/models.yaml` |
| `HF_TOKEN` | Optional; avoids HF rate limits on Modal/droplet |
