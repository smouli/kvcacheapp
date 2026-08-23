#!/usr/bin/env bash
# Bootstrap CUDA + NCCL + vLLM on a fresh Linux GPU host (DO GPU droplet, etc.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/venv-kvcache}"
PYTHON="${PYTHON:-python3.11}"

echo "== KV Cache App · GPU stack setup =="
echo "Root:  $ROOT"
echo "Venv:  $VENV"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver + CUDA first."
  echo "  DO GPU droplets: use the CUDA-ready GPU image."
  exit 1
fi

echo ""
echo "--- NVIDIA driver / CUDA ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
echo "GPU count: $GPU_COUNT"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Installing Python 3.11..."
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl
fi

if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo ""
echo "--- PyTorch (CUDA 12.4 wheels) ---"
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "--- vLLM (brings NCCL via PyTorch / CUDA runtime) ---"
pip install "vllm>=0.8.0"

echo ""
echo "--- Benchmark harness deps ---"
pip install -r "$ROOT/benchmarks/requirements.txt"

echo ""
echo "--- NCCL / CUDA sanity (PyTorch) ---"
python3 "$ROOT/benchmarks/scripts/verify_nccl.py" || true

if [[ "$GPU_COUNT" -ge 2 ]]; then
  echo ""
  echo "--- NCCL multi-GPU all-reduce smoke test ---"
  if command -v torchrun >/dev/null 2>&1; then
    NCCL_DEBUG=WARN torchrun --nproc_per_node="$GPU_COUNT" \
      "$ROOT/benchmarks/scripts/nccl_smoke.py" || echo "WARN: NCCL smoke failed — check IB/RDMA/firewall"
  else
    echo "torchrun not on PATH; skip distributed NCCL test"
  fi
fi

echo ""
echo "=== Setup complete ==="
echo "Activate:  source $VENV/bin/activate"
echo "Serve:     bash $ROOT/benchmarks/scripts/serve_vllm.sh qwen7b_bf16"
echo "Profiles:  cat $ROOT/benchmarks/config/models.yaml"
echo "Bench:     bash $ROOT/benchmarks/scripts/run_engine_benchmark.sh"
