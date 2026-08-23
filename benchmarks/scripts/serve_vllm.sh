#!/usr/bin/env bash
# Start vLLM with a named profile (tensor parallel / quantization / dtype).
# Usage:
#   bash benchmarks/scripts/serve_vllm.sh qwen7b_bf16
#   bash benchmarks/scripts/serve_vllm.sh qwen32b_tp2_bf16
#   BENCH_PROFILE=qwen72b_awq_int4 bash benchmarks/scripts/serve_vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${1:-${BENCH_PROFILE:-qwen7b_bf16}}"
MODELS_YAML="$ROOT/benchmarks/config/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
  echo "Missing $MODELS_YAML"
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM not found. Run: bash benchmarks/scripts/setup_gpu_stack.sh"
  exit 1
fi

read -r MODEL TP DTYPE QUANT MAX_LEN PREFIX GPU_UTIL GPU_COUNT NOTES <<EOF
$(python3 - "$MODELS_YAML" "$PROFILE" <<'PY'
import sys
import yaml

path, profile = sys.argv[1:3]
cfg = yaml.safe_load(open(path))
if profile not in cfg.get("profiles", {}):
    avail = ", ".join(sorted(cfg["profiles"]))
    raise SystemExit(f"Unknown profile '{profile}'. Available: {avail}")

p = cfg["profiles"][profile]
d = cfg.get("defaults", {})
print(
    p["model"],
    p.get("tensor_parallel", 1),
    p.get("dtype", "bfloat16"),
    p.get("quantization") or "",
    p.get("max_model_len", 32768),
    "1" if p.get("enable_prefix_caching", d.get("enable_prefix_caching", False)) else "0",
    p.get("gpu_memory_utilization", d.get("gpu_memory_utilization", 0.92)),
    p.get("gpu_count", p.get("tensor_parallel", 1)),
    p.get("notes", ""),
)
PY
)
EOF

GPU_COUNT_ACTUAL="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT_ACTUAL" -lt "$TP" ]]; then
  echo "ERROR: profile '$PROFILE' needs tensor_parallel=$TP but only $GPU_COUNT_ACTUAL GPU(s) visible."
  exit 1
fi

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

ARGS=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_LEN"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
)

if [[ "$PREFIX" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$QUANT" ]]; then
  ARGS+=(--quantization "$QUANT")
fi

# NCCL tuning for multi-GPU on cloud VMs (no InfiniBand)
if [[ "$TP" -gt 1 ]]; then
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  echo "NCCL env: IB_DISABLE=$NCCL_IB_DISABLE P2P_DISABLE=$NCCL_P2P_DISABLE"
fi

echo "== vLLM serve · profile: $PROFILE =="
echo "Model:  $MODEL"
echo "TP:     $TP (NCCL)"
echo "Dtype:  $DTYPE"
echo "Quant:  ${QUANT:-none}"
echo "Notes:  $NOTES"
echo ""
echo "Export for benchmarks:"
echo "  export BENCH_MODEL='$MODEL'"
echo "  export BENCH_PROFILE='$PROFILE'"
echo "  export OPENAI_BASE_URL='http://127.0.0.1:$PORT/v1'"
echo ""

exec vllm "${ARGS[@]}"
