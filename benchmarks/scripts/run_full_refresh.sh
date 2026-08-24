#!/usr/bin/env bash
# Full refresh: Modal hardware → model → vLLM + SGLang serving → report → build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/benchmarks/results/full_refresh.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "Full refresh started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(uname -a)"
echo "=============================================="

cd "$ROOT"

echo ""
echo ">>> [1/6] Hardware — CUDA memcpy/GEMM + NCCL on Modal A100:2"
npm run benchmark:modal:hardware:nccl || echo "WARN: hardware/NCCL failed"

echo ""
echo ">>> [2/6] Model layer — transformers on Modal (T4/A10G)"
npm run benchmark:modal || echo "WARN: model layer failed"

echo ""
echo ">>> [3/6] Serving — vLLM TP=1 on Modal A100 (7B · prefix + concurrency)"
npm run benchmark:modal:vllm || echo "WARN: vLLM TP=1 failed"

echo ""
echo ">>> [4/6] Serving — SGLang TP=1 on Modal A100 (same matrix as vLLM)"
npm run benchmark:modal:sglang || echo "WARN: SGLang TP=1 failed"

echo ""
echo ">>> [5/6] Serving — vLLM TP=2 on Modal A100×2 (32B · NCCL tensor parallel)"
npm run benchmark:modal:vllm:tp2 || echo "WARN: vLLM TP=2 failed"

echo ""
echo ">>> [6/6] Regenerate report + production build"
npm run report && npm run build

echo ""
echo "=============================================="
echo "Full refresh finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log:     $LOG"
echo "CSVs:    $ROOT/benchmarks/results/"
echo "Data:    $ROOT/public/benchmark-data.json"
echo "Report:  $ROOT/dist/benchmark-report.html"
echo "=============================================="
