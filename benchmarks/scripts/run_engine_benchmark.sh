#!/usr/bin/env bash
# Phase A engine benchmark on a GPU host (DigitalOcean droplet, RunPod, etc.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${BENCH_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROFILE="${BENCH_PROFILE:-qwen7b_bf16}"
BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OUT="${BENCH_OUT:-$ROOT/benchmarks/results/engine_runs.csv}"
LLMPERF_DIR="${LLMPERF_DIR:-$ROOT/.tools/llmperf}"
MODELS_YAML="$ROOT/benchmarks/config/models.yaml"

echo "== KV Cache App · Engine benchmark =="
echo "Model:   $MODEL"
echo "Profile: $PROFILE"
echo "API:     $BASE_URL"
echo "Out:     $OUT"

if ! curl -sf "${BASE_URL%/v1}/health" >/dev/null 2>&1 && ! curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
  echo ""
  echo "Start vLLM first (on your DO GPU droplet):"
  echo "  bash benchmarks/scripts/setup_gpu_stack.sh"
  echo "  bash benchmarks/scripts/serve_vllm.sh $PROFILE"
  echo ""
  exit 1
fi

if [[ ! -d "$LLMPERF_DIR" ]]; then
  echo "Cloning llmperf into $LLMPERF_DIR ..."
  git clone --depth 1 https://github.com/run-ai/llmperf.git "$LLMPERF_DIR"
  pip install -q -r "$LLMPERF_DIR/requirements.txt" 2>/dev/null || pip install ray pandas numpy tqdm
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
touch "$OUT"

run_case() {
  local phase="$1" layer="$2" inp="$3" out="$4" conc="$5" workload="$6"
  local tag="p${phase}_in${inp}_out${out}_c${conc}_${workload}"
  local dir="$ROOT/benchmarks/results/raw/$tag"
  mkdir -p "$dir"

  echo "→ $tag"
  python3 "$LLMPERF_DIR/token_benchmark_ray.py" \
    --model "$MODEL" \
    --llm-api openai \
    --base-url "$BASE_URL" \
    --mean-input-tokens "$inp" --stddev-input-tokens 0 \
    --mean-output-tokens "$out" --stddev-output-tokens 0 \
    --num-concurrent-requests "$conc" \
    --max-num-completed-requests $((conc * 50)) \
    --results-dir "$dir" \
    --timeout 1800 || true

  python3 - "$OUT" "$phase" "$layer" "$inp" "$out" "$conc" "$workload" "$dir" "$MODEL" "$PROFILE" "$MODELS_YAML" <<'PY'
import csv, json, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

(out, phase, layer, inp, out_t, conc, workload, raw,
 model, profile, models_yaml) = sys.argv[1:12]
raw = Path(raw)
ttft = tpot = tok_s = 0.0
for p in raw.glob("*.json"):
    try:
        d = json.loads(p.read_text())
        ttft = d.get("ttft_ms", d.get("mean_ttft_ms", ttft)) or ttft
        tpot = d.get("tpot_ms", d.get("mean_tpot_ms", tpot)) or tpot
        tok_s = d.get("output_throughput", d.get("mean_output_tokens_per_s", tok_s)) or tok_s
    except Exception:
        pass

seq = int(inp)
layers = n_kv = head_dim = hidden = None
tp = quant = ""
engine = "vLLM+llmperf"
try:
    import yaml
    cfg = yaml.safe_load(Path(models_yaml).read_text())
    prof = cfg.get("profiles", {}).get(profile, {})
    tp = str(prof.get("tensor_parallel", 1))
    quant = prof.get("quantization") or ""
    kv = cfg.get("kv_params", {}).get(model) or cfg.get("kv_params", {}).get(
        prof.get("model", model), {}
    )
    layers = kv.get("layers")
    n_kv = kv.get("n_kv")
    head_dim = kv.get("head_dim")
    hidden = kv.get("hidden")
    if tp and int(tp) > 1:
        engine = f"vLLM+llmperf+TP{tp}"
    if quant:
        engine += f"+{quant.upper()}"
except Exception:
    pass

if layers and n_kv and head_dim and hidden:
    kv_gqa = 2 * layers * seq * n_kv * head_dim * 2 / (1024**3)
    kv_dense = 2 * layers * seq * hidden * 2 / (1024**3)
else:
    kv_gqa = 2 * 28 * seq * 4 * 128 * 2 / (1024**3)
    kv_dense = 2 * 28 * seq * 3584 * 2 / (1024**3)

row = {
    "run_id": f"eng-{uuid.uuid4().hex[:8]}",
    "phase": phase, "stack_layer": "serving", "layer": layer, "provider": "digitalocean_gpu",
    "model": model, "engine": engine,
    "input_tokens_target": inp, "output_tokens_target": out_t,
    "concurrency": conc, "workload": workload, "cache_mode": "off",
    "session_affinity": "", "ttft_ms_p50": ttft, "ttft_ms_p95": ttft * 1.2,
    "tpot_ms_p50": tpot, "output_tok_s_p50": tok_s,
    "prompt_tokens": inp, "completion_tokens": out_t, "cached_prompt_tokens": 0,
    "kv_gib_modeled_gqa": round(kv_gqa, 4), "kv_gib_modeled_dense": round(kv_dense, 4),
    "tensor_parallel": tp, "quantization": quant,
    "notes": f"profile={profile} llmperf {raw.name}",
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
path = Path(out)
new = not path.exists() or path.stat().st_size == 0
with path.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if new:
        w.writeheader()
    w.writerow(row)
PY
}

for inp in 1024 8192 10240; do
  for out in 128 512; do
    for conc in 1 8; do
      [[ "$inp" == 1024 && "$out" == 512 && "$conc" == 8 ]] && continue
      run_case A engine "$inp" "$out" "$conc" single
    done
  done
done

python3 "$ROOT/benchmarks/scripts/generate_report.py"
echo "Done. Open /benchmark-report.html after npm run build"
