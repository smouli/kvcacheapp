#!/usr/bin/env bash
# Run all unattended benchmarks available from a laptop (Modal + demo data).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/benchmarks/results/overnight.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "Overnight run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(uname -a)"
echo "=============================================="

cd "$ROOT"

echo ""
echo ">>> [1/4] Demo API benchmark (dry-run, no key needed)"
npm run benchmark:api:dry || echo "WARN: api dry-run failed"

echo ""
echo ">>> [2/4] Full Modal model-layer matrix (0.5B + 7B, all S shapes)"
echo "    This is the long step — each 7B/A10G case may take several minutes."
npm run benchmark:modal || echo "WARN: modal benchmark failed"

echo ""
echo ">>> [3/4] Regenerate HTML report"
npm run report

echo ""
echo ">>> [4/4] Production build"
npm run build

echo ""
echo "=============================================="
echo "Overnight run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log:     $LOG"
echo "CSV:     $ROOT/benchmarks/results/modal_runs.csv"
echo "Report:  $ROOT/public/benchmark-report.html"
echo "=============================================="

# Summary counts
python3 <<'PY'
import csv
from pathlib import Path
root = Path(".")
rows = []
for p in [root/"benchmarks/sample_results/demo_runs.csv",
          root/"benchmarks/results/modal_runs.csv"]:
    if p.exists():
        with p.open() as f:
            rows.extend(list(csv.DictReader(f)))
model = sum(1 for r in rows if r.get("stack_layer")=="model" or r.get("provider")=="modal")
serving = sum(1 for r in rows if r.get("stack_layer")=="serving" or (r.get("provider") or "") not in ("modal","") and r.get("stack_layer")!="model")
print(f"Total CSV rows: {len(rows)} (approx model={model}, serving={serving})")
PY
