#!/usr/bin/env bash
# Create GitHub repo + push. Non-interactive.
#   bash scripts/bootstrap_github.sh [repo-name] [--public] [--owner USER]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="kvcacheapp"
VISIBILITY="--private"
OWNER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --public) VISIBILITY="--public"; shift ;;
    --owner) OWNER="$2"; shift 2 ;;
    -*) echo "Unknown flag: $1"; exit 1 ;;
    *) NAME="$1"; shift ;;
  esac
done

cd "$ROOT"

if [[ ! -d .git ]]; then
  git init -b main
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "$(cat <<'EOF'
Initial KV cache explorer and inference benchmark harness.

Includes GQA sizing UI, model vs serving layer benchmarks, Modal GPU runs,
vLLM/NCCL/quant profiles, and DigitalOcean deploy docs.
EOF
)"
fi

TARGET="${OWNER:+$OWNER/}$NAME"
if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin already set: $(git remote get-url origin)"
  git push -u origin HEAD
else
  gh repo create "$TARGET" $VISIBILITY --source=. --remote=origin --push
fi

echo ""
echo "Repo: $(gh repo view --json url -q .url)"
echo "Next — DigitalOcean Apps → Create → this GitHub repo"
echo "  Build:  npm ci && npm run report && npm run build"
echo "  Output: dist"
