#!/usr/bin/env bash
# Full 1k triage + merge loop. Run in WSL with GH_TOKEN exported:
#   export GH_TOKEN=<token>
#   bash triage-1k.sh            (from this repo checkout)
# Resumable: re-running continues from lab-state.json.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=~/llama-pr-lab
cd "$LAB"

if [ -z "${GH_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GH_TOKEN not set. Create one at https://github.com/settings/tokens"
  echo "(classic, no scopes needed for public repos), then: export GH_TOKEN=xxx"
  exit 1
fi

echo "== stage 1+2: triage 1000 open PRs, detail top 200 (with CI status) =="
python3 -u "$HERE/fetch_prs.py" --limit 1000 --top 200 --out candidates-1k.json --include-ci

echo "== merge loop: 10-PR batches until candidates exhausted =="
python3 -u "$HERE/merge_lab.py" \
  --candidates candidates-1k.json \
  --repo ~/llama-pr-lab/llama.cpp \
  --state-dir ~/llama-pr-lab/state \
  --batch 10 --max-prs 100000 \
  --smoke-model ~/llama-pr-lab/models/tinyllama.gguf \
  --bench-model ~/llama-pr-lab/models/qwen2.5-7b-00001-of-00002.gguf \
  --pp 32 --tg 32 --regression-pct 15 2>&1 | tee run-1k.log
