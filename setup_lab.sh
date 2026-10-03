#!/usr/bin/env bash
# Provision a PR-merge worktree WITHOUT a full clone (run in WSL).
#
# A full llama.cpp clone pulls every historical blob; merging PRs only
# needs the base tree, each PR's diff, and enough history for merge-base.
# So this clones:
#   --depth N           shallow history (merge-base for active PRs is recent)
#   --filter=blob:none  partial clone: file contents fetched on demand
#   sparse checkout      optional working-tree subset (faster checkout)
# PR refs are fetched per-PR later by merge_lab.py (full ref, but blobless,
# so only commits+trees travel until checkout/merge needs the contents).
#
# Usage:
#   bash setup_lab.sh [repo-url] [dest-dir]
#   CLONE_DEPTH=200 SPARSE_DIRS="src ggml common" LAB_BASE=master bash setup_lab.sh
#
# If a merge ever fails on a shallow boundary, merge_lab.py deepens
# automatically (--deepen) and retries once.
set -euo pipefail
URL="${1:-https://github.com/ggml-org/llama.cpp.git}"
DEST="${2:-$HOME/llama-pr-lab/llama.cpp}"
DEPTH="${CLONE_DEPTH:-100}"
BASE="${LAB_BASE:-master}"
SPARSE="${SPARSE_DIRS:-}"

if [ -d "$DEST/.git" ]; then
  echo "exists (skip clone): $DEST"
  cd "$DEST"
  git rev-parse --is-shallow-repository
  exit 0
fi

echo "== partial shallow clone (depth=$DEPTH, blob:none) =="
echo "   $URL -> $DEST"
git clone --depth "$DEPTH" --filter=blob:none --no-checkout "$URL" "$DEST"
cd "$DEST"

if [ -n "$SPARSE" ]; then
  # shellcheck disable=SC2086
  echo "== sparse checkout: $SPARSE =="
  git sparse-checkout init --cone
  # shellcheck disable=SC2086
  git sparse-checkout set $SPARSE
fi

git checkout "$BASE"
echo "== ready =="
git rev-parse --is-shallow-repository
git count-objects -v -H
