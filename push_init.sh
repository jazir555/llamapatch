#!/usr/bin/env bash
# Initial push of the lab into jazir555/llamapatch (empty repo).
# Auth entirely at runtime from ~/.config/llama-pr-lab/token; no secret here.
set -euo pipefail
SRC=/mnt/c/Users/mmeadow/Documents/LLMDroid/pr-lab
DST=/home/mmeadow/llamapatch-push
TOKFILE="$HOME/.config/llama-pr-lab/token"
CREDFILE="$HOME/.config/llama-pr-lab/git-cred"
test -s "$TOKFILE" || { echo "missing $TOKFILE"; exit 1; }
GH_TOKEN=$(cat "$TOKFILE")
export GH_TOKEN

rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC"/README.md "$SRC"/candidates-sample.json "$SRC"/check_token.sh \
   "$SRC"/config.json "$SRC"/create_repo.sh "$SRC"/fetch_models.sh \
   "$SRC"/fetch_prs.py "$SRC"/merge_lab.py "$SRC"/push_init.sh \
   "$SRC"/triage-1k.sh "$DST"/
chmod 644 "$DST"/README.md "$DST"/candidates-sample.json "$DST"/config.json
printf '%s\n' '*token*' '*.pem' '*.key' '.env' '__pycache__/' > "$DST/.gitignore"

cd "$DST"
git init -b main 2>/dev/null || { git init; git checkout -b main; }
git add -A
git -c user.name="jazir555" -c user.email="techtutormm@gmail.com" \
  commit -m "Initial commit: llama.cpp PR mass-merge lab

- fetch_prs.py: 2-stage triage of open PRs by perf keywords/paths
- merge_lab.py: one verified commit per PR, build/smoke/bench gates,
  auto-quarantine with bisect, --doctor state reconciliation
- 22 upstream PRs merged with clean-base parity (see README results)"
git remote add origin https://github.com/jazir555/llamapatch.git

printf 'https://%s:%s@github.com\n' "jazir555" "$GH_TOKEN" > "$CREDFILE"
chmod 600 "$CREDFILE"
unset GH_TOKEN
git -c credential.helper="store --file $CREDFILE" push -u origin main
echo "PUSH-OK"
