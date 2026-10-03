#!/usr/bin/env bash
# Creates the llamapatch repo under the token owner's account.
# Auth happens entirely at runtime from ~/.config/llama-pr-lab/token;
# this file carries no secret. Usage: bash create_repo.sh [name]
set -euo pipefail
NAME="${1:-llamapatch}"
TOKFILE="$HOME/.config/llama-pr-lab/token"
test -s "$TOKFILE" || { echo "missing $TOKFILE"; exit 1; }
GH_TOKEN=$(cat "$TOKFILE")
export GH_TOKEN

LOGIN=$(curl -s -H "Authorization: Bearer $GH_TOKEN" https://api.github.com/user \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('login',''))")
if [ -z "$LOGIN" ]; then echo "auth failed"; exit 1; fi
echo "owner: $LOGIN"

RESP=$(curl -s -w "\n%{http_code}" -X POST \
  -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$NAME\",\"private\":false,\"description\":\"Automated llama.cpp PR mass-merge lab: triage, verified per-PR merges, self-heal, bench gates\"}")
CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | head -n -1)
echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print('full_name:',d.get('full_name')); print('clone_url:',d.get('clone_url')); print('errors:',d.get('errors'))" 2>/dev/null || echo "$BODY" | head -c 500
echo "HTTP:$CODE"
