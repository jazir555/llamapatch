#!/usr/bin/env python3
"""Fetch open llama.cpp PRs and triage to perf-relevant merge candidates.

Stage 1 (cheap, 1 API call per 100 PRs): list endpoint only, keyword score.
Stage 2 (expensive, 2 calls per PR): detail + files, only for top candidates.

Usage (in WSL):
  GH_TOKEN=xxx python3 fetch_prs.py --limit 1000 --top 100 --out candidates.json
  python3 fetch_prs.py --limit 200 --top 50 --out candidates.json   # unauth (60 req/hr)

Output: candidates.json sorted by score desc, each with number/title/score/reasons.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

try:
    from pr_intent import classify_intent
except ImportError:  # pragma: no cover
    def classify_intent(cand):
        return {"backends": [], "area": "other", "expects_bench_gain": False,
                "verifiable_on_cpu": False, "reason": "pr_intent missing"}

API = "https://api.github.com/repos/ggml-org/llama.cpp"

def req(url, token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "llama-pr-lab"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            return json.load(resp), resp.headers
    except urllib.error.HTTPError as e:
        if e.code == 403 and "rate" in str(e.headers).lower():
            reset = e.headers.get("X-RateLimit-Reset", "?")
            print(f"RATE LIMITED. Reset at {reset}. Set GH_TOKEN to raise limit.", file=sys.stderr)
        raise

def score_title(title, body, cfg):
    t = ((title or "") + " " + (body or "")).lower()
    hits = [k for k in cfg["perf_keywords"] if k.lower() in t]
    return len(hits), hits

def is_rate_limit(exc):
    """Pure: True when exc is a GitHub rate-limit response: 429, or 403
    with an exhausted quota (X-RateLimit-Remaining: 0) or an explicit
    rate-limit message. Other 403s (e.g. per-PR access issues) return
    False so one bad PR can't abort the whole triage. On True the caller
    saves partial output and exits 2 instead of sleeping 30s per PR."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return True
        if exc.code == 403:
            try:
                rem = exc.headers.get("X-RateLimit-Remaining") if exc.headers else None
                if rem is not None and int(rem) == 0:
                    return True
            except Exception:
                pass
            return "rate limit" in str(getattr(exc, "msg", "") or "").lower()
    return False

def touches_perf(paths, cfg):
    """Prefix-aware perf-path check (no substring false positives)."""
    norm = [p.strip("/").lower() for p in paths]
    for pp in cfg["perf_paths"]:
        base = pp.strip("/").lower().rstrip("/")
        for p in norm:
            if p == base or p.startswith(base + "/"):
                return True
    return False

def fetch_ci_state(head_sha, token):
    """Combined commit status for a PR head. Returns (state, total)."""
    try:
        data, _ = req(f"{API}/commits/{head_sha}/status", token)
        return data.get("state"), len(data.get("statuses", []))
    except Exception as e:
        return f"error:{e}"[:120], 0

def collect_pages(fetch_one, limit, sleep=time.sleep, retries=3):
    """Resilient pagination: fetch_one(page) -> (items, is_last).

    Transient failures retry with exponential backoff (1s, 2s, 4s);
    persistent failure returns partial items + the error instead of
    discarding everything fetched so far. Returns (items, error)."""
    items, page, error = [], 1, None
    while len(items) < limit:
        for attempt in range(retries + 1):
            try:
                data, is_last = fetch_one(page)
                break
            except Exception as e:
                if attempt >= retries:
                    error = e
                    data, is_last = None, True
                else:
                    sleep(2 ** attempt)
        if error is not None:
            break
        items.extend(data or [])
        if is_last or not data:
            break
        page += 1
    return items[:limit], error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--top", type=int, default=100, help="how many go to stage-2 detail fetch")
    ap.add_argument("--out", default="candidates.json")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--include-ci", action="store_true",
                    help="fetch combined commit status per top-PR (extra API calls)")
    ap.add_argument("--ci-sleep", type=float, default=0.7)
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    with open(os.path.join(os.path.dirname(__file__), a.config)) as f:
        cfg = json.load(f)

    def _one(page):
        url = f"{API}/pulls?state=open&per_page=100&page={page}"
        data, _ = req(url, token)
        print(f"page {page}: got {len(data or [])}", flush=True)
        if len(data or []) < 100:
            return data or [], True
        time.sleep(1 if token else 3)
        return data, False

    all_prs, page_err = collect_pages(_one, a.limit)
    print(f"total open fetched: {len(all_prs)}", file=sys.stderr)
    if page_err is not None:
        print(f"stage-1 pages failed after retries ({page_err}); "
              f"triaging partial {len(all_prs)} PRs", file=sys.stderr)

    staged = []
    for pr in all_prs:
        labels = [l["name"].lower() for l in pr.get("labels", [])]
        if any(x in labels for x in cfg["exclude_labels"]):
            continue
        if pr.get("draft"):
            continue  # skip drafts in auto-merge; review manually later
        s, hits = score_title(pr.get("title"), pr.get("body"), cfg)
        staged.append({"number": pr["number"], "title": pr["title"],
                       "user": pr["user"]["login"], "updated": pr["updated_at"],
                       "labels": labels, "score": s, "hits": hits,
                       "head": pr["head"]["sha"][:8]})
    staged.sort(key=lambda x: (-x["score"], x["updated"]))
    print(f"stage-1 survivors: {len(staged)} (drafts/excluded removed)")

    # Stage 2: enrich top-N with mergeable + files (+ optional CI status)
    top = staged[:a.top]
    rate_limited = False
    for i, c in enumerate(top):
        n = c["number"]
        try:
            detail, _ = req(f"{API}/pulls/{n}", token)
            c["mergeable"] = detail.get("mergeable")
            c["mergeable_state"] = detail.get("mergeable_state")
            c["additions"] = detail.get("additions"); c["deletions"] = detail.get("deletions")
            c["changed_files"] = detail.get("changed_files")
            c["head_full"] = detail.get("head", {}).get("sha", "")
            files, _ = req(f"{API}/pulls/{n}/files?per_page=100", token)
            paths = [f["filename"] for f in files]
            c["files"] = paths[:20]
            c["touches_perf_path"] = touches_perf(paths, cfg)
            c["intent"] = classify_intent(c)
            if c["touches_perf_path"]:
                c["score"] += 2
                c["hits"] = c["hits"] + ["touches-perf-path"]
            if (c["additions"] or 0) + (c["deletions"] or 0) > 2000:
                c["score"] -= 1
                c["hits"] = c["hits"] + ["large-diff-penalty"]
            if c.get("mergeable_state") == "dirty":
                c["score"] -= 1
                c["hits"] = c["hits"] + ["dirty-penalty"]
            if a.include_ci and c.get("head_full"):
                ci_state, ci_n = fetch_ci_state(c["head_full"], token)
                c["ci_state"] = ci_state
                c["ci_count"] = ci_n
                if ci_state in ("failure", "error"):
                    c["score"] -= 2
                    c["hits"] = c["hits"] + ["ci-red-penalty"]
                print(f"[{i+1}/{len(top)}] #{n} mergeable={c['mergeable_state']} ci={c.get('ci_state')} files={c['changed_files']} score={c['score']}")
            else:
                print(f"[{i+1}/{len(top)}] #{n} mergeable={c['mergeable_state']} files={c['changed_files']} score={c['score']}")
        except urllib.error.HTTPError as e:
            c["error"] = f"HTTP {e.code}: {e.reason}"
            if is_rate_limit(e):
                print(f"[{i+1}/{len(top)}] #{n} RATE LIMITED — saving partial "
                      f"({i}/{len(top)} enriched), re-run later with GH_TOKEN set. "
                      f"Reset at {e.headers.get('X-RateLimit-Reset', '?') if e.headers else '?'}")
                rate_limited = True
                break
            print(f"[{i+1}/{len(top)}] #{n} ERROR {c['error']} (backing off 30s)")
            time.sleep(30)
        except Exception as e:
            c["error"] = str(e)
            print(f"[{i+1}/{len(top)}] #{n} ERROR {e}")
        time.sleep(a.ci_sleep if token else 2.5)

    top.sort(key=lambda x: (-x["score"], x["updated"]))
    with open(a.out, "w") as f:
        json.dump(top, f, indent=2)
    print(f"wrote {a.out} with {len(top)} candidates")
    if rate_limited:
        print("EXIT: rate-limited, partial triage saved. Export GH_TOKEN and re-run.",
              file=sys.stderr)
        sys.exit(2)
    print("Top 15:")
    for c in top[:15]:
        print(f"  #{c['number']} s={c['score']} {c.get('mergeable_state')} {c['title'][:70]}")

if __name__ == "__main__":
    main()
