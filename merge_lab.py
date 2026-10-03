#!/usr/bin/env python3
"""Automated mass-merge lab: merge PR batches (10 at a time), build, smoke, bench, self-heal.

Design for 1k+ open PRs: never merge blind. Every batch is:
  1. git fetch origin pull/N/head:pr/N  (no API token needed, no fork remotes)
  2. git merge --no-ff pr/N + amend 'pr-lab: merge #N <title>' (per-PR commit)
  3. on conflict -> revert, quarantine, try next (self-heal #1)
  4. cmake build (ccache if present)
  5. on build fail -> revert, quarantine bad (self-heal #2)
  6. smoke: llama-bench pp32/tg32 tiny model; optional llama-perplexity gate
  7. on runtime fail / >threshold regression -> revert + quarantine (self-heal #3)
  8. bench improvement logged when >threshold above baseline (proves the fix
     helped); empty (already-upstream) merges recorded as noop, not gated.

Usage (in WSL, from llama.cpp checkout):
  python3 merge_lab.py --candidates candidates.json --base master --batch 10 --max-prs 50
  python3 merge_lab.py --candidates candidates.json --dry-run   # plan only, no builds

State files (in lab dir): lab-state.json, lab-log.jsonl, quarantined.json
Branching: starts from --base, creates pr-lab/base-<sha>, each good PR commits
on top. To resume: re-run, it loads lab-state.json.

Bisect: not used. Each PR is merged and gated alone on top of good-so-far,
so a bad PR reverts cleanly without losing the rest of its 10-batch.
Interaction failures (A+B fail but each passes alone) are logged as
batch-overlap-warn for manual review — pairs are never retried (avoids
combinatorial explosion across 1k PRs).
"""
import argparse, json, os, re, subprocess, time, datetime

try:
    from pr_intent import classify_intent, verdict_for
except ImportError:  # pragma: no cover - standalone fallback
    def classify_intent(cand):
        return {"backends": [], "area": "other", "expects_bench_gain": False,
                "verifiable_on_cpu": False, "reason": "pr_intent missing"}
    def verdict_for(intent, base, val, regression_pct):
        if not base or not val:
            return "no-baseline"
        if val < base * (1 - regression_pct / 100):
            return "regression"
        if val > base * (1 + regression_pct / 100):
            return "improvement"
        return "parity"

PURE_HELPERS = True  # marker: plan_batches/parse_bench/should_skip_ci are import-safe


def plan_batches(pending, batch, max_prs):
    """Pure batch planner: chunk pending into batches of `batch`, cap max_prs."""
    capped = pending[:max_prs]
    return [capped[i:i + batch] for i in range(0, len(capped), batch)]


def parse_bench_output(out):
    """Pure bench parser: returns tg throughput float or None.

    Handles: JSON list, {"results": [...]}, log-prefixed JSON, trailing
    non-JSON lines, and regex fallbacks ("avg_throughput"/tok-s).
    tg row = dict with n_gen>0 and avg_ts.
    """
    val = None
    try:
        start = out.rfind("[")
        obj = None
        if start >= 0:
            try:
                obj = json.loads(out[start:])
            except Exception:
                end = out.rfind("]")
                if end > start:
                    obj = json.loads(out[start:end + 1])
        rows = obj.get("results", obj) if isinstance(obj, dict) else obj
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and (row.get("n_gen") or 0) > 0 and row.get("avg_ts"):
                    val = float(row["avg_ts"])
                    break
    except Exception:
        val = None
    if val is None:
        for pat in (r'"avg_throughput"\s*:\s*([\d.]+)', r'"throughput"\s*:\s*([\d.]+)',
                    r'([\d.]+)\s*tok/s'):
            m = re.search(pat, out)
            if m:
                try:
                    val = float(m.group(1))
                    break
                except ValueError:
                    continue
    return val


def should_skip_ci(cand, skip_flag):
    """Pure CI-red decision: skip only when flagged and state is red."""
    return bool(skip_flag) and (cand or {}).get("ci_state") in ("failure", "error")


def sanitize_merge_msg(n, title):
    """Shell-safe one-line merge message. Strips newlines/quotes/`$`."""
    clean = re.sub(r"[\r\n'\x60$\"\\;|&<>!()]+", " ", title or "")
    clean = re.sub(r"\s+", " ", clean).strip()[:100]
    return f"pr-lab: merge #{n} {clean}".strip()


def doctor_match(subjects, n):
    """True when a merge commit for PR n exists (v4 '#N' or legacy 'pr/N')."""
    return f"#{n}" in subjects or f"pr/{n}" in subjects


def is_transient_quarantine(reason):
    """True for quarantine reasons worth retrying on a later run: failed
    fetches and hung git reflect network/weather, not PR content. A truly
    dead PR (deleted fork) simply re-quarantines next run — self-stabilizing,
    unlike permanent merge-conflict quarantines."""
    r = reason or ""
    return r == "fetch-failed" or r.startswith("git-timeout") or r.startswith("git-error")


def gate_error_reason(exc):
    """Pure mapping: hung gate -> 'timeout', missing binary/cwd -> 'infra-error'."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, OSError):
        return "infra-error"
    return "error"


def lock_path(state_dir):
    return os.path.join(state_dir, "lab.lock")


def acquire_lock(state_dir):
    """Fail-fast mutual exclusion for one state-dir.

    Two concurrent runs (cron + manual, two shells) sharing lab-state.json
    would interleave merges/quarantines and corrupt state. The lock file
    holds pid+timestamp for forensics. Stale locks (killed run) must be
    removed by the operator after verifying no run is active — never
    auto-stolen, since stealing risks the corruption this prevents.
    """
    lp = lock_path(state_dir)
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"lab lock exists: {lp} — another run may be active; "
            f"remove it only after verifying no merge loop is running")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": os.getpid(),
                            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}))


def release_lock(state_dir):
    try:
        os.remove(lock_path(state_dir))
    except FileNotFoundError:
        pass


def build_report(state, quar, cands):
    """Pure Markdown report: what merged, what each PR proved, what got
    quarantined and why. Evidence for the 'did it improve what it claimed'
    review after every 10-batch. Never raises on missing keys."""
    state = state or {}
    quar = quar or []
    cands = cands or []
    titles = {c.get("number"): c.get("title", "") for c in cands if isinstance(c, dict)}
    merged = state.get("merged", []) or []
    bench = state.get("bench_results", {}) or {}
    base = state.get("bench_baseline")
    base_sha = (state.get("bench_baseline_sha") or "")[:8] or "?"
    cand_nums = {c.get("number") for c in cands
                 if isinstance(c, dict) and isinstance(c.get("number"), int)}
    pending_n = len(cand_nums - set(merged) - set(state.get("quarantined", []) or []))
    L = [f"# llamapatch report",
         f"",
         f"- merged: {len(merged)} | quarantined: {len(state.get('quarantined', []) or [])} | "
         f"pending: {pending_n} | batches: {state.get('batches_done', 0)} | "
         f"baseline tg: {base} @ {base_sha}",
         f""]
    late = []
    for q in quar:
        if isinstance(q, dict) and str(q.get("reason", "")).startswith("late-") \
           and q.get("pr") not in late:
            late.append(q.get("pr"))
    if late:
        L.append(f"- post-run healed (regressions caught AFTER merges): "
                 f"{', '.join(f'#{n}' for n in late)}")
        L.append(f"")
    L.append("## merged")
    if not merged:
        L.append("(none yet)")
    else:
        L.append("| PR | title | area | backends | verdict | tg vs base |")
        L.append("|---|---|---|---|---|---|")
        for n in merged:
            b = bench.get(str(n), {}) if isinstance(bench, dict) else {}
            L.append(f"| #{n} | {(titles.get(n, '') or '')[:60]} | {b.get('area', '?')} | "
                     f"{','.join(b.get('backends', []) or []) or '-'} | "
                     f"{b.get('verdict', 'unverified')} | "
                     f"{b.get('tg', '-')} vs {b.get('base', '-')} |")
    L.append("")
    L.append("## quarantined")
    if not quar:
        L.append("(none)")
    else:
        L.append("| PR | reason | detail |")
        L.append("|---|---|---|")
        for q in quar:
            if isinstance(q, dict):
                L.append(f"| #{q.get('pr', '?')} | {q.get('reason', '?')} | "
                         f"{(q.get('detail', '') or '')[:80]} |")
    L.append("")
    gains = [n for n in merged
             if isinstance(bench, dict) and bench.get(str(n), {}).get("verdict") == "improvement"]
    if gains:
        L.append(f"## proven improvements ({len(gains)})")
        for n in gains:
            b = bench[str(n)]
            L.append(f"- #{n} {titles.get(n, '')[:70]}: {b.get('tg')} vs base {b.get('base')}")
        L.append("")
    return "\n".join(L) + "\n"


def smoke_ok(rc, out):
    """Pure smoke verdict: rc 0 + any throughput token."""
    return rc == 0 and bool(re.search(r"(tg\d+|throughput|tok/s|avg_ts)", out))

def sh(cmd, cwd, check=False, capture=True, timeout=1800):
    exe = "/bin/bash" if os.path.exists("/bin/bash") else None
    r = subprocess.run(cmd, cwd=cwd, shell=True, text=True, executable=exe,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None,
                       timeout=timeout)
    out = r.stdout if capture else ""
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{out[-4000:]}")
    return r.returncode, (out or "")[-8000:]

class Lab:
    def __init__(self, a):
        self.a = a
        self.repo = os.path.expanduser(a.repo)
        self.state_f = os.path.join(a.state_dir, "lab-state.json")
        self.log_f = os.path.join(a.state_dir, "lab-log.jsonl")
        self.quar_f = os.path.join(a.state_dir, "quarantined.json")
        os.makedirs(a.state_dir, exist_ok=True)
        try:
            with open(a.candidates) as f:
                raw = json.load(f)
        except Exception as e:
            raise RuntimeError(f"cannot load candidates {a.candidates}: {e}")
        if not isinstance(raw, list):
            raise RuntimeError(f"candidates {a.candidates} must be a JSON list")
        self.cands = [c for c in raw
                      if isinstance(c, dict) and isinstance(c.get("number"), int)]
        self.cands_skipped = len(raw) - len(self.cands)
        if self.cands_skipped:
            print(f"candidates: skipped {self.cands_skipped} malformed entries", flush=True)
        self.state = {"base_sha": None, "branch": None, "merged": [], "quarantined": [],
                      "bench_baseline": None, "batches_done": 0}
        if os.path.exists(self.state_f):
            try:
                with open(self.state_f) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.state = loaded
            except Exception as e:
                bak = self.state_f + f".corrupt-{int(time.time())}"
                try:
                    os.replace(self.state_f, bak)
                except Exception:
                    pass
                print(f"state corrupt, backed up to {bak}: {e}", flush=True)
        if os.path.exists(self.quar_f):
            try:
                with open(self.quar_f) as f:
                    loaded = json.load(f)
                self.quar = loaded if isinstance(loaded, list) else []
            except Exception as e:
                bak = self.quar_f + f".corrupt-{int(time.time())}"
                try:
                    os.replace(self.quar_f, bak)
                except Exception:
                    pass
                print(f"quarantine file corrupt, backed up to {bak}: {e}", flush=True)
                self.quar = []
        else:
            self.quar = []

    def save(self):
        with open(self.state_f, "w") as f:
            json.dump(self.state, f, indent=2)
        with open(self.quar_f, "w") as f:
            json.dump(self.quar, f, indent=2)

    def log(self, **kw):
        kw["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.log_f, "a") as f:
            f.write(json.dumps(kw) + "\n")
        print(f"[{kw.get('event')}] {json.dumps({k: v for k, v in kw.items() if k not in ('ts','event')})[:220]}")

    def git(self, cmd, check=False, timeout=300):
        # Git plumbing is fast; a hung fetch/merge must fail into retry and
        # timeout-quarantine within minutes, never stall a 10-batch for the
        # 30-minute bulk-command default.
        return sh(f"git {cmd}", self.repo, check=check, timeout=timeout)

    def git_args(self, args):
        """Shell-free git invocation (for messages with arbitrary titles)."""
        exe = None
        r = subprocess.run(["git"] + args, cwd=self.repo, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=120)
        return r.returncode, (r.stdout or "")[-8000:]

    def ensure_identity(self):
        rc, out = self.git("config user.email")
        if rc != 0 or not out.strip():
            self.git("config user.email 'pr-lab@localhost'")
            self.git("config user.name 'pr-lab'")
            self.log(event="git-identity-set")

    def clean_tree(self):
        # abort any in-progress merge/rebase, leave branch head intact
        self.git("merge --abort")
        self.git("rebase --abort")
        rc, out = self.git("status --porcelain")
        return rc, out

    def tracked_clean(self):
        """True when no staged/worktree tracked modifications (ignores
        untracked build/ output). Used to verify reverts actually worked."""
        rc, out = self.git("status --porcelain --untracked-files=no")
        return rc == 0 and not out.strip()

    def ensure_clean_start(self):
        """After kills/restarts the worktree may hold modified tracked files
        (builds regenerate checked-in files). Merging over that dirt fails
        with bogus merge-conflicts that wrongly quarantine good PRs, so
        reset to HEAD — all real work is committed — and verify. Untracked
        files (build/, ccache) are never touched. Raises loudly when dirt
        persists instead of corrupting the batch."""
        if self.tracked_clean():
            return
        self.log(event="dirty-start-reset")
        rc, out = self.git("reset --hard HEAD")
        if rc != 0 or not self.tracked_clean():
            _, out2 = self.git("status --porcelain --untracked-files=no")
            raise RuntimeError(
                "worktree dirty at start and reset failed; refusing to merge "
                f"over dirt:\n{(out2 or out or '')[-2000:]}")

    def revert_last(self):
        """Revert the just-merged commit (local branch, never pushed: safe)
        and verify the tree is clean. Raises RuntimeError when the tree
        stays dirty — continuing would corrupt every later PR, so the run
        must stop loudly for operator inspection instead of mass-
        quarantining good PRs."""
        rc, out = self.git("reset --hard HEAD~1")
        if rc != 0:
            raise RuntimeError(f"reset --hard HEAD~1 failed:\n{out[-2000:]}")
        if not self.tracked_clean():
            rc2, out2 = self.git("status --porcelain --untracked-files=no")
            raise RuntimeError(
                "tree still dirty after revert (build modified tracked files?); "
                f"stopping to protect later PRs:\n{(out2 or '')[-2000:]}")

    def preflight(self):
        """Fail-fast environment checks before touching the repo.

        Hard failures (raise): repo missing/not-a-repo, base ref
        unresolvable, empty candidate list. Soft (log-only warnings):
        missing cmake/ninja/ccache, low disk, skipped malformed entries.
        Dry-run never reaches here, so planning works anywhere.
        """
        import shutil
        if not os.path.isdir(self.repo):
            raise RuntimeError(f"repo dir missing: {self.repo}")
        rc, out = self.git("rev-parse --git-dir")
        if rc != 0:
            raise RuntimeError(f"not a git repo: {self.repo}\n{out[-1000:]}")
        if not self.cands:
            raise RuntimeError("no usable candidates (empty or all malformed)")
        if not self.state.get("branch"):
            rc, out = self.git(f"rev-parse --verify {self.a.base}")
            if rc != 0:
                raise RuntimeError(f"base ref {self.a.base!r} not found:\n{out[-1000:]}")
        tools = {}
        for t in ("cmake", "ninja", "ccache"):
            r = subprocess.run(f"command -v {t}" if os.path.exists("/bin/bash") else f"where {t}",
                               shell=True, capture_output=True, text=True, timeout=30)
            tools[t] = r.returncode == 0
        missing = [t for t, ok in tools.items() if not ok]
        if missing:
            self.log(event="preflight-warn", missing_tools=missing)
        try:
            free_gb = shutil.disk_usage(self.a.state_dir).free / 1e9
            if free_gb < 5:
                self.log(event="preflight-warn", low_disk_gb=round(free_gb, 1))
        except Exception:
            pass
        if self.cands_skipped:
            self.log(event="preflight-warn", skipped_candidates=self.cands_skipped)
        self.log(event="preflight-ok", tools=tools, candidates=len(self.cands))

    def setup_branch(self):
        if self.state["branch"]:
            rc, out = self.git(f"checkout {self.state['branch']}")
            if rc != 0:
                raise RuntimeError(f"checkout {self.state['branch']} failed:\n{out[-2000:]}")
            return
        rc, base = self.git(f"rev-parse {self.a.base}")
        if rc != 0 or not base.strip():
            raise RuntimeError(f"rev-parse {self.a.base} failed:\n{base[-2000:]}")
        base = base.strip()
        br = f"pr-lab/base-{base[:8]}"
        rc, out = self.git(f"checkout -B {br} {base}")
        if rc != 0:
            raise RuntimeError(f"checkout -B {br} failed:\n{out[-2000:]}")
        self.state["base_sha"] = base
        self.state["branch"] = br
        self.save()
        self.log(event="branch", branch=br, base=base)

    def has_local_pr(self, n):
        rc, _ = self.git(f"rev-parse --verify pr/{n}")
        return rc == 0

    def fetch_pr(self, n, retries=2):
        for attempt in range(retries + 1):
            rc, out = self.git(f"fetch origin pull/{n}/head:pr/{n} --force")
            if rc == 0:
                return True, out
            # Resilience: fork deleted / offline / mock repo — a local
            # pr/N branch (or a prior fetch) is still mergeable.
            if self.has_local_pr(n):
                return True, f"local-pr/{n}-fallback"
            if attempt < retries:
                time.sleep(5)
        # final fallback check before giving up
        if self.has_local_pr(n):
            return True, f"local-pr/{n}-fallback"
        return False, out

    def build(self):
        btimeout = getattr(self.a, "build_timeout", 3600)
        cfg = f"cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE={self.a.build_type}"
        if sh("which ccache", self.repo)[0] == 0:
            cfg += " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        rc, out = sh(cfg, self.repo, timeout=btimeout)
        if rc != 0:
            return False, "configure-failed\n" + out
        rc, out = sh(f"cmake --build build --target {' '.join(self.a.targets)} -j {self.a.jobs}",
                     self.repo, timeout=btimeout)
        return (rc == 0), out

    def smoke(self):
        # Smoke gate: llama-bench pp32/tg32 on the tiny model proves inference
        # works AND yields a perf number. (llama-cli 0.5.0-dev is silent
        # non-interactively, so bench is the gate.)
        bench = os.path.join(self.repo, "build", "bin", "llama-bench")
        model = os.path.expanduser(self.a.smoke_model)
        if not os.path.exists(bench) or not os.path.exists(model):
            return False, f"missing bench({os.path.exists(bench)}) or model({model})"
        rc, out = sh(f"set -o pipefail; timeout 300 '{bench}' -m '{model}' -p 32 -n 32 2>&1 | tail -25", self.repo)
        return smoke_ok(rc, out), out

    def bench(self):
        # llama-bench -o json emits a list (or {"results": [...]});
        # tg entry has n_gen>0, metric is avg_ts. Falls back to regexes.
        # Writes last-bench.txt for forensics; caller decides pass/fail.
        bench = os.path.join(self.repo, "build", "bin", "llama-bench")
        model = os.path.expanduser(self.a.bench_model)
        rc, out = sh(f"set -o pipefail; timeout 600 '{bench}' -m '{model}' -p {self.a.pp} -n {self.a.tg} -o json 2>&1 | tail -80", self.repo)
        val = parse_bench_output(out)
        try:
            with open(os.path.join(self.a.state_dir, "last-bench.txt"), "w") as f:
                f.write(out)
        except Exception:
            pass
        return rc == 0, val, out

    def quarantine(self, n, reason, detail=""):
        # Dedupe identical (pr, reason) rows: transient requeue/retry cycles
        # would otherwise grow quarantined.json unboundedly over a 1k run.
        # A new reason for the same PR is kept as history.
        if not any(isinstance(q, dict) and q.get("pr") == n and q.get("reason") == reason
                   for q in self.quar):
            self.quar.append({"pr": n, "reason": reason, "detail": detail[:1000]})
        if n not in self.state["quarantined"]:
            self.state["quarantined"].append(n)
        self.save()
        self.log(event="quarantine", pr=n, reason=reason)

    def pr_files(self, n):
        """Files changed by pr/N vs HEAD (for overlap warnings)."""
        rc, out = self.git(f"diff --name-only HEAD...pr/{n}")
        if rc != 0:
            return []
        return [l.strip() for l in out.strip().splitlines() if l.strip()]

    def merge_one_committed(self, n, batch_tag):
        """Fetch + merge + amend-message a single PR. Returns (ok, reason).

        One commit per PR (git cannot stack uncommitted merges). The merge
        message is amended to 'pr-lab: merge #N <title>' so --doctor can
        reconcile state vs repo. Empty (already-upstream) merges are
        detected pre-amend via HEAD-unchanged and reported as 'noop-empty'
        so the caller can record without bench-gating a no-op — and without
        renaming the previous PR's commit. Hung git (TimeoutExpired) and
        missing git (OSError) return git-timeout/git-error instead of
        crashing the batch.
        """
        try:
            return self._merge_one_inner(n, batch_tag)
        except subprocess.TimeoutExpired as e:
            try:
                self.clean_tree()
            except Exception:
                pass
            return False, f"git-timeout: {str(e)[-500:]}"
        except OSError as e:
            try:
                self.clean_tree()
            except Exception:
                pass
            return False, f"git-error: {str(e)[-500:]}"

    def _merge_one_inner(self, n, batch_tag):
        self.clean_tree()
        ok, _ = self.fetch_pr(n)
        if not ok:
            return False, "fetch-failed"
        rc, old_head = self.git("rev-parse HEAD")
        old_head = old_head.strip() if rc == 0 else ""
        rc, merge_out = self.git(f"merge --no-ff --no-edit pr/{n}")
        if rc != 0:
            self.clean_tree()
            return False, f"merge-conflict: {merge_out[-1500:]}"
        rc, out = self.git("rev-parse HEAD")
        if rc != 0:
            self.clean_tree()
            return False, "rev-parse-failed"
        new_head = out.strip()
        if (old_head and new_head == old_head) or "already up to date" in merge_out.lower():
            # No new commit: PR already contained in HEAD (already-upstream
            # or empty branch). Must NOT amend — that would rename the
            # previous PR's merge commit. Record as noop, no gating.
            self.clean_tree()
            return True, "noop-empty"
        # verify the merge actually landed (2 parents) and tree is clean
        rc, out = self.git(f"rev-list --parents -n 1 {new_head}")
        if rc != 0 or len(out.strip().split()) < 3:
            self.clean_tree()
            return False, f"empty-or-nonmerge: {out[-500:]}"
        # amend message for traceability / doctor matching (shell-free:
        # PR titles may contain $, quotes, backticks — never pass via shell).
        title = next((c.get("title", "") for c in self.cands if c.get("number") == n), "")
        msg = sanitize_merge_msg(n, title)
        arc, aout = self.git_args(["commit", "--amend", "-m", msg])
        if arc != 0:
            self.log(event="amend-failed", pr=n, detail=aout[-500:])
        # empty-merge check: no diff vs first parent => already upstream
        rc, out = self.git("diff HEAD^1 HEAD --stat")
        if rc == 0 and not out.strip():
            return True, "noop-empty"
        return True, None

    def perplexity(self):
        """Optional correctness gate: llama-perplexity on a WikiText sample.
        Skipped (pass) when binary/model/sample absent. Returns (ok, detail).
        Fail quarantines the PR (correctness, not just speed)."""
        import os
        ppl = os.path.join(self.repo, "build", "bin", "llama-perplexity")
        model = os.path.expanduser(self.a.smoke_model)
        sample = os.path.expanduser(getattr(self.a, "ppl_sample", "") or "")
        if not getattr(self.a, "ppl_threshold", 0):
            return True, "ppl-gate-disabled"
        if not os.path.exists(ppl) or not os.path.exists(model) or not sample or not os.path.exists(sample):
            return True, "ppl-gate-skipped-missing-infra"
        rc, out = sh(f"timeout 600 '{ppl}' -m '{model}' -f '{sample}' 2>&1 | tail -10", self.repo)
        import re
        m = re.search(r"(?:perplexity|ppl)[:\s]+([\d.]+)", out, re.I)
        if rc != 0 or not m:
            return False, out[-1500:]
        try:
            val = float(m.group(1))
        except ValueError:
            return False, out[-1500:]
        if val > float(self.a.ppl_threshold):
            return False, f"ppl {val} > threshold {self.a.ppl_threshold}\n{out[-800:]}"
        return True, f"ppl {val}"

    def run_gates(self, n, intent):
        """Build/smoke/ppl/bench gates for one merged commit.

        Returns True when the PR counts (merged, possibly unverified-perf)
        and False when it was quarantined. All gate failures revert the
        commit and quarantine. Hung/crashed gates (TimeoutExpired/OSError)
        quarantine as *-timeout/*-error instead of crashing the 10-batch
        run; bench timeouts never punish the PR (infra flake ->
        merged-unverified-perf). revert_last() failures propagate loudly.
        """
        try:
            okb, bout = self.build()
        except Exception as e:
            self.revert_last()
            self.quarantine(n, f"build-{gate_error_reason(e)}", str(e)[-2000:])
            return False
        if not okb:
            self.revert_last()
            self.quarantine(n, "build-failed", bout[-2000:])
            return False
        try:
            oks, sout = self.smoke()
        except Exception as e:
            self.revert_last()
            self.quarantine(n, f"smoke-{gate_error_reason(e)}", str(e)[-2000:])
            return False
        if not oks:
            self.revert_last()
            self.quarantine(n, "smoke-failed", sout[-2000:])
            return False
        try:
            okp, pdetail = self.perplexity()
        except Exception as e:
            self.revert_last()
            self.quarantine(n, f"perplexity-{gate_error_reason(e)}", str(e)[-2000:])
            return False
        if not okp:
            self.revert_last()
            self.quarantine(n, "perplexity-failed", pdetail[-2000:])
            return False
        if self.a.bench_model and self.a.regression_pct > 0 and \
           os.path.exists(os.path.expanduser(self.a.bench_model)):
            try:
                okbench, val, bout = self.bench()
            except Exception as e:
                self.log(event="bench-failed", pr=n,
                         detail=f"bench-{gate_error_reason(e)}: {str(e)[-800:]}")
                self.state["merged"].append(n)
                self.state["batches_done"] += 1
                self.save()
                self.log(event="merged-unverified-perf", pr=n,
                         total=len(self.state["merged"]))
                return True
            if not okbench or val is None:
                self.log(event="bench-failed", pr=n,
                         detail=bout[-1000:])
                # bench infra flake (OOM/timeout/parse): do NOT punish
                # the PR, but record the merge without a perf verdict.
                self.state["merged"].append(n)
                self.state["batches_done"] += 1
                self.save()
                self.log(event="merged-unverified-perf", pr=n,
                         total=len(self.state["merged"]))
                return True
            base = self.state.get("bench_baseline")
            verdict = verdict_for(intent, base, val, self.a.regression_pct)
            br = self.state.setdefault("bench_results", {})
            br[str(n)] = {"tg": val, "base": base, "verdict": verdict,
                          "area": intent.get("area"),
                          "backends": intent.get("backends"),
                          "expects_gain": intent.get("expects_bench_gain")}
            self.save()
            if verdict == "regression":
                self.revert_last()
                self.log(event="regression", pr=n, val=val, base=base,
                         area=intent.get("area"), backends=intent.get("backends"))
                self.quarantine(n, "perf-regression", f"{val} vs {base}")
                return False
            if verdict == "improvement":
                self.log(event="improvement", pr=n, val=val, base=base,
                         area=intent.get("area"), backends=intent.get("backends"))
            elif verdict == "parity" and intent.get("expects_bench_gain") \
                    and base and val:
                # Claimed a gain this box can measure, produced none:
                # roll back. Only measured improvements stay merged.
                self.revert_last()
                self.log(event="no-improvement", pr=n, val=val, base=base,
                         area=intent.get("area"), backends=intent.get("backends"))
                self.quarantine(n, "no-improvement",
                                f"{val} vs {base} (no measured gain)")
                return False
            else:
                # parity with no gain expected (backend-only perf on a CPU
                # box, fixes, features) or no baseline to judge against:
                # green gates + no regression is this PR's proof. Record
                # why, don't punish.
                self.log(event="parity", pr=n, val=val, base=base,
                         area=intent.get("area"), backends=intent.get("backends"),
                         expects_gain=intent.get("expects_bench_gain"))
        self.state["merged"].append(n)
        self.state["batches_done"] += 1
        self.save()
        self.log(event="merged", pr=n, total=len(self.state["merged"]),
                 area=intent.get("area"), backends=intent.get("backends"))
        return True

    def ensure_baseline(self):
        """Build + bench the clean base once per base SHA.

        The baseline must belong to the base under test: after a rebase or
        base move, old numbers would mislabel parity as improvement (or vice
        versa), so a SHA mismatch rebuilds. Records bench_baseline_sha even
        when the bench is unparsed so a broken bench doesn't rebuild every
        run — the next base change retries.

        Never baselines a dirty HEAD: if campaign merges are already on the
        branch (model arrived late, first baseline failed), benching now
        would poison every future verdict with PR improvements baked in —
        a machine for false perf-regressions. Defer loudly instead; gates
        fall back to unverified, never to wrong numbers.
        """
        if not self.a.bench_model or (self.a.regression_pct or 0) <= 0:
            return
        if self.state.get("bench_baseline_sha") is not None and \
           self.state.get("bench_baseline_sha") == self.state.get("base_sha"):
            return
        model = os.path.expanduser(self.a.bench_model)
        if not (os.path.exists(model) or os.path.exists(model + ".1")):
            print("bench model absent, skipping baseline bench", flush=True)
            return
        rc, head = None, ""
        try:
            # Best-effort only: preflight already validated the repo, so a
            # failure here must not crash the run — fall through to the
            # normal baseline path.
            rc, head = self.git("rev-parse HEAD")
            head = head.strip() if rc == 0 else ""
        except Exception:
            head = ""
        if head and self.state.get("base_sha") and head != self.state["base_sha"]:
            print("baseline deferred: HEAD holds campaign merges, "
                  "baseline must come from the clean base", flush=True)
            self.log(event="baseline-deferred", head=head[:8],
                     base=(self.state.get("base_sha") or "")[:8])
            return
        ok, out = self.build()
        print(f"baseline build: {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            return
        okb, val, bout = self.bench()
        self.state["bench_baseline"] = val
        self.state["bench_baseline_sha"] = self.state.get("base_sha")
        print(f"baseline bench tg: {val}", flush=True)
        if val is None:
            self.log(event="baseline-bench-unparsed", detail=bout[-1000:])
        self.save()

    def maybe_force_unlock(self):
        """Operator-explicit stale-lock override (--force-unlock).

        Returns True when a pre-existing lock was cleared. Only for use
        after verifying no merge loop is running (e.g. SIGKILL left a
        stale lock); the event is logged for forensics.
        """
        if not getattr(self.a, "force_unlock", False):
            return False
        if not os.path.exists(lock_path(self.a.state_dir)):
            return False
        self.log(event="force-unlock")
        release_lock(self.a.state_dir)
        return True

    def run(self):
        pending = [c["number"] for c in self.cands
                   if c["number"] not in self.state["merged"]
                   and c["number"] not in self.state["quarantined"]]
        print(f"pending: {len(pending)} (merged={len(self.state['merged'])} quar={len(self.state['quarantined'])})", flush=True)
        if self.a.dry_run:
            for chunk in plan_batches(pending, self.a.batch, self.a.max_prs):
                print(f"batch: {chunk}")
            return
        if getattr(self.a, "report", False):
            rep = build_report(self.state, self.quar, self.cands)
            out = getattr(self.a, "report_out", "")
            if out:
                with open(os.path.expanduser(out), "w") as f:
                    f.write(rep)
                print(f"wrote {out}")
            else:
                print(rep, end="")
            return
        self.maybe_force_unlock()
        acquire_lock(self.a.state_dir)
        try:
            self._run_locked(pending)
        finally:
            release_lock(self.a.state_dir)

    def _drop_merged(self, n):
        """Remove a reverted culprit from the merged list (quarantine() only
        appends to quarantined; without this the state claims a PR whose
        commit is gone — exactly the phantom doctor would later drop)."""
        if n in self.state["merged"]:
            self.state["merged"] = [m for m in self.state["merged"] if m != n]

    def _mark_late(self, n, verdict):
        """Stamp a heal-reverted culprit's bench verdict so reports and the
        post-heal gain check never credit a PR whose commit is gone."""
        br = self.state.setdefault("bench_results", {})
        entry = br.get(str(n))
        if isinstance(entry, dict):
            entry["verdict"] = verdict
        else:
            br[str(n)] = {"verdict": verdict}
        self.save()

    def final_verify_and_heal(self, new_merges):
        """Post-run gate: per-PR gates pass at merge time, but regressions
        can surface AFTER merges (interactions between PRs in the batch).
        Re-verifies the final tree (build + smoke + bench vs baseline) and
        walks HEAD back — reverting + quarantining each culprit as
        late-{build,smoke,regression}-failed — until clean or the heal-walk
        cap. Returns (status, detail) with status in clean/healed/capped/
        skipped. A cap is loud (needs operator), never silent. Each PR
        proved its own intent at merge time, so this gate only guards
        no-breakage, it never re-litigates gains."""
        if not new_merges:
            return "skipped", "nothing merged this run"
        cap = max(1, int(getattr(self.a, "heal_walk", 10) or 10))
        remaining = list(new_merges)
        steps = 0
        while remaining and steps < cap:
            try:
                okb, bout = self.build()
            except Exception as e:
                okb, bout = False, f"build-{gate_error_reason(e)}: {e}"
            if not okb:
                culprit = remaining.pop()
                self.revert_last()
                self._drop_merged(culprit)
                self._mark_late(culprit, "late-build-failed")
                self.quarantine(culprit, "late-build-failed", str(bout)[-2000:])
                self.log(event="late-heal", pr=culprit, gate="build")
                steps += 1
                continue
            try:
                oks, sout = self.smoke()
            except Exception as e:
                oks, sout = False, f"smoke-{gate_error_reason(e)}: {e}"
            if not oks:
                culprit = remaining.pop()
                self.revert_last()
                self._drop_merged(culprit)
                self._mark_late(culprit, "late-smoke-failed")
                self.quarantine(culprit, "late-smoke-failed", str(sout)[-2000:])
                self.log(event="late-heal", pr=culprit, gate="smoke")
                steps += 1
                continue
            if self.a.bench_model and (self.a.regression_pct or 0) > 0 and \
               os.path.exists(os.path.expanduser(self.a.bench_model)) and \
               self.state.get("bench_baseline"):
                try:
                    okbench, val, bout = self.bench()
                except Exception as e:
                    self.log(event="final-bench-flake", detail=str(e)[-500:])
                    return ("healed" if steps else "unverified"), "bench infra flake"
                if not okbench or val is None:
                    self.log(event="final-bench-flake", detail=str(bout)[-1000:])
                    return ("healed" if steps else "unverified"), "bench unparsed"
                base = self.state["bench_baseline"]
                if val < base * (1 - self.a.regression_pct / 100):
                    culprit = remaining.pop()
                    self.revert_last()
                    self._drop_merged(culprit)
                    self._mark_late(culprit, "late-regression")
                    self.quarantine(culprit, "late-regression", f"{val} vs {base}")
                    self.log(event="late-heal", pr=culprit, gate="bench",
                             val=val, base=base)
                    steps += 1
                    continue
            break
        if remaining and steps >= cap:
            return "capped", f"still failing after {steps} reverts; needs operator"
        return ("healed" if steps else "clean"), "final tree verified"

    def verify_improvements_final(self, new_merges):
        """Post-heal improvement check: per-PR gains were proven at merge
        time, but later merges and heal-reverts can dilute them below the
        threshold. Re-benches the final tree once against the baseline.

        Returns (status, detail): verified (gain holds in aggregate),
        lost (measured, gain gone — report only), unverified (bench flake),
        skipped (no model/baseline), na (no improvement claims this run).

        Never auto-reverts: combined gains need not stack linearly, so a
        lost aggregate gain misattributes blame. The operator decides from
        the logged claim list + final numbers.
        """
        br = self.state.get("bench_results", {}) or {}
        # Only PRs still merged count: heal-reverted culprits were stamped
        # late-* by _mark_late, but filter by merged membership too so a
        # ghost claim can never pass verification.
        alive = set(self.state.get("merged", []) or [])
        claimed = [n for n in new_merges
                   if n in alive
                   and isinstance(br.get(str(n)), dict)
                   and br.get(str(n)).get("verdict") == "improvement"]
        if not claimed:
            return "na", "no improvement claims this run"
        if not self.a.bench_model or not os.path.exists(os.path.expanduser(self.a.bench_model)):
            return "skipped", "no bench model"
        base = self.state.get("bench_baseline")
        if not base:
            return "skipped", "no baseline"
        try:
            okbench, val, bout = self.bench()
        except Exception as e:
            return "unverified", f"bench flake: {e}"[:300]
        if not okbench or val is None:
            return "unverified", f"bench unparsed: {str(bout)[-300:]}"
        if val > base * (1 + self.a.regression_pct / 100):
            return "verified", f"final {val} vs base {base} holds gains from {claimed}"
        return "lost", f"final {val} vs base {base} loses gains claimed by {claimed}"

    def _run_locked(self, pending):
        self.preflight()
        self.ensure_identity()
        self.setup_branch()
        self.clean_tree()
        self.ensure_clean_start()
        if self.a.doctor:
            self.doctor()
            return
        # baseline bench (7B model required; skip if absent; rebuilds when
        # the base SHA moved so verdicts compare against the right base)
        self.ensure_baseline()
        merged_before = set(self.state["merged"])
        done = 0
        i = 0
        cand_by_num = {c["number"]: c for c in self.cands}
        while i < len(pending) and done < self.a.max_prs:
            batch = [n for n in pending[i:i+self.a.batch]
                     if n not in self.state["quarantined"]]
            if not batch:
                i += self.a.batch
                continue
            self.log(event="batch-start", prs=batch)
            batch_files = set()
            for n in batch:
                if done >= self.a.max_prs:
                    break
                # CI gate: skip red PRs before paying for a build (annotated
                # by fetch_prs.py --include-ci; override with --no-skip-ci-red).
                # Skipped, NOT quarantined: CI flips green on reruns/pushes,
                # and the next triage refresh updates ci_state, so the PR is
                # retried automatically. The skip costs one dict lookup.
                if should_skip_ci(cand_by_num.get(n), getattr(self.a, "skip_ci_red", True)):
                    ci = (cand_by_num.get(n) or {}).get("ci_state")
                    self.log(event="ci-red-skipped", pr=n, ci_state=ci)
                    continue
                intent = classify_intent(cand_by_num.get(n) or {"number": n})
                self.log(event="pr-intent", pr=n, area=intent.get("area"),
                         backends=intent.get("backends"),
                         expects_gain=intent.get("expects_bench_gain"))
                # overlap warning (pre-merge: HEAD...pr/N is only valid
                # before HEAD advances; post-merge the diff is empty).
                # Same files twice in one batch of 10 raises conflict odds;
                # still tried sequentially (self-heals on conflict).
                try:
                    files = set(self.pr_files(n))
                    overlap = files & batch_files
                    if overlap:
                        self.log(event="batch-overlap-warn", pr=n,
                                 overlap=sorted(overlap)[:10])
                except Exception:
                    files = set()
                ok, reason = self.merge_one_committed(n, f"batch-{self.state['batches_done']}")
                if not ok:
                    # Transient fetch/git failures quarantine under their own
                    # reason so --doctor can requeue them for retry; real
                    # content conflicts quarantine permanently.
                    if is_transient_quarantine(reason):
                        self.quarantine(n, reason.split(":")[0], reason)
                    else:
                        self.quarantine(n, "merge-conflict", reason)
                    continue
                batch_files |= files
                if reason == "noop-empty":
                    # Already upstream: record without bench-gating a no-op.
                    self.state["merged"].append(n)
                    self.state["batches_done"] += 1
                    self.save()
                    self.log(event="merged-noop-empty", pr=n,
                             total=len(self.state["merged"]))
                    done += 1
                    continue
                # incremental gates: build + smoke (+ppl) + bench on top of
                # this commit. Hung/crashed gates self-heal via run_gates
                # (quarantine *-timeout, never crash the batch).
                if self.run_gates(n, intent):
                    done += 1
                continue
            i += self.a.batch
        # Post-run gate: per-PR gates pass at merge time, but interactions
        # can regress the tree AFTER merges. Verify the final tree and
        # self-heal (revert + quarantine culprits) before reporting DONE.
        new_merges = [n for n in self.state["merged"] if n not in merged_before]
        if new_merges:
            status, detail = self.final_verify_and_heal(new_merges)
            self.log(event="final-verify", status=status, detail=str(detail)[:300])
            print(f"FINAL {status}: {detail}", flush=True)
            if status in ("clean", "healed"):
                istatus, idetail = self.verify_improvements_final(new_merges)
                self.log(event="final-improvements", status=istatus,
                         detail=str(idetail)[:300])
                print(f"IMPROVEMENTS {istatus}: {idetail}", flush=True)
        print(f"DONE merged={self.state['merged']} quarantined={len(self.state['quarantined'])}", flush=True)

    def doctor(self):
        """Reconcile state files vs repo. Drops phantom 'merged' entries whose
        commit is absent, requeues quarantines caused by the MERGE_HEAD
        stacking bug (never truly conflict-tested), and requeues transient
        fetch/git quarantines for retry (a dead fork simply re-quarantines
        next run). Matches both '#N' (v4 amended messages) and legacy
        'pr/N' (v3 default merge messages)."""
        rc, out = self.git(f"log --format=%s {self.state.get('base_sha', self.a.base)}..HEAD")
        subjects = out if rc == 0 else ""
        fixed_merged, fixed_quar, requeued = [], [], []
        for n in self.state["merged"]:
            if doctor_match(subjects, n):
                fixed_merged.append(n)
            else:
                print(f"doctor: drop phantom merged #{n} (no commit on branch)")
        for q in list(self.quar):
            if "MERGE_HEAD exists" in q.get("detail", ""):
                print(f"doctor: requeue #{q['pr']} (bogus MERGE_HEAD quarantine)")
                requeued.append(q["pr"])
            elif is_transient_quarantine(q.get("reason", "")):
                print(f"doctor: requeue #{q['pr']} (transient {q.get('reason')}, retry next run)")
                requeued.append(q["pr"])
            elif q.get("reason") == "ci-red-skipped":
                # Migration: skips no longer quarantine, so release old
                # entries back to pending; fresh triage re-scores their CI.
                print(f"doctor: requeue #{q['pr']} (ci-red skip is retryable)")
                requeued.append(q["pr"])
            else:
                fixed_quar.append(q)
        self.state["merged"] = fixed_merged
        self.state["quarantined"] = [n for n in self.state["quarantined"] if n not in requeued]
        self.quar = fixed_quar
        self.save()
        print(f"doctor: merged={fixed_merged} requeued={requeued}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.json")
    ap.add_argument("--repo", default="~/llama-pr-lab/llama.cpp")
    ap.add_argument("--state-dir", default=".")
    ap.add_argument("--base", default="master")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-prs", type=int, default=50)
    ap.add_argument("--targets", nargs="+", default=["llama-cli", "llama-bench"])
    ap.add_argument("--build-type", default="Release")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--build-timeout", type=int, default=3600,
                    help="seconds per cmake configure/build before build-timeout quarantine")
    ap.add_argument("--heal-walk", type=int, default=10,
                    help="max post-run final-verify reverts before stopping loudly")
    ap.add_argument("--smoke-model", default="~/llama-pr-lab/models/tinyllama.gguf")
    ap.add_argument("--bench-model", default="~/llama-pr-lab/models/qwen2.5-7b-00001-of-00002.gguf")
    ap.add_argument("--pp", type=int, default=32)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--regression-pct", type=float, default=15.0)
    ap.add_argument("--skip-ci-red", dest="skip_ci_red", action="store_true", default=True)
    ap.add_argument("--no-skip-ci-red", dest="skip_ci_red", action="store_false")
    ap.add_argument("--ppl-threshold", type=float, default=0.0,
                    help="perplexity ceiling; 0 disables the ppl gate")
    ap.add_argument("--ppl-sample", default="",
                    help="path to WikiText sample for llama-perplexity gate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-unlock", action="store_true",
                    help="clear a stale lab.lock after verifying no run is active")
    ap.add_argument("--report", action="store_true",
                    help="print/write Markdown evidence report from state files, then exit")
    ap.add_argument("--report-out", default="",
                    help="write report to this path instead of stdout")
    ap.add_argument("--doctor", action="store_true",
                    help="reconcile state vs repo, requeue bogus quarantines, then exit")
    a = ap.parse_args()
    Lab(a).run()

if __name__ == "__main__":
    main()
