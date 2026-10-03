#!/usr/bin/env python3
"""Mock-repo end-to-end test: proves the 10-batch self-heal loop works
against a real git repo without llama.cpp, models, or network.

Builds a temp repo:
  base: a.txt=v1
  pr/1: clean change (a.txt=v1+one)          -> should merge
  pr/2: conflicting change (a.txt=v1+two)    -> conflicts after pr/1, quarantined
  pr/3: independent new file (b.txt)          -> should merge
  pr/4: branch at HEAD (no new commits)       -> noop-empty, must NOT rename #3

Drives Lab.merge_one_committed / pr_files / quarantine / doctor /
plan_batches, asserting: merged == [1, 3], quarantined == [2],
noop detected for 4 with #3's message intact.

Run: python3 test_e2e_mock.py (needs git on PATH; ~15s)
"""
import json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import merge_lab as M

FAIL = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" :: {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

def git(repo, *args):
    r = subprocess.run(["git"] + list(args), cwd=repo, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    return r.returncode, (r.stdout or "")

def make_lab(repo, statedir, cands):
    cf = os.path.join(statedir, "cands.json")
    json.dump(cands, open(cf, "w"))
    class A: pass
    a = A()
    a.repo = repo; a.state_dir = statedir; a.candidates = cf
    a.base = "master"; a.batch = 10; a.max_prs = 50
    a.targets = ["t"]; a.build_type = "Release"; a.jobs = 2
    a.smoke_model = ""; a.bench_model = ""; a.pp = 32; a.tg = 32
    a.regression_pct = 0; a.skip_ci_red = True
    a.ppl_threshold = 0.0; a.ppl_sample = ""
    a.dry_run = False; a.doctor = False
    return M.Lab(a)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo")
    statedir = os.path.join(td, "state")
    os.makedirs(repo); os.makedirs(statedir)
    git(repo, "init", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "a.txt"), "w").write("v1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "base")

    git(repo, "checkout", "-b", "pr/1")
    open(os.path.join(repo, "a.txt"), "w").write("v1\none\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr1")
    git(repo, "checkout", "master")
    git(repo, "checkout", "-b", "pr/2")
    open(os.path.join(repo, "a.txt"), "w").write("v1\ntwo\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr2")
    git(repo, "checkout", "master")
    git(repo, "checkout", "-b", "pr/3")
    open(os.path.join(repo, "b.txt"), "w").write("hello\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr3")
    git(repo, "checkout", "master")

    cands = [{"number": i, "title": f"mock pr {i}"} for i in (1, 2, 3)]
    lab = make_lab(repo, statedir, cands)
    lab.ensure_identity()
    lab.setup_branch()
    check("branch-created", lab.state["branch"].startswith("pr-lab/base-"),
          lab.state["branch"])

    batches = M.plan_batches([1, 2, 3], 10, 50)
    check("one-batch", batches == [[1, 2, 3]], str(batches))

    # pre-merge file lists (run() overlap order: before HEAD advances)
    files1 = set(lab.pr_files(1))
    check("pr1-files", "a.txt" in files1, str(files1))
    ok1, reason1 = lab.merge_one_committed(1, "batch-0")
    check("pr1-merges", ok1 and reason1 is None, f"{ok1} {reason1}")

    files2 = set(lab.pr_files(2))
    check("overlap-seen", bool(files1 & files2), f"{files1} vs {files2}")
    ok2, reason2 = lab.merge_one_committed(2, "batch-0")
    check("pr2-conflicts", not ok2 and str(reason2).startswith("merge-conflict"),
          f"{ok2} {str(reason2)[:120]}")
    lab.quarantine(2, "merge-conflict", str(reason2))
    check("pr2-quarantined", 2 in lab.state["quarantined"])

    files3 = set(lab.pr_files(3))
    check("pr3-independent", "b.txt" in files3 and not (files3 & files1), str(files3))
    ok3, reason3 = lab.merge_one_committed(3, "batch-0")
    check("pr3-merges", ok3 and reason3 is None, f"{ok3} {reason3}")

    lab.state["merged"].extend([1, 3])
    lab.save()
    subjects = git(repo, "log", "--format=%s",
                   f"{lab.state['base_sha']}..HEAD")[1]
    check("doctor-keep-1", M.doctor_match(subjects, 1), subjects[:200])
    check("doctor-keep-3", M.doctor_match(subjects, 3), subjects[:200])
    lab.doctor()
    check("doctor-state", lab.state["merged"] == [1, 3], str(lab.state["merged"]))

    # noop: branch at HEAD has nothing new -> noop-empty, #3 message intact
    git(repo, "checkout", lab.state["branch"])
    git(repo, "checkout", "-b", "pr/4", "HEAD")
    lab.cands.append({"number": 4, "title": "noop"})
    ok4, reason4 = lab.merge_one_committed(4, "batch-0")
    check("noop-detected", ok4 and reason4 == "noop-empty", f"{ok4} {reason4}")
    tip = git(repo, "log", "--format=%s", "-1")[1].strip()
    check("noop-no-clobber", "#3" in tip and "#4" not in tip, tip)

    import time as _t
    t0 = _t.time()
    okf, _ = lab.fetch_pr(1)
    dt = _t.time() - t0
    check("fetch-local-fallback", okf, f"{okf}")
    check("fetch-fast", dt < 8, f"{dt:.1f}s")

# Scenario 2: full run() orchestration — one 10-batch, gates stubbed green.
# pr/101 clean, pr/102 conflicts with 101, pr/103 independent.
# Expect merged == [101, 103], quarantined == [102], no crash, state saved.
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo")
    statedir = os.path.join(td, "state")
    os.makedirs(repo); os.makedirs(statedir)
    git(repo, "init", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "a.txt"), "w").write("v1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "base")
    git(repo, "checkout", "-b", "pr/101")
    open(os.path.join(repo, "a.txt"), "w").write("v1\none\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr101")
    git(repo, "checkout", "master")
    git(repo, "checkout", "-b", "pr/102")
    open(os.path.join(repo, "a.txt"), "w").write("v1\ntwo\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr102")
    git(repo, "checkout", "master")
    git(repo, "checkout", "-b", "pr/103")
    open(os.path.join(repo, "b.txt"), "w").write("hello\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr103")
    git(repo, "checkout", "master")

    cands = [{"number": i, "title": f"mock pr {i}"} for i in (101, 102, 103, 104)]
    lab = make_lab(repo, statedir, cands)
    lab.a.batch = 10; lab.a.max_prs = 10
    lab.a.bench_model = ""  # skip bench + baseline (no models in mock)
    lab.a.regression_pct = 0
    lab.build = lambda: (True, "mock build ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s mock")
    lab.run()
    check("run-merged", lab.state["merged"] == [101, 103], str(lab.state["merged"]))
    # 102 conflicts with 101; 104 has no local branch and no origin, so its
    # fetch fails transiently -> quarantined as fetch-failed, NOT conflict.
    check("run-quarantined", lab.state["quarantined"] == [102, 104],
          str(lab.state["quarantined"]))
    check("run-quar-reason",
          any(q["pr"] == 102 and q["reason"] == "merge-conflict" for q in lab.quar)
          and any(q["pr"] == 104 and q["reason"] == "fetch-failed" for q in lab.quar),
          str(lab.quar))
    check("run-batches", lab.state["batches_done"] == 2, str(lab.state["batches_done"]))
    # every kept merge records the exact head SHA that was gated.
    h101 = git(repo, "rev-parse", "pr/101")[1].strip()
    h103 = git(repo, "rev-parse", "pr/103")[1].strip()
    check("run-heads",
          lab.state.get("merged_heads", {}) == {"101": h101, "103": h103},
          str(lab.state.get("merged_heads")))
    # post-run final gate ran on the merged tree (stubs green -> clean)
    check("run-final-verify",
          '"final-verify"' in open(lab.log_f).read()
          and '"clean"' in open(lab.log_f).read())
    # resume: second run() is a no-op (nothing pending) and stays green
    lab2 = make_lab(repo, statedir, cands)
    lab2.a.batch = 10; lab2.a.max_prs = 10
    lab2.a.bench_model = ""
    lab2.a.regression_pct = 0
    lab2.build = lambda: (True, "mock")
    lab2.smoke = lambda: (True, "tg32 mock")
    lab2.run()
    check("run-resume-stable",
          lab2.state["merged"] == [101, 103] and lab2.state["quarantined"] == [102, 104],
          f"{lab2.state['merged']} {lab2.state['quarantined']}")
    # doctor requeues the transient fetch-failed 104 for retry, keeps the
    # genuine conflict 102 quarantined.
    lab2.doctor()
    check("run-doctor-requeues-fetch",
          lab2.state["quarantined"] == [102]
          and [q["pr"] for q in lab2.quar] == [102],
          f"{lab2.state['quarantined']} {lab2.quar}")

# Scenario 3: strict improvement policy inside run(). PR 201 claims a gain
# ("faster kernel" -> perf intent, no backend files -> expects_gain) but
# benches parity, so the loop must revert it as no-improvement and merge
# nothing. Proves only-measured-gains-stay end to end.
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo")
    statedir = os.path.join(td, "state")
    os.makedirs(repo); os.makedirs(statedir)
    git(repo, "init", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "a.txt"), "w").write("v1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "base")
    git(repo, "checkout", "-b", "pr/201")
    open(os.path.join(repo, "a.txt"), "w").write("v1\nfaster\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr201")
    git(repo, "checkout", "master")

    model = os.path.join(td, "m.gguf")
    open(model, "w").write("x")
    lab = make_lab(repo, statedir, [{"number": 201, "title": "faster kernel"}])
    lab.a.batch = 10; lab.a.max_prs = 10
    lab.a.bench_model = model
    lab.a.regression_pct = 15
    lab.build = lambda: (True, "mock build ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s mock")
    lab.bench = lambda: (True, 40.0, "parity")
    lab.run()
    check("strict-merged-empty", lab.state["merged"] == [], str(lab.state["merged"]))
    check("strict-quarantined", lab.state["quarantined"] == [201],
          str(lab.state["quarantined"]))
    check("strict-reason",
          any(q["pr"] == 201 and q["reason"] == "no-improvement" for q in lab.quar),
          str(lab.quar))
    check("strict-tree-clean",
          open(os.path.join(repo, "a.txt")).read() == "v1\n")
    check("strict-baseline", lab.state["bench_baseline"] == 40.0,
          str(lab.state.get("bench_baseline")))

# Scenario 4: CI-red skip is retryable, not terminal. PR 301 is red in run
# 1 (skipped pre-build, never quarantined); after the author fixes CI and
# fresh triage flips it green, run 2 merges it. PR 302 merges in run 1.
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo")
    statedir = os.path.join(td, "state")
    os.makedirs(repo); os.makedirs(statedir)
    git(repo, "init", "-b", "master")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "a.txt"), "w").write("v1\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "base")
    git(repo, "checkout", "-b", "pr/302")
    open(os.path.join(repo, "b.txt"), "w").write("hello\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr302")
    git(repo, "checkout", "master")

    def _runlab(cands):
        lb = make_lab(repo, statedir, cands)
        lb.a.batch = 10; lb.a.max_prs = 10
        lb.a.bench_model = ""
        lb.a.regression_pct = 0
        lb.build = lambda: (True, "mock build ok")
        lb.smoke = lambda: (True, "tg32 : 40 t/s mock")
        lb.run()
        return lb

    # run 1: 301 red -> skipped without quarantine; 302 merges.
    lab = _runlab([{"number": 301, "title": "red", "ci_state": "failure"},
                   {"number": 302, "title": "green"}])
    check("ci-skip-merges-green", lab.state["merged"] == [302],
          str(lab.state["merged"]))
    check("ci-skip-no-quarantine", lab.state["quarantined"] == []
          and lab.quar == [], f"{lab.state['quarantined']} {lab.quar}")
    check("ci-skip-logged", '"ci-red-skipped"' in open(lab.log_f).read())
    check("ci-skip-stays-pending", 301 not in lab.state["merged"]
          and 301 not in lab.state["quarantined"])

    # author fixes CI; new branch appears (pushed fix). Fresh triage sees
    # green 301, run 2 merges it.
    git(repo, "checkout", "-b", "pr/301", "master")
    open(os.path.join(repo, "c.txt"), "w").write("fixed\n")
    git(repo, "add", "-A"); git(repo, "commit", "-m", "pr301")
    git(repo, "checkout", lab.state["branch"])
    lab2 = _runlab([{"number": 301, "title": "red", "ci_state": "success"},
                    {"number": 302, "title": "green"}])
    check("ci-flip-retried", lab2.state["merged"] == [302, 301],
          str(lab2.state["merged"]))
    check("ci-flip-clean", lab2.state["quarantined"] == [],
          str(lab2.state["quarantined"]))

print(f"\n{len(FAIL)} failures")
sys.exit(1 if FAIL else 0)
