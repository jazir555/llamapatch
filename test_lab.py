#!/usr/bin/env python3
"""Self-tests for llamapatch pure logic. No network, no WSL, no models.

Run: python3 test_lab.py
Covers the gates that previously broke auto-merge: bench parsing,
smoke verdict, batch planning, CI skip, doctor matching, merge-message
sanitization, perf-path matching, and corrupt-state self-heal.
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))
import merge_lab as M
from fetch_prs import touches_perf, score_title

FAIL = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" :: {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

# 1. bench parsing
check("bench-list", M.parse_bench_output('[{"n_gen":0,"avg_ts":99},{"n_gen":32,"avg_ts":38.05}]') == 38.05)
check("bench-results-obj", M.parse_bench_output('{"results": [{"n_gen":16,"avg_ts":15.79}]}') == 15.79)
check("bench-log-prefix", M.parse_bench_output('build ok\n[{"n_gen":128,"avg_ts":42.5}]') == 42.5)
check("bench-trailing-line", M.parse_bench_output('[{"n_gen":32,"avg_ts":40.0}]\nsummary done') == 40.0)
check("bench-regex-fallback", M.parse_bench_output('speed: 160.39 tok/s') == 160.39)
check("bench-none", M.parse_bench_output('no numbers here') is None)

# 2. smoke verdict
check("smoke-tg", M.smoke_ok(0, "tg32 : 42 t/s"))
check("smoke-throughput", M.smoke_ok(0, "throughput 160 t/s"))
check("smoke-avgts", M.smoke_ok(0, '{"avg_ts": 38.05}'))
check("smoke-rc-fail", not M.smoke_ok(1, "tg32 : 42 t/s"))
check("smoke-empty", not M.smoke_ok(0, "silent output"))

# 3. batch planning (10 at a time)
pending = list(range(1, 26))
batches = M.plan_batches(pending, 10, 50)
check("batches-10", batches == [list(range(1, 11)), list(range(11, 21)), list(range(21, 26))])
check("batches-cap", M.plan_batches(pending, 10, 15) == [list(range(1, 11)), list(range(11, 16))])
check("batches-empty", M.plan_batches([], 10, 50) == [])

# 4. CI skip
check("ci-skip-failure", M.should_skip_ci({"ci_state": "failure"}, True))
check("ci-skip-error", M.should_skip_ci({"ci_state": "error"}, True))
check("ci-keep-success", not M.should_skip_ci({"ci_state": "success"}, True))
check("ci-keep-pending", not M.should_skip_ci({"ci_state": "pending"}, True))
check("ci-keep-unset", not M.should_skip_ci({}, True))
check("ci-override", not M.should_skip_ci({"ci_state": "failure"}, False))

# 5. doctor matching (v4 #N + legacy pr/N)
check("doctor-hash", M.doctor_match("pr-lab: merge #29110 metal kernels", 29110))
check("doctor-legacy", M.doctor_match("Merge branch 'pr/29110' into x", 29110))
check("doctor-miss", not M.doctor_match("pr-lab: merge #9999 other", 29110))

# 6. merge message sanitization (shell-injection safe)
msg = M.sanitize_merge_msg(1, "fix $(rm -rf /) `evil` 'q' \"d\"; | & stuff\nnewline")
check("msg-no-shell", all(c not in msg for c in ["$", "`", "'", '"', ";", "|", "&", "\n"]),
      msg)
check("msg-has-num", "#1" in msg)
check("msg-trunc", len(M.sanitize_merge_msg(2, "x" * 500)) <= len("pr-lab: merge #2 ") + 100)

# 7. perf-path prefix matching (no substring false positives)
cfg = {"perf_paths": ["ggml/src", "ggml/include", "src/", "common/", "tools/server", "examples/bench"],
       "perf_keywords": []}
check("perf-cuda", touches_perf(["ggml/src/ggml-cuda/foo.cu"], cfg))
check("perf-src", touches_perf(["src/llama.cpp"], cfg))
check("perf-docs-miss", not touches_perf(["docs/README.md"], cfg))
check("perf-substr-miss", not touches_perf(["my_src_fake/file.cpp"], cfg))

# 8. scoring
s, hits = score_title("CUDA perf: faster flash attention kernel", "", {"perf_keywords": ["perf", "cuda", "flash", "kernel"]})
check("score-hits", s == 4 and "cuda" in hits, f"{s} {hits}")

# 9. corrupt-state self-heal
class A: pass
with tempfile.TemporaryDirectory() as td:
    cand = os.path.join(td, "c.json")
    json.dump([{"number": 1, "title": "t"}], open(cand, "w"))
    open(os.path.join(td, "lab-state.json"), "w").write("{corrupt!")
    open(os.path.join(td, "quarantined.json"), "w").write("[corrupt!")
    a = A(); a.repo = td; a.state_dir = td; a.candidates = cand
    try:
        lab = M.Lab(a)
        check("corrupt-state-fresh", lab.state["merged"] == [] and lab.quar == [])
        bak = [f for f in os.listdir(td) if "corrupt-" in f]
        check("corrupt-backup", len(bak) == 2, str(bak))
    except Exception as e:
        check("corrupt-no-crash", False, str(e))

# 10. candidates-sample sanity (batch-10 planning over real triage)
try:
    sample = json.load(open(os.path.join(os.path.dirname(__file__), "candidates-sample.json")))
    nums = [c["number"] for c in sample]
    check("sample-count", len(nums) >= 20, str(len(nums)))
    check("sample-batches", M.plan_batches(nums, 10, 30)[0][:3] == nums[:3])
except Exception as e:
    check("sample-load", False, str(e))

# 11. intent classification (proves the right thing per PR kind)
from pr_intent import classify_intent, verdict_for
cuda_perf = {"number": 1, "title": "CUDA: faster TOP_K kernel", "labels": ["ggml", "cuda"],
             "files": ["ggml/src/ggml-cuda/top-k.cu"]}
i = classify_intent(cuda_perf)
check("intent-cuda-backend", "cuda" in i["backends"], str(i))
check("intent-cuda-area", i["area"] == "perf", str(i))
check("intent-cuda-no-cpu-gain", not i["expects_bench_gain"], str(i))
check("verdict-cuda-parity", verdict_for(i, 40.0, 40.5, 15) == "parity")
check("verdict-cuda-regression", verdict_for(i, 40.0, 30.0, 15) == "regression")
check("verdict-cuda-big-gain-parity", verdict_for(i, 40.0, 60.0, 15) == "parity",
      "backend-only gain on CPU box must not claim improvement")
cpu_perf = {"number": 2, "title": "ggml-cpu: faster sgemm", "labels": ["ggml"],
            "files": ["ggml/src/ggml-cpu/llamafile/sgemm.cpp"]}
j = classify_intent(cpu_perf)
check("intent-cpu-area", j["area"] == "perf", str(j))
check("intent-cpu-expects-gain", j["expects_bench_gain"], str(j))
check("verdict-cpu-improvement", verdict_for(j, 40.0, 60.0, 15) == "improvement")
fix = {"number": 3, "title": "CUDA: fix crash for MTP decoding", "labels": ["ggml", "cuda"],
       "files": ["ggml/src/ggml-cuda/ggml-cuda.cu"]}
k = classify_intent(fix)
check("intent-fix-area", k["area"] == "fix", str(k))
check("verdict-fix-parity", verdict_for(k, 40.0, 40.2, 15) == "parity")
check("verdict-no-baseline", verdict_for(k, None, 40.0, 15) == "no-baseline")
# real sample: every triaged PR classifies without crash
try:
    bad = [c["number"] for c in sample
           if not isinstance(classify_intent(c).get("area"), str)]
    check("sample-intent-all", bad == [], str(bad[:5]))
except Exception as e:
    check("sample-intent", False, str(e))

# 12. preflight + candidates validation + safe revert
import subprocess as _sp

def _git(repo, *args):
    r = _sp.run(["git"] + list(args), cwd=repo, text=True,
                stdout=_sp.PIPE, stderr=_sp.STDOUT, timeout=60)
    return r.returncode, (r.stdout or "")

class _A: pass

def _mklab(repo, statedir, cands_obj, write_raw=None):
    cf = os.path.join(statedir, "c.json")
    if write_raw is not None:
        open(cf, "w").write(write_raw)
    else:
        json.dump(cands_obj, open(cf, "w"))
    a = _A(); a.repo = repo; a.state_dir = statedir; a.candidates = cf
    return a

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 1, "title": "t"}])
    a.base = "master"
    lab = M.Lab(a)
    try:
        lab.preflight()
        check("preflight-ok", True)
    except Exception as e:
        check("preflight-ok", False, str(e)[:200])
    check("preflight-logged",
          "preflight-ok" in open(lab.log_f).read(), "missing preflight-ok event")
    # revert_last restores HEAD and clean tree
    _git(repo, "checkout", "-b", "work")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "change")
    h1 = _git(repo, "rev-parse", "HEAD")[1].strip()
    try:
        lab.revert_last()
        h2 = _git(repo, "rev-parse", "HEAD")[1].strip()
        check("revert-head", h2 != h1, f"{h1[:8]} vs {h2[:8]}")
        check("revert-clean", lab.tracked_clean())
        check("revert-content", open(os.path.join(repo, "f.txt")).read() == "v1\n")
    except Exception as e:
        check("revert-last", False, str(e)[:200])
    _git(repo, "checkout", "master")
    # tracked dirt detected (build touching a tracked file must not slip by)
    open(os.path.join(repo, "f.txt"), "w").write("dirty\n")
    check("dirty-detected", not lab.tracked_clean())
    _git(repo, "checkout", "--", ".")
    check("recleaned", lab.tracked_clean())

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    repo = os.path.join(td, "repo"); os.makedirs(repo)
    _git(repo, "init", "-b", "master")
    raw = [{"number": 10, "title": "good"}, {"n": 2}, "oops",
           {"number": "x"}, {"number": 11, "title": "good2"}]
    a = _mklab(repo, sd, raw)
    try:
        lab = M.Lab(a)
        check("cands-filtered", [c["number"] for c in lab.cands] == [10, 11],
              str(lab.cands))
        check("cands-skipped-count", lab.cands_skipped == 3, str(lab.cands_skipped))
    except Exception as e:
        check("cands-validation", False, str(e)[:200])

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    a = _mklab(os.path.join(td, "nope"), sd, [{"number": 1}])
    a.base = "master"
    try:
        M.Lab(a).preflight()
        check("preflight-missing-repo", False, "should have raised")
    except RuntimeError as e:
        check("preflight-missing-repo", "missing" in str(e).lower(), str(e)[:150])
    a2 = _mklab(sd, sd, [{"number": 1}])  # state dir is not a git repo
    a2.base = "master"
    try:
        M.Lab(a2).preflight()
        check("preflight-not-a-repo", False, "should have raised")
    except RuntimeError as e:
        check("preflight-not-a-repo", "not a git repo" in str(e).lower(), str(e)[:150])
    a3 = _mklab(sd, sd, [{"number": 1}])
    a3.base = "master"
    cf = a3.candidates
    open(cf, "w").write("{not json!")
    try:
        M.Lab(a3)
        check("cands-corrupt", False, "should have raised")
    except RuntimeError as e:
        check("cands-corrupt", "cannot load candidates" in str(e).lower(), str(e)[:150])

# 13. gate timeouts self-heal (hung build/smoke/git quarantines, never crashes)
import subprocess as _sp2
check("gate-reason-timeout",
      M.gate_error_reason(_sp2.TimeoutExpired("cmake", 3600)) == "timeout")
check("gate-reason-infra",
      M.gate_error_reason(OSError("no such file")) == "infra-error")
check("gate-reason-other",
      M.gate_error_reason(ValueError("x")) == "error")

def _gatelab(repo, sd, num=99):
    a = _mklab(repo, sd, [{"number": num, "title": "t"}])
    a.base = "master"; a.batch = 10; a.max_prs = 50
    a.targets = ["t"]; a.build_type = "Release"; a.jobs = 2
    a.smoke_model = ""; a.bench_model = ""; a.pp = 32; a.tg = 32
    a.regression_pct = 0; a.skip_ci_red = True
    a.ppl_threshold = 0.0; a.ppl_sample = ""; a.build_timeout = 3600
    return M.Lab(a)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")[1].strip()
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "merged-pr")
    lab = _gatelab(repo, sd)
    def _hung(*a, **k):
        raise _sp2.TimeoutExpired("cmake --build", 3600)
    lab.build = _hung
    lab._build_ok = True  # streak case: infra proven, failure is the PR's
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": True}
    counted = lab.run_gates(99, intent)
    check("build-timeout-quarantined", counted is False)
    check("build-timeout-reason",
          any(q["pr"] == 99 and q["reason"] == "build-timeout" for q in lab.quar),
          str(lab.quar))
    check("build-timeout-reverted",
          _git(repo, "rev-parse", "HEAD")[1].strip() == base)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")[1].strip()
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "merged-pr")
    lab = _gatelab(repo, sd)
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (False, "no throughput lines")
    lab._smoke_green = True  # streak case: infra proven, failure is the PR's
    counted = lab.run_gates(99, {"area": "fix", "backends": [], "expects_bench_gain": False})
    check("smoke-fail-quarantined", counted is False)
    check("smoke-fail-reason",
          any(q["pr"] == 99 and q["reason"] == "smoke-failed" for q in lab.quar),
          str(lab.quar))
    check("smoke-fail-reverted",
          _git(repo, "rev-parse", "HEAD")[1].strip() == base)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "merged-pr")
    head = _git(repo, "rev-parse", "HEAD")[1].strip()
    lab = _gatelab(repo, sd)
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    counted = lab.run_gates(99, {"area": "fix", "backends": [], "expects_bench_gain": False})
    check("gates-pass-counted", counted is True)
    check("gates-pass-kept", _git(repo, "rev-parse", "HEAD")[1].strip() == head)
    check("gates-pass-merged", 99 in lab.state["merged"])

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    lab = _gatelab(repo, sd)
    def _dead(*a, **k):
        raise _sp2.TimeoutExpired("git", 1800)
    lab.git = _dead
    ok, reason = lab.merge_one_committed(99, "b")
    check("merge-git-timeout", ok is False and str(reason).startswith("git-timeout"),
          f"{ok} {reason}")

# 14. evidence report (per-batch review: what improved, what didn't, why)
state = {"merged": [29110, 29806], "quarantined": [28952],
         "batches_done": 1, "bench_baseline": 40.0,
         "bench_results": {
             "29110": {"tg": 40.5, "base": 40.0, "verdict": "parity",
                       "area": "perf", "backends": ["metal"], "expects_gain": False},
             "29806": {"tg": 48.0, "base": 40.0, "verdict": "improvement",
                       "area": "perf", "backends": ["cpu"], "expects_gain": True}}}
quar = [{"pr": 28952, "reason": "merge-conflict", "detail": "CONFLICT in ggml.c"}]
cands = [{"number": 29110, "title": "metal kernels"}, {"number": 29806, "title": "tinyBLAS"},
         {"number": 28952, "title": "FP8 support"}]
rep = M.build_report(state, quar, cands)
check("report-counts", "merged: 2" in rep and "quarantined: 1" in rep, rep[:200])
check("report-pending", "pending: 0" in rep, rep[:200])
check("report-pending-open",
      "pending: 2" in M.build_report({"merged": [1], "quarantined": []},
                                     [], [{"number": 1}, {"number": 2}, {"number": 3}]))
check("report-rows", "#29110" in rep and "#29806" in rep and "#28952" in rep)
check("report-verdicts", "parity" in rep and "improvement" in rep)
check("report-gains", "proven improvements (1)" in rep and "#29806" in rep.split("proven")[-1])
check("report-empty-safe", "(none yet)" in M.build_report({"merged": []}, [], []))
check("report-nonelist-safe", "merged: 0" in M.build_report(None, None, None))

# 15. triage rate-limit abort (partial save + exit, not hours of 30s sleeps)
from urllib.error import HTTPError as _HE
from email.message import Message as _Msg
from fetch_prs import is_rate_limit as _rl
def _mkhttp(code, msg, remaining=None):
    h = _Msg()
    if remaining is not None:
        h["X-RateLimit-Remaining"] = str(remaining)
    return _HE("http://x", code, msg, h, None)
check("rl-429", _rl(_mkhttp(429, "Too Many Requests")) is True)
check("rl-403-empty", _rl(_mkhttp(403, "Forbidden", 0)) is True)
check("rl-403-msg", _rl(_mkhttp(403, "API rate limit exceeded")) is True)
check("rl-403-other", _rl(_mkhttp(403, "Forbidden", 59)) is False)
check("rl-404", _rl(_mkhttp(404, "Not Found")) is False)
check("rl-other-exc", _rl(ValueError("x")) is False)

# 16. resilient pagination (transient retry, partial on persistent failure)
from fetch_prs import collect_pages as _pages
calls = {"n": 0}
def _flaky(page):
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("transient blip")
    return ([{"number": page}], True)
sleeps = []
items, err = _pages(_flaky, 1000, sleep=lambda s: sleeps.append(s))
check("pages-retry-ok", items == [{"number": 1}] and err is None, f"{items} {err}")
check("pages-backoff", sleeps == [1, 2], str(sleeps))
def _dead2(page):
    if page == 1:
        return ([{"number": 1}], False)
    raise ConnectionError("down")
items2, err2 = _pages(_dead2, 1000, sleep=lambda s: None)
check("pages-partial", items2 == [{"number": 1}] and isinstance(err2, ConnectionError),
      f"{items2} {err2}")
def _full(page):
    return ([{"number": page * 100 + i} for i in range(100)], False)
items3, err3 = _pages(_full, 250, sleep=lambda s: None)
check("pages-limit", len(items3) == 250 and err3 is None, str(len(items3)))
items4, err4 = _pages(lambda p: ([], True), 1000, sleep=lambda s: None)
check("pages-empty", items4 == [] and err4 is None)

# 17. baseline follows the base SHA (stale baseline mislabels verdicts)
def _baselab(sd, model):
    a = _mklab("/tmp/llamapatch-norepo", sd, [{"number": 1, "title": "t"}])
    a.base = "master"; a.batch = 10; a.max_prs = 50
    a.targets = ["t"]; a.build_type = "Release"; a.jobs = 2
    a.smoke_model = ""; a.bench_model = model; a.pp = 32; a.tg = 32
    a.regression_pct = 15; a.skip_ci_red = True
    a.ppl_threshold = 0.0; a.ppl_sample = ""; a.build_timeout = 3600
    return M.Lab(a)

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.state["base_sha"] = "abc123"
    calls = {"build": 0, "bench": 0}
    lab.build = lambda: (calls.__setitem__("build", calls["build"] + 1) or (True, "ok"))
    lab.bench = lambda: (calls.__setitem__("bench", calls["bench"] + 1) or (True, 42.0, "out"))
    lab.ensure_baseline()
    check("baseline-recorded",
          lab.state["bench_baseline"] == 42.0 and lab.state["bench_baseline_sha"] == "abc123",
          str({k: lab.state.get(k) for k in ("bench_baseline", "bench_baseline_sha")}))
    lab.ensure_baseline()
    check("baseline-cached", calls == {"build": 1, "bench": 2}, str(calls))
    lab.state["base_sha"] = "def456"
    lab.ensure_baseline()
    check("baseline-rebuilt",
          calls == {"build": 2, "bench": 4} and lab.state["bench_baseline_sha"] == "def456",
          f"{calls} {lab.state.get('bench_baseline_sha')}")

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.state["base_sha"] = "abc123"
    lab.build = lambda: (False, "configure boom")
    lab.bench = lambda: (_ for _ in ()).throw(AssertionError("bench must not run"))
    lab.ensure_baseline()
    check("baseline-build-fail",
          lab.state["bench_baseline"] is None and "bench_baseline_sha" not in lab.state)

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.state["base_sha"] = "abc123"
    n = {"bench": 0}
    lab.build = lambda: (True, "ok")
    lab.bench = lambda: (n.__setitem__("bench", n["bench"] + 1) or (False, None, "unparsed"))
    lab.ensure_baseline()
    check("baseline-unparsed-sha",
          lab.state["bench_baseline"] is None and lab.state["bench_baseline_sha"] == "abc123")
    lab.ensure_baseline()
    check("baseline-unparsed-no-loop", n == {"bench": 2}, str(n))

# 18. clean start after kills (dirty tracked files must not fake conflicts)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 1, "title": "t"}])
    lab = M.Lab(a)
    lab.ensure_clean_start()
    check("clean-start-clean", True)
    open(os.path.join(repo, "f.txt"), "w").write("build regenerated me\n")
    open(os.path.join(repo, "build-out.o"), "w").write("untracked artifact\n")
    check("clean-start-sees-dirt", not lab.tracked_clean())
    lab.ensure_clean_start()
    check("clean-start-reset",
          lab.tracked_clean() and open(os.path.join(repo, "f.txt")).read() == "v1\n")
    check("clean-start-keeps-untracked",
          os.path.exists(os.path.join(repo, "build-out.o")),
          "untracked build outputs must survive")
    check("clean-start-logged", "dirty-start-reset" in open(lab.log_f).read())

# 19. model + bench-gate naming consistency (setup drift breaks every run)
_here = os.path.dirname(__file__)
_fm = open(os.path.join(_here, "fetch_models.sh")).read()
_triage = open(os.path.join(_here, "triage-1k.sh")).read()
_ml = open(os.path.join(_here, "merge_lab.py")).read()
_cfg = json.load(open(os.path.join(_here, "config.json")))
PART1 = "qwen2.5-7b-00001-of-00002.gguf"
PART2 = "qwen2.5-7b-00002-of-00002.gguf"
check("models-fetch-part1", PART1 in _fm and "q4_k_m-00001-of-00002" in _fm)
check("models-fetch-part2", PART2 in _fm and "q4_k_m-00002-of-00002" in _fm)
check("models-no-single-qwen", "qwen2.5-7b.gguf" not in _fm and "qwen2.5-7b.gguf" not in _ml,
      "single-file Qwen URL 404s upstream; split parts required")
check("models-triage-part1", PART1 in _triage)
check("models-default-part1", PART1 in _ml)
check("models-config-part1", _cfg["perf_model"]["file"] == PART1, str(_cfg["perf_model"]))
check("models-tiny-url", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" in _fm)
check("bench-defaults", "default=32" in _ml and "default=15.0" in _ml)

# 20. run lock (concurrent loops sharing one state-dir corrupt state)
with tempfile.TemporaryDirectory() as td:
    M.acquire_lock(td)
    lp = os.path.join(td, "lab.lock")
    check("lock-file", os.path.exists(lp))
    check("lock-pid", json.load(open(lp)).get("pid") == os.getpid())
    try:
        M.acquire_lock(td)
        check("lock-second", False, "second acquire must raise")
    except RuntimeError as e:
        check("lock-second", "another run may be active" in str(e), str(e)[:150])
    M.release_lock(td)
    check("lock-released", not os.path.exists(lp))
    M.acquire_lock(td)
    check("lock-reacquire", os.path.exists(lp))
    M.release_lock(td)
    M.release_lock(td)  # missing file must not raise
    check("lock-release-idempotent", True)

# 21. force-unlock override (explicit operator intent only)
with tempfile.TemporaryDirectory() as td:
    cf = os.path.join(td, "c.json")
    json.dump([{"number": 1}], open(cf, "w"))
    class _AF: pass
    a = _AF(); a.repo = os.path.join(td, "norepo"); a.state_dir = td
    a.candidates = cf; a.force_unlock = False
    lab = M.Lab(a)
    M.acquire_lock(td)
    check("force-off-keeps-lock", lab.maybe_force_unlock() is False
          and os.path.exists(os.path.join(td, "lab.lock")))
    a.force_unlock = True
    check("force-on-clears", lab.maybe_force_unlock() is True
          and not os.path.exists(os.path.join(td, "lab.lock")))
    check("force-logged", "force-unlock" in open(lab.log_f).read())
    check("force-absent", lab.maybe_force_unlock() is False)

# 22. transient quarantines retry via doctor (network blips must not kill PRs)
check("transient-fetch", M.is_transient_quarantine("fetch-failed") is True)
check("transient-timeout", M.is_transient_quarantine("git-timeout: blah") is True)
check("transient-giterr", M.is_transient_quarantine("git-error: blah") is True)
check("transient-conflict", M.is_transient_quarantine("merge-conflict: blah") is False)
check("transient-build", M.is_transient_quarantine("build-failed") is False)
check("transient-empty", M.is_transient_quarantine("") is False)
check("transient-none", M.is_transient_quarantine(None) is False)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 5}, {"number": 6}, {"number": 8}])
    a.base = "master"
    lab = M.Lab(a)
    lab.state["base_sha"] = _git(repo, "rev-parse", "HEAD")[1].strip()
    lab.state["merged"] = [7]
    lab.state["quarantined"] = [5, 6, 8, 9]
    lab.quar = [{"pr": 5, "reason": "fetch-failed", "detail": "fetch-failed"},
                {"pr": 6, "reason": "merge-conflict", "detail": "CONFLICT"},
                {"pr": 8, "reason": "git-timeout", "detail": "git-timeout: hung"},
                {"pr": 9, "reason": "ci-red-skipped", "detail": "ci_state=failure"}]
    lab.save()
    lab.doctor()
    check("doctor-drops-phantom", lab.state["merged"] == [], str(lab.state["merged"]))
    check("doctor-keeps-conflict",
          [q["pr"] for q in lab.quar] == [6], str(lab.quar))
    check("doctor-requeues-transient", lab.state["quarantined"] == [6],
          str(lab.state["quarantined"]))

# 38. reverted/phantom drops prune recorded head SHAs (no orphans)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 5}, {"number": 6}])
    a.base = "master"
    lab = M.Lab(a)
    lab.state["base_sha"] = _git(repo, "rev-parse", "HEAD")[1].strip()
    lab.state["merged"] = [5]
    lab.state["merged_heads"] = {"5": "a" * 40}
    lab.save()
    lab._drop_merged(5)
    check("drop-prunes-head",
          lab.state["merged"] == [] and lab.state.get("merged_heads") == {},
          str(lab.state.get("merged_heads")))
    lab.state["merged"] = [5, 6]
    lab.state["merged_heads"] = {"5": "a" * 40, "6": "b" * 40}
    lab.save()
    lab.doctor()
    check("doctor-prunes-heads", lab.state.get("merged_heads") == {},
          str(lab.state.get("merged_heads")))

# 39. doctor verifies merge parents against tested SHAs
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 7}, {"number": 8}])
    a.base = "master"
    lab = M.Lab(a)
    lab.state["base_sha"] = _git(repo, "rev-parse", "HEAD")[1].strip()
    _git(repo, "checkout", "-b", "pr/7")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "pr7")
    head7 = _git(repo, "rev-parse", "HEAD")[1].strip()
    _git(repo, "checkout", "master")
    _git(repo, "merge", "--no-ff", "--no-edit", "pr/7")
    _git(repo, "commit", "--amend", "-m", "pr-lab: merge #7 seven")
    _git(repo, "checkout", "-b", "pr/8")
    open(os.path.join(repo, "f.txt"), "w").write("v3\n")
    _git(repo, "commit", "-am", "pr8")
    head8 = _git(repo, "rev-parse", "HEAD")[1].strip()
    _git(repo, "checkout", "master")
    _git(repo, "merge", "--no-ff", "--no-edit", "pr/8")
    _git(repo, "commit", "--amend", "-m", "Merge branch 'pr/8' into master")
    check("merge-parents",
          lab.merge_parents() == {7: head7, 8: head8}, str(lab.merge_parents()))
    lab.state["merged"] = [7]
    lab.state["merged_heads"] = {"7": "0" * 40}
    lab.save()
    lab.doctor()
    check("doctor-warns-moved",
          lab.state["merged"] == [7] and any("#7" in w for w in lab.doctor_warnings),
          str(lab.doctor_warnings))
    lab.state["merged_heads"] = {"7": head7}
    lab.save()
    lab.doctor()
    check("doctor-quiet-match", lab.doctor_warnings == [], str(lab.doctor_warnings))

# 40. boundary bench verdicts get one confirmation run (noise guard)
_g = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": True}
check("confirm-reg-boundary", M.needs_confirm(_g, 100, 83, 15, 5.0) is True)
check("confirm-reg-clear", M.needs_confirm(_g, 100, 70, 15, 5.0) is False)
check("confirm-imp-boundary", M.needs_confirm(_g, 100, 118, 15, 5.0) is True)
check("confirm-imp-clear", M.needs_confirm(_g, 100, 130, 15, 5.0) is False)
check("confirm-gainline-below", M.needs_confirm(_g, 100, 112, 15, 5.0) is True)
check("confirm-parity-far-noexpect-rollback-stands",
      M.needs_confirm(_g, 100, 100, 15, 5.0) is False)
_nox = {"area": "perf", "backends": ["cuda"], "expects_bench_gain": False}
check("confirm-noexpect-above", M.needs_confirm(_nox, 100, 118, 15, 5.0) is False)
check("confirm-disabled", M.needs_confirm(_g, 100, 88, 15, 0) is False)
check("confirm-nobase", M.needs_confirm(_g, 0, 88, 15, 5.0) is False)

def _scripted(vals):
    vals = list(vals)
    def _b():
        v = vals.pop(0) if len(vals) > 1 else vals[0]
        return (True, v, f"v={v}")
    return _b

def _confirmlab(sd, model, repo):
    # Local twin of _benchlab (defined later in file): same fake args +
    # stubbed gates, but bound to a real mock repo for revert_last.
    a = _mklab(repo, sd, [{"number": 77, "title": "t"}])
    a.base = "master"; a.batch = 10; a.max_prs = 50
    a.targets = ["t"]; a.build_type = "Release"; a.jobs = 2
    a.smoke_model = ""; a.bench_model = model; a.pp = 32; a.tg = 32
    a.regression_pct = 15; a.skip_ci_red = True
    a.bench_noise_pct = 5.0
    a.ppl_threshold = 0.0; a.ppl_sample = ""; a.build_timeout = 3600
    lab = M.Lab(a)
    lab.state["base_sha"] = "base1"
    lab.state["bench_baseline"] = 40.0
    lab.state["bench_baseline_sha"] = "base1"
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    return lab

def _confirmrepo(td):
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "merged-pr")
    return repo, sd

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.bench = _scripted([33.5, 40.0])  # boundary regression, then clean
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": False}
    check("confirm-saves-noisy-pr", lab.run_gates(77, intent) is True)
    check("confirm-mean-stored",
          lab.state["bench_results"]["77"]["runs"] == [33.5, 40.0]
          and lab.state["bench_results"]["77"]["verdict"] == "parity",
          str(lab.state["bench_results"].get("77")))

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.bench = _scripted([30.0, 31.0])  # real regression, twice
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": False}
    check("confirm-upholds-real", lab.run_gates(77, intent) is False)
    check("confirm-real-quarantined",
          any(q["pr"] == 77 and q["reason"] == "perf-regression" for q in lab.quar),
          str(lab.quar))

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    calls = {"n": 0}
    def _flake():
        calls["n"] += 1
        return (True, 33.5, "first") if calls["n"] == 1 else (False, None, "flaked")
    lab.bench = _flake
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": False}
    check("confirm-flake-keeps-first", lab.run_gates(77, intent) is False)
    check("confirm-flake-runs",
          lab.state["bench_results"]["77"]["runs"] == [33.5],
          str(lab.state["bench_results"].get("77")))

# 41. final-verify confirms boundary readings before reverting a culprit
with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.bench = _scripted([33.5, 40.0])  # boundary, then clean: noise, not guilt
    lab.state["merged"] = [71, 72]
    st, _ = lab.final_verify_and_heal([71, 72])
    check("final-confirm-saves", st == "clean", st)
    check("final-confirm-no-revert", lab.state["merged"] == [71, 72],
          str(lab.state["merged"]))

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.bench = _scripted([30.0, 31.0])  # real regression, twice
    lab.state["merged"] = [71, 72]
    st, _ = lab.final_verify_and_heal([72])
    check("final-confirm-upholds", st == "healed", st)
    check("final-confirm-reverts", lab.state["merged"] == [71],
          str(lab.state["merged"]))
    check("final-confirm-runs",
          any("runs=" in q.get("detail", "") for q in lab.quar
              if q.get("reason") == "late-regression"),
          str(lab.quar))

# 23. quarantine dedup (retry cycles must not grow the file unboundedly)
with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    a = _mklab(os.path.join(td, "norepo"), sd, [{"number": 9}])
    lab = M.Lab(a)
    lab.quarantine(9, "fetch-failed", "blip")
    lab.quarantine(9, "fetch-failed", "blip again")
    check("quar-dedup",
          len(lab.quar) == 1 and lab.state["quarantined"] == [9],
          f"{lab.quar} {lab.state['quarantined']}")
    lab.quarantine(9, "merge-conflict", "now a real conflict")
    check("quar-new-reason-kept",
          len(lab.quar) == 2
          and [q["reason"] for q in lab.quar] == ["fetch-failed", "merge-conflict"],
          str(lab.quar))

# 24. stem-aware triage scoring (no more debug-is-bug)
from fetch_prs import score_title as _st
_check_cfg = lambda *kws: {"perf_keywords": list(kws)}
check("score-word", _st("Prefix parsing fix", "", _check_cfg("fix")) == (1, ["fix"]))
check("score-stem",
      _st("Fixes prefixes in suffixes", "", _check_cfg("fix")) == (1, ["fix"]))
check("score-debug-not-bug", _st("debug build", "", _check_cfg("bug")) == (0, []))
check("score-phrase", _st("add support for X", "", _check_cfg("add ")) == (1, ["add "]))
check("score-multiword",
      _st("flash attention kernel", "", _check_cfg("flash attention", "kernel"))[0] == 2)
check("score-case-plural", _st("CUDA kernels", "", _check_cfg("cuda")) == (1, ["cuda"]))
check("score-punct-substr",
      _st("top-k loop", "", _check_cfg("top-k")) == (1, ["top-k"]))

# 25. dead code stays dead (stacked-merge stubs confused every reader)
check("no-merge-one", not hasattr(M.Lab, "merge_one"))
check("no-try-merge", not hasattr(M.Lab, "try_merge"))
check("no-commit-batch", not hasattr(M.Lab, "commit_batch"))

# 26. triage transport never hangs (timeout reaches urlopen; headers right)
import io as _io
import urllib.request as _urlreq
import fetch_prs as F
_orig_open = _urlreq.urlopen
_seen = {}
class _FakeResp:
    headers = {"X": "1"}
    def __init__(self, payload):
        self._io = _io.StringIO(payload)
    def read(self, *a):
        return self._io.read(*a)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
def _fake_open(req_obj, **kw):
    _seen.clear()
    _seen.update(kw)
    _seen["request"] = req_obj
    return _FakeResp('{"a": 1}')
_urlreq.urlopen = _fake_open
try:
    data, hdrs = F.req("http://example/x", "tok123")
    check("req-parses", data == {"a": 1}, str(data))
    check("req-timeout", _seen.get("timeout") == 60, str(_seen))
    check("req-auth", _seen["request"].get_header("Authorization") == "Bearer tok123")
    check("req-accept", "github+json" in (_seen["request"].get_header("Accept") or ""))
    data2, _ = F.req("http://example/x", None)
    check("req-noauth", data2 == {"a": 1}
          and _seen["request"].get_header("Authorization") is None)
finally:
    _urlreq.urlopen = _orig_open

# 27. run_gates bench verdicts (regression reverts, improvement credits)
def _benchlab(sd, model, baseline):
    a = _mklab("/tmp/llamapatch-norepo", sd, [{"number": 77, "title": "t"}])
    a.base = "master"; a.batch = 10; a.max_prs = 50
    a.targets = ["t"]; a.build_type = "Release"; a.jobs = 2
    a.smoke_model = ""; a.bench_model = model; a.pp = 32; a.tg = 32
    a.regression_pct = 15; a.skip_ci_red = True
    a.ppl_threshold = 0.0; a.ppl_sample = ""; a.build_timeout = 3600
    lab = M.Lab(a)
    lab.state["base_sha"] = "base1"
    lab.state["bench_baseline"] = baseline
    lab.state["bench_baseline_sha"] = "base1"
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    return lab

def _gitrepo(td, extra="v2\n"):
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")[1].strip()
    open(os.path.join(repo, "f.txt"), "w").write(extra)
    _git(repo, "commit", "-am", "merged-pr")
    return repo, sd, base

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _benchlab(sd, model, 40.0)
    lab.repo = repo  # real repo for revert; gates stubbed except bench
    lab.bench = lambda: (True, 30.0, "slow")
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": True}
    check("regression-quarantined", lab.run_gates(77, intent) is False)
    check("regression-reason",
          any(q["pr"] == 77 and q["reason"] == "perf-regression" for q in lab.quar),
          str(lab.quar))
    check("regression-reverted",
          _git(repo, "rev-parse", "HEAD")[1].strip() == base)
    check("regression-verdict-stored",
          lab.state["bench_results"]["77"]["verdict"] == "regression",
          str(lab.state["bench_results"]))

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _benchlab(sd, model, 40.0)
    lab.repo = repo
    head = _git(repo, "rev-parse", "HEAD")[1].strip()
    lab.bench = lambda: (True, 50.0, "fast")
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": True}
    check("improvement-counted", lab.run_gates(77, intent) is True)
    check("improvement-kept", _git(repo, "rev-parse", "HEAD")[1].strip() == head)
    check("improvement-merged", 77 in lab.state["merged"])
    check("improvement-verdict-stored",
          lab.state["bench_results"]["77"]["verdict"] == "improvement")
    check("improvement-logged", '"improvement"' in open(lab.log_f).read())

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _benchlab(sd, model, 40.0)
    lab.repo = repo
    lab.bench = lambda: (True, 40.5, "parity")
    intent = {"area": "perf", "backends": ["cuda"], "expects_bench_gain": False}
    check("backend-parity-counted", lab.run_gates(77, intent) is True)
    check("backend-parity-verdict",
          lab.state["bench_results"]["77"]["verdict"] == "parity")

# 28. shell entrypoints stay syntactically valid (broken bash = dead lab)
import shutil as _shutil
import subprocess as _sp3
_bash = _shutil.which("bash")
_scripts = ["triage-1k.sh", "llamapatch", "fetch_models.sh", "setup_lab.sh"]
if _bash:
    import re as _re2
    for _s in _scripts:
        # The bash on PATH may be WSL (Windows files under /mnt/c) or
        # Git-Bash (accepts C:/...). Try each spelling; pass on the first
        # that parses.
        _p = os.path.join(_here, _s).replace("\\", "/")
        _m = _re2.match(r"^([A-Za-z]):/(.*)$", _p)
        _cands = ([f"/mnt/{_m.group(1).lower()}/{_m.group(2)}", _p] if _m else [_p])
        _ok, _err = False, ""
        for _c in _cands:
            _r = _sp3.run([_bash, "-n", _c],
                          capture_output=True, text=True, timeout=60)
            if _r.returncode == 0:
                _ok = True
                break
            _err = _r.stderr[:300]
        check(f"bash-syntax-{_s}", _ok, _err)
    _tri = open(os.path.join(_here, "triage-1k.sh")).read()
    check("triage-emits-report", "--report --report-out" in _tri)
    _lp = open(os.path.join(_here, "llamapatch")).read()
    check("entrypoint-report-cmd", "report)" in _lp and "--report" in _lp)
else:
    check("bash-syntax-skipped", True)

# 29. post-run final verify: catch regressions AFTER merges, heal by revert
def _heallab(sd, model, baseline=40.0):
    lab = _benchlab(sd, model, baseline)
    lab.a.heal_walk = 10
    return lab

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    lab.bench = lambda: (True, 40.5, "parity")
    lab.state["merged"] = [77]
    st, _ = lab.final_verify_and_heal([77])
    check("final-clean", st == "clean", st)
    check("final-clean-keeps", lab.state["merged"] == [77])

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "A")
    open(os.path.join(repo, "f.txt"), "w").write("v3\n")
    _git(repo, "commit", "-am", "B")
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 ok")
    def _headbench():
        content = open(os.path.join(repo, "f.txt")).read()
        return (True, 30.0, "slow") if "v3" in content else (True, 40.0, "ok")
    lab.bench = _headbench
    lab.state["merged"] = [71, 72]
    st, _ = lab.final_verify_and_heal([71, 72])
    check("final-healed", st == "healed", st)
    check("final-heal-drops-culprit", lab.state["merged"] == [71],
          str(lab.state["merged"]))
    check("final-heal-quarantines",
          any(q["pr"] == 72 and q["reason"] == "late-regression" for q in lab.quar),
          str(lab.quar))
    check("final-heal-content", open(os.path.join(repo, "f.txt")).read() == "v2\n")

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td, extra="v3\n")
    open(os.path.join(repo, "f.txt"), "w").write("v4\n")
    _git(repo, "commit", "-am", "C")
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.a.heal_walk = 1
    lab.repo = repo
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 30.0, "always slow")
    lab.state["merged"] = [71, 72]
    st, detail = lab.final_verify_and_heal([71, 72])
    check("final-capped", st == "capped", f"{st} {detail}")
    check("final-cap-shrinks-one", lab.state["merged"] == [71],
          str(lab.state["merged"]))

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    calls = {"build": 0}
    def _flaky_build():
        calls["build"] += 1
        return (False, "boom") if calls["build"] == 1 else (True, "ok")
    lab.build = _flaky_build
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 40.0, "parity")
    lab.state["merged"] = [71]
    st, _ = lab.final_verify_and_heal([71])
    check("final-build-healed", st == "healed", st)
    check("final-build-quarantines",
          any(q["pr"] == 71 and q["reason"] == "late-build-failed" for q in lab.quar),
          str(lab.quar))
    check("final-skipped", lab.final_verify_and_heal([])[0] == "skipped")

# 30. only measured improvements stay (perf claims must prove the gain)
with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _benchlab(sd, model, 40.0)
    lab.repo = repo
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 40.5, "parity-noise")
    intent = {"area": "perf", "backends": ["cpu"], "expects_bench_gain": True}
    check("no-improvement-rolled-back", lab.run_gates(77, intent) is False)
    check("no-improvement-quarantined",
          any(q["pr"] == 77 and q["reason"] == "no-improvement" for q in lab.quar),
          str(lab.quar))
    check("no-improvement-reverted",
          _git(repo, "rev-parse", "HEAD")[1].strip() == base)
    check("no-improvement-not-merged", 77 not in lab.state["merged"])

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _benchlab(sd, model, 40.0)
    lab.repo = repo
    head = _git(repo, "rev-parse", "HEAD")[1].strip()
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 40.2, "parity")
    intent = {"area": "fix", "backends": [], "expects_bench_gain": False}
    check("fix-parity-kept", lab.run_gates(77, intent) is True)
    check("fix-parity-head", _git(repo, "rev-parse", "HEAD")[1].strip() == head)

# 31. post-heal improvement verification (gains must survive the batch)
def _implab(sd, model, results):
    lab = _benchlab(sd, model, 40.0)
    lab.state["bench_results"] = dict(results)
    lab.state["merged"] = [int(k) for k, v in results.items()
                           if isinstance(v, dict) and v.get("verdict") == "improvement"]
    return lab

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _implab(sd, model, {"77": {"tg": 48.0, "verdict": "improvement"}})
    lab.bench = lambda: (True, 47.0, "holds")
    st, detail = lab.verify_improvements_final([77])
    check("improvements-verified", st == "verified", f"{st} {detail}")

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _implab(sd, model, {"77": {"tg": 48.0, "verdict": "improvement"}})
    lab.bench = lambda: (True, 41.0, "diluted")
    st, detail = lab.verify_improvements_final([77])
    check("improvements-lost", st == "lost" and "77" in detail, f"{st} {detail}")

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _implab(sd, model, {"77": {"tg": 40.2, "verdict": "parity"}})
    lab.bench = lambda: (_ for _ in ()).throw(AssertionError("must not bench"))
    st, _ = lab.verify_improvements_final([77])
    check("improvements-na", st == "na", st)

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _implab(sd, model, {"77": {"tg": 48.0, "verdict": "improvement"}})
    lab.a.bench_model = ""
    lab.bench = lambda: (_ for _ in ()).throw(AssertionError("must not bench"))
    st, _ = lab.verify_improvements_final([77])
    check("improvements-skipped", st == "skipped", st)

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _implab(sd, model, {"77": {"tg": 48.0, "verdict": "improvement"}})
    import subprocess as _sp4
    lab.bench = lambda: (_ for _ in ()).throw(_sp4.TimeoutExpired("bench", 600))
    st, _ = lab.verify_improvements_final([77])
    check("improvements-flake", st == "unverified", st)

# 32. no ghost gains: heal-reverted PRs must not pass the final gain check
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "A")
    open(os.path.join(repo, "f.txt"), "w").write("v3\n")
    _git(repo, "commit", "-am", "B")
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 ok")
    def _hb():
        content = open(os.path.join(repo, "f.txt")).read()
        return (True, 30.0, "slow") if "v3" in content else (True, 48.0, "ok")
    lab.bench = _hb
    lab.state["merged"] = [71, 72]
    lab.state["bench_results"] = {"71": {"tg": 47.0, "verdict": "improvement"},
                                  "72": {"tg": 48.0, "verdict": "improvement"}}
    st, _ = lab.final_verify_and_heal([71, 72])
    check("ghost-healed", st == "healed", st)
    check("ghost-marked",
          lab.state["bench_results"]["72"]["verdict"] == "late-regression",
          str(lab.state["bench_results"]))
    st2, detail2 = lab.verify_improvements_final([71, 72])
    check("ghost-excluded", st2 == "verified" and "72" not in detail2,
          f"{st2} {detail2}")

with tempfile.TemporaryDirectory() as td:
    repo, sd, base = _gitrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.build = lambda: (False, "boom")
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 40.0, "x")
    lab._build_ok = True  # streak case: infra proven, failure marks the PR
    lab.state["merged"] = [71]
    lab.state["bench_results"] = {"71": {"tg": 48.0, "verdict": "improvement"}}
    st, _ = lab.final_verify_and_heal([71])
    check("ghost-build-marked",
          st == "healed"
          and lab.state["bench_results"]["71"]["verdict"] == "late-build-failed",
          f"{st} {lab.state['bench_results']}")

# 33. never baseline a merged tree (poisoned baseline => false regressions)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")[1].strip()
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "campaign-merge")
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.repo = repo
    lab.state["base_sha"] = base
    calls = {"build": 0, "bench": 0}
    lab.build = lambda: (calls.__setitem__("build", calls["build"] + 1) or (True, "ok"))
    lab.bench = lambda: (calls.__setitem__("bench", calls["bench"] + 1) or (True, 99.0, "x"))
    lab.ensure_baseline()
    check("baseline-deferred", calls == {"build": 0, "bench": 0}
          and lab.state["bench_baseline"] is None, f"{calls}")
    check("baseline-deferred-logged", "baseline-deferred" in open(lab.log_f).read())
    _git(repo, "checkout", base)
    lab.ensure_baseline()
    check("baseline-clean-head",
          calls == {"build": 1, "bench": 2} and lab.state["bench_baseline"] == 99.0,
          f"{calls} {lab.state.get('bench_baseline')}")

# 34. report surfaces post-run heals (caught AFTER merges, not at merge time)
_rq = [{"pr": 72, "reason": "late-regression", "detail": "30 vs 40"},
       {"pr": 73, "reason": "merge-conflict", "detail": "x"}]
_rep = M.build_report({"merged": [71], "quarantined": [72, 73], "batches_done": 1},
                      _rq, [{"number": 71}, {"number": 72}, {"number": 73}])
check("report-late-healed", "post-run healed" in _rep and "#72" in _rep
      and "#73" not in _rep.split("post-run healed")[1].split("\n")[0])
check("report-no-late",
      "post-run healed" not in M.build_report({"merged": [71]}, [], [{"number": 71}]))

# 35. git ops time out in minutes, not half-hours (hung fetch stalls batches)
with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    a = _mklab(os.path.join(td, "norepo"), sd, [{"number": 1}])
    lab = M.Lab(a)
    _orig_sh, _seen = M.sh, {}
    def _cap_sh(cmd, cwd, **kw):
        _seen.clear()
        _seen.update(kw)
        _seen["cmd"] = cmd
        return 0, ""
    M.sh = _cap_sh
    try:
        rc, _ = lab.git("status --porcelain")
        check("git-timeout-default", rc == 0 and _seen.get("timeout") == 300,
              str(_seen))
        lab.git("fetch origin pull/1/head:pr/1 --force", timeout=60)
        check("git-timeout-override", _seen.get("timeout") == 60, str(_seen))
    finally:
        M.sh = _orig_sh

# 36. tested head SHAs recorded per merge (forensics for stale fallbacks)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "pr/77")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "pr77")
    _git(repo, "checkout", "master")
    a = _mklab(repo, sd, [{"number": 77}])
    lab = M.Lab(a)
    expect = _git(repo, "rev-parse", "pr/77")[1].strip()
    check("pr-head", lab.pr_head(77) == expect and len(expect) == 40, expect[:8])
    check("pr-head-missing", lab.pr_head(999) == "")
    lab.record_merged(77)
    lab.record_merged(77)
    check("recorded-head",
          lab.state["merged_heads"].get("77") == expect, str(lab.state["merged_heads"]))
    check("recorded-once", lab.state["merged"] == [77], str(lab.state["merged"]))
    rep = M.build_report(lab.state, [], [{"number": 77, "title": "t"}])
    check("report-head-col", expect[:8] in rep)

# 37. stale local fallback refused; current fallback merges
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "pr/5")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "pr5")
    _git(repo, "checkout", "master")
    head5 = _git(repo, "rev-parse", "pr/5")[1].strip()
    a = _mklab(repo, sd, [{"number": 5, "head_full": "0" * 40, "title": "stale"}])
    lab = M.Lab(a)
    ok, reason = lab.merge_one_committed(5, "batch-0")
    check("stale-refused", not ok and (reason or "").startswith("fetch-failed: stale"),
          str(reason))
    check("stale-transient", M.is_transient_quarantine(reason))
    a2 = _mklab(repo, sd, [{"number": 5, "head_full": head5, "title": "cur"}])
    lab2 = M.Lab(a2)
    ok2, reason2 = lab2.merge_one_committed(5, "batch-0")
    check("current-fallback-merges", ok2, str(reason2))

# 42. baseline anchors on the mean of two runs (one number, 50 verdicts)
with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.state["base_sha"] = "abc123"
    lab.build = lambda: (True, "ok")
    vals = [40.0, 44.0]
    lab.bench = lambda: (True, vals.pop(0) if vals else 44.0, "out")
    lab.ensure_baseline()
    check("baseline-mean",
          lab.state["bench_baseline"] == 42.0
          and lab.state.get("bench_baseline_runs") == [40.0, 44.0],
          str({k: lab.state.get(k) for k in ("bench_baseline", "bench_baseline_runs")}))

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.a.bench_noise_pct = 0  # operator-disabled: single run
    lab.state["base_sha"] = "abc123"
    n = {"bench": 0}
    lab.build = lambda: (True, "ok")
    lab.bench = lambda: (n.__setitem__("bench", n["bench"] + 1) or (True, 42.0, "out"))
    lab.ensure_baseline()
    check("baseline-single-when-disabled",
          n == {"bench": 1} and lab.state["bench_baseline"] == 42.0, str(n))

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _baselab(sd, model)
    lab.state["base_sha"] = "abc123"
    lab.build = lambda: (True, "ok")
    calls = {"n": 0}
    def _oneflake():
        calls["n"] += 1
        return (True, 42.0, "ok") if calls["n"] == 1 else (False, None, "flaked")
    lab.bench = _oneflake
    lab.ensure_baseline()
    check("baseline-flake-keeps-first", lab.state["bench_baseline"] == 42.0,
          str(lab.state.get("bench_baseline")))

# 43. doctor warnings persist into state and render in the report
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 7, "title": "seven"}])
    a.base = "master"
    lab = M.Lab(a)
    lab.state["base_sha"] = _git(repo, "rev-parse", "HEAD")[1].strip()
    _git(repo, "checkout", "-b", "pr/7")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "pr7")
    _git(repo, "checkout", "master")
    _git(repo, "merge", "--no-ff", "--no-edit", "pr/7")
    _git(repo, "commit", "--amend", "-m", "pr-lab: merge #7 seven")
    lab.state["merged"] = [7]
    lab.state["merged_heads"] = {"7": "0" * 40}  # wrong on purpose
    lab.save()
    lab.doctor()
    check("doctor-warnings-persisted",
          lab.state.get("doctor_warnings") == lab.doctor_warnings
          and any("#7" in w for w in lab.doctor_warnings),
          str(lab.state.get("doctor_warnings")))
    rep = M.build_report(lab.state, [], [{"number": 7, "title": "seven"}])
    check("report-warnings-section",
          "## doctor warnings" in rep and "#7 tree parent" in rep)
    check("report-no-warnings-quiet",
          "## doctor warnings" not in M.build_report({"merged": []}, [], []))

# 48. report surfaces confirmation runs (invisible guards get distrusted)
_rep48 = M.build_report(
    {"merged": [7], "bench_baseline": 40.0, "bench_baseline_sha": "abc123",
     "bench_baseline_runs": [40.0, 44.0],
     "bench_results": {"7": {"tg": 36.75, "base": 40.0, "verdict": "parity",
                             "runs": [33.5, 40.0]}}},
    [], [{"number": 7, "title": "seven"}])
check("report-runs-row", "36.75 vs 40.0 (2 runs)" in _rep48, _rep48)
check("report-runs-base", "baseline tg: 40.0 @ abc123 (n=2)" in _rep48, _rep48)
_rep48s = M.build_report(
    {"merged": [7], "bench_results": {"7": {"tg": 40.0, "verdict": "parity",
                                            "runs": [40.0]}}},
    [], [{"number": 7, "title": "seven"}])
check("report-runs-single-quiet", "(2 runs)" not in _rep48s and "(n=2)" not in _rep48s)

# 49. crash-safe state saves (a kill mid-write must not corrupt campaigns)
with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    p = os.path.join(sd, "x.json")
    M.atomic_write_json(p, {"a": [1, 2, 3]})
    check("atomic-roundtrip", json.load(open(p)) == {"a": [1, 2, 3]})
    check("atomic-no-tmp", [f for f in os.listdir(sd) if ".tmp-" in f] == [],
          str(os.listdir(sd)))
    M.atomic_write_json(p, {"b": "overwritten"})
    check("atomic-replace", json.load(open(p)) == {"b": "overwritten"})

with tempfile.TemporaryDirectory() as td:
    sd = os.path.join(td, "st"); os.makedirs(sd)
    a = _mklab(os.path.join(td, "norepo"), sd, [{"number": 9}])
    lab = M.Lab(a)
    lab.state["merged"] = [8]
    lab.state["bench_baseline_runs"] = [40.0, 44.0]
    lab.quarantine(9, "fetch-failed", "blip")
    lab2 = M.Lab(_mklab(os.path.join(td, "norepo"), sd, [{"number": 9}]))
    check("save-state-survives",
          lab2.state["merged"] == [8]
          and lab2.state.get("bench_baseline_runs") == [40.0, 44.0],
          str(lab2.state))
    check("save-quar-survives",
          [q["pr"] for q in lab2.quar] == [9]
          and lab2.state["quarantined"] == [9],
          f"{lab2.quar} {lab2.state['quarantined']}")
    check("save-no-tmp-left", [f for f in os.listdir(sd) if ".tmp-" in f] == [],
          str(os.listdir(sd)))

# 50. base_ref fallback chain (no eager-.get crash on minimal args)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 1}])  # no .base attr on purpose
    lab = M.Lab(a)
    check("base-ref-default", lab.base_ref() == "master", lab.base_ref())
    lab.state["base_sha"] = "abc123"
    check("base-ref-state", lab.base_ref() == "abc123")
    a.base = "develop"
    lab.state["base_sha"] = None
    check("base-ref-arg", lab.base_ref() == "develop")
    lab.doctor()  # must not crash on minimal args
    check("doctor-minimal-args", lab.state["merged"] == [])

# 51. intent priority edges (wrong expects_gain rolls back good PRs)
from pr_intent import classify_intent as _ci, detect_backends as _db
_fixperf = _ci({"number": 1, "title": "fix CUDA crash, faster path",
                "labels": [], "files": ["ggml/src/ggml-cuda/x.cu"]})
check("intent-fix-beats-perf", _fixperf["area"] == "fix"
      and not _fixperf["expects_bench_gain"], str(_fixperf))
_metal = _ci({"number": 2, "title": "faster metal kernels",
              "labels": [], "files": ["ggml/src/ggml-metal/x.metal"]})
check("intent-metal-backend-only", "metal" in _metal["backends"]
      and not _metal["expects_bench_gain"], str(_metal))
_vk = _ci({"number": 3, "title": "vulkan: pack FMA",
           "labels": ["vulkan"], "files": []})
check("intent-vulkan-label", "vulkan" in _vk["backends"]
      and _vk["area"] != "fix", str(_vk))
_feat = _ci({"number": 4, "title": "add Qwen3 support", "labels": [], "files": []})
check("intent-feature", _feat["area"] == "feature"
      and not _feat["expects_bench_gain"], str(_feat))
_empty = _ci({})
check("intent-empty-safe", _empty["area"] == "other"
      and not _empty["expects_bench_gain"] and _empty["backends"] == [],
      str(_empty))
_srv = _ci({"number": 5, "title": "faster parallel decoding",
            "labels": [], "files": ["tools/server/server.cpp"]})
check("intent-server-cpu-visible", _srv["expects_bench_gain"], str(_srv))
check("intent-backends-pure", _db({"labels": ["CUDA"], "title": "", "files": []}) == ["cuda"])

# 52. stage-2 enrichment end to end (mocked API, real main())
import urllib.request as _ur
_PRS52 = [
    {"number": 11, "title": "faster sgemm kernels", "body": "",
     "user": {"login": "a"}, "updated_at": "2026-01-02T00:00:00Z",
     "labels": [{"name": "ggml"}], "draft": False, "head": {"sha": "a" * 40}},
    {"number": 12, "title": "fix crash on load", "body": "",
     "labels": [{"name": "bug"}, "oops"], "draft": False},  # no head/user
    {"number": "x", "title": "junk"},  # skipped: no int number
    "junk-string",                      # skipped: not a dict
    {"number": 13, "title": "wip experiment", "body": "", "user": {"login": "b"},
     "updated_at": "2026-01-03T00:00:00Z", "labels": [], "draft": True},
]
_DET52 = {
    11: {"mergeable": True, "mergeable_state": "clean", "additions": 100,
         "deletions": 50, "changed_files": 2, "head": {"sha": "b" * 40}},
    12: {"mergeable": False, "mergeable_state": "dirty", "additions": 1500,
         "deletions": 800, "changed_files": 3, "head": {"sha": "c" * 40}},
}
_FILES52 = {11: ["ggml/src/ggml-cpu/sgemm.cpp", "ggml/src/ggml-cpu/x.cpp"],
            12: ["src/llama.cpp"]}
_CI52 = {"b" * 40: "success", "c" * 40: "failure"}
def _mockreq52(url, token, timeout=60):
    if url.endswith("/files?per_page=100"):
        n = int(url.split("/pulls/")[1].split("/")[0])
        return ([{"filename": p} for p in _FILES52[n]], None)
    if "/commits/" in url:
        sha = url.rstrip("/").split("/")[-2]
        return ({"state": _CI52[sha], "statuses": [{}, {}]}, None)
    if "/pulls/" in url and "state=open" not in url:
        return (_DET52[int(url.rstrip("/").split("/")[-1])], None)
    return (list(_PRS52), None)
_argv52, _req52, _sleep52 = sys.argv, F.req, F.time.sleep
F.time.sleep = lambda s: None
try:
    with tempfile.TemporaryDirectory() as td52:
        _out52 = os.path.join(td52, "cands.json")
        sys.argv = ["fetch_prs.py", "--limit", "10", "--top", "5",
                    "--out", _out52, "--include-ci", "--ci-sleep", "0"]
        F.req = _mockreq52
        F.main()
        _got52 = {c["number"]: c for c in json.load(open(_out52))}
        check("stage2-skips-malformed", sorted(_got52) == [11, 12], str(sorted(_got52)))
        check("stage2-head-default", _got52[12]["head"] == "" and _got52[12]["user"] == "")
        _c11, _c12 = _got52[11], _got52[12]
        check("stage2-head-full", _c11["head_full"] == "b" * 40 and _c12["head_full"] == "c" * 40)
        check("stage2-perf-bonus", "touches-perf-path" in _c11["hits"] and _c11["score"] >= 4,
              f"{_c11['score']} {_c11['hits']}")
        check("stage2-penalties", "dirty-penalty" in _c12["hits"]
              and "large-diff-penalty" in _c12["hits"]
              and "ci-red-penalty" in _c12["hits"] and _c12.get("ci_state") == "failure",
              f"{_c12['score']} {_c12['hits']}")
        check("stage2-intent", _c11["intent"]["expects_bench_gain"] is True
              and _c12["intent"]["area"] == "fix",
              f"{_c11['intent']} {_c12['intent']}")
        check("stage2-order", _c11["score"] > _c12["score"])

    # breaker: systematic stage-2 failure aborts fast with partial output
    _plist52 = [{"number": 20 + i, "title": f"pr {i}", "body": "",
                 "user": {"login": "u"}, "updated_at": "2026-01-01T00:00:00Z",
                 "labels": [], "draft": False, "head": {"sha": "d" * 40}}
                for i in range(6)]
    def _deadreq52(url, token, timeout=60):
        if "state=open" in url:
            return (list(_plist52), None)
        raise _ur.HTTPError(url, 404, "Not Found", None, None)
    with tempfile.TemporaryDirectory() as td52b:
        _out52b = os.path.join(td52b, "cands.json")
        sys.argv = ["fetch_prs.py", "--limit", "10", "--top", "6",
                    "--out", _out52b, "--ci-sleep", "0"]
        F.req = _deadreq52
        try:
            F.main()
            check("breaker-exits", False, "main must sys.exit(2)")
        except SystemExit as e:
            check("breaker-exits", e.code == 2, f"exit={e.code}")
        _part52 = json.load(open(_out52b))
        check("breaker-partial", len(_part52) == 6
              and all("error" in c for c in _part52[:5]),
              str([(c["number"], c.get("error")) for c in _part52]))
finally:
    sys.argv, F.req, F.time.sleep = _argv52, _req52, _sleep52

# 44. fallback without triaged head_full merges (unenriched candidates skip
# the freshness check instead of blocking on it)
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "pr/6")
    open(os.path.join(repo, "f.txt"), "w").write("v2\n")
    _git(repo, "commit", "-am", "pr6")
    _git(repo, "checkout", "master")
    a = _mklab(repo, sd, [{"number": 6, "title": "nohead"}])  # no head_full
    lab = M.Lab(a)
    ok, reason = lab.merge_one_committed(6, "batch-0")
    check("missing-head-skips-check", ok, str(reason))

# 45. threshold + wrapper consistency (drift between knobs breaks campaigns)
_here45 = os.path.dirname(os.path.abspath(__file__))
_ml45 = open(os.path.join(_here45, "merge_lab.py")).read()
_triage45 = open(os.path.join(_here45, "triage-1k.sh")).read()
_dispatcher45 = open(os.path.join(_here45, "llamapatch")).read()
_cfg45 = json.load(open(os.path.join(_here45, "config.json")))
check("config-noise-knob", _cfg45.get("bench_noise_pct") == 5.0, str(_cfg45.get("bench_noise_pct")))
check("config-noise-default-match", "--bench-noise-pct" in _ml45
      and "default=5.0" in _ml45)
check("triage-forwards-flags",
      _triage45.count('"$@"') >= 1, "merge line must forward operator flags")
check("dispatcher-forwards-flags",
      _dispatcher45.count('"$@"') >= 3, "merge/dry/doctor/report forward flags")

# 47. triage scoring contract (wrong picks waste whole 10-batches)
_tcfg = {"perf_keywords": ["fix", "bug", "cuda", "flash attention", "q4_0"],
         "perf_paths": ["ggml/src", "src/", "common/"]}
check("score-hits", F.score_title("vulkan cuda kernels", "", _tcfg)[0] >= 1)
check("score-stem", "fix" in F.score_title("fixes crash on load", "", _tcfg)[1])
check("score-no-debug", "bug" not in F.score_title("debug helper", "", _tcfg)[1])
check("score-no-prefix", "fix" not in F.score_title("prefix handling", "", _tcfg)[1])
check("score-phrase", "flash attention" in F.score_title("Flash Attention kernel", "", _tcfg)[1])
check("score-punct", "q4_0" in F.score_title("q4_0 quant fix", "", _tcfg)[1])
check("score-empty", F.score_title("", "", _tcfg) == (0, []))
check("perf-prefix", F.touches_perf(["ggml/src/foo.c"], _tcfg) is True)
check("perf-exact-dir", F.touches_perf(["common"], _tcfg) is True)
check("perf-no-substring", F.touches_perf(["srcfoo/x.c", "mycommon/y.c"], _tcfg) is False)
import urllib.error as _ue
from email.message import Message as _Msg
_h = _Msg(); _h["X-RateLimit-Remaining"] = "0"
check("ratelimit-429", F.is_rate_limit(_ue.HTTPError("u", 429, "too many", None, None)) is True)
check("ratelimit-quota", F.is_rate_limit(_ue.HTTPError("u", 403, "forbidden", _h, None)) is True)
check("ratelimit-msg", F.is_rate_limit(_ue.HTTPError("u", 403, "rate limit exceeded", None, None)) is True)
check("ratelimit-other403", F.is_rate_limit(_ue.HTTPError("u", 403, "forbidden", None, None)) is False)
check("ratelimit-nontype", F.is_rate_limit(ValueError("x")) is False)
sleeps = []
def _flaky(page):
    if len(sleeps) < 2:
        raise ConnectionError("blip")
    return ([f"p{page}"], True)
items, err = F.collect_pages(_flaky, 10, sleep=sleeps.append, retries=3)
check("pages-retry", items == ["p1"] and err is None and sleeps == [1, 2],
      f"{items} {err} {sleeps}")
def _dead(page):
    if page == 1:
        return (["a"], False)
    raise ConnectionError("down")
items, err = F.collect_pages(_dead, 10, sleep=lambda s: None, retries=1)
check("pages-partial", items == ["a"] and isinstance(err, ConnectionError),
      f"{items} {err}")
items, err = F.collect_pages(lambda p: ([1, 2, 3], False), 2, sleep=lambda s: None)
check("pages-limit", items == [1, 2] and err is None, f"{items}")

# 46. lifecycle details: lock names the hatch, identity never clobbers
with tempfile.TemporaryDirectory() as td:
    M.acquire_lock(td)
    try:
        M.acquire_lock(td)
        check("lock-names-hatch", False, "second acquire must raise")
    except RuntimeError as e:
        check("lock-names-hatch", "--force-unlock" in str(e)
              and "another run may be active" in str(e), str(e)[:200])
    finally:
        M.release_lock(td)

with tempfile.TemporaryDirectory() as td:
    # Isolate from the operator's global git identity: HOME/USERPROFILE
    # redirect + NOSYSTEM so effective config starts empty.
    home = os.path.join(td, "home"); os.makedirs(home)
    saved = {k: os.environ.get(k) for k in
             ("HOME", "USERPROFILE", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM")}
    os.environ["HOME"] = home; os.environ["USERPROFILE"] = home
    os.environ.pop("GIT_CONFIG_GLOBAL", None)
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
        os.makedirs(repo); os.makedirs(sd)
        _git(repo, "init", "-b", "master")
        _git(repo, "config", "user.email", "mine@example.com")
        a = _mklab(repo, sd, [{"number": 1}])
        lab = M.Lab(a)
        lab.ensure_identity()
        check("identity-keeps-email",
              _git(repo, "config", "user.email")[1].strip() == "mine@example.com")
        check("identity-fills-name",
              _git(repo, "config", "user.name")[1].strip() == "pr-lab")
        _git(repo, "config", "--unset", "user.email")
        _git(repo, "config", "--unset", "user.name")
        lab.ensure_identity()
        check("identity-fills-both",
              _git(repo, "config", "user.email")[1].strip() == "pr-lab@localhost"
              and _git(repo, "config", "user.name")[1].strip() == "pr-lab")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

# 53. main() CLI wiring (argv -> dry/report without touching the repo)
import contextlib as _cl
_argv53 = sys.argv
try:
    with tempfile.TemporaryDirectory() as td53:
        cf53 = os.path.join(td53, "c.json")
        json.dump([{"number": 1, "title": "t"}, {"number": 2, "title": "u"}],
                  open(cf53, "w"))
        sd53 = os.path.join(td53, "st"); os.makedirs(sd53)
        sys.argv = ["merge_lab.py", "--candidates", cf53, "--state-dir", sd53,
                    "--batch", "1", "--max-prs", "1", "--dry-run"]
        _buf53 = _io.StringIO()
        with _cl.redirect_stdout(_buf53):
            M.main()
        check("cli-dry-run", "batch: [1]" in _buf53.getvalue(), _buf53.getvalue()[:200])
        rep53 = os.path.join(td53, "r.md")
        sys.argv = ["merge_lab.py", "--candidates", cf53, "--state-dir", sd53,
                    "--report", "--report-out", rep53]
        with _cl.redirect_stdout(_io.StringIO()):
            M.main()
        _r53 = open(rep53).read()
        check("cli-report", "llamapatch report" in _r53 and "pending: 2" in _r53,
              _r53[:200])
        check("cli-no-lock", not os.path.exists(os.path.join(sd53, "lab.lock")),
              str(os.listdir(sd53)))
finally:
    sys.argv = _argv53

# 55. review fixes: generic errors trip the breaker, None intent safe,
# doctor_warnings always defined
_argv55, _req55, _sleep55 = sys.argv, F.req, F.time.sleep
F.time.sleep = lambda s: None
try:
    with tempfile.TemporaryDirectory() as td55:
        _out55 = os.path.join(td55, "cands.json")
        _plist55 = [{"number": 30 + i, "title": f"pr {i}", "body": "",
                     "user": {"login": "u"}, "updated_at": "2026-01-01T00:00:00Z",
                     "labels": [], "draft": False, "head": {"sha": "e" * 40}}
                    for i in range(6)]
        def _conndead55(url, token, timeout=60):
            if "state=open" in url:
                return (list(_plist55), None)
            raise ConnectionError("proxy down")
        sys.argv = ["fetch_prs.py", "--limit", "10", "--top", "6",
                    "--out", _out55, "--ci-sleep", "0"]
        F.req = _conndead55
        try:
            F.main()
            check("breaker-generic-exits", False, "main must sys.exit(2)")
        except SystemExit as e:
            check("breaker-generic-exits", e.code == 2, f"exit={e.code}")
        _part55 = json.load(open(_out55))
        check("breaker-generic-partial", len(_part55) == 6
              and all("error" in c for c in _part55[:5]),
              str([(c["number"], c.get("error")) for c in _part55]))
finally:
    sys.argv, F.req, F.time.sleep = _argv55, _req55, _sleep55
check("confirm-none-intent-reg", M.needs_confirm(None, 100, 83, 15, 5.0) is True)
check("confirm-none-intent-above", M.needs_confirm(None, 100, 118, 15, 5.0) is False)
with tempfile.TemporaryDirectory() as td55b:
    sd55b = os.path.join(td55b, "st"); os.makedirs(sd55b)
    a55b = _mklab(os.path.join(td55b, "norepo"), sd55b, [{"number": 1}])
    check("doctor-warnings-default", M.Lab(a55b).doctor_warnings == [])

# 58. PR content without full repos: shallow-boundary deepen+retry,
# setup_lab.sh provisioning shape
_setup58 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "setup_lab.sh")).read()
check("setup-partial", "--filter=blob:none" in _setup58)
check("setup-shallow", "--depth" in _setup58)
check("setup-sparse", "sparse-checkout" in _setup58)
check("setup-base", 'LAB_BASE' in _setup58 and 'checkout "$BASE"' in _setup58)
check("setup-idempotent", ".git" in _setup58 and "exists (skip clone)" in _setup58)

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 61, "title": "shallow-pr"}])
    lab = M.Lab(a)
    _calls58, _merges58 = [], {"n": 0}
    def _fakegit58(cmd, check=False, timeout=300):
        _calls58.append(cmd)
        if cmd.startswith("fetch --deepen"):
            return 0, "deepened"
        if cmd.startswith("fetch origin"):
            return 0, ""
        if cmd.startswith("merge --no-ff"):
            _merges58["n"] += 1
            if _merges58["n"] == 1:
                return 1, "fatal: refusing to merge unrelated histories (shallow boundary)"
            return 0, "Merge made by test"
        if cmd == "rev-parse HEAD":
            return 0, "aaa\n" if _merges58["n"] <= 1 else "bbb\n"
        if cmd.startswith("rev-list"):
            return 0, "bbb p1 p2\n"
        if cmd.startswith("diff HEAD"):
            return 0, " f.txt | 2 +-"
        return 0, ""
    lab.git = _fakegit58
    lab.git_args = lambda args: (0, "")
    ok, reason = lab._merge_one_inner(61, "batch-0")
    check("shallow-deepens-retries", ok and reason is None, f"{ok} {reason}")
    check("shallow-deepen-called",
          any(c.startswith("fetch --deepen") for c in _calls58), str(_calls58))
    check("shallow-deepen-logged", '"shallow-deepen"' in open(lab.log_f).read())

with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 62, "title": "conflict-pr"}])
    lab = M.Lab(a)
    _calls58b = []
    def _fakegit58b(cmd, check=False, timeout=300):
        _calls58b.append(cmd)
        if cmd.startswith("fetch origin"):
            return 0, ""
        if cmd.startswith("merge --no-ff"):
            return 1, "Auto-merging f.txt\nCONFLICT (content): Merge conflict"
        return 0, ""
    lab.git = _fakegit58b
    ok, reason = lab._merge_one_inner(62, "batch-0")
    check("conflict-no-deepen", not ok and str(reason).startswith("merge-conflict"),
          f"{ok} {str(reason)[:80]}")
    check("conflict-skips-deepen",
          not any(c.startswith("fetch --deepen") for c in _calls58b))

# 56. first-ever smoke failure verifies infra before blaming the PR
_fix56 = {"area": "fix", "backends": [], "expects_bench_gain": False}
def _smokes56(seq):
    seq = list(seq)
    def _s():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return _s

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.smoke = _smokes56([(False, "nope"), (True, "tg32 : 40 t/s")])
    lab.bench = lambda: (True, 40.0, "v=40")
    check("smoke-infra-quarantines-pr", lab.run_gates(77, _fix56) is False)
    check("smoke-infra-reason",
          any(q["pr"] == 77 and q["reason"] == "smoke-failed" for q in lab.quar),
          str(lab.quar))
    check("smoke-infra-reverted", open(os.path.join(repo, "f.txt")).read() == "v1\n")
    check("smoke-infra-flag", lab._smoke_green is True)

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.smoke = lambda: (False, "dead")
    try:
        lab.run_gates(77, _fix56)
        check("smoke-infra-aborts", False, "must raise, not quarantine")
    except RuntimeError as e:
        check("smoke-infra-aborts", "NOT quarantined" in str(e), str(e)[:200])
    check("smoke-infra-no-burn", lab.state["quarantined"] == [] and lab.quar == [])

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    lab.bench = lambda: (True, 40.0, "v=40")
    check("smoke-streak-pass", lab.run_gates(77, _fix56) is True)
    calls = {"build": 0}
    _b = lab.build
    lab.build = lambda: (calls.__setitem__("build", calls["build"] + 1) or _b())
    lab.smoke = lambda: (False, "broke")
    check("smoke-streak-fails-pr", lab.run_gates(78, _fix56) is False)
    check("smoke-streak-no-reverify", calls == {"build": 1}, str(calls))
    check("smoke-streak-quarantined",
          any(q["pr"] == 78 and q["reason"] == "smoke-failed" for q in lab.quar),
          str(lab.quar))

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.smoke = lambda: (False, "dead")
    lab.bench = lambda: (True, 40.0, "v=40")
    lab.state["merged"] = [71]
    try:
        lab.final_verify_and_heal([71])
        check("final-smoke-infra-aborts", False, "must raise, not quarantine")
    except RuntimeError as e:
        check("final-smoke-infra-aborts", "NOT" in str(e) and "quarantined" in str(e),
              str(e)[:200])
    check("final-smoke-no-burn",
          lab.state["merged"] == [71] and lab.state["quarantined"] == [],
          f"{lab.state['merged']} {lab.state['quarantined']}")

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.smoke = _smokes56([(False, "nope"), (True, "tg32 : 40 t/s")])
    lab.bench = lambda: (True, 40.0, "v=40")
    lab.state["merged"] = [71]
    st, _ = lab.final_verify_and_heal([71])
    check("final-smoke-heals-pr", st == "healed", st)
    check("final-smoke-late-quar",
          any(q["pr"] == 71 and q["reason"] == "late-smoke-failed" for q in lab.quar),
          str(lab.quar))

# 57. first-ever build failure verifies infra before blaming the PR
def _builds57(seq):
    seq = list(seq)
    def _b():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return _b

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.build = _builds57([(False, "boom"), (True, "ok")])
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    lab.bench = lambda: (True, 40.0, "v=40")
    check("build-infra-quarantines-pr", lab.run_gates(77, _fix56) is False)
    check("build-infra-reason",
          any(q["pr"] == 77 and q["reason"] == "build-failed" for q in lab.quar),
          str(lab.quar))
    check("build-infra-flag", lab._build_ok is True)

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.build = lambda: (False, "no cmake")
    try:
        lab.run_gates(77, _fix56)
        check("build-infra-aborts", False, "must raise, not quarantine")
    except RuntimeError as e:
        check("build-infra-aborts", "NOT quarantined" in str(e), str(e)[:200])
    check("build-infra-no-burn", lab.state["quarantined"] == [] and lab.quar == [])

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _confirmlab(sd, model, repo)
    lab.build = lambda: (True, "ok")
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    lab.bench = lambda: (True, 40.0, "v=40")
    check("build-streak-pass", lab.run_gates(77, _fix56) is True)
    calls = {"build": 0}
    _bb = lab.build
    lab.build = lambda: (calls.__setitem__("build", calls["build"] + 1)
                         or (False, "boom"))
    lab.smoke = lambda: (True, "tg32 : 40 t/s")
    check("build-streak-fails-pr", lab.run_gates(78, _fix56) is False)
    check("build-streak-no-reverify", calls == {"build": 1}, str(calls))

with tempfile.TemporaryDirectory() as td:
    repo, sd = _confirmrepo(td)
    model = os.path.join(td, "m.gguf"); open(model, "w").write("x")
    lab = _heallab(sd, model)
    lab.repo = repo
    lab.build = _builds57([(False, "boom"), (True, "ok")])
    lab.smoke = lambda: (True, "tg32 ok")
    lab.bench = lambda: (True, 40.0, "parity")
    lab.state["merged"] = [71]
    st, _ = lab.final_verify_and_heal([71])
    check("final-build-heals-pr", st == "healed", st)
    check("final-build-late-quar",
          any(q["pr"] == 71 and q["reason"] == "late-build-failed" for q in lab.quar),
          str(lab.quar))

# 60. patch manager API (stdlib GUI backend, live localhost server)
import patch_manager as PM
import socketserver as _ss60
import http.client as _hc60
import threading as _th60
import time as _t60
import urllib.parse as _up60
_srv60 = _ss60.ThreadingTCPServer(("127.0.0.1", 0), PM.Handler)
_port60 = _srv60.server_address[1]
_th60.Thread(target=_srv60.serve_forever, daemon=True).start()
def _get60(path):
    c = _hc60.HTTPConnection("127.0.0.1", _port60, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, r.read().decode()
def _post60(path, obj):
    raw = json.dumps(obj).encode()
    c = _hc60.HTTPConnection("127.0.0.1", _port60, timeout=30)
    c.request("POST", path, raw, {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.read().decode()
try:
    st, body = _get60("/")
    check("gui-index", st == 200 and "checkbox" in body and "Merge selected" in body,
          str(st))
    with tempfile.TemporaryDirectory() as td60:
        cf60 = os.path.join(td60, "c.json")
        json.dump([{"number": 1, "title": "one", "score": 5,
                    "intent": {"area": "perf"}},
                   {"number": 2, "title": "two", "score": 1},
                   {"number": 3, "title": "three", "score": 0}], open(cf60, "w"))
        sd60 = os.path.join(td60, "st"); os.makedirs(sd60)
        json.dump({"merged": [1], "quarantined": [2], "bench_baseline": None,
                   "batches_done": 0}, open(os.path.join(sd60, "lab-state.json"), "w"))
        json.dump([], open(os.path.join(sd60, "quarantined.json"), "w"))
        st, body = _get60("/api/candidates?file=" + _up60.quote(cf60)
                          + "&state_dir=" + _up60.quote(sd60))
        _d60 = json.loads(body)
        check("gui-candidates",
              st == 200 and {c["number"]: c["status"] for c in _d60["candidates"]}
              == {1: "merged", 2: "quarantined", 3: "pending"}, body[:300])
        check("gui-area", [c for c in _d60["candidates"] if c["number"] == 1][0]["area"] == "perf")
        st, _ = _get60("/api/candidates?file=" + os.path.join(td60, "nope.json"))
        check("gui-missing", st == 404, str(st))
        bad60 = os.path.join(td60, "bad.json")
        json.dump({"not": "a list"}, open(bad60, "w"))
        st, _ = _get60("/api/candidates?file=" + _up60.quote(bad60))
        check("gui-malformed", st == 422, str(st))
        st, _ = _post60("/api/merge", {"candidates": cf60, "repo": td60,
                                       "numbers": []})
        check("gui-empty-selection", st == 400, str(st))
        st, _ = _post60("/api/merge", {"candidates": cf60, "repo": td60,
                                       "numbers": [999]})
        check("gui-unknown-pr", st == 400, str(st))
        st, _ = _post60("/api/merge", {"candidates": cf60, "repo": "",
                                       "numbers": [3]})
        check("gui-no-repo", st == 400, str(st))
        st, _ = _post60("/api/triage", {"slug": "bogus"})
        check("gui-bad-slug", st == 400, str(st))
        st, _ = _post60("/api/setup", {"url": "https://example/x.git", "dest": ""})
        check("gui-setup-validation", st == 400, str(st))
        st, body = _post60("/api/merge", {"candidates": cf60, "repo": td60,
                                         "state_dir": os.path.join(td60, "stm"),
                                         "numbers": [2, 3], "dry_run": True,
                                         "batch": 10, "max_prs": 10})
        _mid60 = json.loads(body).get("id")
        check("gui-merge-accepted", st == 200 and isinstance(_mid60, int),
              f"{st} {body[:150]}")
        _done60, _tail60 = False, ""
        for _ in range(60):
            _t60.sleep(0.5)
            st, body = _get60(f"/api/runs/{_mid60}")
            _r60 = json.loads(body)
            if _r60.get("status") != "running":
                _done60 = _r60.get("status") == "done"
                _tail60 = _r60.get("log_tail", "")
                break
        check("gui-merge-dry-run", _done60 and "batch: [2, 3]" in _tail60,
              _tail60[-300:])
        _sel60 = json.load(open(os.path.join(td60, "stm",
                                             "candidates-selected.json")))
        check("gui-selected-file", sorted(c["number"] for c in _sel60) == [2, 3],
              str(_sel60))
        st, body = _get60("/api/report?candidates=" + _up60.quote(cf60)
                          + "&state_dir=" + _up60.quote(sd60))
        check("gui-report", st == 200 and "llamapatch report" in body, body[:150])
finally:
    pass  # server stays up for section 61; shut down at file end

# 61. run cancel kills the tree and only our own stale lock
st, _ = _post60("/api/runs/999999/cancel", {})
check("cancel-unknown", st == 404, str(st))
with tempfile.TemporaryDirectory() as td61:
    sd61 = os.path.join(td61, "st"); os.makedirs(sd61)
    _log61 = os.path.join(td61, "t.log")
    _rid61 = PM.Handler.app.start_run(
        "test", [sys.executable, "-c", "import time; time.sleep(120)"],
        _log61, state_dir=sd61)
    for _ in range(100):
        _pid61 = PM.Handler.app.runs[_rid61].get("pid")
        if _pid61:
            break
        _t60.sleep(0.1)
    json.dump({"pid": _pid61 or 0}, open(os.path.join(sd61, "lab.lock"), "w"))
    st, body = _post60(f"/api/runs/{_rid61}/cancel", {})
    _c61 = json.loads(body)
    check("cancel-ok", st == 200 and _c61.get("status") == "cancelled",
          f"{st} {body[:200]}")
    _proc61 = PM.Handler.app.runs[_rid61].get("proc")
    check("cancel-dead", _proc61 is not None and _proc61.poll() is not None)
    check("cancel-owned-lock",
          not os.path.exists(os.path.join(sd61, "lab.lock"))
          and "cleared" in _c61.get("note", ""), _c61.get("note", ""))
    st, body = _post60(f"/api/runs/{_rid61}/cancel", {})
    check("cancel-idempotent", st == 200 and "already finished" in body, body[:150])
    _rid61b = PM.Handler.app.start_run(
        "test", [sys.executable, "-c", "import time; time.sleep(120)"],
        os.path.join(td61, "t2.log"), state_dir=sd61)
    for _ in range(100):
        if PM.Handler.app.runs[_rid61b].get("pid"):
            break
        _t60.sleep(0.1)
    json.dump({"pid": os.getpid()}, open(os.path.join(sd61, "lab.lock"), "w"))
    st, body = _post60(f"/api/runs/{_rid61b}/cancel", {})
    check("cancel-foreign-lock",
          os.path.exists(os.path.join(sd61, "lab.lock"))
          and "left alone" in body, body[:200])

# 59. triage works against any upstream slug (generic patch manager)
check("api-base", F.api_base("acme/widgets") == "https://api.github.com/repos/acme/widgets")
check("api-default", F.API == F.api_base(F.DEFAULT_REPO)
      and F.DEFAULT_REPO == "ggml-org/llama.cpp")
check("slug-ok", F.check_slug("ggml-org/llama.cpp") and F.check_slug("a/b-c_d.e"))
check("slug-bad", not F.check_slug("") and not F.check_slug("noslash")
      and not F.check_slug("a/b/c") and not F.check_slug(None))
_cfg59 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "config.json")))
check("config-repo", _cfg59.get("repo") == "ggml-org/llama.cpp", str(_cfg59.get("repo")))
_argv59, _req59, _sleep59 = sys.argv, F.req, F.time.sleep
F.time.sleep = lambda s: None
try:
    with tempfile.TemporaryDirectory() as td59:
        _out59 = os.path.join(td59, "c.json")
        _seen59 = []
        def _slugged59(url, token, timeout=60):
            _seen59.append(url)
            if "state=open" in url:
                return ([{"number": 41, "title": "fix widget", "body": "",
                          "user": {"login": "u"},
                          "updated_at": "2026-01-01T00:00:00Z",
                          "labels": [], "draft": False,
                          "head": {"sha": "f" * 40}}], None)
            if url.endswith("/files?per_page=100"):
                return ([], None)
            return ({"mergeable": True, "mergeable_state": "clean",
                     "additions": 1, "deletions": 1, "changed_files": 1,
                     "head": {"sha": "f" * 40}}, None)
        sys.argv = ["fetch_prs.py", "--limit", "5", "--top", "1",
                    "--out", _out59, "--ci-sleep", "0",
                    "--repo", "acme/widgets"]
        F.req = _slugged59
        F.main()
        check("slug-urls",
              _seen59 and all(u.startswith("https://api.github.com/repos/acme/widgets/")
                              for u in _seen59), str(_seen59))
        _got59 = json.load(open(_out59))
        check("slug-output", [c["number"] for c in _got59] == [41],
              str(_got59))
    sys.argv = ["fetch_prs.py", "--limit", "5", "--top", "1",
                "--out", os.path.join(tempfile.gettempdir(), "nope.json"),
                "--repo", "bogus"]
    try:
        F.main()
        check("slug-bad-exits", False, "must sys.exit(2)")
    except SystemExit as e:
        check("slug-bad-exits", e.code == 2, f"exit={e.code}")
finally:
    sys.argv, F.req, F.time.sleep = _argv59, _req59, _sleep59

# 54. preflight refuses to burn a campaign on a missing smoke model
with tempfile.TemporaryDirectory() as td:
    repo = os.path.join(td, "repo"); sd = os.path.join(td, "st")
    os.makedirs(repo); os.makedirs(sd)
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    open(os.path.join(repo, "f.txt"), "w").write("v1\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-m", "base")
    a = _mklab(repo, sd, [{"number": 1, "title": "t"}])
    a.base = "master"
    a.smoke_model = os.path.join(td, "no-such-model.gguf")
    lab = M.Lab(a)
    try:
        lab.preflight()
        check("preflight-missing-model", False, "must raise, not quarantine later")
    except RuntimeError as e:
        check("preflight-missing-model", "smoke model missing" in str(e), str(e)[:150])
    check("preflight-no-burn", lab.state["quarantined"] == [] and lab.quar == [])
    a.smoke_model = os.path.join(td, "m.gguf"); open(a.smoke_model, "w").write("x")
    lab2 = M.Lab(a)
    try:
        lab2.preflight()
        check("preflight-model-present", True)
    except RuntimeError as e:
        check("preflight-model-present", False, str(e)[:200])
    a.smoke_model = os.path.join(td, "no-such-model.gguf")
    a.doctor = True
    lab3 = M.Lab(a)
    try:
        lab3.preflight()
        check("preflight-doctor-skips", True)
    except RuntimeError as e:
        check("preflight-doctor-skips", False, str(e)[:200])

_srv60.shutdown()
_srv60.server_close()

print(f"\n{len(FAIL)} failures")
sys.exit(1 if FAIL else 0)
