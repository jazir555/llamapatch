# llama.cpp PR mass-merge lab

Goal: merge 50+ (eventually 1k+) open upstream PRs for performance, 10 at a
time, with fully automated build/smoke/bench gates and self-healing (conflict /
build / runtime failures auto-quarantine the bad PR and keep going).

## Why this design (for 1k PRs)

Manual merging is impossible: PRs step on each other (same `ggml/src/*`,
`src/*` files), go stale, or break the build. So:

1. **Triage first, merge second.** `fetch_prs.py` scores all open PRs by
   perf keywords (stem-aware word matching: `fix` hits fixed/fixes but not
   prefix, `bug` never matches debug) + perf paths, penalizes huge diffs,
   skips drafts / `do-not-merge`. Only the top-N get expensive detail API
   calls.
2. **Merge via `pull/N/head` refspec, not fork remotes.** `git fetch origin
   pull/N/head:pr/N` needs no token, no per-fork remote (unlike
   `scripts/pr2wt.sh`, which is for interactive single-PR worktrees).
3. **Batches of 10.** Limits blast radius while making steady progress;
   each PR gets its own verified commit, so a bad PR reverts cleanly
   (`git reset --hard HEAD~1`) without losing the other 9.
4. **Gates:** configure+build → bench smoke on TinyLlama → `llama-bench`
   pp/tg on 7B Q4_K_M (+ optional `llama-perplexity` correctness gate) →
   optional CI-red skip (annotated by `fetch_prs.py --include-ci`).
   Verdicts are intent-aware (`pr_intent.py`): a CUDA/Metal/Vulkan/SYCL perf
   PR showing CPU parity is the *correct* outcome on a CPU box (gain lives
   on that backend) — recorded as `parity`, never punished; only a true
   >15% tg regression quarantines. **Only measured improvements stay: a
   CPU/generic perf PR that claims a gain but benches within noise is
   reverted and quarantined as `no-improvement`.** Fixes/features prove
   their intent via green correctness gates + no regression. The bench
   baseline is pinned to the base SHA (`bench_baseline_sha`) and rebuilds
   automatically after any rebase/base move, so verdicts never compare
   against a stale base — and it is never taken on a branch already holding
   campaign merges (that would bake PR gains into the baseline and
   manufacture false regressions); then it defers loudly and gates fall
   back to unverified.
5. **Self-heal:** merge conflict / build fail / smoke fail /
   perplexity fail / regression / no-improvement / empty (already-upstream)
   noop all quarantine or record the culprit(s) and continue. CI-red PRs are
   **skipped, never quarantined** — CI flips green on reruns/pushes, so the
   next triage refresh retries them automatically (`--doctor` also releases
   old `ci-red-skipped` entries). Transient fetch/git
   failures quarantine as `fetch-failed`/`git-timeout` (never as conflicts)
   and `--doctor` requeues them for retry — a dead fork simply
   re-quarantines next run. After the loop, a **final post-run gate**
   re-verifies the merged tree (interactions can regress after individual
   gates pass) and walks back culprits as `late-*`, then **re-verifies that
   claimed gains still hold in aggregate** (`IMPROVEMENTS verified/lost` —
   report-only, never auto-reverts since combined gains need not stack).
   Full log in `lab-log.jsonl`.

## Layout

- Lab checkout (fast Linux fs, NOT /mnt/c): `~/llama-pr-lab/llama.cpp`
- Models: `~/llama-pr-lab/models/{tinyllama.gguf,qwen2.5-7b-00001-of-00002.gguf (+ -00002- part)}`
- This dir (`pr-lab/` in LLMDroid, mirrored to WSL) holds the automation.
- State (in WSL lab dir or here): `lab-state.json`, `lab-log.jsonl`,
  `quarantined.json`, `candidates.json`

## Run (all in WSL Ubuntu)

```bash
# 0. toolchain (once)
sudo apt-get update && sudo apt-get install -y cmake ninja-build gcc g++ curl jq pkg-config libcurl4-openssl-dev ccache
pip3 install huggingface_hub

# 1. clone + models
git clone https://github.com/ggml-org/llama.cpp.git ~/llama-pr-lab/llama.cpp
bash pr-lab/fetch_models.sh   # tiny (~0.7GB) + 7B Q4_K_M (~4.7GB)

# 2. triage (set GH_TOKEN to avoid 60/hr limit)
export GH_TOKEN=xxx
python3 pr-lab/fetch_prs.py --limit 1000 --top 100 --out candidates.json

# 3. plan only
python3 pr-lab/merge_lab.py --candidates candidates.json --dry-run --batch 10 --max-prs 50

# 4. baseline build + bench (proves harness before any PR)
cmake -S ~/llama-pr-lab/llama.cpp -B ~/llama-pr-lab/llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama-pr-lab/llama.cpp/build --target llama-cli llama-bench -j4
~/llama-pr-lab/llama.cpp/build/bin/llama-cli -m ~/llama-pr-lab/models/tinyllama.gguf -p "Hello" -n 20 --temp 0
~/llama-pr-lab/llama.cpp/build/bin/llama-bench -m ~/llama-pr-lab/models/qwen2.5-7b-00001-of-00002.gguf -p 128 -n 128 -o json

# 5. auto-merge loop (10 at a time, up to 50)
python3 pr-lab/merge_lab.py --candidates candidates.json --batch 10 --max-prs 50
```

Self-test (no network, no models, runs on Windows):
```bash
python3 test_lab.py   # 266 checks: bench/smoke/batch/CI/doctor/sanitize/state-heal/intent/preflight/timeouts/report/ratelimit/pages/baseline/cleanstart/models/lock/transient/quar/scoring/deadcode/transport/gates/final/improvement/ghost/defer/late/gittimeout/heads/stale/headprune/parents/confirm/finalconfirm/basemean/warnings/headskip/knobs/lifecycle/triage/runs
python3 test_e2e_mock.py  # 39 checks: merge/conflict/noop/doctor + full run() 10-batch + resume + fetch-retry + strict-no-improvement + ci-flip-retry + head-shas + confirm-e2e
```

## Evidence report (review after every 10-batch)
```bash
bash llamapatch report                    # stdout
bash llamapatch report --report-out r1.md # file
```

`build_report()` renders merged/quarantined counts, per-PR area/backend/
verdict/tg-vs-base rows, and a `proven improvements` list — the artifact the
operator reviews to confirm each batch improved what it claimed.

## Gate timeouts (hung builds never kill a batch)

Every gate runs under a timeout (`--build-timeout 3600` for cmake
configure+build; 300s/600s shell timeouts for smoke/bench/perplexity). A hung
or crashed gate quarantines as `build-timeout`/`smoke-timeout`/
`perplexity-timeout` (or `*-infra-error` for missing binaries) and the loop
continues; bench timeouts never punish the PR (`merged-unverified-perf`).
Hung git inside merges returns `git-timeout` the same way. Reverts still go
through verified `revert_last()`.

## Preflight & safe revert

`run()` (non-dry, non-report) takes an exclusive `lab.lock` in the state dir
first — a second concurrent loop fails fast instead of interleaving merges
into shared state (a SIGKILL-stale lock is cleared explicitly with
`--force-unlock`, only after verifying no run is active; the override is
logged). Then `preflight()`: hard-fails on missing repo,
non-repo dir, unresolvable `--base`, or empty/malformed candidates (malformed
entries are skipped with a count, corrupt JSON aborts loudly); missing
cmake/ninja/ccache and low disk are logged warnings. After checkout it runs
`ensure_clean_start()`: a kill-restarted worktree may hold modified tracked
files that would fake merge-conflicts, so it resets to HEAD (untracked
`build/` output untouched) and refuses loudly if dirt persists. Every gate revert goes
through `revert_last()`, which verifies the tracked tree is clean afterwards
— a persistently dirty tree raises instead of silently corrupting the next 9
PRs of the batch.

Resume: re-run step 5; it loads `lab-state.json` and skips merged/quarantined.

## What to expect

- ~17 builds for 50 PRs in batches of 3, ~5–10 min each on 4 cores → a few hours.
  Use `ccache` (auto-detected) to cut rebuilds.
- Most of 1k+ PRs will NOT be perf PRs and many conflict; expect high quarantine
  rate. That's normal — the log tells you which ones and why.
- Interaction conflicts (A+B fail but each passes alone) are quarantined as a
  pair for manual review — not retried automatically.

## Rolling fixes (learned the hard way)

- **v4 — batch-10 hardening.** Default `--batch 10` (was 3); merge message
  amended to `pr-lab: merge #N <title>` so `--doctor` reconciles state
  (matches both `#N` and legacy `pr/N`); already-upstream merges detected
  pre-amend via HEAD-unchanged/`Already up to date` guard (never renames the
  previous PR's commit) and recorded as `merged-noop-empty`; in-batch
  file-overlap warnings computed pre-merge; robust bench JSON parsing (last
  array, `{"results":[...]}` shape); smoke accepts any throughput token;
  `--include-ci` annotates red PRs and the merge loop skips them pre-build;
  optional `--ppl-threshold/--ppl-sample` perplexity correctness gate;
  per-PR `bench_results` + `improvement` events prove the gain; corrupt
  state files backed up (`.corrupt-<ts>`) instead of crashing; `fetch_pr`
  falls back to a local `pr/N` branch (deleted-fork/offline resilient).
- **v1 bug — stacked uncommitted merges are impossible.** `git merge --no-commit`
  twice in a row fails with "MERGE_HEAD exists"; the 2nd `merge --abort` wipes
  the 1st PR too, and an unchecked `git commit` on the clean tree silently
  records a phantom merge. Fixed in v3: **one commit per PR**
  (`merge_one_committed`, verified 2-parent merge commit), batch = N commits.
  Revert-one = `git reset --hard HEAD~1` (local branch, never pushed: safe).
- **Always check `git commit` rc + verify HEAD advanced.** State must never
  record uncommitted work. `--doctor` reconciles state vs repo and requeues
  quarantines whose detail contains "MERGE_HEAD exists".
- **Stream logs unbuffered** (`python3 -u`, log file + `tail -f`, never
  `| tail -30` on a live process — it buffers until exit and the run looks dead).
- **llama-cli 0.5.0-dev is silent non-interactively** (server-backend refactor);
  smoke gate uses `llama-bench -p 32 -n 32` on TinyLlama instead.
- **Models: curl-direct, not HF API.** `huggingface_hub` hit 401 for public
  repos; `curl -L <repo>/resolve/main/<file>?download=true` works. Qwen2.5-7B
  Q4_K_M is **split** (`qwen2.5-7b-instruct-q4_k_m-00001-of-00002` +
  `-00002-of-00002`, verified against the HF API file list); `fetch_models.sh`
  saves both as `qwen2.5-7b-0000{1,2}-of-00002.gguf` and every script/config
  points bench gates at part 1 (`test_lab.py` pins this agreement).
  TinyLlama: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
- **WSL `~/.wslconfig` duplicate-key warnings** (`wsl2.memory`/`processors`)
  are harmless noise from the user's config; ignore.

## Results (2026-10-02, WSL Ubuntu, 4 vCPU, CPU backend, threads=2)

22/30 triaged PRs merged (`pr-lab/base-bed0a856` atop `bed0a8566`), 8
quarantined with genuine content conflicts, `EXIT:0`, state==repo verified.

| model | clean base `bed0a8566` | 22-merge HEAD `84244bf6d` | verdict |
|---|---|---|---|
| TinyLlama-1.1B Q4_K_M pp32 | 164.39 ± 5.56 | 160.39 ± 3.83 | parity (-2.4%, in noise) |
| TinyLlama-1.1B Q4_K_M tg32 | 42.22 ± 0.20 | 38.05 ± 1.54 | parity (-10%, tg noisy on this box) |
| Qwen2.5-7B Q4_K_M pp16/tg16 | — | 15.79 / 3.71 | loads + runs (regression reference: tg32 1.93) |

No CPU regression (expected: merged diffs are metal/cuda/sycl backends).
CPU gains not expected from this batch; GPU-backend gains (metal multi-column
mat-vec #29110, CUDA top-k #28671, cooperative softmax #28982, SYCL FA #29171)
materialize on those backends. Early "baseline" 121/21 numbers were
machine-load noise (downloads+builds concurrent) — always compare same-state.

## Next scale-up

- Full 1k triage needs `GH_TOKEN` (unauthenticated API caps at 60 req/hr;
  1k PRs need ~2000 detail calls). Then: `fetch_prs.py --limit 1000 --top 200`.
  On rate-limit, triage saves partial output and exits 2 immediately instead
  of sleeping 30s per PR — re-run with a token to continue. Stage-1 page
  fetches retry transient failures (1s/2s/4s backoff) and triage partial
  results rather than discarding all pages on a persistent failure.
- `candidates.json` (30) is exhausted: 22 merged + 8 quarantined. Next run needs
  fresh triage output.
- Keep lab worktrees under `~/llama-pr-lab/` (persistent); WSL `/tmp` is wiped
  on restart (lost one base comparison to this).

- Add `llama-perplexity` gate on WikiText-2 sample for correctness.
  (v4: flag exists as `--ppl-threshold/--ppl-sample`; wire a default sample path.)
- Nightly rebase: `git fetch origin master`, re-triage (PRs close/merge daily).
