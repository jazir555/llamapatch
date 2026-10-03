#!/usr/bin/env python3
"""PR intent classification for llamapatch (pure, offline, no network).

Problem it solves: a CUDA perf PR merged on a CPU-only lab box will show
bench *parity*, not gain — that is expected, not failure. Without intent,
the tool either over-claims ("no improvement?") or under-checks (a CPU
perf PR that shows parity should be scrutinized; a backend PR showing
parity is fine). Intent drives the verdict:

  perf + cpu/generic files  -> expects_bench_gain on this box
  perf + cuda/metal/vulkan/sycl/hexagon-only files -> gain only observable
      on that backend; on CPU the correct verdict is parity (no regression)
      + smoke/ppl green, recorded as backend-unverifiable
  fix   -> correctness: smoke/ppl green + parity is a pass
  feature/model -> loads+runs: smoke green + parity is a pass

Signals: labels, title keywords, file paths. Deterministic, testable.
"""
import re

# backend token -> (label tokens, title tokens, path fragments)
BACKENDS = {
    "cuda": (("cuda",), ("cuda",), ("ggml-cuda", "cuda")),
    "metal": (("metal", "apple metal"), ("metal",), ("ggml-metal", ".metal")),
    "vulkan": (("vulkan",), ("vulkan",), ("ggml-vulkan",)),
    "sycl": (("sycl",), ("sycl",), ("ggml-sycl",)),
    "cpu": (("cpu",), ("cpu", "avx", "neon", "sve", "sme", "kleidiai", "tinyblas", "llamafile"),
            ("ggml-cpu", "kleidiai", "llamafile", "sgemm")),
    "hexagon": (("hexagon",), ("hexagon", "htp", "hvx"), ("ggml-hexagon",)),
    "rpc": (("rpc",), ("rpc",), ("ggml-rpc", "ggml/include/ggml-rpc")),
    "server": (("server",), ("server",), ("tools/server", "tools/parallel-decision")),
    "model": (("model", "conversion"), ("model", "support", "add "), ("src/models", "conversion", "gguf")),
}

PERF_WORDS = ("perf", "speed", "faster", "optimi", "accelerat", "throughput",
              "latency", "bench", "kernel", "flash")
FIX_WORDS = ("fix", "crash", "bug", "incorrect", "wrong", "overflow", "leak",
             "hang", "broken", "regress", "assert", "oob", "race")
FEATURE_WORDS = ("feat", "add ", "support", "new ", "introduc", "implement", "enable ")

# backends whose perf gains are NOT observable on a CPU-only box
BACKEND_ONLY_BOXES = ("cuda", "metal", "vulkan", "sycl", "hexagon")


def detect_backends(cand):
    labels = " ".join(cand.get("labels", [])).lower()
    title = (cand.get("title") or "").lower()
    files = [f.lower() for f in cand.get("files", [])]
    found = []
    for name, (lab_toks, title_toks, path_frags) in BACKENDS.items():
        if any(t in labels for t in lab_toks):
            found.append(name)
            continue
        if any(t in title for t in title_toks):
            found.append(name)
            continue
        if any(fr in f for f in files for fr in path_frags):
            found.append(name)
    return found


def detect_area(cand):
    text = ((cand.get("title") or "") + " " + " ".join(cand.get("labels", []))).lower()
    if any(w in text for w in FIX_WORDS):
        return "fix"
    if any(w in text for w in FEATURE_WORDS):
        return "feature"
    if any(w in text for w in PERF_WORDS):
        return "perf"
    return "other"


def classify_intent(cand):
    """Return {backends, area, expects_bench_gain, verifiable_on_cpu, reason}."""
    backends = detect_backends(cand or {})
    area = detect_area(cand or {})
    hw = [b for b in backends if b in BACKEND_ONLY_BOXES]
    if area == "perf" and not hw:
        expects = True
        reason = "perf without backend-only files: gain should show on CPU"
    elif area == "perf" and hw:
        expects = False
        reason = f"perf for {','.join(hw)}: gain needs that backend; CPU verdict is parity"
    elif area == "fix":
        expects = False
        reason = "fix: correctness (smoke/ppl) + parity is a pass"
    elif area == "feature":
        expects = False
        reason = "feature/model: loads+runs + parity is a pass"
    else:
        expects = False
        reason = "no perf claim: parity + green gates is a pass"
    return {
        "backends": backends,
        "area": area,
        "expects_bench_gain": expects,
        "verifiable_on_cpu": expects,
        "reason": reason,
    }


def verdict_for(intent, base, val, regression_pct):
    """Intent-aware bench verdict: 'regression' | 'improvement' | 'parity' | 'no-baseline'.

    Regression is universal (any PR). Improvement only counts when the gain
    is expected to be visible on this box; backend-only perf showing parity
    is 'parity', not failure.
    """
    if not base or not val:
        return "no-baseline"
    if val < base * (1 - regression_pct / 100):
        return "regression"
    if val > base * (1 + regression_pct / 100):
        return "improvement" if intent.get("expects_bench_gain") else "parity"
    return "parity"


def needs_confirm(intent, base, val, regression_pct, noise_pct=5.0):
    """True when an action-taking verdict rests on a measurement within
    noise_pct of its threshold line. Single bench runs are noisy: a
    regression just under the line (revert+quarantine), an improvement
    just over it (credit), or a parity just under the gain line (which
    rolls back a claimed gain) each earn one confirmation run before the
    tool acts. Clear results and no-op parity need no second run."""
    try:
        pct = float(regression_pct or 0)
        noise = float(noise_pct or 0)
    except (TypeError, ValueError):
        return False
    if not base or not val or pct <= 0 or noise <= 0:
        return False
    lo = base * (1 - pct / 100)    # regression line
    hi = base * (1 + pct / 100)    # improvement line
    band = base * noise / 100
    if val < lo:
        return val >= lo - band
    if (intent or {}).get("expects_bench_gain") and abs(val - hi) <= band:
        return True
    return False
