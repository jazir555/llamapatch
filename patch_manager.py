#!/usr/bin/env python3
"""Local GUI patch manager for llamapatch (stdlib only, no dependencies).

The operator points the manager at any checkout (any version of llama.cpp,
or any other repo the pipeline is configured for), ticks checkboxes for
the PRs they want, and merges the selection — triage, merge, log tail and
evidence report from one page.

Endpoints (JSON unless noted):
  GET  /                              the page (HTML)
  GET  /api/candidates?file=&state_dir=
                                      PRs annotated merged/quarantined/pending
  POST /api/triage   {slug, limit, top, out, include_ci}
  POST /api/setup    {url, dest, depth, sparse, base}
  POST /api/merge    {candidates, repo, base, state_dir, numbers, batch,
                      max_prs, dry_run}
  GET  /api/runs/<id>                 {status, rc, log_tail}
  GET  /api/report?candidates=&state_dir=
                                      Markdown evidence report (text)

Runs execute merge_lab.py / fetch_prs.py / setup_lab.sh as subprocesses
with argv arrays (never a shell); output streams to per-run log files
the page polls. Binds 127.0.0.1 only and refuses anything else:
localhost-only by design, no auth layer.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

try:
    import merge_lab as _M
except ImportError:  # pragma: no cover - standalone fallback
    _M = None

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>llamapatch manager</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:1em auto;padding:0 1em}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.25em .5em;text-align:left}
tr.merged td{background:#e6f4ea}tr.quarantined td{background:#fce8e6}
input[type=text]{width:26em;max-width:90%}pre{background:#111;color:#eee;padding:.5em;overflow:auto;max-height:30em;white-space:pre-wrap}
.row{margin:.5em 0}button{margin-right:.5em}
</style></head><body>
<h1>llamapatch manager</h1>
<div class="row"><label>Upstream slug <input id="slug" type="text" value="ggml-org/llama.cpp"></label></div>
<div class="row"><label>Checkout (repo) <input id="repo" type="text" placeholder="/home/user/llama-pr-lab/llama.cpp"></label>
<label>Base <input id="base" type="text" value="master" style="width:8em"></label></div>
<div class="row"><label>Candidates file <input id="cands" type="text" value="candidates-1k.json"></label>
<label>State dir <input id="statedir" type="text" value="" placeholder="(default: alongside candidates)"></label>
<button onclick="load()">Load</button></div>
<div class="row"><label>Limit <input id="limit" type="text" value="200" style="width:5em"></label>
<label>Top <input id="top" type="text" value="50" style="width:5em"></label>
<label>Out <input id="out" type="text" value="candidates.json"></label>
<button onclick="triage()">Triage</button></div>
<div class="row"><button onclick="checkAll(true)">All</button><button onclick="checkAll(false)">None</button>
<button onclick="merge(false)">Merge selected</button><button onclick="merge(true)">Dry run</button>
<button onclick="report()">Report</button></div>
<table><thead><tr><th></th><th>PR</th><th>title</th><th>score</th><th>area</th><th>verdict</th><th>status</th></tr></thead>
<tbody id="rows"></tbody></table>
<h2>Log <span id="runid"></span></h2><pre id="log">(no run)</pre>
<h2>Report</h2><pre id="rep">(no report)</pre>
<script>
let RUN=null, TIMER=null;
async function api(path, opts){const r=await fetch(path,opts);const t=await r.text();let j=null;try{j=JSON.parse(t)}catch(e){}if(!r.ok)throw new Error((j&&j.error)||t.slice(0,300));return j}
async function load(){const q=new URLSearchParams({file:val('cands'),state_dir:val('statedir')});const d=await api('/api/candidates?'+q);const tb=document.getElementById('rows');tb.innerHTML='';for(const c of d.candidates){const tr=document.createElement('tr');if(c.status!=='pending')tr.className=c.status;tr.innerHTML=`<td><input type="checkbox" data-n="${c.number}" ${c.status==='pending'?'checked':''} ${c.status!=='pending'?'disabled':''}></td><td>#${c.number}</td><td>${esc(c.title||'')}</td><td>${c.score??''}</td><td>${c.area||''}</td><td>${c.verdict||''}</td><td>${c.status}</td>`;tb.appendChild(tr)}}
function val(id){return document.getElementById(id).value.trim()}
function esc(s){return s.replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
function checkAll(v){document.querySelectorAll('#rows input[type=checkbox]:not(:disabled)').forEach(c=>c.checked=v)}
function selected(){return [...document.querySelectorAll('#rows input[type=checkbox]:checked')].map(c=>+c.dataset.n)}
async function triage(){const b={slug:val('slug'),limit:+val('limit')||200,top:+val('top')||50,out:val('out')||'candidates.json',include_ci:false};const r=await api('/api/triage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id)}
async function merge(dry){const b={candidates:val('cands'),repo:val('repo'),base:val('base')||'master',state_dir:val('statedir'),numbers:selected(),dry_run:dry};const r=await api('/api/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id);load()}
async function report(){const q=new URLSearchParams({candidates:val('cands'),state_dir:val('statedir')});const r=await fetch('/api/report?'+q);document.getElementById('rep').textContent=await r.text()}
function watch(id){RUN=id;document.getElementById('runid').textContent='run '+id;clearInterval(TIMER);TIMER=setInterval(poll,1000);poll()}
async function poll(){if(RUN==null)return;const d=await api('/api/runs/'+RUN);document.getElementById('log').textContent=d.log_tail||'(running…)';if(d.status!=='running'){clearInterval(TIMER);load()}}
</script></body></html>
"""


def _json(handler, code, obj):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text(handler, code, body, ctype="text/plain; charset=utf-8"):
    raw = body.encode()
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class PatchApp:
    """Run registry + filesystem helpers. All subprocesses get argv arrays."""

    def __init__(self):
        self.lock = threading.Lock()
        self.runs = {}
        self.seq = 0

    # -- inputs ---------------------------------------------------------
    def load_candidates(self, path):
        if not path:
            return None, "candidates file required"
        if not os.path.exists(path):
            return None, f"candidates file not found: {path}"
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            return None, f"candidates file unreadable: {e}"
        if not isinstance(raw, list):
            return None, "candidates file must hold a JSON list"
        cands = [c for c in raw
                 if isinstance(c, dict) and isinstance(c.get("number"), int)]
        return cands, None

    def load_state(self, state_dir):
        state, quar = {"merged": [], "quarantined": []}, []
        if state_dir:
            for name, default in (("lab-state.json", state),
                                  ("quarantined.json", quar)):
                p = os.path.join(state_dir, name)
                if os.path.exists(p):
                    try:
                        with open(p) as f:
                            loaded = json.load(f)
                        if name.startswith("lab-") and isinstance(loaded, dict):
                            state = loaded
                        elif isinstance(loaded, list):
                            quar = loaded
                    except Exception:
                        pass
        bench = state.get("bench_results", {}) if isinstance(state, dict) else {}
        merged = state.get("merged", []) if isinstance(state, dict) else []
        return state, quar, bench, merged

    def default_state_dir(self, candidates):
        d = os.path.dirname(os.path.abspath(candidates))
        return os.path.join(d, "state")

    # -- runs -----------------------------------------------------------
    def start_run(self, kind, cmd, log_path, env=None):
        with self.lock:
            self.seq += 1
            rid = self.seq
            self.runs[rid] = {"id": rid, "kind": kind, "status": "running",
                              "cmd": cmd, "log": log_path, "rc": None,
                              "started": True}
        def _bg():
            rc = 2
            try:
                with open(log_path, "w") as f:
                    p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                       text=True, timeout=86400, env=env,
                                       cwd=HERE)
                    rc = p.returncode
            except Exception as e:
                try:
                    with open(log_path, "a") as f:
                        f.write(f"\n[manager] runner failed: {e}\n")
                except Exception:
                    pass
            with self.lock:
                self.runs[rid]["status"] = "done" if rc == 0 else "failed"
                self.runs[rid]["rc"] = rc
        threading.Thread(target=_bg, daemon=True).start()
        return rid

    def run_info(self, rid):
        with self.lock:
            rec = dict(self.runs.get(rid, {}))
        if not rec:
            return None
        tail = ""
        try:
            with open(rec["log"]) as f:
                lines = f.read().splitlines()
            tail = "\n".join(lines[-200:])
        except Exception as e:
            tail = f"(log unavailable: {e})"
        rec["log_tail"] = tail
        return rec


class Handler(http.server.BaseHTTPRequestHandler):
    app = PatchApp()
    server_version = "llamapatch-manager/1"

    def log_message(self, *a):
        pass

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            return json.loads(raw.decode() or "{}")
        except Exception:
            return None

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/":
            return _text(self, 200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/candidates":
            cands, err = self.app.load_candidates(q.get("file", ""))
            if err:
                return _json(self, 404 if "not found" in err else 422,
                             {"error": err})
            sd = q.get("state_dir") or (self.app.default_state_dir(q["file"])
                                        if q.get("file") else "")
            state, quar, bench, merged = self.app.load_state(sd)
            qset = set(state.get("quarantined", []) or [])
            rows = []
            for c in cands:
                n = c["number"]
                b = bench.get(str(n), {}) if isinstance(bench, dict) else {}
                status = ("merged" if n in merged else
                          "quarantined" if n in qset else "pending")
                rows.append({"number": n, "title": c.get("title", ""),
                             "score": c.get("score"),
                             "area": (c.get("intent") or {}).get("area", "")
                             if isinstance(c.get("intent"), dict) else "",
                             "verdict": b.get("verdict", ""),
                             "status": status})
            return _json(self, 200, {"candidates": rows, "state_dir": sd})
        if u.path.startswith("/api/runs/"):
            try:
                rid = int(u.path.rsplit("/", 1)[1])
            except ValueError:
                return _json(self, 400, {"error": "bad run id"})
            rec = self.app.run_info(rid)
            if rec is None:
                return _json(self, 404, {"error": "unknown run"})
            return _json(self, 200, rec)
        if u.path == "/api/report":
            if _M is None:
                return _json(self, 500, {"error": "merge_lab unavailable"})
            cands, err = self.app.load_candidates(q.get("candidates", ""))
            if err:
                return _json(self, 404 if "not found" in err else 422,
                             {"error": err})
            sd = q.get("state_dir") or self.app.default_state_dir(
                q.get("candidates", ""))
            state, quar, _, _ = self.app.load_state(sd)
            try:
                rep = _M.build_report(state, quar, cands)
            except Exception as e:
                return _json(self, 500, {"error": f"report failed: {e}"})
            return _text(self, 200, rep, "text/markdown; charset=utf-8")
        return _json(self, 404, {"error": "unknown path"})

    def do_POST(self):
        body = self._body()
        if body is None:
            return _json(self, 400, {"error": "invalid JSON body"})
        if self.path == "/api/triage":
            slug = (body.get("slug") or "").strip()
            if not SLUG_RE.fullmatch(slug):
                return _json(self, 400, {"error": "bad slug, want owner/name"})
            try:
                limit = int(body.get("limit", 200))
                top = int(body.get("top", 50))
            except (TypeError, ValueError):
                return _json(self, 400, {"error": "limit/top must be integers"})
            out = (body.get("out") or "candidates.json").strip() or "candidates.json"
            cmd = [sys.executable, "-u", os.path.join(HERE, "fetch_prs.py"),
                   "--repo", slug, "--limit", str(limit), "--top", str(top),
                   "--out", out]
            if body.get("include_ci"):
                cmd.append("--include-ci")
            rid = self._spawn("triage", cmd, None)
            return _json(self, 200, {"id": rid})
        if self.path == "/api/setup":
            url = (body.get("url") or "").strip()
            dest = (body.get("dest") or "").strip()
            if not url or not dest:
                return _json(self, 400, {"error": "url and dest required"})
            env = dict(os.environ)
            if body.get("depth"):
                env["CLONE_DEPTH"] = str(body["depth"])
            if body.get("sparse"):
                env["SPARSE_DIRS"] = str(body["sparse"])
            if body.get("base"):
                env["LAB_BASE"] = str(body["base"])
            rid = self._spawn("setup",
                              ["bash", os.path.join(HERE, "setup_lab.sh"),
                               url, dest], None, env=env)
            return _json(self, 200, {"id": rid})
        if self.path == "/api/merge":
            return self._merge(body)
        return _json(self, 404, {"error": "unknown path"})

    def _spawn(self, kind, cmd, log_hint, env=None):
        logdir = tempfile.mkdtemp(prefix="llamapatch-run-")
        log_path = os.path.join(logdir, f"{kind}.log")
        return self.app.start_run(kind, cmd, log_path, env=env)

    def _merge(self, body):
        cands, err = self.app.load_candidates(body.get("candidates", ""))
        if err:
            return _json(self, 404 if "not found" in err else 422,
                         {"error": err})
        repo = (body.get("repo") or "").strip()
        if not repo:
            return _json(self, 400, {"error": "repo checkout path required"})
        base = (body.get("base") or "master").strip() or "master"
        nums = body.get("numbers", [])
        if not nums:
            return _json(self, 400, {"error": "no PRs selected"})
        if (not isinstance(nums, list) or
                any(not isinstance(n, int) for n in nums)):
            return _json(self, 400, {"error": "numbers must be int list"})
        known = {c["number"] for c in cands}
        unknown = [n for n in nums if n not in known]
        if unknown:
            return _json(self, 400, {"error": f"unknown PRs: {unknown[:10]}"})
        try:
            batch = int(body.get("batch", 10))
            max_prs = int(body.get("max_prs", 50))
        except (TypeError, ValueError):
            return _json(self, 400, {"error": "batch/max_prs must be integers"})
        sd = (body.get("state_dir") or "").strip() or self.app.default_state_dir(
            body.get("candidates", ""))
        os.makedirs(sd, exist_ok=True)
        sel_path = os.path.join(sd, "candidates-selected.json")
        with open(sel_path, "w") as f:
            json.dump([c for c in cands if c["number"] in set(nums)], f,
                      indent=2)
        cmd = [sys.executable, "-u", os.path.join(HERE, "merge_lab.py"),
               "--candidates", sel_path, "--repo", repo, "--base", base,
               "--state-dir", sd, "--batch", str(batch),
               "--max-prs", str(max_prs)]
        if body.get("dry_run"):
            cmd.append("--dry-run")
        rid = self._spawn("merge", cmd, None)
        return _json(self, 200, {"id": rid, "selected": os.path.basename(sel_path)})


def serve(host="127.0.0.1", port=8123):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"refusing non-loopback host {host!r}")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, port), Handler) as srv:
        print(f"llamapatch manager at http://{host}:{srv.server_address[1]}",
              flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args(argv)
    serve(a.host, a.port)


if __name__ == "__main__":
    main()
