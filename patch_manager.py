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


def _sh_path(p):
    """bash-ready path. WSL's bash (C:\\Windows\\system32\\bash.exe) only
    understands /mnt/<drive>/... spellings; Git-Bash takes C:/...; POSIX
    is a no-op. URLs never match the drive-letter pattern."""
    p = (p or "").replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m and _is_wsl_bash():
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return p


_WSL_BASH = None


def _is_wsl_bash():
    """True when `bash` on PATH is the WSL launcher (cached probe)."""
    global _WSL_BASH
    if _WSL_BASH is None:
        import shutil
        try:
            exe = shutil.which("bash") or "bash"
            p = subprocess.run([exe, "-c", "echo $WSL_DISTRO_NAME"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=30)
            _WSL_BASH = bool((p.stdout or "").strip())
        except Exception:
            _WSL_BASH = False
    return _WSL_BASH

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
<label>Base <input id="base" type="text" value="master" style="width:8em" list="refs"></label>
<datalist id="refs"></datalist><button onclick="versions()">Versions</button></div>
<div class="row"><label>Candidates file <input id="cands" type="text" value="candidates-1k.json"></label>
<label>State dir <input id="statedir" type="text" value="" placeholder="(default: alongside candidates)"></label>
<button onclick="load()">Load</button></div>
<div class="row"><label>Limit <input id="limit" type="text" value="200" style="width:5em"></label>
<label>Top <input id="top" type="text" value="50" style="width:5em"></label>
<label>Out <input id="out" type="text" value="candidates.json"></label>
<label><input id="ci" type="checkbox"> include CI</label>
<button onclick="triage()">Triage</button></div>
<div class="row"><label>Clone URL <input id="cloneurl" type="text" placeholder="https://github.com/ggml-org/llama.cpp.git"></label>
<label>Dest <input id="clonedest" type="text" placeholder="/home/user/llama-pr-lab/llama.cpp"></label>
<button onclick="setup()">Setup checkout</button></div>
<div class="row"><button onclick="checkAll(true)">All</button><button onclick="checkAll(false)">None</button>
<label>Batch <input id="batch" type="text" value="10" style="width:4em"></label>
<label>Max PRs <input id="maxprs" type="text" value="50" style="width:4em"></label>
<button onclick="merge(false)">Merge selected</button><button onclick="merge(true)">Dry run</button>
<button onclick="cancel()">Cancel run</button><button onclick="doctor()">Doctor</button><button onclick="report()">Report</button></div>
<div class="row"><label>Filter <input id="flt" type="text" placeholder="text…" oninput="render()"></label>
<label>Status <select id="fltstatus" onchange="render()"><option value="">all</option><option>pending</option><option>merged</option><option>quarantined</option></select></label></div>
<table><thead><tr><th></th><th><a href="#" onclick="return sort('number')">PR</a></th><th><a href="#" onclick="return sort('title')">title</a></th><th><a href="#" onclick="return sort('score')">score</a></th><th>area</th><th>files</th><th>CI</th><th>verdict</th><th><a href="#" onclick="return sort('status')">status</a></th></tr></thead>
<tbody id="rows"></tbody></table>
<h2>Runs</h2><div class="row"><button onclick="runs()">Refresh</button></div><pre id="runs">(no runs)</pre>
<h2>Log <span id="runid"></span></h2><pre id="log">(no run)</pre>
<h2>Report</h2><pre id="rep">(no report)</pre>
<script>
let RUN=null, TIMER=null, ROWS=[], SORTK='number', SORTD=1;
async function api(path, opts){const r=await fetch(path,opts);const t=await r.text();let j=null;try{j=JSON.parse(t)}catch(e){}if(!r.ok)throw new Error((j&&j.error)||t.slice(0,300));return j}
async function load(){const q=new URLSearchParams({file:val('cands'),state_dir:val('statedir')});const d=await api('/api/candidates?'+q);saveFields();ROWS=d.candidates;render()}
function render(){const tb=document.getElementById('rows');tb.innerHTML='';const keep=savedChecks();const f=(val('flt')||'').toLowerCase(),fs=val('fltstatus');const rows=ROWS.filter(c=>(!fs||c.status===fs)&&(!f||((c.title||'')+' '+(c.area||'')+' #'+c.number).toLowerCase().includes(f)));rows.sort((a,b)=>{const x=a[SORTK]??'',y=b[SORTK]??'';return (x<y?-1:x>y?1:0)*SORTD});for(const c of rows){const tr=document.createElement('tr');if(c.status!=='pending')tr.className=c.status;const files=(c.files||[]).join(', ')+(c.file_count>(c.files||[]).length?` +${c.file_count-(c.files||[]).length} more`:'');const checked=c.status==='pending'&&(keep===null||keep.has(c.number));tr.innerHTML=`<td><input type="checkbox" data-n="${c.number}" ${checked?'checked':''} ${c.status!=='pending'?'disabled':''} onchange="saveChecks()"></td><td>#${c.number} <button onclick="preview(${c.number})" title="diff vs base">diff</button></td><td>${esc(c.title||'')}<br><small>${esc(c.head||'')}</small></td><td>${c.score??''}</td><td title="${esc(c.intent_reason||'')}">${c.area||''}</td><td><small>${esc(files)}</small></td><td>${c.ci||''}</td><td>${c.verdict||''}</td><td>${c.status}${c.quar_reason?` (${esc(c.quar_reason)})`:''}${c.status==='quarantined'?` <button onclick="release([${c.number}])">release</button>`:''}</td>`;tb.appendChild(tr)}}
async function preview(n){const b={repo:val('repo'),base:val('base')||'master',number:n};const d=await api('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});document.getElementById('rep').textContent=`#${n} vs ${d.base}\n${d.stat}\n---\n${d.diff}`}
async function release(ns){const b={candidates:val('cands'),state_dir:val('statedir'),numbers:ns};const r=await api('/api/release',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});load()}
async function doctor(){const b={candidates:val('cands'),repo:val('repo'),state_dir:val('statedir')};const r=await api('/api/doctor',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id)}
function sort(k){if(SORTK===k)SORTD*=-1;else{SORTK=k;SORTD=1}render();return false}
async function runs(){const d=await api('/api/runs');document.getElementById('runs').textContent=d.runs.length?d.runs.map(r=>`#${r.id} ${r.kind} ${r.status}${r.rc==null?'':' rc='+r.rc}${r.merged_n==null?'':` M${r.merged_n}/Q${r.quar_n??0}`}`).join(String.fromCharCode(10)):'(no runs)'}
async function versions(){const q=new URLSearchParams({repo:val('repo')});const d=await api('/api/refs?'+q);const dl=document.getElementById('refs');dl.innerHTML='';for(const b of [...(d.branches||[]),...(d.tags||[])]){const o=document.createElement('option');o.value=b;dl.appendChild(o)}const slug=val('slug');const origin=(d.origin||'').toLowerCase();if(slug&&origin&&!origin.includes(slug.toLowerCase())){document.getElementById('rep').textContent=`WARNING: checkout origin ${d.origin} does not match slug ${slug} — merging across repos produces nonsense. Fix one of them before merging.`}}
function selKey(){return 'llamapatch-sel:'+val('cands')}
function savedChecks(){try{const s=JSON.parse(localStorage.getItem(selKey())||'null');return Array.isArray(s)?new Set(s):null}catch(e){return null}}
function saveChecks(){try{localStorage.setItem(selKey(),JSON.stringify(selected()))}catch(e){}}
function saveFields(){try{localStorage.setItem('llamapatch-fields',JSON.stringify({slug:val('slug'),repo:val('repo'),base:val('base'),cands:val('cands'),statedir:val('statedir'),batch:val('batch'),maxprs:val('maxprs')}))}catch(e){}}
function restoreFields(){try{const f=JSON.parse(localStorage.getItem('llamapatch-fields')||'null');if(!f)return;for(const k of ['slug','repo','base','cands','statedir','batch','maxprs']){if(f[k]!==undefined)document.getElementById(k).value=f[k]}}catch(e){}}
restoreFields();
function val(id){return document.getElementById(id).value.trim()}
function esc(s){return s.replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
function checkAll(v){document.querySelectorAll('#rows input[type=checkbox]:not(:disabled)').forEach(c=>c.checked=v);saveChecks()}
function selected(){return [...document.querySelectorAll('#rows input[type=checkbox]:checked')].map(c=>+c.dataset.n)}
async function triage(){const b={slug:val('slug'),limit:+val('limit')||200,top:+val('top')||50,out:val('out')||'candidates.json',include_ci:document.getElementById('ci').checked};const r=await api('/api/triage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id)}
async function setup(){const b={url:val('cloneurl'),dest:val('clonedest'),base:val('base')||'master'};const r=await api('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id)}
async function merge(dry){saveChecks();const b={candidates:val('cands'),repo:val('repo'),base:val('base')||'master',state_dir:val('statedir'),numbers:selected(),batch:+val('batch')||10,max_prs:+val('maxprs')||50,dry_run:dry};const r=await api('/api/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});watch(r.id);load()}
async function report(){const q=new URLSearchParams({candidates:val('cands'),state_dir:val('statedir')});const r=await fetch('/api/report?'+q);document.getElementById('rep').textContent=await r.text()}
async function cancel(){if(RUN==null)return;const d=await api('/api/runs/'+RUN+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll()}
function watch(id){RUN=id;document.getElementById('runid').textContent='run '+id;clearInterval(TIMER);TIMER=setInterval(poll,1000);poll()}
async function poll(){if(RUN==null)return;const d=await api('/api/runs/'+RUN);document.getElementById('log').textContent=d.log_tail||'(running…)';if(d.status!=='running'){clearInterval(TIMER);load();report()}}
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

    @staticmethod
    def git_refs(repo):
        """Branches + recent tags + HEAD for the version picker. All errors
        become messages: a bad path must explain itself, not 500."""
        if not repo:
            return None, "repo checkout path required"
        if not os.path.isdir(repo):
            return None, f"repo dir missing: {repo}"
        def _run(args):
            try:
                p = subprocess.run(["git", "-C", repo] + args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=60)
            except Exception as e:
                return None
            if p.returncode != 0:
                return None
            return [l.strip() for l in (p.stdout or "").splitlines()
                    if l.strip()]
        branches = _run(["branch", "--format=%(refname:short)"])
        if branches is None:
            return None, f"not a git repo: {repo}"
        tags = _run(["tag", "--sort=-creatordate"]) or []
        head = _run(["rev-parse", "--short", "HEAD"]) or [""]
        origin = _run(["config", "--get", "remote.origin.url"]) or [""]
        return {"branches": branches[:100], "tags": tags[:50],
                "head": head[0], "origin": origin[0]}, None

    @staticmethod
    def verify_ref(repo, ref):
        """True when ref resolves in the checkout (fail fast on typos
        before a multi-hour merge run starts)."""
        if not repo or not ref:
            return False
        try:
            p = subprocess.run(["git", "-C", repo, "rev-parse", "--verify",
                                ref], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               timeout=60)
        except Exception:
            return False
        return p.returncode == 0

    # -- runs -----------------------------------------------------------
    def start_run(self, kind, cmd, log_path, env=None, state_dir=""):
        """Spawn cmd (argv array) in background with output to log_path.
        Popen (not run) so cancel can terminate the whole process tree."""
        with self.lock:
            self.seq += 1
            rid = self.seq
            self.runs[rid] = {"id": rid, "kind": kind, "status": "running",
                              "cmd": cmd, "log": log_path, "rc": None,
                              "pid": None, "state_dir": state_dir,
                              "cancel_note": ""}
            self._prune_locked()
        def _bg():
            rc = 2
            proc = None
            try:
                with self.lock:
                    if self.runs[rid]["status"] != "running":
                        return  # cancelled before spawn: never start the child
                with open(log_path, "w") as f:
                    kw = {}
                    if os.name != "nt":
                        kw["start_new_session"] = True
                    else:
                        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                            text=True, env=env, cwd=HERE, **kw)
                    with self.lock:
                        self.runs[rid]["pid"] = proc.pid
                        self.runs[rid]["proc"] = proc
                    rc = proc.wait(timeout=86400)
            except Exception as e:
                try:
                    with open(log_path, "a") as f:
                        f.write(f"\n[manager] runner failed: {e}\n")
                except Exception:
                    pass
            with self.lock:
                rec = self.runs[rid]
                if rec["status"] == "running":
                    rec["status"] = "done" if rc == 0 else "failed"
                rec["rc"] = rc
        threading.Thread(target=_bg, daemon=True).start()
        return rid

    def _prune_locked(self, keep=20):
        """Retention: finished runs beyond the newest `keep` lose their
        records AND log dirs (TEMP would otherwise grow one dir per run
        forever). Never prune a running run."""
        done = sorted((i for i, r in self.runs.items()
                       if r.get("status") != "running"))
        for i in done[:max(0, len(done) - keep)]:
            rec = self.runs.pop(i, {})
            try:
                log = rec.get("log") or ""
                d = os.path.dirname(log)
                if d and os.path.basename(d).startswith("llamapatch-run-"):
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    def cancel_run(self, rid):
        """Terminate the run's process tree, then clear OUR stale lock only:
        lab.lock is removed iff its pid is our reaped child (never another
        run's). Returns (http_code, dict). Liveness comes from our own
        Popen handle (poll), never signal probing — portable, no pid-reuse
        race."""
        with self.lock:
            rec = self.runs.get(rid)
            if rec is None:
                return 404, {"error": "unknown run"}
            if rec["status"] != "running":
                return 200, {"id": rid, "status": rec["status"],
                             "note": "already finished"}
            proc = rec.get("proc")
            pid = rec.get("pid")
        note = []
        if proc is None:
            note.append("process not started yet; marked cancelled")
        elif proc.poll() is not None:
            note.append("process already exited; marked cancelled")
        else:
            try:
                import signal
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                    note.append(f"process {pid} terminated")
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(pid),
                                        "/T", "/F"],
                                       capture_output=True, timeout=30)
                    else:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    try:
                        proc.wait(timeout=10)
                        note.append(f"process tree {pid} killed")
                    except subprocess.TimeoutExpired:
                        note.append(f"process tree {pid} would not die")
            except Exception as e:
                note.append(f"terminate failed: {e}")
        with self.lock:
            rec = self.runs.get(rid, {})
            rec["status"] = "cancelled"
            rec["cancel_note"] = "; ".join(note)
        lock_note = self._clear_owned_lock(rec)
        if lock_note:
            with self.lock:
                self.runs[rid]["cancel_note"] += "; " + lock_note
            note.append(lock_note)
        try:
            with open(rec.get("log", ""), "a") as f:
                f.write(f"\n[manager] cancelled: {'; '.join(note)}\n")
        except Exception:
            pass
        return 200, {"id": rid, "status": "cancelled", "note": "; ".join(note)}

    @staticmethod
    def _clear_owned_lock(rec):
        """Remove lab.lock iff it belongs to our reaped child. Never touch
        another run's lock: a pid mismatch means hands off, and liveness
        comes from our own Popen handle (no pid-reuse race, no signals)."""
        sd = rec.get("state_dir") or ""
        pid = rec.get("pid")
        proc = rec.get("proc")
        if not sd or not pid or proc is None:
            return ""
        lp = os.path.join(sd, "lab.lock")
        try:
            with open(lp) as f:
                info = json.load(f)
        except Exception:
            return ""
        if not isinstance(info, dict) or info.get("pid") != pid:
            return "lock belongs to another run; left alone"
        if proc.poll() is None:
            return "lock holder still alive; left alone"
        try:
            os.remove(lp)
            return "stale owned lock cleared"
        except Exception as e:
            return f"lock removal failed: {e}"

    def run_info(self, rid):
        with self.lock:
            rec = dict(self.runs.get(rid, {}))
        if not rec:
            return None
        rec.pop("proc", None)  # Popen handle is not JSON-serializable
        rec["log_tail"] = self.tail_log(rec.get("log", ""))
        return rec

    @staticmethod
    def tail_log(path, max_bytes=65536, max_lines=200):
        """Last lines without reading the whole file: a 50-merge run's log
        is megabytes of bench output; polling it every second must not
        load it all into memory each time."""
        try:
            size = os.path.getsize(path)
        except Exception as e:
            return f"(log unavailable: {e})"
        try:
            with open(path, "rb") as f:
                f.seek(max(0, size - max_bytes))
                chunk = f.read().decode("utf-8", "replace")
            if size > max_bytes:
                chunk = chunk.split("\n", 1)[-1]  # drop partial first line
            return "\n".join(chunk.splitlines()[-max_lines:])
        except Exception as e:
            return f"(log unavailable: {e})"


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
            qreason = {}
            for q in quar:
                if isinstance(q, dict) and isinstance(q.get("pr"), int):
                    qreason.setdefault(q["pr"], q.get("reason", ""))
            for c in cands:
                n = c["number"]
                b = bench.get(str(n), {}) if isinstance(bench, dict) else {}
                status = ("merged" if n in merged else
                          "quarantined" if n in qset else "pending")
                intent = c.get("intent")
                intent = intent if isinstance(intent, dict) else {}
                files = c.get("files", []) or []
                rows.append({"number": n, "title": c.get("title", ""),
                             "score": c.get("score"),
                             "area": intent.get("area", ""),
                             "intent_reason": intent.get("reason", ""),
                             "head": c.get("head", "") or "",
                             "files": files[:8], "file_count": len(files),
                             "ci": c.get("ci_state", "") or "",
                             "verdict": b.get("verdict", ""),
                             "quar_reason": qreason.get(n, ""),
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
        if u.path == "/api/runs":
            with self.app.lock:
                runs = [{"id": r["id"], "kind": r.get("kind", ""),
                         "status": r.get("status", ""),
                         "rc": r.get("rc")} for r in self.app.runs.values()]
            for r in runs:
                sd = self.app.runs.get(r["id"], {}).get("state_dir") or ""
                if sd:
                    try:
                        _st, _, _, _mg = self.app.load_state(sd)
                        r["merged_n"] = len(_mg or [])
                        r["quar_n"] = len(_st.get("quarantined", []) or [])
                    except Exception:
                        pass
            return _json(self, 200, {"runs": runs})
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
        if u.path == "/api/refs":
            info, err = self.app.git_refs(q.get("repo", ""))
            if err:
                return _json(self, 400, {"error": err})
            return _json(self, 200, info)
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
                              ["bash", _sh_path(os.path.join(HERE, "setup_lab.sh")),
                               _sh_path(url), _sh_path(dest)], None, env=env)
            return _json(self, 200, {"id": rid})
        if self.path == "/api/merge":
            return self._merge(body)
        if self.path == "/api/release":
            return self._release(body)
        if self.path == "/api/doctor":
            return self._doctor(body)
        if self.path == "/api/preview":
            return self._preview(body)
        if self.path.startswith("/api/runs/") and self.path.endswith("/cancel"):
            try:
                rid = int(self.path.split("/")[3])
            except (ValueError, IndexError):
                return _json(self, 400, {"error": "bad run id"})
            code, obj = self.app.cancel_run(rid)
            return _json(self, code, obj)
        return _json(self, 404, {"error": "unknown path"})

    def _spawn(self, kind, cmd, log_hint, env=None, state_dir=""):
        logdir = tempfile.mkdtemp(prefix="llamapatch-run-")
        log_path = os.path.join(logdir, f"{kind}.log")
        return self.app.start_run(kind, cmd, log_path, env=env,
                                  state_dir=state_dir)

    def _merge(self, body):
        cands, err = self.app.load_candidates(body.get("candidates", ""))
        if err:
            return _json(self, 404 if "not found" in err else 422,
                         {"error": err})
        repo = (body.get("repo") or "").strip()
        if not repo:
            return _json(self, 400, {"error": "repo checkout path required"})
        base = (body.get("base") or "master").strip() or "master"
        if not body.get("dry_run") and not self.app.verify_ref(repo, base):
            return _json(self, 400,
                         {"error": f"base ref {base!r} not found in {repo}"})
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
        rid = self._spawn("merge", cmd, None, state_dir=sd)
        return _json(self, 200, {"id": rid, "selected": os.path.basename(sel_path)})

    def _release(self, body):
        """Release quarantined PRs back to pending (retry after upstream
        fixes, rebases, or CI flips). Removes membership + detail rows
        with quarantine-file-first atomic ordering, like Lab.save."""
        if _M is None:
            return _json(self, 500, {"error": "merge_lab unavailable"})
        cands, err = self.app.load_candidates(body.get("candidates", ""))
        if err:
            return _json(self, 404 if "not found" in err else 422,
                         {"error": err})
        nums = body.get("numbers", [])
        if not nums or not isinstance(nums, list) or \
                any(not isinstance(n, int) for n in nums):
            return _json(self, 400, {"error": "numbers must be non-empty int list"})
        sd = (body.get("state_dir") or "").strip() or self.app.default_state_dir(
            body.get("candidates", ""))
        state, quar, _, _ = self.app.load_state(sd)
        qset = set(state.get("quarantined", []) or [])
        hit = [n for n in nums if n in qset]
        if not hit:
            return _json(self, 400, {"error": "none of the selected PRs are quarantined"})
        state["quarantined"] = [n for n in qset if n not in set(hit)]
        quar = [q for q in quar
                if not (isinstance(q, dict) and q.get("pr") in set(hit))]
        os.makedirs(sd, exist_ok=True)
        _M.atomic_write_json(os.path.join(sd, "quarantined.json"), quar)
        _M.atomic_write_json(os.path.join(sd, "lab-state.json"), state)
        return _json(self, 200, {"released": hit})

    def _doctor(self, body):
        """Run merge_lab --doctor as a tracked run (reconcile + requeue)."""
        repo = (body.get("repo") or "").strip()
        if not repo:
            return _json(self, 400, {"error": "repo checkout path required"})
        cands, err = self.app.load_candidates(body.get("candidates", ""))
        if err:
            return _json(self, 404 if "not found" in err else 422,
                         {"error": err})
        sd = (body.get("state_dir") or "").strip() or self.app.default_state_dir(
            body.get("candidates", ""))
        os.makedirs(sd, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.join(HERE, "merge_lab.py"),
               "--candidates", os.path.abspath(body.get("candidates", "")),
               "--repo", repo, "--state-dir", sd, "--doctor"]
        rid = self._spawn("doctor", cmd, None, state_dir=sd)
        return _json(self, 200, {"id": rid})

    def _preview(self, body):
        """Diff a PR against the base without touching merge state. Fetches
        into a dedicated preview/N namespace (never the loop's pr/N refs),
        tolerating fetch failure when the ref already exists locally —
        same resilience as the merge loop's own fetch."""
        repo = (body.get("repo") or "").strip()
        base = (body.get("base") or "master").strip() or "master"
        n = body.get("number")
        if not repo or not isinstance(n, int):
            return _json(self, 400, {"error": "repo and int number required"})
        if not self.app.verify_ref(repo, base):
            return _json(self, 400,
                         {"error": f"base ref {base!r} not found in {repo}"})
        ref = f"preview/{n}"
        frc = subprocess.run(["git", "-C", repo, "fetch", "origin",
                              f"pull/{n}/head:{ref}", "--force"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=300)
        if frc.returncode != 0 and not self.app.verify_ref(repo, ref):
            return _json(self, 404, {"error": f"PR #{n} not fetchable: "
                                              f"{(frc.stdout or '')[-300:]}"})
        stat = subprocess.run(["git", "-C", repo, "diff", "--stat",
                               f"{base}...{ref}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
        diff = subprocess.run(["git", "-C", repo, "diff",
                               f"{base}...{ref}", "--", "."],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120)
        if stat.returncode != 0:
            return _json(self, 422, {"error": f"diff failed: "
                                              f"{(stat.stdout or '')[-300:]}"})
        lines = (diff.stdout or "").splitlines()
        if len(lines) > 300:
            lines = lines[:300] + [f"... ({len(lines) - 300} more lines)"]
        return _json(self, 200, {"base": base, "stat": stat.stdout or "",
                                 "diff": "\n".join(lines)})


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
