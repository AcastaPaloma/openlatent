"""The harness UI: a local web app over the files each stage writes. Standard library only, so it runs on a laptop.

    python -m privharness ui [--port 8765] [--no-open]

Runs are read from two places: results/harness/<name> (saved in this repo) and a local mirror of every VM's
~/work/harness (remote.sync), refreshed in the background. Starting and stopping work goes through remote.py. VMs
added here are kept in ~/.privharness/hosts.json, outside the repo.

Binds to 127.0.0.1 only. Requests whose Host or Origin is not this server are refused, so a web page open in another
tab cannot use it to start work on a VM.
"""

import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import remote

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL = "local"
LOCAL_ROOT = os.path.join(HERE, "results", "harness")
STORE = os.path.expanduser(os.environ.get("PRIVHARNESS_UI_HOME", "~/.privharness"))
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
REFRESH_S = 20
LOG_RE = re.compile(r"(harness\.log|console\.log|logs/[A-Za-z0-9._-]+\.log)")
RERUNS = ("_s1", "_s2", "_e24", "_dino")
STAGES = [("ingest", "Read the files"), ("profile", "Find where the information lives"), ("plan", "Choose disguises to try"),
          ("screen", "Quick test of every disguise"), ("evaluate", "Attack the shortlist"),
          ("robustness", "Re-attack the winner"), ("report", "Write the report")]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tail(path, n=200):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def failed_candidates(root):
    """Candidates whose latest evaluation in harness.log failed."""
    last = {}
    for line in tail(os.path.join(root, "harness.log"), 5000):
        m = re.search(r"evaluate (\S+): (done|FAILED|already done|started)", line)
        if m:
            last[m[1]] = m[2]
    return sorted(k for k, v in last.items() if v == "FAILED")


def evaluated(root):
    d = os.path.join(root, "eval")
    return sorted(n for n in os.listdir(d) if os.path.exists(os.path.join(d, n, "result.json"))) if os.path.isdir(d) else []


def progress(root, running):
    """Stage list and overall status from the files on disk, so it holds however the run was started."""
    have = lambda f: os.path.exists(os.path.join(root, f))  # noqa: E731
    done = set(evaluated(root))
    short = (read_json(os.path.join(root, "shortlist.json")) or {}).get("shortlist", [])
    rep = read_json(os.path.join(root, "report.json")) or {}
    best = (rep.get("recommendation") or {}).get("best")
    reruns = [best + s for s in RERUNS] if best else []
    n_rerun = sum(r in done for r in reruns)
    reran = bool(best) and n_rerun == len(reruns)
    code = (tail(os.path.join(root, "exit_code.txt"), 1) or [""])[0].strip()
    # a report also exists partway through, before the re-attacks; only these say the search got to its end
    completed = reran or code == "0" or (code == "" and "] report: " in "".join(tail(os.path.join(root, "harness.log"), 1)))
    flags = dict(ingest=have("ingest_report.json") or have("dataset.json"), profile=have("profile.json"), plan=have("plan.json"),
                 screen=have("screen.json") and have("shortlist.json"), evaluate=bool(short) and set(short) <= done,
                 robustness=reran, report=have("report.md") and completed and not (running and not reran))
    detail = dict(evaluate=f"{sum(n in done for n in short)} of {len(short)}" if short else "",
                  robustness=f"{n_rerun} of {len(reruns)}" if reruns else "")
    finished = flags["report"]
    current = None if finished else next((k for k, _ in STAGES if not flags[k]), None)
    status = "running" if running else "finished" if finished else "failed" if code not in ("", "0", "stopped") else "stopped" if flags["ingest"] or have("console.log") else "starting"
    stages = []
    for k, label in STAGES:
        state = "done" if flags[k] else "active" if running and k == current else "halted" if k == current else "skipped" if k == "robustness" and finished else "waiting"
        stages.append(dict(id=k, label=label, state=state, detail=detail.get(k, "")))
    return dict(status=status, stages=stages, current=current, exit_code=code, evaluated=sorted(done), shortlist=short,
                failed=failed_candidates(root))


def certified(rep):
    """report.json's certified scores; reports written before they existed get the same rule (worst over the re-runs)."""
    rec, scored = rep.get("recommendation") or {}, rep.get("scored") or {}
    best = rec.get("best")
    if not best or best not in scored:
        return None, None
    runs = [scored[n] for n in [best, *rec.get("robustness", [])] if n in scored]
    p = rec.get("certified_privacy", min((v["privacy"] for v in runs if v.get("privacy") is not None), default=None))
    u = rec.get("certified_utility", min((v["utility"]["retention"] for v in runs if v["utility"].get("retention") is not None), default=None))
    return p, u


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.path = os.path.join(STORE, "hosts.json")
        self.hosts = read_json(self.path) or []
        self.status = {}
        self.host_locks = {}

    def save(self):
        os.makedirs(STORE, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.hosts, f, indent=1)

    def mirror(self, host):
        return os.path.join(STORE, "mirror", host)

    def add_host(self, host):
        remote.check_host(host)
        with self.lock:
            if host not in self.hosts:
                self.hosts.append(host)
                self.save()

    def remove_host(self, host):
        with self.lock:
            if host in self.hosts:
                self.hosts.remove(host)
                self.save()
            self.status.pop(host, None)

    def refresh(self, host):
        """Probe the VM and mirror its results. One refresh per VM at a time; a second caller just waits for it."""
        if host not in self.hosts:
            raise remote.RemoteError("That VM is not on your list.")
        with self.host_locks.setdefault(host, threading.Lock()):
            st = dict(self.status.get(host, {}), checked=time.time())
            try:
                st.update(probe=remote.probe(host), error=None)
                if st["probe"]["runs"]:
                    remote.sync(host, self.mirror(host))
            except remote.RemoteError as e:
                st.update(error=str(e))
            if host in self.hosts:
                self.status[host] = st
            return st

    def background(self):
        while True:
            for host in list(self.hosts):
                try:
                    self.refresh(host)
                except Exception as e:  # keep refreshing the other VMs
                    self.status[host] = dict(self.status.get(host, {}), error=f"{type(e).__name__}: {e}", checked=time.time())
            time.sleep(REFRESH_S)

    def root(self, host, name):
        base = LOCAL_ROOT if host == LOCAL else self.mirror(host)
        if not remote.NAME_RE.fullmatch(name) or (host != LOCAL and host not in self.hosts):
            raise remote.RemoteError("There is no such test.")
        return os.path.join(base, name)

    def is_running(self, host, name):
        return host != LOCAL and name in ((self.status.get(host) or {}).get("probe") or {}).get("running", [])

    def summary(self, host, name):
        root = self.root(host, name)
        prog = progress(root, self.is_running(host, name))
        rep = read_json(os.path.join(root, "report.json")) or {}
        plan = read_json(os.path.join(root, "plan.json")) or {}
        rec = rep.get("recommendation") or {}
        p, u = certified(rep)
        scored = rep.get("scored") or {}
        cand = plan.get("candidates") or {}
        ranked = sorted((n for n in scored if n in cand and n not in ("raw", "black")), key=lambda n: -(scored[n].get("pus") or -1))
        order = (ranked or prog["shortlist"] or plan.get("screen") or [])[:6]
        return dict(host=host, name=name, place="This computer" if host == LOCAL else host, **prog, best=rec.get("best"),
                    privacy=p, utility=u, reachable=host == LOCAL or not (self.status.get(host) or {}).get("error"),
                    frames=[dict(name=n, config=(cand.get(n) or {}).get("config")) for n in order])

    def runs(self):
        out = []
        for host in [LOCAL, *self.hosts]:
            base = LOCAL_ROOT if host == LOCAL else self.mirror(host)
            names = set(os.listdir(base)) if os.path.isdir(base) else set()
            if host != LOCAL:
                names |= set(((self.status.get(host) or {}).get("probe") or {}).get("runs", []))
            for n in sorted(names):
                if remote.NAME_RE.fullmatch(n) and (host != LOCAL or os.path.isdir(os.path.join(base, n))):
                    out.append(self.summary(host, n))
        return out

    def state(self):
        hosts = [dict(host=h, **{k: v for k, v in (self.status.get(h) or {}).items()}) for h in self.hosts]
        return dict(hosts=hosts, runs=self.runs(), adapters=sorted(remote.ADAPTERS))

    def run(self, host, name):
        root = self.root(host, name)
        out = self.summary(host, name)
        for key, f in (("report", "report.json"), ("plan", "plan.json"), ("profile", "profile.json"), ("screen", "screen.json"),
                       ("ingest", "ingest_report.json")):
            out[key] = read_json(os.path.join(root, f))
        if out["report"]:
            out["report"].pop("generated_from", None)
        logs = os.path.join(root, "logs")
        out["logs"] = ["harness.log", "console.log"] + (sorted(f"logs/{f}" for f in os.listdir(logs)) if os.path.isdir(logs) else [])
        out["console_tail"] = tail(os.path.join(root, "console.log"), 12) if out["status"] == "failed" else []
        return out

    def log(self, host, name, file):
        if not LOG_RE.fullmatch(file or ""):
            raise remote.RemoteError("There is no such log.")
        return dict(file=file, lines=tail(os.path.join(self.root(host, name), file), 300))

    def post(self, path, b):
        host = b.get("host")
        if path == "/api/hosts/add":
            self.add_host(host)
            return self.refresh(host)
        if host not in self.hosts:
            raise remote.RemoteError("That VM is not on your list. Add it first.")
        if path == "/api/hosts/remove":
            self.remove_host(host)
            return dict(ok=True)
        if path == "/api/hosts/refresh":
            return self.refresh(host)
        if path == "/api/hosts/setup":
            out = remote.setup(host)
            self.refresh(host)
            return out
        if path == "/api/files":
            return remote.list_dir(host, b.get("path"))
        if path == "/api/table":
            return remote.peek_table(host, b.get("path") or "")
        if path == "/api/launch":
            desc = b.get("description")
            if desc is not None:
                check_description(desc)
                desc["name"] = b.get("name")
            out = remote.launch(host, b.get("name"), b.get("options") or {}, desc, b.get("adapt"), b.get("split"), bool(b.get("resume")))
            self.refresh(host)
            return out
        if path == "/api/stop":
            out = remote.stop(host, b.get("name"))
            self.refresh(host)
            return out
        raise remote.RemoteError("Unknown request.")


def check_description(d):
    """The checks ingest would otherwise fail on hours later or on the VM, phrased for the person filling in the form."""
    if not isinstance(d, dict) or not (d.get("folder") or d.get("table")):
        raise remote.RemoteError("Say where the files are: a folder or a table.")
    if d.get("folder"):
        try:
            names = re.compile(d["folder"].get("pattern", "")).groupindex
        except re.error as e:
            raise remote.RemoteError(f"The file pattern is not valid: {e}") from None
        cols = {t.get("column") for t in d.get("targets", [])}
        if not cols <= set(names):
            raise remote.RemoteError(f"The file pattern does not capture: {', '.join(sorted(str(c) for c in cols - set(names)))}.")
    roles = [t.get("role") for t in d.get("targets", [])]
    if "utility" not in roles:
        raise remote.RemoteError("Say what must stay usable: mark one label as the thing to keep.")
    if "secret" not in roles:
        raise remote.RemoteError("Say what must stay hidden: mark at least one label as the thing to hide.")


class Handler(BaseHTTPRequestHandler):
    server_version = "privharness"
    app = None

    def log_message(self, *args):
        pass

    def allowed(self):
        """Only this server's own pages: blocks DNS rebinding and other sites posting to the local port."""
        own = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in own and (origin is None or urlparse(origin).netloc in own)

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def guarded(self, fn):
        if not self.allowed():
            return self.send(403, dict(error="Refused: this request did not come from the harness UI."))
        try:
            self.send(200, fn())
        except remote.RemoteError as e:
            self.send(400, dict(error=str(e)))
        except Exception as e:
            self.send(500, dict(error=f"The UI hit an unexpected error: {type(e).__name__}: {e}"))

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            with open(INDEX, "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        routes = {"/api/state": self.app.state, "/api/run": lambda: self.app.run(q.get("host"), q.get("name", "")),
                  "/api/log": lambda: self.app.log(q.get("host"), q.get("name", ""), q.get("file"))}
        if u.path in routes:
            return self.guarded(routes[u.path])
        self.send(404, dict(error="Not found."))

    def do_POST(self):
        def handle():
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise remote.RemoteError("Requests must be JSON.")
            n = int(self.headers.get("Content-Length") or 0)
            return self.app.post(urlparse(self.path).path, json.loads(self.rfile.read(n) or b"{}"))
        self.guarded(handle)


def serve(port=8765, open_browser=True):
    app = App()
    Handler.app = app
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=app.background, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"privharness UI: {url}  (Ctrl-C to stop; runs on VMs keep going)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
