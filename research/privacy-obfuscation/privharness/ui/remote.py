"""SSH layer of the harness UI: everything the laptop asks of a GPU VM, with the user's own keys and ssh config.

Every remote step is a small Python script piped to `python3 -` over ssh, so user input travels as JSON inside the
script and never through a remote shell. Searches run in tmux on the VM and survive the laptop closing.

    probe     is the VM reachable, what GPU and disk does it have, is the harness installed, which runs are alive
    sync      pull scores and logs into a local mirror (never frames, weights or predictions)
    list_dir  / peek_table   look at a dataset on the VM for the guided setup
    launch / stop / setup    start or resume a search, stop it, install the harness (scripts/setup_vm.sh)

Layout on the VM, as used by every earlier run: ~/openlatent (repo), ~/work/harness/<name> (one dataset).
"""

import json
import os
import re
import shlex
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# BatchMode: never prompt, so a missing key fails with a message instead of hanging. accept-new: trust a VM the first
# time it is added, as `ssh` would after answering yes, and still refuse a host key that later changes.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "StrictHostKeyChecking=accept-new"]
HOST_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*(@[A-Za-z0-9_][A-Za-z0-9._-]*)?")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
ADAPTERS = {"kth": "kth", "ucf101_vp": "ucf101", "hmdb51_vp": "hmdb51", "charades": "charades", "utdmhad": "utdmhad"}
MEDIA_EXT = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v", ".wmv", ".flv",  # ingest.VIDEO_EXT
             ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]  # ingest.IMAGE_EXT


class RemoteError(Exception):
    """A failure worded for the person at the screen."""


def check_host(host):
    if not isinstance(host, str) or not HOST_RE.fullmatch(host):
        raise RemoteError("Write the VM as user@address (for example ubuntu@203.0.113.7) or a name from your ssh config.")
    return host


def check_name(name):
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise RemoteError("Use a short name with lowercase letters, numbers, dashes or underscores.")
    return name


def ssh_opts():
    cfg = os.environ.get("PRIVHARNESS_SSH_CONFIG")
    return (["-F", cfg] if cfg else []) + SSH_OPTS


def explain(stderr):
    """ssh's stderr -> what happened and what to do about it."""
    e = stderr.strip()
    last = e.splitlines()[-1] if e else "no error text"
    if "Permission denied" in e:
        return "The VM refused your SSH key. Check the username, and that your key can log in with plain ssh."
    if "Could not resolve" in e:
        return "That address could not be found. Check it for typos, or that the VM still exists."
    if "timed out" in e or "Operation timed out" in e or "No route to host" in e or "Connection refused" in e:
        return "The VM did not answer. It may be off or terminated, or its firewall may block SSH."
    if "Host key verification failed" in e or "REMOTE HOST IDENTIFICATION" in e:
        return "This VM's identity changed since you last connected. If you recreated it, remove its old entry with ssh-keygen -R."
    if "python3" in e and "not found" in e:
        return "The VM has no python3, which the harness needs."
    return f"SSH failed: {last[:300]}"


def run_py(host, script, args=None, timeout=60):
    """Run `script` on the VM with the JSON-safe `args` as the variable A; it prints one JSON line."""
    check_host(host)
    src = f"import json\nA = json.loads({json.dumps(args or {})!r})\n{script}"
    try:
        p = subprocess.run(["ssh", *ssh_opts(), host, "python3", "-"], input=src, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RemoteError("The VM took too long to answer. Try again.") from None
    except FileNotFoundError:
        raise RemoteError("This computer has no ssh command.") from None
    if p.returncode == 255:
        raise RemoteError(explain(p.stderr))
    lines = p.stdout.strip().splitlines()
    if p.returncode != 0 or not lines:
        raise RemoteError(f"The VM reported an error: {(p.stderr.strip().splitlines() or ['no error text'])[-1][:300]}")
    out = json.loads(lines[-1])
    if isinstance(out, dict) and out.get("error"):
        raise RemoteError(out["error"])
    return out


PROBE = r"""
import importlib.util as iu, os, shutil, subprocess
H = os.path.expanduser("~")
REPO, ROOT = H + "/openlatent/research/privacy-obfuscation", H + "/work/harness"
def sh(*c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""
names = sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
ps = [l.split() for l in sh("ps", "-eo", "args").splitlines() if "privharness" in l]
sessions = sh("tmux", "ls", "-F", "#S").splitlines() if shutil.which("tmux") else []
running = [n for n in names if os.path.isdir(ROOT + "/" + n) and
           ("ph_" + n in sessions or any(ROOT + "/" + n in l for l in ps))]
gpus = []
for l in sh("nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits").splitlines():
    p = [x.strip() for x in l.split(",")]
    if len(p) == 4:
        gpus.append(dict(name=p[0], mem_total=p[1], mem_used=p[2], util=p[3]))
setup = H + "/work/privharness_setup"
tail = lambda f, n=12: open(f, errors="replace").read().splitlines()[-n:] if os.path.exists(f) else []
print(json.dumps(dict(
    hostname=os.uname().nodename, gpus=gpus, disk_free_gb=round(shutil.disk_usage(H).free / 1e9),
    tmux=bool(shutil.which("tmux")), harness=os.path.exists(REPO + "/privharness/__main__.py"),
    libraries=all(iu.find_spec(m) for m in ("torch", "torchvision", "timm", "cv2")),
    commit=sh("git", "-C", H + "/openlatent", "log", "-1", "--format=%h %s")[:90],
    home=H, runs=names, running=running,
    data={n: os.path.isdir(H + "/work/data/" + d + "/frames") for n, d in A["adapters"].items()},
    setup=dict(running=sh("pgrep", "-f", "privharness_setup[.]sh") != "", log=tail(setup + ".log"),
               exit=tail(setup + ".exit", 1)))))
"""


def probe(host):
    return run_py(host, PROBE, dict(adapters=ADAPTERS), timeout=45)


def sync(host, dest):
    """Mirror ~/work/harness: scores, plans and logs only. Frames stay on the VM (dataset licences may forbid copying)."""
    os.makedirs(dest, exist_ok=True)
    filters = ["--exclude=frames/", "--exclude=dataset.json", "--include=*/", "--include=*.json", "--include=*.md",
               "--include=*.svg", "--include=*.log", "--include=exit_code.txt", "--exclude=*"]
    cmd = ["rsync", "-a", "--prune-empty-dirs", *filters, "-e", shlex.join(["ssh", *ssh_opts()]), f"{check_host(host)}:work/harness/", dest + "/"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RemoteError("Copying results from the VM took too long. It will try again.") from None
    if p.returncode == 255:
        raise RemoteError(explain(p.stderr))
    if p.returncode not in (0, 23, 24):  # 23: no ~/work/harness yet; 24: a file vanished mid-copy
        raise RemoteError(f"Copying results failed: {(p.stderr.strip().splitlines() or ['rsync error'])[-1][:300]}")


LIST_DIR = r"""
import os
path = os.path.expanduser(A["path"] or "~")
if not os.path.isdir(path):
    print(json.dumps(dict(error=f"There is no folder at {path} on this VM.")))
    raise SystemExit
files, total, ext = [], 0, tuple(A["ext"])
try:
    dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith("."))
    tables = sorted(f for f in os.listdir(path) if f.lower().endswith((".csv", ".jsonl")))
except PermissionError:
    print(json.dumps(dict(error=f"This login cannot read {path}.")))
    raise SystemExit
for dirpath, dirnames, names in os.walk(path):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    for n in sorted(names):
        if n.lower().endswith(ext):
            total += 1
            if len(files) < A["limit"]:
                files.append(os.path.relpath(os.path.join(dirpath, n), path))
print(json.dumps(dict(path=path, parent=os.path.dirname(path), dirs=dirs[:200], tables=[os.path.join(path, t) for t in tables[:50]], total=total, files=files)))
"""


def list_dir(host, path):
    return run_py(host, LIST_DIR, dict(path=str(path or "")[:1024], ext=MEDIA_EXT, limit=20000), timeout=120)


PEEK_TABLE = r"""
import csv, os
path = os.path.expanduser(A["path"])
if not os.path.isfile(path):
    print(json.dumps(dict(error=f"There is no file at {path} on this VM.")))
    raise SystemExit
rows = [json.loads(l) for l in open(path) if l.strip()] if path.endswith(".jsonl") else list(csv.DictReader(open(path, newline="")))
if not rows:
    print(json.dumps(dict(error="That table has no rows.")))
    raise SystemExit
cols = {}
for c in rows[0]:
    vals = {str(r.get(c, "")) for r in rows}
    cols[c] = dict(distinct=len(vals), sample=sorted(vals)[:6])
print(json.dumps(dict(path=path, rows=len(rows), columns=cols, first=[str(rows[0].get(c, ""))[:120] for c in rows[0]])))
"""


def peek_table(host, path):
    return run_py(host, PEEK_TABLE, dict(path=str(path)[:1024]), timeout=120)


LAUNCH = r"""
import os, shlex, shutil, subprocess
H = os.path.expanduser("~")
REPO, name = H + "/openlatent/research/privacy-obfuscation", A["name"]
root, session = H + "/work/harness/" + name, "ph_" + name
fail = lambda msg: (print(json.dumps(dict(error=msg))), exit())
if not os.path.exists(REPO + "/privharness/__main__.py"):
    fail("The harness is not installed on this VM yet. Set the VM up first.")
if not shutil.which("tmux"):
    fail("This VM has no tmux. Set the VM up first.")
if subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0:
    fail(f"{name} is already running on this VM.")
ready = os.path.exists(root + "/dataset.json")
if os.path.exists(root) and not A["resume"]:
    fail(f"This VM already has a test called {name}. Pick another name, or open that test and resume it.")
if A["resume"] and not ready and not (A.get("description") or A.get("adapt")):
    A["description"] = json.load(open(root + "/description.json")) if os.path.exists(root + "/description.json") else None
    if not A["description"]:
        fail(f"{name} never finished reading its files and has no saved setup. Start it again as a new test.")
os.makedirs(root, exist_ok=True)
py = "python3 -m privharness"
steps = []
if not ready:
    if A.get("description"):
        json.dump(A["description"], open(root + "/description.json", "w"), indent=1)
        steps.append(f"{py} ingest {shlex.quote(root + '/description.json')} {shlex.quote(root)}")
    elif A.get("adapt"):
        steps.append(f"{py} adapt {shlex.quote(A['adapt'])} {shlex.quote(root)}" + (f" --split {shlex.quote(A['split'])}" if A.get("split") else ""))
o = A["options"]
saved = root + "/run_options.json"
if A["resume"] and os.path.exists(saved):  # keep what the user chose; a different shortlist size would change the search
    o = {**json.load(open(saved)), "parallel": o["parallel"]}
json.dump(o, open(saved, "w"))
search = f"{py} search {shlex.quote(root)} --parallel {int(o['parallel'])}"
if o.get("budget"):
    search += f" --budget {int(o['budget'])}"
if not o.get("robustness", True):
    search += " --no-robustness"
steps.append(search)
q = shlex.quote
cmd = f"cd {q(REPO)} && export PYTHONUNBUFFERED=1 && ( {' && '.join(steps)} ) >> {q(root + '/console.log')} 2>&1; echo $? > {q(root + '/exit_code.txt')}"
if os.path.exists(root + "/exit_code.txt"):
    os.remove(root + "/exit_code.txt")
subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash -lc " + q(cmd)], check=True)
print(json.dumps(dict(ok=True, command=" && ".join(steps), session=session)))
"""


def launch(host, name, options, description=None, adapt=None, split=None, resume=False):
    check_name(name)
    if adapt is not None and adapt not in ADAPTERS:
        raise RemoteError("That is not one of the prepared research datasets.")
    if split is not None and not re.fullmatch(r"[a-z0-9_]{1,32}", str(split)):
        raise RemoteError("That split name is not valid.")
    if not resume and not description and not adapt:
        raise RemoteError("Choose a dataset first.")
    o = dict(parallel=max(1, min(8, int(options.get("parallel", 2)))), robustness=bool(options.get("robustness", True)))
    if options.get("budget"):
        o["budget"] = max(2, min(30, int(options["budget"])))
    return run_py(host, LAUNCH, dict(name=name, options=o, description=description, adapt=adapt, split=split, resume=bool(resume)))


STOP = r"""
import os, subprocess
root = os.path.expanduser("~/work/harness/") + A["name"]
subprocess.run(["tmux", "kill-session", "-t", "=ph_" + A["name"]], capture_output=True)
subprocess.run(["pkill", "-f", "privharness.* " + root.replace(".", "[.]") + "( |$)"], capture_output=True)
if os.path.isdir(root):  # the UI tells a run someone stopped from one that crashed or lost its VM
    open(root + "/exit_code.txt", "w").write("stopped\n")
print(json.dumps(dict(ok=True)))
"""


def stop(host, name):
    return run_py(host, STOP, dict(name=check_name(name)))


SETUP = r"""
import os, subprocess
base = os.path.expanduser("~/work/privharness_setup")
os.makedirs(os.path.dirname(base), exist_ok=True)
if subprocess.run(["pgrep", "-f", "privharness_setup[.]sh"], capture_output=True).returncode == 0:
    print(json.dumps(dict(error="Setup is already running on this VM.")))
    raise SystemExit
open(base + ".sh", "w").write(A["script"])
if os.path.exists(base + ".exit"):
    os.remove(base + ".exit")
cmd = f"bash {base}.sh > {base}.log 2>&1; echo $? > {base}.exit"
subprocess.Popen(["bash", "-c", cmd], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(json.dumps(dict(ok=True)))
"""


def setup(host):
    """Run scripts/setup_vm.sh on the VM, detached; progress comes back through probe()."""
    script = open(os.path.join(HERE, "scripts", "setup_vm.sh")).read()
    return run_py(host, SETUP, dict(script=script))
