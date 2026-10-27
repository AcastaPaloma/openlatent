#!/usr/bin/env bash
# Snapshot of every VM's queue as JSON (for the dashboard's live panel).
# Usage: scripts/live.sh > live.json
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/hosts.sh"
echo '{"hosts": ['
first=1
for ip in $HOSTS; do
  j=$($SSH ubuntu@$ip 'python3 - <<EOF
import json, os, glob, subprocess
Q = os.path.expanduser("~/work/queue")
def cfg(p):
    try: return json.load(open(p))
    except Exception: return {"name": os.path.basename(p)}
def name(p): return cfg(p)["name"]
def last(n):
    try:
        lines = [l for l in open(os.path.expanduser(f"~/work/logs/{n}.log"), errors="replace") if l.startswith("[")]
        return lines[-1].strip()[:200] if lines else ""
    except Exception: return ""
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(json.dumps({"name": os.uname().nodename, "gpu": gpu,
  "running": [{"name": name(p), "config": cfg(p), "last_line": last(name(p))} for p in sorted(glob.glob(Q + "/running/*.json"))],
  "pending": [name(p) for p in sorted(glob.glob(Q + "/pending/*.json"))],
  "failed": [name(p) for p in sorted(glob.glob(Q + "/failed/*.json"))]}))
EOF' 2>/dev/null)
  [ -z "$j" ] && continue
  [ $first -eq 0 ] && echo ","
  echo "$j"; first=0
done
echo "], \"time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
