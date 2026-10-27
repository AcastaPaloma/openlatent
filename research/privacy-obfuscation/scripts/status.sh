#!/usr/bin/env bash
# One screen of loop state: per VM, GPU use, worker, queue, running jobs' last log line, failures.
# Usage: scripts/status.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/hosts.sh"
for ip in $HOSTS; do
  $SSH ubuntu@$ip 'Q=~/work/queue
    echo "=== $(hostname)  gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)  worker: $(pgrep -f "^python3 .*worker.py" >/dev/null && echo up || echo DOWN)"
    echo "  pending: $(ls $Q/pending 2>/dev/null | tr "\n" " ")"
    for j in $(ls $Q/running 2>/dev/null); do n=$(python3 -c "import json;print(json.load(open(\"$Q/running/$j\"))[\"name\"])"); echo "  running $n: $(tail -1 ~/work/logs/$n.log 2>/dev/null | cut -c1-150)"; done
    echo "  done: $(ls $Q/done 2>/dev/null | wc -l)  failed: $(ls $Q/failed 2>/dev/null | tr "\n" " ")"' 2>&1
done
