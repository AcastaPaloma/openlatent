#!/usr/bin/env bash
# One look at every machine.  Usage: bash tick.sh   (hosts in hosts.txt, one user@ip per line)
cd "$(dirname "$0")"
while read -r H; do
  [ -z "$H" ] && continue
  echo "=== $H  $(date -u +%H:%MZ)"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$H" '
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed "s/^/gpu /"
    echo "workers: $(pgrep -fc "worker.py --gpu")"
    for d in pending running done failed; do echo "$d: $(ls ~/queue/$d 2>/dev/null | wc -l)"; done
    ls ~/queue/failed 2>/dev/null | head -5
    df -h ~ | tail -1 | awk "{print \"disk used \" \$5}"
    for f in ~/queue/running/*; do [ -e "$f" ] && echo "run: $(basename $f)"; done
  ' 2>&1 || echo "UNREACHABLE $H"
done < hosts.txt
