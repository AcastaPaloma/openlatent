#!/usr/bin/env bash
# One look at every machine.  Usage: bash tick.sh   (hosts in hosts.txt, one user@ip per line)
cd "$(dirname "$0")"
while read -r H; do
  [ -z "$H" ] && continue
  echo "=== $H  $(date -u +%H:%MZ)"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$H" '
    git -C ~/openlatent pull -q --ff-only origin fields-research 2>/dev/null
    python3 ~/openlatent/research/fields/scripts/normalize_queue.py
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed "s/^/gpu /"
    echo "workers: $(pgrep -fc "[w]orker.py --gpu")"
    for d in pending running done failed; do echo "$d: $(ls ~/queue/$d 2>/dev/null | wc -l)"; done
    ls ~/queue/failed 2>/dev/null | head -5
    df -h ~ | tail -1 | awk "{print \"disk used \" \$5}"
    for f in ~/queue/running/*; do [ -e "$f" ] && echo "run: $(basename $f)"; done
    true
  ' 2>&1 || echo "UNREACHABLE $H"
done < hosts.txt
