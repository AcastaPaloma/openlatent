#!/usr/bin/env bash
# Pull every finished job's result JSON (not weights; those live on HF) into research/fields/results/.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
while read -r H; do
  [ -z "$H" ] && continue
  rsync -aq --include='*/' --include='job_result.json' --include='metrics.json' --include='result.json' --exclude='*' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$H:work/runs/" results/ 2>&1 | head -3
done < scripts/hosts.txt
ls results | wc -l | sed 's/^/result dirs: /'
