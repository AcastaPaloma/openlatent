#!/usr/bin/env bash
# Pull every finished job's result JSON (not weights; those live on HF) from the VM.
# Programme-1 prefixes (a_ b_ c_ d_ e_ f_ smoke) -> research/fields/results/
# Programme-2 prefixes (h_ t_ c3_ r_)           -> research/adversarial/results/
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p research/fields/results research/adversarial/results
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
KEEP=(--include='*/' --include='job_result.json' --include='metrics.json' --include='result.json' --include='result_s*.json' --exclude='*')
while read -r H; do
  [ -z "$H" ] && continue
  # programme 2 first (explicit prefixes), then everything else is programme 1
  rsync -aq -e "$SSH" --include='h_*/' --include='t_*/' --include='c3_*/' --include='r_*/' \
    --include='job_result.json' --include='metrics.json' --include='result.json' --include='result_s*.json' --exclude='*' \
    "$H:work/runs/" research/adversarial/results/ 2>&1 | head -3
  rsync -aq -e "$SSH" --exclude='h_*' --exclude='t_*' --exclude='c3_*' --exclude='r_*' "${KEEP[@]}" \
    "$H:work/runs/" research/fields/results/ 2>&1 | head -3
done < research/fields/scripts/hosts.txt
echo "result dirs: fields $(ls research/fields/results | wc -l | tr -d ' '), adversarial $(ls research/adversarial/results | wc -l | tr -d ' ')"
