#!/usr/bin/env bash
# Pull every finished job's result JSON (not weights; those live on HF) from the VM.
# Programme-1 prefixes (a_ b_ c_ d_ e_ f_ smoke) -> research/fields/results/; programme-2 prefixes (h_ t_ c3_ r_) -> research/adversarial/results/.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p research/fields/results research/adversarial/results
INC=(--include='*/' --include='job_result.json' --include='metrics.json' --include='result.json' --include='result_s*.json' --exclude='*')
while read -r H; do
  [ -z "$H" ] && continue
  rsync -aq "${INC[@]}" --exclude='h_*' --exclude='t_*' --exclude='c3_*' --exclude='r_*' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$H:work/runs/" research/fields/results/ 2>&1 | head -3
  rsync -aq --include='h_*/' --include='t_*/' --include='c3_*/' --include='r_*/' --include='h_*/**' --include='t_*/**' --include='c3_*/**' --include='r_*/**' --exclude='*' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$H:work/runs/" /tmp/adv_runs_$$/ 2>/dev/null
  if [ -d /tmp/adv_runs_$$ ]; then
    rsync -aq "${INC[@]}" /tmp/adv_runs_$$/ research/adversarial/results/ ; rm -rf /tmp/adv_runs_$$
  fi
done < research/fields/scripts/hosts.txt
echo "result dirs: fields $(ls research/fields/results | wc -l), adversarial $(ls research/adversarial/results | wc -l)"
