#!/usr/bin/env bash
# Pull every finished job's small result JSONs (not weights; those live on HF) from the VM.
# Programme-1 prefixes (a_ b_ c_ d_ e_ f_ smoke) -> research/fields/results/
# Programme-2 prefixes (h_ t_ c3_ r_)           -> research/adversarial/results/
# The remote side LISTS candidate files at depth <= 2 (rsync filter rules would traverse whole run trees: slow with large outputs);
# rsync then copies exactly that list.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p research/fields/results research/adversarial/results
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
while read -r H; do
  [ -z "$H" ] && continue
  LIST=$($SSH "$H" "cd ~/work/runs && find . -maxdepth 2 -type f \( -name job_result.json -o -name metrics.json -o -name result.json -o -name 'result_s*.json' \) -size -2000k | sed 's|^\./||'") || { echo "collect: ssh failed for $H"; continue; }
  echo "$LIST" | grep -E '^(h_|t_|c3_|r_)' > /tmp/collect_adv.$$ ; echo "$LIST" | grep -vE '^(h_|t_|c3_|r_)' | grep -v '^$' > /tmp/collect_fld.$$
  [ -s /tmp/collect_adv.$$ ] && rsync -aq -e "$SSH" --files-from=/tmp/collect_adv.$$ "$H:work/runs/" research/adversarial/results/ 2>&1 | head -3
  [ -s /tmp/collect_fld.$$ ] && rsync -aq -e "$SSH" --files-from=/tmp/collect_fld.$$ "$H:work/runs/" research/fields/results/ 2>&1 | head -3
  rm -f /tmp/collect_adv.$$ /tmp/collect_fld.$$
done < research/fields/scripts/hosts.txt
echo "result dirs: fields $(ls research/fields/results | wc -l | tr -d ' '), adversarial $(ls research/adversarial/results | wc -l | tr -d ' ')"
