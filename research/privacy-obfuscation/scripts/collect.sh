#!/usr/bin/env bash
# Pull finished results (result.json + run.log of runs that have a result.json) from every VM
# into results/runs/<name>/, then rebuild results/README.md. Runs still in progress are skipped.
# Usage: scripts/collect.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/hosts.sh"
DEST="$HERE/../results/runs"
mkdir -p "$DEST"
for ip in $HOSTS; do
  list=$($SSH ubuntu@$ip 'cd ~/work/runs 2>/dev/null && for d in */; do [ -f "$d/result.json" ] && printf "%sresult.json\n%srun.log\n" "$d" "$d"; done' 2>/dev/null)
  [ -n "$list" ] && rsync -a -e "$SSH" --files-from=<(echo "$list") "ubuntu@$ip:work/runs/" "$DEST/" 2>/dev/null
done
python3 "$HERE/summarize.py" "$DEST" "$HERE/../results/README.md"
