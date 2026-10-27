#!/usr/bin/env bash
# Pull finished results (result.json, run.log) from every VM into results/runs/<name>/,
# then rebuild results/README.md.
# Usage: scripts/collect.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/hosts.sh"
DEST="$HERE/../results/runs"
mkdir -p "$DEST"
for ip in $HOSTS; do
  rsync -a -e "$SSH" --include='*/' --include='result.json' --include='run.log' --exclude='*' \
    "ubuntu@$ip:work/runs/" "$DEST/" 2>/dev/null
done
python3 "$HERE/summarize.py" "$DEST" "$HERE/../results/README.md"
