#!/bin/bash
# pull_runs.sh DEST: copy finished d_* job outputs (records, units, meta; not logs) from the VM into DEST/<job>/
VM=ubuntu@192.222.52.237
DEST=${1:?dest}
mkdir -p "$DEST"
rsync -a -e "ssh -o BatchMode=yes" --include='d_*/' --include='records.jsonl' --include='units.npz' --include='units_index.json' --include='meta.json' --include='job_result.json' --exclude='*' "$VM:work/runs/" "$DEST/"
ls "$DEST" | grep "^d_"
