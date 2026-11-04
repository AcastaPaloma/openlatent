#!/bin/bash
# usage: sync_results.sh <dest-dir> [name-glob]   -- copy finished/partial A results (jsonl, preds, split arrays; no embeddings) from the VM
dest=${1:?dest}; glob=${2:-a_*}
mkdir -p "$dest"
ssh -o BatchMode=yes ubuntu@192.222.52.237 "cd ~/work/runs && ls -d $glob" | grep -v -e extract -e smoke | while read d; do
  rsync -a -e "ssh -o BatchMode=yes" --exclude train.log ubuntu@192.222.52.237:~/work/runs/$d/ "$dest/$d/"
done
ls "$dest" | wc -l
