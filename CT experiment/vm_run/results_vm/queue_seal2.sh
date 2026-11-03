#!/bin/bash
cd ~/ct
while [ $(grep -l DONE logs/ext_seal2_*.log 2>/dev/null | wc -l) -lt 8 ]; do sleep 20; done
echo "extraction complete $(date)" > logs/seal2_started.txt; ls feats | wc -l >> logs/seal2_started.txt
cat jobs_seal2.txt | xargs -P 10 -L1 bash -c 'g=$((RANDOM%8)); SEAL2=1 EXTRA=99999 T=12 ./runone.sh $g $0 $1 $2 $3'
echo done > logs/queue_seal2.done
