#!/bin/bash
cd ~/ct
cat jobs_w.txt | xargs -P 9 -L1 bash -c 'g=$((RANDOM%8)); SAVEW=1 SEAL2=1 SEALMAN=manifest_valid.csv TAG=_w EXTRA=99999 T=12 ./runone.sh $g $0 $1 $2 $3'
echo done > logs/queue_w.done
