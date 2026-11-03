#!/bin/bash
cd ~/ct
cat jobs15.txt | xargs -P 9 -L1 bash -c 'g=$((RANDOM%8)); SEAL2=1 SEALMAN=manifest_valid.csv TAG=_sealv2 EXTRA=99999 T=12 ./runone.sh $g $0 $1 $2 $3'
echo done > logs/queue16.done
