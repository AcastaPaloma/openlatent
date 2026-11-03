#!/bin/bash
cd ~/ct
cat jobs14.txt | xargs -P 9 -L1 bash -c 'g=$((RANDOM%8)); TAG=_ss EXTRA=99999 T=12 ./runone.sh $g $0 $1 $2 $3'
echo done > logs/queue14.done
