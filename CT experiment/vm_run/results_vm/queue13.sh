#!/bin/bash
cd ~/ct
cat jobs13.txt | xargs -P 10 -L1 bash -c 'g=$((RANDOM%8)); EXTRA=99999 T=12 ./runone.sh $g $0 $1 $2 $3'
echo QUEUE13_DONE > logs/queue13.done
