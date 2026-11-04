#!/bin/bash
# queue_job.sh jobs/specs/d_x.yaml ... : validate with VM PyYAML, then scp into ~/queue/pending (run after commit+push+VM pull)
VM=ubuntu@192.222.52.237
for f in "$@"; do
  n=$(basename $f .yaml)
  scp -q -o BatchMode=yes "$f" $VM:/tmp/_vj_$n.yaml || { echo "scp fail $f"; continue; }
  ssh -o BatchMode=yes $VM "python3 -c \"
import yaml
c=yaml.safe_load(open('/tmp/_vj_$n.yaml'))
assert isinstance(c,dict) and c['name']=='$n' and isinstance(c['cmd'],str) and isinstance(c.get('priority',50),int), c
\" && cp /tmp/_vj_$n.yaml ~/queue/pending/$n.yaml.tmp && mv ~/queue/pending/$n.yaml.tmp ~/queue/pending/$n.yaml && echo queued $n" || echo "BAD $f"
done
