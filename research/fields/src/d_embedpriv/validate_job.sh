#!/bin/bash
# validate job YAML(s) with the VM's PyYAML before queueing:  scripts/validate_job.sh specs/d_*.yaml
VM=ubuntu@192.222.52.237
for f in "$@"; do
  scp -q -o BatchMode=yes "$f" $VM:/tmp/_vj.yaml && ssh -o BatchMode=yes $VM "python3 -c \"
import yaml,sys
c=yaml.safe_load(open('/tmp/_vj.yaml'))
assert isinstance(c,dict) and isinstance(c['name'],str) and isinstance(c['cmd'],str) and isinstance(c.get('priority',50),int), c
assert c['name']=='$(basename $f .yaml)', ('name!=file', c['name'])
print('ok', c['name'], c['priority'], len(c['cmd']))
\"" || echo "BAD $f"
done
