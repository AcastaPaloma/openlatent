#!/usr/bin/env python3
"""Fair-share the pending queue across focuses (run on the VM by the main agent every tick).

Builders set priorities inside their own focus (a_ / b_ / c_ prefix); across focuses nobody gets to jump the queue.
Within a focus the existing priority order is kept (ties by name).  Across focuses the jobs are interleaved
round-robin, so each focus with pending work gets about one job in every len(focuses) claims.  New priorities are 10, 11, 12, ...
(priority 0-4 is reserved for the main agent: smoke tests, hand-promoted jobs).  Idempotent.
"""
import re, sys
from collections import defaultdict
from pathlib import Path

import yaml

pend = Path.home() / "queue" / "pending"
groups = defaultdict(list)
for p in sorted(pend.glob("*.yaml")):
    try:
        cfg = yaml.safe_load(p.read_text())
        assert isinstance(cfg, dict) and "cmd" in cfg
    except Exception:
        continue  # the worker quarantines these
    pr = cfg.get("priority", 50)
    if pr < 5:  # 0-4 reserved for the main agent
        continue
    groups[p.name.split("_")[0]].append((pr, p.name, p))
for g in groups:
    groups[g].sort()
order = sorted(groups)
n = 0
rank = 10
while any(groups[g] for g in order):
    for g in order:
        if groups[g]:
            _, _, p = groups[g].pop(0)
            txt = p.read_text()
            new = re.sub(r"(?m)^priority:.*$", f"priority: {rank}", txt) if re.search(r"(?m)^priority:", txt) else f"priority: {rank}\n" + txt
            if new != txt:
                p.write_text(new)
                n += 1
            rank += 1
print(f"normalised {n} specs across {order}; {rank - 10} jobs ranked")
