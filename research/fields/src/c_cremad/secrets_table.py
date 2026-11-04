"""Secondary-secret table (sex BA, race BA, age MAE + leakage vs raw) for a list of run dirs: python secrets_table.py RAW DIR..."""
import sys, os
import aggregate as ag

raw = ag.load(sys.argv[1])
for d in sys.argv[2:]:
    runs = ag.load(d)
    if not runs:
        continue
    S = ag.summarize(runs, raw)
    sec = S.get("secrets")
    if not sec:
        continue
    parts = []
    for k in ("sex", "race"):
        e = sec[k]
        L = e.get("leakage", {}).get("point", float("nan"))
        parts.append("%s BA %.3f [%.3f,%.3f] L=%.2f" % (k, e["point"], e["ci"][0], e["ci"][1], L))
    a = sec["age"]
    parts.append("age MAE %.1f (const %.1f) L=%.2f" % (a["mae"], a["const_mean_mae"], a.get("leakage", {}).get("point", float("nan"))))
    print(S["name"], " | ".join(parts))
