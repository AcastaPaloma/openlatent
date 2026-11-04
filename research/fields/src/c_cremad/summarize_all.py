"""Summarise every Focus-C run directory under a runs root into one table + JSON (bootstrap over actors, paired resamples).

  python summarize_all.py --runs ~/work/runs --out_json c_summary.json --out_md c_summary.md
Runs named <base>_s<k> (one job per seed) are merged into one row <base> with all their seeds.  Trimmed runs (c_trim_*, *_trim)
are normalised against c_ref_raw_trim, everything else against c_ref_raw.
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
import aggregate as ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/work/runs"))
    ap.add_argument("--out_json", default="c_summary.json")
    ap.add_argument("--out_md", default="c_summary.md")
    a = ap.parse_args()
    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.runs, "c_*"))):
        base = re.sub(r"_s\d+$", "", os.path.basename(d))
        if base.startswith("c_feats"):
            continue
        groups[base].append(d)
    raw = ag.load(os.path.join(a.runs, "c_ref_raw")) if os.path.isdir(os.path.join(a.runs, "c_ref_raw")) else None
    raw_trim = ag.load(os.path.join(a.runs, "c_ref_raw_trim")) if os.path.isdir(os.path.join(a.runs, "c_ref_raw_trim")) else None
    sums = []
    for base, dirs in groups.items():
        runs = [r for d in dirs for r in ag.load(d)]
        if not runs or "id" not in runs[0] and "emo" not in runs[0]:
            continue
        if "layers" in runs[0]:
            continue
        runs = [dict(r, name=base) for r in runs]
        if not runs[0].get("id"):
            continue
        ref = raw_trim if ("trim" in base and raw_trim) else raw
        try:
            sums.append(ag.summarize(runs, ref))
        except Exception as e:  # keep going, report
            print("skip", base, repr(e))
    json.dump(sums, open(a.out_json, "w"), indent=1)
    order = lambda s: (s["name"].split("_")[1], s["cfg"].get("dz", 0) or 0, s["name"])
    md = ag.HEAD + "\n" + "\n".join(ag.row(s) for s in sorted(sums, key=order))
    open(a.out_md, "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
