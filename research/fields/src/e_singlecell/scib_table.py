"""Markdown table of the scIB-style context scores (mean over folds) from scib_ctx_<dataset>.json files.  python scib_table.py a.json b.json ..."""
import json
import sys
from pathlib import Path

import numpy as np


def main(paths):
    for p in paths:
        d = json.loads(Path(p).read_text())
        ds = Path(p).stem.replace("scib_ctx_", "")
        folds = [f for f, v in d.items() if "error" not in v]
        errs = {f: v["error"] for f, v in d.items() if "error" in v}
        arms = sorted({a for f in folds for a in d[f]}, key=lambda a: list(d[folds[0]]).index(a) if a in d[folds[0]] else 99)
        print(f"\n### scIB-style context, {ds} (scib-metrics 0.5.6, held-out cells, mean over folds {folds}; NOT comparable with published scIB totals)")
        if errs:
            print(f"folds that failed: {errs}")
        print("| arm | bio conservation | batch correction | total (0.6 bio + 0.4 batch) | iLISI | KBET | graph conn. | cLISI | isolated labels | KMeans NMI |\n|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            def m(k):
                v = [d[f][a][k] for f in folds if a in d[f] and d[f][a].get(k) is not None]
                return f"{np.mean(v):.3f}" if v else ""
            print(f"| {a} | {m('Bio conservation')} | {m('Batch correction')} | {m('Total')} | {m('iLISI')} | {m('KBET')} | {m('Graph connectivity')} | {m('cLISI')} | {m('Isolated labels')} | {m('KMeans NMI')} |")


if __name__ == "__main__":
    main(sys.argv[1:])
