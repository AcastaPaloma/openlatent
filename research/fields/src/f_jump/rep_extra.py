"""POST-HOC descriptive add-on to the controlled replication (F.md section 19.2): the adapter vs the CE-MLP floor WITHOUT the discrete leakage-tolerance
matching (paired donor bootstrap of dU and dL at r = 0 and r = 0.5) and a linearly interpolated matched-leakage dU along the CE dither curve.
Not pre-registered; reported beside, never instead of, the rule-based verdict.

  python rep_extra.py --runs <dirs...> --B 1000 --json out.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if p and Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "e_singlecell"))

import numpy as np  # noqa: E402

import aggregate as AG  # noqa: E402  (E)

spec = importlib.util.spec_from_file_location("agg_rep", HERE / "aggregate_hlca_rep.py")
AR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--budgets", default="10,100")
    ap.add_argument("--tags", default="ce")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    D = AG.Data(a.dataset, a.runs)
    folds = list(D.fall)
    tags = a.tags.split(",")
    ce_of = lambda b, t: [f"{t}_f{b}_r0", f"{t}_f{b}_r0.5"]  # noqa: E731
    arms = ["raw50"] + [x for b in budgets for x in AR.AD_of(b) + [AR.X_of(b)] + [y for t in tags for y in ce_of(b, t)]]
    F = AR.Fast(D, arms, folds)
    ones = np.ones(D.n)
    out = {}
    Ws = AG.resamples(D, a.B)

    def stats(w):
        U, L, _ = F.UL([x for x in arms if F.has(x)], w)
        return U, L
    U0, L0 = stats(ones)
    boots = [stats(w) for w in Ws]
    for b in budgets:
        ad0, ad5 = AR.AD_of(b)
        row = {}
        for t in tags:
            ce0, ce5 = ce_of(b, t)

            def interp(U, L):                                 # CE utility at the adapter's r = 0 leakage, linear in (L, U) between the r0.5 and r0 CE points
                l0, l5 = L[ce0], L[ce5]
                if not (np.isfinite(l0) and np.isfinite(l5)) or abs(l0 - l5) < 1e-6:
                    return np.nan
                tt = (L[ad0] - l5) / (l0 - l5)
                return U[ce5] + tt * (U[ce0] - U[ce5])
            for tag, ad, ce in (("r0", ad0, ce0), ("r0.5", ad5, ce5)):
                dU = [u[ad] - u[ce] for u, l in boots]
                dL = [l[ad] - l[ce] for u, l in boots]
                row[f"{t}_dU_{tag}"] = dict(pt=U0[ad] - U0[ce], ci=AG.ci(dU))
                row[f"{t}_dL_{tag}"] = dict(pt=L0[ad] - L0[ce], ci=AG.ci(dL))
                row[f"{t}_U_{tag}"] = dict(adapter=U0[ad], ce=U0[ce])
                row[f"{t}_L_{tag}"] = dict(adapter=L0[ad], ce=L0[ce])
            di = [u[ad0] - interp(u, l) for u, l in boots]
            row[f"{t}_dU_interp_matched_L"] = dict(pt=U0[ad0] - interp(U0, L0), ci=AG.ci(di), ce_interp_U=interp(U0, L0))
        row["ldaX"] = dict(U=U0[AR.X_of(b)], L=L0[AR.X_of(b)])
        out[b] = row
        print(b, json.dumps(row, default=float), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
