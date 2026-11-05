"""H3-002 Part A aggregator (results_notes/F.md section 15): dU against the SAME-label LDA as a function of the number of label classes K.

  python aggregate_k.py --runs d0=dir[,d1=dir,d2=dir] --stored $HOME/work/f3_regen/merge/merged3_s0 --out $SCR/k [--B 1000]
`--stored` supplies the raw-profile attacker predictions (normalisation of L) of the SAME V/H wells from the S5 run; nothing is written next to the inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregate as AG

DELTA_U = 0.010
MATCH = 0.05
KS = ["K30", "K100", "K300", "K1000", "K3000", "K10000", "K32000"]


def plate_ci(P, a, b, key, rng, B):
    ax, ay = P[f"apq|H|{a}{key}"].astype(np.float64), P[f"apq|H|{b}{key}"].astype(np.float64)
    ok = ~np.isnan(ax) & ~np.isnan(ay)
    ids, inv = np.unique(P["plate|H"], return_inverse=True)
    ds = np.bincount(inv[ok], weights=(ax - ay)[ok], minlength=len(ids))
    n = np.bincount(inv[ok], minlength=len(ids)).astype(np.float64)
    bs = [(w @ ds) / (w @ n) for w in (np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(B))]
    return AG.ci(bs)


def wls(x, y, se):
    w = 1.0 / np.maximum(se, 1e-6) ** 2
    X = np.stack([np.ones_like(x), x], 1)
    A = X.T @ (X * w[:, None])
    beta = np.linalg.solve(A, X.T @ (w * y))
    fit = X @ beta
    ybar = np.sum(w * y) / np.sum(w)
    r2 = 1 - np.sum(w * (y - fit) ** 2) / np.sum(w * (y - ybar) ** 2)
    return beta, float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    runs = dict(x.split("=") for x in a.runs.split(","))
    zs = np.load(Path(a.stored) / "preds_s0.npz")
    res = dict(quantity="dU against the SAME-label LDA (ldaM_K_8); NOT the break-even against the 100%-label envelope of F4",
               verdict_original="NO-GO at the S4 gate (JK1 v0 literal); stage 1 partial survival; F4 PARTIAL (F.md 5, 12.6, 13)", draws={})
    md = [f"# H3-002 Part A aggregate (pre-registration: results_notes/F.md section 15); dU vs the SAME-label LDA, not the F4 break-even; B={a.B}\n"]
    cells = []
    for di, (dn, dp) in enumerate(runs.items()):
        d = Path(dp)
        P = {k: v for k, v in np.load(d / "preds_s0.npz").items()}
        meta = json.loads((d / "meta_s0.json").read_text())
        for k in ("idx|H", "idx|V"):
            assert np.array_equal(P[k], zs[k]), k
        for sname in ("V", "H"):                                   # inject raw attacker predictions (same wells) for the normalisation of L
            for kind in meta["kinds"]:
                P[f"att|{sname}|raw|{kind}"] = zs[f"att|{sname}|raw|{kind}"]
        meta["arms"] = meta["arms"] + ["raw"]
        meta["families"]["raw"] = "floor"
        meta["dims"]["raw"] = 1040
        tmp = Path(a.out + f"_tmp_{dn}")
        tmp.mkdir(parents=True, exist_ok=True)
        np.savez(tmp / "preds_s0.npz", **P)
        (tmp / "meta_s0.json").write_text(json.dumps(meta))
        sp = AG.Split(tmp, "s0", a.B, 900 + di, True, "ldaT2")
        rng = np.random.default_rng(910 + di)
        counts = {}
        for p in ("", "|Pabc"):
            nC = len(P["cid|H|xs" if p == "" else "cid|H|Pabc"])
            counts[p] = rng.multinomial(nC, np.ones(nC) / nC, size=a.B).astype(np.float32)
        Uc = lambda x, p: P[f"apc|H|{x}|xs" if p == "" else f"apc|H|{x}{p}"].astype(np.float64)  # noqa: E731
        Uv = lambda x, p: float(P[f"apc|V|{x}|xs" if p == "" else f"apc|V|{x}{p}"].mean())  # noqa: E731
        regs = [r for r in meta["k_info"]]
        out_d = {}
        md.append(f"## draw {dn}: wells per class {json.dumps({r: round(v['wells_per_class'], 2) for r, v in meta['k_info'].items()})}\n")
        rows = ["| regime | K | trained | a* | U_H(a*) [L_H] | U_H(ldaM_8) [L_H] | dU_same P0 [95% CI] (plate CI) | dU_same Pabc [CI] | fixed-arm gap P0 | U_V - U_V(perm) | flags |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in regs:
            K = meta["k_info"][r]["K"]
            info = meta["k_train"][r]
            c1 = info["loss_last200"] <= 0.8 * info["loss_first20"]
            c2 = info["eff_rank"] >= 2
            pn = f"perm_{r}_d64" if f"perm_{r}_d64" in meta["arms"] else (f"perm_K3000_d64" if r.startswith("K3000") else None)
            up = Uv(pn, "") if pn else float("nan")
            c3 = bool(Uv(f"mlp_{r}_d8", "") - up >= DELTA_U) if pn else True
            trained = bool(c1 and c2 and c3)
            lda = f"ldaM_{r}_8"
            arms_r = [f"mlp_{r}_d8", f"mlp_{r}_d8_r0.5"]
            cand = [x for x in arms_r if sp.L("V", x) <= sp.L("V", lda) + MATCH]
            flag = ""
            if not cand:
                cand = [min(arms_r, key=lambda x: sp.L("V", x))]
                flag = "leakage-unmatched"
            astar = max(cand, key=lambda x: Uv(x, ""))
            cell = dict(regime=r, K=K, trained=trained, astar=astar, flag=flag, wells_per_class=meta["k_info"][r]["wells_per_class"], U_V_minus_perm=Uv(f"mlp_{r}_d8", "") - up if pn else None)
            for p, nm in (("", "P0"), ("|Pabc", "Pabc")):
                dv = Uc(astar, p) - Uc(lda, p)
                bb = (counts[p] @ dv.astype(np.float32)) / counts[p].shape[1]
                cell[nm] = dict(dU=float(dv.mean()), se=float(np.std(bb)), ci=AG.ci(bb), plate_ci=plate_ci(P, astar, lda, "" if p == "" else "|Pabc", rng, min(a.B, 500)),
                                U_H_adapter=float(Uc(astar, p).mean()), U_H_lda=float(Uc(lda, p).mean()))
            cell["L_H_adapter"], cell["L_H_lda"], cell["L_V_adapter"], cell["L_V_lda"] = sp.L("H", astar), sp.L("H", lda), sp.L("V", astar), sp.L("V", lda)
            cell["fixed_arm_gap_P0"] = float(Uc(f"mlp_{r}_d8", "").mean() - Uc(lda, "").mean())
            out_d[r] = cell
            rows.append(f"| {r} | {K} | {trained} | {astar} | {AG.fmt(cell['P0']['U_H_adapter'])} [{AG.fmt(cell['L_H_adapter'], 2)}] | {AG.fmt(cell['P0']['U_H_lda'])} [{AG.fmt(cell['L_H_lda'], 2)}] | {AG.fci(cell['P0']['dU'], cell['P0']['ci'], 4)} ({AG.fmt(cell['P0']['plate_ci'][0], 4)}, {AG.fmt(cell['P0']['plate_ci'][1], 4)}) | "
                        f"{AG.fci(cell['Pabc']['dU'], cell['Pabc']['ci'], 4)} | {AG.fmt(cell['fixed_arm_gap_P0'], 4)} | {AG.fmt(cell['U_V_minus_perm'])} | {flag}{'' if trained else ' UNTRAINED'} |")
            if r in KS:
                cells.append(dict(draw=dn, **cell))
        res["draws"][dn] = out_d
        md.append("\n".join(rows) + "\n")
    # ---- K law
    good = [c for c in cells if c["trained"]]
    v = {"cells_used": len(good), "cells_total": len(cells)}
    for nm in ("P0", "Pabc"):
        x = np.log10([c["K"] for c in good])
        y = np.array([c[nm]["dU"] for c in good])
        se = np.array([c[nm]["se"] for c in good])
        beta, r2 = wls(x, y, se)
        rngb = np.random.default_rng(5)
        bs = []
        for _ in range(a.B):
            ii = rngb.integers(0, len(good), len(good))
            if len(np.unique(x[ii])) < 3:
                continue
            bs.append(wls(x[ii], y[ii], se[ii])[0][1])
        v[nm] = dict(alpha=float(beta[0]), beta=float(beta[1]), beta_ci=AG.ci(bs), r2=r2)
        mean_by_K = {}
        for c in good:
            mean_by_K.setdefault(c["K"], []).append(c[nm]["dU"])
        seq = [(K, float(np.mean(m))) for K, m in sorted(mean_by_K.items())]
        v[nm]["dU_by_K"] = seq
        v[nm]["monotone"] = bool(all(b[1] >= a_[1] - 0.005 for a_, b in zip(seq[:-1], seq[1:])))
    wl = {}
    for dn, out_d in res["draws"].items():
        if all(r in out_d for r in ("K3000", "K3000_w1", "K3000_w2")):
            wl[dn] = dict(w1=out_d["K3000_w1"]["P0"]["dU"] - out_d["K3000"]["P0"]["dU"], w2=out_d["K3000_w2"]["P0"]["dU"] - out_d["K3000"]["P0"]["dU"])
    v["wells_per_class_effect"] = wl
    wmax = max([abs(x) for d_ in wl.values() for x in d_.values()] or [0.0])
    held = bool(v["P0"]["monotone"] and 0.015 <= v["P0"]["beta"] <= 0.09 and v["P0"]["r2"] >= 0.7 and wmax < 0.015 and v["Pabc"]["beta"] >= 0.015)
    falsified = bool(v["P0"]["beta"] < 0.015 or v["P0"]["r2"] < 0.7 or wmax >= 0.015)
    v["verdict_tightened"] = ("NON-MONOTONE" if not v["P0"]["monotone"] else ("FALSIFIED" if falsified else ("HOLDS" if held else "INCONCLUSIVE (law in range but a secondary condition fails)")))
    res["law"] = v
    md.append("## K law\n\n```\n" + json.dumps(dict(verdict_original=res["verdict_original"], **v), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
