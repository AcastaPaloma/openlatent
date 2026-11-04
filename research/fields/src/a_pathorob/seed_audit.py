#!/usr/bin/env python
"""Power audit (H2-016 step 1) and per-seed replication of the post-hoc AE picks (H2-001) on the saved predictions.

  python seed_audit.py --runs DIR [DIR ...] --out audit.md [--B 1000]

For every (encoder, dataset, seed) cell of the held-out-class protocol and every ae configuration:
  Delta   = held-class accuracy of the ae config minus the BEST inductive floor (by point estimate on C, so optimistic for the floor) with
            L_worst <= L(config) + 0.05; floors = raw, centre_whiten, leace, lda, lda_pca, pca, randproj and the dithered codes.
  SE      = bootstrap SD of the paired difference over the sealed test units;  MDE80 = 2.8 x SE (80% power, two-sided 5%).
  P3      = number of floor points within +-0.05 of the config's L (< 2 means "no comparator": the point cannot be called a win).
Note the sealed test units C are IDENTICAL across seeds in this protocol (RandomState(1234)); seeds replicate training noise only.
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_p2 import Cell, load, family, ci, fmt, KC, KX   # noqa

FLOORS = ("raw", "centre_whiten", "leace", "lda", "lda_pca", "pca", "randproj", "dither")
PICKS = ("ae__beta0.001_dl32_lam10.0", "ae__beta0.001_dl8_lam0.0")


def build(cell, floor, raw_c, raw_all):
    tab = {}
    for cfg in sorted(cell.by):
        if cfg.startswith("ctrl_"):
            continue
        wc = cell.centre(cfg, KC)
        Lb = (wc[1] - floor) / np.maximum(raw_c[1] - floor, 1e-6)
        Lp = (wc[0] - floor) / max(raw_c[0] - floor, 1e-6)
        t = dict(L=(Lp, Lb), held=cell.label(cfg, "held", "linear", "C"), ret=cell.ret(cfg, "knn_xc"))
        t["V_held"] = cell.vmean(cfg, lambda m: m.get("labelheld/linear/V", float("nan")))
        t["V_L"] = cell.vmean(cfg, lambda m: max(m[f"centre/{k}/V"] for k in KC))
        tab[cfg] = t
    return tab


def envelope(tab, c, key="held", tol=0.05):
    refs = [x for x in tab if family(x) in FLOORS and tab[x]["L"][0] <= tab[c]["L"][0] + tol]
    near = [x for x in tab if family(x) in FLOORS and abs(tab[x]["L"][0] - tab[c]["L"][0]) <= tol]
    if not refs:
        return None, len(near)
    return max(refs, key=lambda x: tab[x][key][0]), len(near)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    recs = load(a.runs)
    cells = sorted({(r["encoder"], r["dataset"], r.get("variant", "crossed")) for r in recs if "holdout" in r.get("variant", "")})
    out = []
    audit_rows = []
    for enc, ds, var in cells:
        seeds = sorted({r["seed"] for r in recs if (r["encoder"], r["dataset"], r.get("variant")) == (enc, ds, var)})
        per_seed = {}
        for s in seeds:
            sub = [r for r in recs if r["seed"] == s]
            cell = Cell(enc, ds, var, sub, a.runs, a.B)
            if "raw" not in cell.by or not any(c.startswith("ae__") for c in cell.by):
                continue
            floor = cell.floor()
            raw_c = cell.centre("raw", KC)
            tab = build(cell, floor, raw_c, None)
            per_seed[s] = (cell, tab)
        if not per_seed:
            continue
        out.append(f"\n### {enc} / {ds} / {var}  (seeds {sorted(per_seed)}; sealed C identical across seeds)\n")
        out.append("Pre-registered picks (the two seed-0 post hoc AE configurations) vs the best inductive floor at L <= L_AE + 0.05:\n")
        out.append("| seed | config | L | held acc [95% CI] | best floor (held acc, L) | n floor pts within +-0.05 (P3) | Δ held [95% CI] | SE | MDE80 | verdict |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        pick_wins = {p: 0 for p in PICKS}
        for s, (cell, tab) in per_seed.items():
            for p in PICKS:
                if p not in tab:
                    continue
                b, npt = envelope(tab, p)
                if b is None:
                    out.append(f"| {s} | {p} | {tab[p]['L'][0]:.2f} | {fmt(*tab[p]['held'][:2])} | (no floor at this leakage) | {npt} | n/a | | | no comparator |")
                    continue
                d = tab[p]["held"][1] - tab[b]["held"][1]
                dp = tab[p]["held"][0] - tab[b]["held"][0]
                se = float(np.nanstd(d))
                win = (npt >= 2) and (ci(d)[0] > 0) and dp >= 0.04
                pick_wins[p] += int(win)
                audit_rows.append((enc, ds, s, p, dp, se))
                out.append(f"| {s} | {p} | {tab[p]['L'][0]:.2f} | {fmt(*tab[p]['held'][:2])} | {b} ({tab[b]['held'][0]:.3f}, {tab[b]['L'][0]:.2f}) | {npt} | {dp:+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}] | {se:.3f} | {2.8*se:.3f} | {'WIN (>= +0.04, CI > 0, P3 ok)' if win else 'no'} |")
        out.append("\nSeeds on which each pre-registered pick wins (Δ >= +0.04, lower CI > 0, >= 2 floor points near its L): " + "; ".join(f"{p}: {n} of {len(per_seed)}" for p, n in pick_wins.items()) + "\n")
        # V-selected AE per seed (within-seed V) and pooled V
        out.append("AE configuration selected on validation (V held-class accuracy among ae configs whose V leakage is within +0.05 of the best inductive floor's V leakage... simplified: max V held-class accuracy among ae configs with V_L <= median V_L of the ae grid), per seed and pooled over seeds:\n")
        out.append("| selection | config | L | held acc | best floor | Δ held [95% CI] | SE | MDE80 |")
        out.append("|---|---|---|---|---|---|---|---|")
        sel = []
        for s, (cell, tab) in per_seed.items():
            aes = [c for c in tab if c.startswith("ae__")]
            medL = np.median([tab[c]["V_L"] for c in aes])
            cand = [c for c in aes if tab[c]["V_L"] <= medL]
            c = max(cand, key=lambda x: tab[x]["V_held"])
            sel.append((s, c))
            b, npt = envelope(tab, c)
            if b is None:
                out.append(f"| seed {s} V-selected | {c} | {tab[c]['L'][0]:.2f} | {tab[c]['held'][0]:.3f} | (none) | n/a | | |")
                continue
            d = tab[c]["held"][1] - tab[b]["held"][1]
            se = float(np.nanstd(d))
            out.append(f"| seed {s} V-selected | {c} | {tab[c]['L'][0]:.2f} | {tab[c]['held'][0]:.3f} | {b} ({tab[b]['held'][0]:.3f}) | {tab[c]['held'][0]-tab[b]['held'][0]:+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}] | {se:.3f} | {2.8*se:.3f} |")
        same = {}
        for s, c in sel:
            same[c] = same.get(c, 0) + 1
        out.append("\nPer-seed V-selected ae configuration: " + ", ".join(f"seed {s}: {c}" for s, c in sel) + f". Most frequent config chosen on {max(same.values())} of {len(sel)} seeds.\n")
        # MDE audit over all ae configs
        ses, Lhw = [], []
        for s, (cell, tab) in per_seed.items():
            for c in [c for c in tab if c.startswith("ae__")]:
                b, npt = envelope(tab, c)
                if b is None:
                    continue
                ses.append(float(np.nanstd(tab[c]["held"][1] - tab[b]["held"][1])))
                lo, hi = ci(tab[c]["L"][1]); Lhw.append((hi - lo) / 2)
        out.append(f"Power audit, all ae configs x seeds with a floor comparator (n={len(ses)}): SE of the paired Δ held: min {min(ses):.3f}, median {np.median(ses):.3f}, max {max(ses):.3f}; "
                   f"**MDE80 = 2.8 x SE: min {2.8*min(ses):.3f}, median {2.8*np.median(ses):.3f}, max {2.8*max(ses):.3f}**; median half-width of the 95% CI of L_worst: {np.median(Lhw):.3f} (the leakage-matching tolerance is 0.05).\n")
    text = "\n".join(out)
    Path(a.out).write_text(text) if a.out != "-" else print(text)


if __name__ == "__main__":
    main()
