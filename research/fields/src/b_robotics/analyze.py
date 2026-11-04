"""Aggregate B' results: python analyze.py DIR [DIR...] > table.md   (reads res_<kind>_<arm>_s<seed>.json)

Kill rule (fixed in advance, see results_notes/B.md): for an action-correlated kind, the learned arm survives only if,
over all seeds, utility NMSE(sam) - NMSE(learned) >= MARGIN (paired by seed, all seeds positive) AND
leak_worst(learned) <= leak_worst(sam) + LEAK_TOL AND the learned arm also beats the rate-bottleneck controls.
"""
import sys, glob, json, os, re
import numpy as np

MARGIN, LEAK_TOL = 0.05, 0.02
ORDER = ["clean", "raw", "rate4", "rateb", "sam", "gtmask", "adv", "adv30", "lmask", "advmask", "sup", "supadv", "smask"]


def load(dirs):
    R = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "res_*.json"), recursive=True):
            r = json.load(open(f))
            R.setdefault((re.match(r"res_(.+?)_" + r["arm"] + r"_s", os.path.basename(f)).group(1), r["arm"]), {})[r["seed"]] = r
    return R


def ms(xs):
    xs = np.array(xs, float)
    return f"{xs.mean():.3f}" + (f" ±{xs.std(ddof=1):.3f}" if len(xs) > 1 else "") + f" (n={len(xs)})"


def main(dirs):
    R = load(dirs)
    kinds = sorted({k for k, _ in R})
    for kind in kinds:
        print(f"\n### train kind: {kind}\n")
        print("| arm | seeds | utility NMSE xyz (2k labels) [episode-bootstrap CI, mean of seeds] | utility, all labels | utility MLP | leak worst | leak resid-vel (lin/mlp) | leak rel-pos (lin/mlp) | leak colour (lin/mlp) | util on test kind: none / indep / corr_det / corr_noisy / corr_glued |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for arm in ORDER:
            rs = R.get((kind, arm))
            if arm == "clean" and kind != "none" and ("none", "clean") in R:   # clean-trained LAM evaluated on this kind's test set (OOD for it)
                S0 = list(R[("none", "clean")].values())
                u = [r["util_by_test_kind"].get(kind) for r in S0]
                print(f"| clean (trained on none; test on {kind}) | {len(S0)} | {ms([x for x in u if x is not None])} | | | | | | | |")
                S1 = [r["util_lin2k_xyz"] for r in S0]
                print(f"| clean (ID: test on none) | {len(S0)} | {ms(S1)} | {ms([r['util_linfull_xyz'] for r in S0])} | {ms([r['util_mlp_xyz'] for r in S0])} | 0 | | | | |")
                continue
            if not rs:
                continue
            S = list(rs.values())
            g = lambda k: [r[k] for r in S if k in r]
            ci = np.mean([r["util_lin2k_xyz_ci"] for r in S], 0)
            lk = lambda n: (" / ".join(f"{np.mean(g(f'leak_{n}_{a}')):.2f}" for a in ("lin", "mlp")) if g(f"leak_{n}_lin") else "n/a")
            bk = {k: [r["util_by_test_kind"].get(k) for r in S] for k in ("none", "indep", "corr_det", "corr_noisy", "corr_glued")}
            bks = " / ".join((f"{np.mean([x for x in v if x is not None]):.2f}" if any(x is not None for x in v) else "-") for v in bk.values())
            print(f"| {arm} | {len(S)} | {ms(g('util_lin2k_xyz'))} [{ci[0]:.3f},{ci[1]:.3f}] | {ms(g('util_linfull_xyz'))} | {ms(g('util_mlp_xyz'))} | {ms(g('leak_worst'))} | {lk('resid_vel')} | {lk('position')} | "
                  f"{(np.mean(g('leak_appearance_lin')) if g('leak_appearance_lin') else float('nan')):.2f} / {(np.mean(g('leak_appearance_mlp')) if g('leak_appearance_mlp') else float('nan')):.2f} | {bks} |")
        # decidability: how much room is there above SAM at all?
        sam_, gt_, raw_ = R.get((kind, "sam"), {}), R.get((kind, "gtmask"), {}), R.get((kind, "raw"), {})
        cl_ = R.get(("none", "clean"), {})
        sd = sorted(set(sam_) & set(gt_))
        if sd:
            print(f"\nheadroom vs SAM on {kind}: exact-silhouette mask gain over SAM = {np.mean([sam_[s]['util_lin2k_xyz'] - gt_[s]['util_lin2k_xyz'] for s in sd]):.3f} "
                  f"(per seed {[round(sam_[s]['util_lin2k_xyz'] - gt_[s]['util_lin2k_xyz'], 3) for s in sd]}); clean-frame oracle gain over SAM = "
                  f"{np.mean([sam_[s]['util_lin2k_xyz'] - cl_[s]['util_lin2k_xyz'] for s in sd if s in cl_]):.3f}; raw-minus-SAM = {np.mean([raw_[s]['util_lin2k_xyz'] - sam_[s]['util_lin2k_xyz'] for s in sd if s in raw_]):.3f}")
        # kill rule
        if kind in ("corr_det", "corr_noisy", "corr_glued", "indep"):
            sam, gt = R.get((kind, "sam"), {}), R.get((kind, "gtmask"), {})
            for cand in ("adv", "adv30", "lmask", "advmask", "sup", "supadv", "smask"):
                c = R.get((kind, cand), {})
                seeds = sorted(set(c) & set(sam))
                if not seeds:
                    continue
                du = [sam[s]["util_lin2k_xyz"] - c[s]["util_lin2k_xyz"] for s in seeds]
                dl = [c[s]["leak_worst"] - sam[s]["leak_worst"] for s in seeds]
                ctrl = [min(R[(kind, a)][s]["util_lin2k_xyz"] for a in ("rate4", "rateb") if (kind, a) in R and s in R[(kind, a)]) - c[s]["util_lin2k_xyz"] for s in seeds
                        if any((kind, a) in R and s in R[(kind, a)] for a in ("rate4", "rateb"))]
                ok = all(x >= MARGIN for x in du) and all(x <= LEAK_TOL for x in dl) and (not ctrl or all(x > 0 for x in ctrl))
                print(f"\nkill rule `{cand}` vs SAM on {kind}: utility gain per seed {[round(x,3) for x in du]} (need >= {MARGIN}), leak diff {[round(x,3) for x in dl]} (need <= {LEAK_TOL}), "
                      f"gain over best rate control {[round(x,3) for x in ctrl]} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:])
