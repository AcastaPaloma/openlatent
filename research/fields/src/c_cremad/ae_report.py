"""Markdown table for the pre-registered AE round: every grid config, 3 seeds, vs the width-matched LDA-4 comparator.
  python ae_report.py RUNS_ROOT"""
import os, sys
import numpy as np
import aggregate as ag

root = os.path.expanduser(sys.argv[1])
load = lambda n: [r for d in sorted(os.listdir(root)) if d == n or d.startswith(n + "_s") for r in ag.load(os.path.join(root, d))]


def uhe(r):
    k = max(ag.kinds_present(r, "emo"), key=lambda k: ag.emo_subset_curve(r, k, [1, 2])[1])
    return ag.emo_subset_curve(r, k, [1, 2]), k


def stat(runs):
    he = ag.stat_over_seeds(runs, lambda r: uhe(r)[0])
    idn = ag.stat_over_seeds(runs, lambda r: ag.worst_id(r, "T1", "unseen")[:2])
    return he, idn


def fmt(b, p):
    lo, hi = ag.ci(b)
    return "%.3f [%.3f,%.3f]" % (p, lo, hi)


def row(name, comp_name):
    runs, comp = load(name), load(comp_name)
    if not runs or not comp:
        return None
    X, L = stat(runs), stat(comp)
    dh, di = X[0][0] - L[0][0], X[1][0] - L[1][0]
    ps = [uhe(r)[0][1] - L_ for r, L_ in zip(runs, [stat(comp)[0][1]] * len(runs))]
    ps_id = [ag.worst_id(r, "T1", "unseen")[1] for r in runs]
    ok1 = [d >= 0.05 for d in ps]
    return ("| %s | %d | %s | %s | %.1fx | %s | %s | %s | %d/%d |" % (
        name, len(runs), fmt(X[0][0], X[0][1]), fmt(X[1][0], X[1][1]), X[1][1] * 91,
        comp_name, fmt(dh, X[0][1] - L[0][1]), fmt(di, X[1][1] - L[1][1]), sum(ok1), len(runs)))


print("| AE config | seeds | U_HE (recall DIS+FEA) [95% CI] | identity T1 [95% CI] | x chance | comparator | dU_HE vs comparator [95% CI] | dID vs comparator [95% CI] | seeds with dU_HE >= +0.05 |")
print("|---|---:|---|---|---:|---|---|---|---|")
for n, c in (("c_ae_d16_l0", "c_lda4_k16"), ("c_ae_d16_l3", "c_lda4_k16"), ("c_ae_d16_l10", "c_lda4_k16"),
             ("c_ae_d32_l0", "c_lda4_k32"), ("c_ae_d32_l3", "c_lda4_k32"), ("c_ae_d32_l10", "c_lda4_k32"),
             ("c_aetrim_d32_l3", "c_lda4trim_k32")):
    r = row(n, c)
    if r:
        print(r)
print()
print("Per-seed point estimates of the selected configuration (U_HE, identity T1 worst attacker):")
for n in ("c_ae_d32_l3", "c_aetrim_d32_l3"):
    for d in sorted(os.listdir(root)):
        if d.startswith(n + "_s"):
            for r in ag.load(os.path.join(root, d)):
                print("  %s seed %d: U_HE %.3f  ID %.3f (%.1fx)" % (n, r["seed"], uhe(r)[0][1], ag.worst_id(r, "T1", "unseen")[1], ag.worst_id(r, "T1", "unseen")[1] * 91))
