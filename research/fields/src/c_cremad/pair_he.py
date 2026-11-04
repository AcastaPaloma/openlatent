"""Held-out-emotion paired comparisons: learned encoder vs an LDA comparator (pre-registered in results_notes/C.md section 2b).
  python pair_he.py RUNS_ROOT LDA_NAME LEARNED_NAME [LEARNED_NAME ...]
Prints U_HE (mean recall of DIS+FEA, best probe), 4-class recall, identity T1 for each and the paired differences (learned - LDA)."""
import os, sys
import numpy as np
import aggregate as ag

root = os.path.expanduser(sys.argv[1])
lda = ag.load(os.path.join(root, sys.argv[2]))


def stat(runs):
    best = max(ag.kinds_present(runs[0], "emo"), key=lambda k: np.mean([ag.emo_subset_curve(r, k, [1, 2])[1] for r in runs]))
    he = ag.stat_over_seeds(runs, lambda r: ag.emo_subset_curve(r, best, [1, 2]))
    tr = ag.stat_over_seeds(runs, lambda r: ag.emo_subset_curve(r, best, [0, 3, 4, 5]))
    idn = ag.stat_over_seeds(runs, lambda r: ag.worst_id(r, "T1", "unseen")[:2])
    return he, tr, idn


def fmt(b, p):
    lo, hi = ag.ci(b)
    return "%.3f [%.3f,%.3f]" % (p, lo, hi)


L = stat(lda)
print("%-24s U_HE %s | 4-class %s | ID %s (%.1fx)" % (sys.argv[2], fmt(L[0][0], L[0][1]), fmt(L[1][0], L[1][1]), fmt(L[2][0], L[2][1]), L[2][1] * 91))
for n in sys.argv[3:]:
    runs = [r for d in sorted(os.listdir(root)) if d == n or d.startswith(n + "_s") for r in ag.load(os.path.join(root, d))]
    if not runs:
        continue
    X = stat(runs)
    dh, dt, di = X[0][0] - L[0][0], X[1][0] - L[1][0], X[2][0] - L[2][0]
    ok1 = (X[0][1] - L[0][1] >= 0.05) and ag.ci(dh)[0] > 0
    ok2 = ag.ci(di)[1] <= 0.02
    ok3 = (X[1][1] - L[1][1]) >= -0.03
    print("%-24s U_HE %s | 4-class %s | ID %s (%.1fx) | dU_HE %s dID %s d4cls %.3f | criteria 1/2/3: %s/%s/%s" % (
        n, fmt(X[0][0], X[0][1]), fmt(X[1][0], X[1][1]), fmt(X[2][0], X[2][1]), X[2][1] * 91, fmt(dh, X[0][1] - L[0][1]), fmt(di, X[2][1] - L[2][1]),
        X[1][1] - L[1][1], ok1, ok2, ok3))
