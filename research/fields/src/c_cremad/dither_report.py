"""Post-hoc H2-014-style control: dithered label-free codes vs the validation-selected AE (U_HE, identity, paired)."""
import os, sys
import aggregate as ag

R = os.path.expanduser(sys.argv[1])
load = lambda n: [r for d in sorted(os.listdir(R)) if d == n or d.startswith(n + "_s") for r in ag.load(os.path.join(R, d))]


def uhe(r):
    k = max(ag.kinds_present(r, "emo"), key=lambda k: ag.emo_subset_curve(r, k, [1, 2])[1])
    return ag.emo_subset_curve(r, k, [1, 2])


def st(runs):
    return ag.stat_over_seeds(runs, uhe), ag.stat_over_seeds(runs, lambda r: ag.worst_id(r, "T1", "unseen")[:2])


def f(b, p):
    lo, hi = ag.ci(b)
    return "%.3f [%.3f,%.3f]" % (p, lo, hi)


sel = sys.argv[2] if len(sys.argv) > 2 else "c_ae_d32_l3"
ae = st(load(sel))
print("%s: U_HE %s | ID %s (%.1fx)" % (sel, f(ae[0][0], ae[0][1]), f(ae[1][0], ae[1][1]), ae[1][1] * 91))
for n in sys.argv[3:]:
    runs = load(n)
    if not runs:
        continue
    X = st(runs)
    print("%-22s seeds %d U_HE %s | ID %s (%.1fx) | vs selected AE: dU_HE %s dID %s" % (
        n, len(runs), f(X[0][0], X[0][1]), f(X[1][0], X[1][1]), X[1][1] * 91, f(X[0][0] - ae[0][0], X[0][1] - ae[0][1]), f(X[1][0] - ae[1][0], X[1][1] - ae[1][1])))
