"""Zero-GPU power audit for R-IMU (no model number is used). Independent unit = subject (24); trials/windows are nested in subjects.
Analytic: Var(delta_hat) = 2 p (1-p) (1-r) / n * (1 + (m-1) rho),  m = units per subject.
Monte-Carlo: Gaussian-copula generator with subject random effect (ICC rho) and cross-arm latent correlation r; the test is the
cluster-robust one-sided 95% rule  delta_hat - t_{0.95,23} * SE_cluster > 0 ; MDE80 = smallest true difference with power >= 0.8.
Usage: python power.py out.json"""
import sys, json
import numpy as np
from scipy.stats import norm, t as tdist

def analytic_mde(p, n_units, m, rho, r, power=0.8, alpha=0.05, n_clusters=24):
    se = np.sqrt(2 * p * (1 - p) * (1 - r) / n_units * (1 + (m - 1) * rho))
    return float((tdist.ppf(1 - alpha, n_clusters - 1) + norm.ppf(power)) * se), float(se)

def mc_power(p_lo, d, n_sub, m, rho, r, nsim, rng):
    """Arm1 accuracy p_lo + d, arm2 accuracy p_lo. Returns power of the one-sided cluster-robust test and mean SE."""
    thr1, thr2 = norm.ppf(min(max(p_lo + d, 1e-6), 1 - 1e-6)), norm.ppf(p_lo)
    U = rng.standard_normal((nsim, n_sub, 1)); E = rng.standard_normal((nsim, n_sub, m))
    G = np.sqrt(rho) * U + np.sqrt(1 - rho) * E
    N1 = rng.standard_normal((nsim, n_sub, m)); N2 = rng.standard_normal((nsim, n_sub, m))
    c1 = (np.sqrt(r) * G + np.sqrt(1 - r) * N1) < thr1
    c2 = (np.sqrt(r) * G + np.sqrt(1 - r) * N2) < thr2
    dsub = (c1.astype(float) - c2.astype(float)).mean(axis=2)  # per-subject mean difference
    est = dsub.mean(axis=1); se = dsub.std(axis=1, ddof=1) / np.sqrt(n_sub)
    return float(np.mean(est - tdist.ppf(0.95, n_sub - 1) * se > 0)), float(se.mean())

def mc_mde(p_lo, n_sub, m, rho, r, nsim=4000, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 0.5, 26)
    pw = []
    for d in grid:
        if p_lo + d >= 0.999: pw.append(1.0); continue
        pw.append(mc_power(p_lo, d, n_sub, m, rho, r, nsim, rng)[0])
    pw = np.array(pw)
    ok = np.where(pw >= 0.8)[0]
    return float(grid[ok[0]]) if len(ok) else None, pw.tolist()

if __name__ == "__main__":
    out = {"analytic": [], "mc": []}
    for n_units, m, tag in [(96, 4, "fold O (96 test trials)"), (264, 11, "pooled cross-fit (264 test trials)"), (24, 1, "subject-level (gender, 24 subjects)")]:
        for p in [0.05, 0.1, 0.2, 0.3, 0.5]:
            for rho in [0.0, 0.3, 0.6]:
                for r in [0.3, 0.5, 0.7]:
                    mde, se = analytic_mde(p, n_units, m, rho, r)
                    out["analytic"].append(dict(design=tag, n_units=n_units, m=m, p=p, rho=rho, r=r, se=round(se, 4), mde80=round(mde, 4)))
    for n_units, m in [(96, 4), (264, 11)]:
        for p in [0.1, 0.3]:
            for rho in [0.3]:
                for r in [0.5]:
                    mde, pw = mc_mde(p, 24, m, rho, r)
                    out["mc"].append(dict(n_units=n_units, m=m, p_lo=p, rho=rho, r_latent=r, mde80_mc=mde, grid_step=0.02))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    # compact print
    print("design | p | rho | r | SE | MDE80")
    for a in out["analytic"]:
        if a["rho"] in (0.3,) and a["r"] == 0.5:
            print(a["design"], a["p"], a["rho"], a["r"], a["se"], a["mde80"])
    for a in out["mc"]: print("MC", a)
