"""Evaluator statistics for R-CXR (instrument S, CXR instance).  numpy/scipy only.

* weighted AUC under patient-cluster bootstrap multiplicities (one resample of PATIENTS shared by every release / attacker, so
  paired differences and worst-case-over-attackers are taken INSIDE each resample),
* paired AUC difference with SE, percentile CI and MDE80 = (z_.975 + z_.80) * SE,
* Hanley-McNeil analytic power table (published patient counts -> smallest resolvable AUC difference),
* floor-envelope comparison at matched leakage (selection on validation only), and the three kill predicates of T.md.
"""
from __future__ import annotations

import numpy as np

Z_MDE = 1.959964 + 0.841621   # two-sided 5% size, 80% power


def auc(y, s) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(s, dtype=float))
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class PatientBootstrap:
    """B resamples of the patient universe.  mult[b, p] = how often patient p is drawn in resample b (multinomial)."""

    def __init__(self, patients, B: int = 1000, seed: int = 0):
        self.patients = np.unique(np.asarray(patients))
        self.index = {p: i for i, p in enumerate(self.patients)}
        rng = np.random.default_rng(seed)
        P = len(self.patients)
        self.mult = np.stack([np.bincount(rng.integers(0, P, size=P), minlength=P) for _ in range(B)]).astype(np.float64)
        self.B = B

    def codes(self, cluster_ids):
        return np.array([self.index[c] for c in cluster_ids])


def auc_boot(y, s, codes, pb: PatientBootstrap) -> np.ndarray:
    """weighted AUC for every resample; units whose patient is not drawn get weight 0.  Returns shape (B,)."""
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s, dtype=float)
    _, g = np.unique(s, return_inverse=True)
    G = g.max() + 1
    out = np.empty(pb.B)
    W = pb.mult[:, codes]                               # (B, n)
    for b in range(pb.B):
        w = W[b]
        wp = np.bincount(g, weights=w * y, minlength=G)
        wn = np.bincount(g, weights=w * (1 - y), minlength=G)
        cn = np.cumsum(wn) - wn
        den = wp.sum() * wn.sum()
        out[b] = np.nan if den == 0 else float((wp * (cn + 0.5 * wn)).sum() / den)
    return out


def summarize(point: float, boot: np.ndarray) -> dict:
    ok = boot[~np.isnan(boot)]
    return dict(point=float(point), se=float(ok.std(ddof=1)), lo=float(np.percentile(ok, 2.5)), hi=float(np.percentile(ok, 97.5)))


def paired(boot_a: np.ndarray, boot_b: np.ndarray, point_a: float, point_b: float) -> dict:
    """a - b with SE from the SHARED resamples; mde80 = 2.80 * SE (the smallest true difference this test resolves with 80% power)."""
    d = boot_a - boot_b
    se = float(np.nanstd(d, ddof=1))
    return dict(diff=float(point_a - point_b), se=se, lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)), mde80=float(Z_MDE * se),
                p_le0=float(np.nanmean(d <= 0)))


# ---------------------------------------------------------------- analytic power (Hanley-McNeil 1982, 1983)
def hm_se(a: float, n1: int, n0: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(np.sqrt(v))


def hm_mde_paired(a: float, n1: int, n0: int, rho: float = 0.6) -> float:
    """MDE80 of the difference between two correlated AUCs (error correlation rho) on the same n1 positives / n0 negatives."""
    return float(Z_MDE * hm_se(a, n1, n0) * np.sqrt(2 * (1 - rho)))


def power_table(units=(("official test, pairs from 1,424 patients", 1424), ("official test, N2 after drops", 1415), ("LAPIS-size test", 4707),
                       ("all multi-visit patients (cross-fit)", 13302), ("PriCheXy 5,000-pair list: 492 patients behind the positives", 492)),
                aucs=(0.55, 0.65, 0.75, 0.85), rhos=(0.3, 0.6, 0.8)) -> list[dict]:
    rows = []
    for name, n in units:
        for a in aucs:
            rows.append(dict(units=name, n_pos=n, n_neg=n, auc=a, se_single=hm_se(a, n, n),
                             **{f"mde80_rho{r}": hm_mde_paired(a, n, n, r) for r in rhos}))
    return rows


# ---------------------------------------------------------------- floors envelope and kill predicates
def pick_floor(D: dict, floors: list[dict], tol_leak: float) -> dict | None:
    """F* = the floor with the highest VALIDATION utility among those whose VALIDATION leakage is <= L_val(D) + tol_leak.  None if no floor
    reaches D's leakage (then D has headroom by definition).  Dicts need keys name, L_val, U_val."""
    ok = [f for f in floors if f["L_val"] <= D["L_val"] + tol_leak]
    return max(ok, key=lambda f: f["U_val"]) if ok else None


def pick_floor_leak(D: dict, floors: list[dict], tol_util: float) -> dict | None:
    """Mirror axis: among floors with U_val >= U_val(D) - tol_util, the one with the lowest VALIDATION leakage."""
    ok = [f for f in floors if f["U_val"] >= D["U_val"] - tol_util]
    return min(ok, key=lambda f: f["L_val"]) if ok else None


def envelope_verdict(D: dict, floors: list[dict], tol_leak: float, tol_util: float, margin_u: float, margin_l: float) -> dict:
    """D beats the envelope iff (utility axis) U_test(D) - U_test(F*) >= margin_u with the paired CI lower bound > 0, OR (leakage axis)
    L_test(F_L) - L_test(D) >= margin_l with CI lower bound > 0.  D/floors carry test-level `U_boot`, `L_boot` (B,) and `U`, `L` points."""
    out = dict(name=D["name"], utility_axis=None, leak_axis=None)
    fs = pick_floor(D, floors, tol_leak)
    if fs is None:
        out["utility_axis"] = dict(floor=None, beats=True, note="no floor reaches D's validation leakage + tol")
    else:
        r = paired(D["U_boot"], fs["U_boot"], D["U"], fs["U"])
        out["utility_axis"] = dict(floor=fs["name"], beats=bool(r["diff"] >= margin_u and r["lo"] > 0), **r)
    fl = pick_floor_leak(D, floors, tol_util)
    if fl is None:
        out["leak_axis"] = dict(floor=None, beats=True, note="no floor holds D's validation utility - tol")
    else:
        r = paired(fl["L_boot"], D["L_boot"], fl["L"], D["L"])           # floor leaks MORE than D  <=>  D better
        out["leak_axis"] = dict(floor=fl["name"], beats=bool(r["diff"] >= margin_l and r["lo"] > 0), **r)
    out["beats_envelope"] = bool(out["utility_axis"]["beats"] or out["leak_axis"]["beats"])
    return out


def kill_floor_envelope(verdicts: list[dict]) -> bool:
    """KT2 fires iff NO learned defence beats the floor envelope on either axis."""
    return not any(v["beats_envelope"] for v in verdicts)


def kill_measurability(raw_worst_auc: float, metadata_floor_auc: float, controls_auc: list[float], min_auc: float = 0.65, min_over_meta: float = 0.03, ctrl_tol: float = 0.03) -> dict:
    """KT1: raw worst-case matched-negative AUC must be >= min_auc AND >= metadata floor + min_over_meta, every control within ctrl_tol of 0.5."""
    c = max(abs(a - 0.5) for a in controls_auc) if controls_auc else 0.0
    return dict(fires=bool(raw_worst_auc < min_auc or raw_worst_auc < metadata_floor_auc + min_over_meta or c > ctrl_tol), raw_worst=raw_worst_auc,
                metadata_floor=metadata_floor_auc, max_control_dev=c)


def kill_demographics(auc_demo: float, auc_worst: float, share: float = 0.90) -> dict:
    """KT3 (random-negative protocol): fires if the image-estimated demographic attacker explains >= `share` of the excess-over-chance of the worst attacker."""
    s = (auc_demo - 0.5) / max(auc_worst - 0.5, 1e-9)
    return dict(fires=bool(s >= share), share=float(s))


if __name__ == "__main__":      # python -m t_cxr.stats  -> power table (markdown)
    print("| units | n_pos=n_neg | AUC | single-AUC SE | MDE80 rho=.3 | rho=.6 | rho=.8 |\n|---|---:|---:|---:|---:|---:|---:|")
    for r in power_table():
        print(f"| {r['units']} | {r['n_pos']} | {r['auc']:.2f} | {r['se_single']:.4f} | {r['mde80_rho0.3']:.4f} | {r['mde80_rho0.6']:.4f} | {r['mde80_rho0.8']:.4f} |")
