"""Evaluator statistics for R-CXR (instrument S, CXR instance).  numpy/scipy only.

* weighted AUC under patient-cluster bootstrap multiplicities (one resample of PATIENTS shared by every release / attacker, so
  paired differences and worst-case-over-attackers are taken INSIDE each resample),
* paired AUC difference with SE, percentile CI and MDE80 = (z_.975 + z_.80) * SE,
* Hanley-McNeil analytic power table (published patient counts -> smallest resolvable AUC difference),
* floor-envelope comparison (FOCUS A2: linear interpolation of the floors' Pareto frontier at the learned point's worst-suite leakage / utility, the frontier recomputed inside every
  patient resample) and the kill predicates of T.md.
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


# ---------------------------------------------------------------- floors envelope (FOCUS A2 matching rule) and kill predicates
def _frontier(L: np.ndarray, U: np.ndarray):
    """Pareto frontier of floor operating points (lower leakage L, higher utility U), sorted by L; returns arrays (Lf, Uf) with Uf non-decreasing."""
    o = np.argsort(L, kind="stable")
    Ls, Us = L[o], U[o]
    keepL, keepU, best = [], [], -np.inf
    for l, u in zip(Ls, Us):
        if u > best + 1e-12:
            keepL.append(l); keepU.append(u); best = u
    return np.array(keepL), np.array(keepU)


def envelope_at(Lf: np.ndarray, Uf: np.ndarray, L_at: float) -> float:
    """utility envelope of the floors at leakage L_at: LINEAR interpolation of the Pareto frontier; constant beyond the ends."""
    fl, fu = _frontier(Lf, Uf)
    return float(np.interp(L_at, fl, fu))


def envelope_leak_at(Lf: np.ndarray, Uf: np.ndarray, U_at: float) -> float:
    """lower envelope of leakage at utility U_at: the smallest leakage reachable at utility >= U_at (linear interpolation of the frontier, inverse)."""
    fl, fu = _frontier(Lf, Uf)
    if U_at <= fu[0]:
        return float(fl[0])
    if U_at > fu[-1]:
        return float("nan")                                  # no floor reaches that utility: D is above the floors' utility range
    return float(np.interp(U_at, fu, fl))


def envelope_boot(floors_L: np.ndarray, floors_U: np.ndarray, L_D: np.ndarray, U_D: np.ndarray) -> dict:
    """floors_L / floors_U: (n_floors, B) bootstrap matrices on the SAME patient resamples as L_D / U_D (B,).  The frontier is recomputed INSIDE each
    resample (the maximum over floors is therefore not given a free pass).  Returns per-resample deltas
       dU = U_D - E_U(L_D)   (positive: D keeps more utility than the floors at D's leakage)
       dL = E_L(U_D) - L_D   (positive: D leaks less than the floors at D's utility)   [nan if D's utility is above every floor's]."""
    B = L_D.shape[0]
    dU, dL = np.empty(B), np.empty(B)
    for b in range(B):
        dU[b] = U_D[b] - envelope_at(floors_L[:, b], floors_U[:, b], L_D[b])
        dL[b] = envelope_leak_at(floors_L[:, b], floors_U[:, b], U_D[b]) - L_D[b]
    return dict(dU=dU, dL=dL)


def envelope_verdict(point_floors_L, point_floors_U, L_D, U_D, boot: dict, margin_u: float, margin_l: float, q: float = 2.5) -> dict:
    """Machine-checkable verdict.  D beats the floor envelope iff the LOWER 95% bound of dU >= margin_u (utility axis) OR the lower 95% bound of dL >= margin_l (leakage axis).
    dL is nan (floors never reach D's utility) -> that axis counts as passed with a note.  margin_* are fixed from the printed MDE80 (T.md 3.8)."""
    pu = U_D - envelope_at(np.asarray(point_floors_L), np.asarray(point_floors_U), L_D)
    pl = envelope_leak_at(np.asarray(point_floors_L), np.asarray(point_floors_U), U_D) - L_D
    dU, dL = boot["dU"], boot["dL"]
    loU, hiU = float(np.nanpercentile(dU, q)), float(np.nanpercentile(dU, 100 - q))
    ok_l = ~np.isnan(dL)
    loL = float(np.nanpercentile(dL, q)) if ok_l.mean() > 0.5 else float("nan")
    util = bool(loU >= margin_u)
    leak = True if np.isnan(pl) else bool(loL >= margin_l)
    return dict(point_dU=float(pu), dU_lo=loU, dU_hi=hiU, point_dL=float(pl), dL_lo=loL, utility_axis_beats=util, leak_axis_beats=leak,
                beats_envelope=bool(util or leak), mde80_dU=float(Z_MDE * np.nanstd(dU, ddof=1)), mde80_dL=float(Z_MDE * np.nanstd(dL, ddof=1)) if ok_l.mean() > 0.5 else float("nan"))


def kill_floor_envelope(verdicts: list[dict]) -> bool:
    """KT2 fires iff NO learned defence beats the floor envelope on either axis."""
    return not any(v["beats_envelope"] for v in verdicts)


def control_tolerance(se: float, n_pos: int, n_neg: int) -> float:
    """FOCUS A2: control tolerance = max(1.96 SE of the control, design-effect-aware MDE of a single AUC at chance)."""
    return float(max(1.96 * se, Z_MDE * hm_se(0.5, n_pos, n_neg)))


def kill_measurability(raw_worst_auc: float, metadata_floor_auc: float, controls: list[tuple[float, float]], n_pos: int, n_neg: int,
                       min_auc: float = 0.65, min_over_meta: float = 0.03) -> dict:
    """KT1: raw worst-case matched-negative AUC must be >= min_auc AND >= metadata floor + min_over_meta, and every control (auc, se) within control_tolerance of 0.5."""
    bad = [(a, se) for a, se in controls if abs(a - 0.5) > control_tolerance(se, n_pos, n_neg)]
    return dict(fires=bool(raw_worst_auc < min_auc or raw_worst_auc < metadata_floor_auc + min_over_meta or bad), raw_worst=raw_worst_auc,
                metadata_floor=metadata_floor_auc, controls_failed=bad)


def kill_demographics(auc_demo: float, auc_worst: float, share: float = 0.90) -> dict:
    """KT3 (random-negative protocol): fires if the image-estimated demographic attacker explains >= `share` of the excess-over-chance of the worst attacker."""
    s = (auc_demo - 0.5) / max(auc_worst - 0.5, 1e-9)
    return dict(fires=bool(s >= share), share=float(s))


if __name__ == "__main__":      # python -m t_cxr.stats  -> power table (markdown)
    print("| units | n_pos=n_neg | AUC | single-AUC SE | MDE80 rho=.3 | rho=.6 | rho=.8 |\n|---|---:|---:|---:|---:|---:|---:|")
    for r in power_table():
        print(f"| {r['units']} | {r['n_pos']} | {r['auc']:.2f} | {r['se_single']:.4f} | {r['mde80_rho0.3']:.4f} | {r['mde80_rho0.6']:.4f} | {r['mde80_rho0.8']:.4f} |")
