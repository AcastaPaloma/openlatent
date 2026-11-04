"""Evidence-budget module: how many clips/windows/images of one person must an attacker pool to reach a target identification rate,
and is the independence assumption behind "evidence adds up" true?

Pooling rule: sum of clipped log-probabilities of the K pooled items (product of experts); a mean-probability rule is available as a
robustness check. A pooled trial = K items of ONE person; its accuracy is whether the pooled argmax is that person.

Pooling schemes (K items of one person):
    random        uniform over all of the person's items (any session)
    same_session  K items from one session (sessions with fewer than K items are not eligible at that K)
    same_record   K items from one recording
    diverse       round-robin over the person's sessions in random order (K <= n_sessions: one item per session)

Independence test: if evidence from different items were independent given the person, all schemes would give the same curve. A shared
session nuisance (offset, scenario, clothing, device) makes same_session pooling gain less than diverse pooling. The module reports the paired
difference diverse - same_session with a person-level cluster bootstrap and its MDE80, the gain ratio and the effective K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import warnings

import numpy as np

from . import stats as st

EPS = 1e-6


def log_evidence(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0))


@dataclass
class Items:
    probs: np.ndarray          # (n, C) per-item class probabilities from the attacker
    person: np.ndarray         # (n,) integer person id == class index
    session: np.ndarray        # (n,) session key (str-able)
    record: Optional[np.ndarray] = None  # (n,) recording key; default: session

    def __post_init__(self):
        self.person = np.asarray(self.person).astype(int)
        self.session = np.asarray(self.session).astype(str)
        self.record = self.session if self.record is None else np.asarray(self.record).astype(str)


def _person_index(items: Items):
    out = {}
    for p in np.unique(items.person):
        idx = np.where(items.person == p)[0]
        sess = {}
        for i in idx:
            sess.setdefault(items.session[i], []).append(i)
        rec = {}
        for i in idx:
            rec.setdefault(items.record[i], []).append(i)
        out[int(p)] = dict(all=idx, sess=[np.array(v) for v in sess.values()], rec=[np.array(v) for v in rec.values()])
    return out


def _draw(pidx: dict, scheme: str, K: int, rs: np.random.RandomState):
    if scheme == "random":
        a = pidx["all"]
        return rs.choice(a, K, replace=False) if len(a) >= K else None
    if scheme in ("same_session", "same_record"):
        groups = pidx["sess"] if scheme == "same_session" else pidx["rec"]
        el = [g for g in groups if len(g) >= K]
        if not el:
            return None
        g = el[rs.randint(len(el))]
        return rs.choice(g, K, replace=False)
    if scheme == "diverse":
        groups = [rs.permutation(g) for g in pidx["sess"]]
        if sum(len(g) for g in groups) < K:
            return None
        order = rs.permutation(len(groups))
        chosen, depth = [], 0
        while len(chosen) < K:
            for gi in order:
                if depth < len(groups[gi]):
                    chosen.append(groups[gi][depth])
                    if len(chosen) == K:
                        break
            depth += 1
        return np.array(chosen)
    raise ValueError(scheme)


def pooled_person_accuracy(items: Items, scheme: str, K: int, n_draws: int = 100, seed: int = 0, rule: str = "logprob"):
    """-> (persons, acc_per_person) with NaN for persons that cannot form a K-pool under this scheme."""
    L = log_evidence(items.probs) if rule == "logprob" else np.asarray(items.probs, dtype=float)
    pi = _person_index(items)
    persons = sorted(pi)
    rs = np.random.RandomState((seed * 1000003 + K * 101 + sum(map(ord, scheme))) % (2 ** 32))
    acc = np.full(len(persons), np.nan)
    for j, p in enumerate(persons):
        ok = []
        for _ in range(n_draws):
            d = _draw(pi[p], scheme, K, rs)
            if d is None:
                break
            ok.append(int(np.argmax(L[d].sum(0))) == p)
        else:
            acc[j] = float(np.mean(ok))
    return np.array(persons), acc


def budget_curve(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _budget_curve(*a, **k)


def _budget_curve(items: Items, K_list: List[int], schemes=("random", "same_session", "same_record", "diverse"), n_draws: int = 100,
                 n_boot: int = 1000, seed: int = 0, rule: str = "logprob", targets=(0.5, 0.9)) -> dict:
    """Pooled accuracy vs K per scheme with a person-level bootstrap, N* per target, and the independence test."""
    res = {"K": list(K_list), "schemes": {}}
    per_person = {}
    for sc in schemes:
        rows = []
        for K in K_list:
            _, a = pooled_person_accuracy(items, sc, K, n_draws, seed, rule)
            rows.append(a)
        per_person[sc] = np.array(rows)  # (len(K_list), n_persons)
    persons = np.unique(items.person)
    P = len(persons)
    rs = np.random.RandomState(seed + 7)
    bidx = rs.randint(0, P, (n_boot, P))

    def boot_mean(M):  # M (nK, P) with NaNs -> mean over eligible persons in each resample, (n_boot, nK)
        out = np.full((n_boot, M.shape[0]), np.nan)
        for i in range(M.shape[0]):
            v = M[i][bidx]
            with np.errstate(all="ignore"):
                out[:, i] = np.nanmean(v, axis=1)
        return out

    for sc, M in per_person.items():
        bm = boot_mean(M)
        with np.errstate(all="ignore"):
            mean = np.nanmean(M, axis=1)
        n_el = (~np.isnan(M)).sum(1)
        res["schemes"][sc] = dict(acc=mean, ci95=[np.nanquantile(bm, 0.025, axis=0), np.nanquantile(bm, 0.975, axis=0)], se=np.nanstd(bm, axis=0, ddof=1),
                                  n_persons_eligible=n_el)
        res["schemes"][sc]["mde80"] = st.Z_MDE * res["schemes"][sc]["se"]
        res["schemes"][sc]["n_star"] = {str(t): _n_star(K_list, mean, t) for t in targets}
        nb = np.array([[_n_star(K_list, bm[b], t) for t in targets] for b in range(min(n_boot, 400))], dtype=float)
        res["schemes"][sc]["n_star_ci95"] = {str(t): [float(np.nanquantile(nb[:, j], 0.025)) if np.isfinite(nb[:, j]).any() else None,
                                                      float(np.nanquantile(nb[:, j], 0.975)) if np.isfinite(nb[:, j]).any() else None]
                                             for j, t in enumerate(targets)}
    # independence test: diverse - same_session, paired over persons
    if "diverse" in per_person and "same_session" in per_person:
        res["independence"] = _independence(K_list, per_person["diverse"], per_person["same_session"], bidx)
    return res


def _n_star(K_list, acc, target):
    """Smallest K reaching `target` (linear interpolation between grid points); None if never reached."""
    acc = np.asarray(acc, dtype=float)
    for i, (K, a) in enumerate(zip(K_list, acc)):
        if np.isfinite(a) and a >= target:
            if i == 0 or not np.isfinite(acc[i - 1]):
                return float(K)
            K0, a0 = K_list[i - 1], acc[i - 1]
            return float(K0 + (target - a0) * (K - K0) / max(a - a0, 1e-12))
    return None


def _independence(K_list, Md, Ms, bidx):
    out = []
    for i, K in enumerate(K_list):
        ok = ~np.isnan(Md[i]) & ~np.isnan(Ms[i])
        if ok.sum() < 3:
            out.append(dict(K=K, available=False))
            continue
        d = np.where(ok, Md[i] - Ms[i], np.nan)
        v = d[bidx]
        with np.errstate(all="ignore"):
            bm = np.nanmean(v, axis=1)
        se = float(np.nanstd(bm, ddof=1))
        lo, hi = float(np.nanquantile(bm, 0.025)), float(np.nanquantile(bm, 0.975))
        mean_d = float(np.nanmean(d))
        mean_s = float(np.nanmean(np.where(ok, Ms[i], np.nan)))
        mean_v = float(np.nanmean(np.where(ok, Md[i], np.nan)))
        out.append(dict(K=K, available=True, n_persons=int(ok.sum()), diverse=mean_v, same_session=mean_s, diff=mean_d, se=se,
                        mde80=st.Z_MDE * se, ci95=[lo, hi], dependence_detected=bool(lo > 0)))
    return out


def effective_k(K_list, acc_diverse, acc_same):
    """K' on the diverse curve that matches each same-session accuracy (interpolated); ratio K'/K < 1 means shared nuisance."""
    Kd, ad = np.asarray(K_list, float), np.asarray(acc_diverse, float)
    ok = np.isfinite(ad)
    out = []
    for K, a in zip(K_list, acc_same):
        if not np.isfinite(a) or ok.sum() < 2:
            out.append(None)
            continue
        out.append(float(np.interp(a, ad[ok], Kd[ok])) if ad[ok].max() >= a >= ad[ok].min() else None)
    return out


def calibrate_temperature(probs, y, grid=None):
    """Temperature T minimising cross-entropy of softmax(log p / T). -> (T, CE nats)."""
    L = log_evidence(probs)
    y = np.asarray(y).astype(int)
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 60)) if grid is None else grid
    best = (None, np.inf)
    for T in grid:
        z = L / T
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        ce = -lp[np.arange(len(y)), y].mean()
        if ce < best[1]:
            best = (float(T), float(ce))
    return best


def predicted_n_star(probs, y, C=None):
    """N*_pred = ln(C) / I_item with I_item = ln(C) - CE of the temperature-calibrated per-item posterior (nats). None if I <= 0."""
    C = C or probs.shape[1]
    T, ce = calibrate_temperature(probs, y)
    I = np.log(C) - ce
    return dict(T=T, ce=ce, info_nats=float(I), n_star_pred=float(np.log(C) / I) if I > 1e-9 else None)


# ---------------------------------------------------------------------------------------------------------------------------
# toy model with a shared session nuisance (validation of the module itself)

def toy_items(C=25, d=16, scale=0.20, rho=0.0, n_sessions=6, per_session=6, seed=0):
    """Person means ~ N(0, scale^2 I_d); per-item noise total variance 1, of which a fraction rho is a SESSION offset shared by the
    session's items. The per-item attacker is the optimal isotropic-Gaussian posterior with total variance 1."""
    rs = np.random.RandomState(seed)
    mu = rs.randn(C, d) * scale
    rows, person, session = [], [], []
    for c in range(C):
        for s in range(n_sessions):
            off = rs.randn(d) * np.sqrt(rho)
            x = mu[c] + off + rs.randn(per_session, d) * np.sqrt(1 - rho)
            rows.append(x)
            person += [c] * per_session
            session += [f"{c}_{s}"] * per_session
    X = np.concatenate(rows)
    d2 = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
    z = -0.5 * d2
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(1, keepdims=True)
    return Items(P, np.array(person), np.array(session))
