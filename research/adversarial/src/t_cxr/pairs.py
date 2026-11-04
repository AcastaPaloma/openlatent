"""Pair construction for R-CXR: PriCheXy-style random negatives (N0) and our matched negatives (N1, N2).

Unit of analysis = patient (the anchor patient owns the positive pair and its matched negative).  One positive per patient on
the test side; up to `per_patient` on the training side.  A *triple* is (anchor i, partner j of the same patient, negative k of
another patient chosen so that the metadata of (i,k) looks like that of (i,j)).  Element order inside a pair is randomised per
triple (identically for the positive and its negative) so that "earlier visit" is not a positional cue.

Matching tiers (N2; the primary design), each tier is tried in order, the first with a candidate wins, `tier` is recorded:
  0  sex, view, age (integer year), pixel spacing sx and (W,H) all equal to the partner's
  1  sex, view, age, sx equal; (W,H) within 5% (log) and the equality flags W==W_anchor, H==H_anchor of the positive preserved
  2  sex, view equal; |age gap| <= 1 year; sx equal; (W,H) within 10%
  3  sex, view equal; |age gap| <= 2 years; (W,H) within 20%
  else: the positive is DROPPED (counted; never silently re-matched).
N1 (demographic only): tiers on (sex, view, age exact -> +-1 -> +-2), device ignored.   N0: any other-patient image.
Because the partner j and the negative k agree on every covariate (up to the tier), a metadata-only attacker has no pair feature
to exploit; `metadata_floor` measures that residual and the pre-registration requires it to be small.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .meta import unit

TIERS_N2 = (
    dict(age=0, sx=True, wh="exact"),
    dict(age=0, sx=True, wh=0.05),
    dict(age=1, sx=True, wh=0.10),
    dict(age=2, sx=False, wh=0.20),
)
TIERS_N1 = (dict(age=0, sx=False, wh=None), dict(age=1, sx=False, wh=None), dict(age=2, sx=False, wh=None))


def _rng(key) -> np.random.Generator:
    return np.random.default_rng(int(unit(key, "t_cxr:rng") * 2**32))


def positives(d: pd.DataFrame, split: str, per_patient: int = 1, seed: str = "p0", require_distinct_fu: bool = True, col: str = "split") -> pd.DataFrame:
    """One (or more) same-patient pairs per multi-visit patient of `split`; i = earlier visit, j = later visit."""
    rows = []
    sub = d[(d[col] == split) & d["ok"]] if "ok" in d else d[d[col] == split]
    dropped_dup = 0
    for pid, g in sub.groupby("pid", sort=True):
        if len(g) < 2:
            continue
        idx = g["row"].to_numpy()
        fu = g["fu"].to_numpy()
        dg = g["dupgroup"].to_numpy() if "dupgroup" in g else np.full(len(g), -1)
        combos = [(a, b) for a, b in combinations(range(len(idx)), 2) if (fu[a] != fu[b] or not require_distinct_fu) and not (dg[a] >= 0 and dg[a] == dg[b])]
        if not combos:
            dropped_dup += 1
            continue
        r = _rng(f"{seed}:{pid}")
        order = r.permutation(len(combos))[:per_patient]
        for o in order:
            a, b = combos[o]
            if fu[a] > fu[b]:
                a, b = b, a
            rows.append((pid, idx[a], idx[b]))
    out = pd.DataFrame(rows, columns=["pid", "i", "j"])
    out.attrs["patients_without_valid_positive"] = dropped_dup
    return out


def _candidates(d_arr, pool_idx_by_cell, pool_all, i, j, tier, used_pid):
    """rows k in the pool that satisfy the tier relative to partner j and anchor i (patient(k) != patient(i))."""
    sex, view, age, W, H, sx, pid = d_arr
    cand = []
    for da in range(-tier["age"], tier["age"] + 1):
        cand.append(pool_idx_by_cell.get((sex[j], view[j], age[j] + da), np.empty(0, dtype=np.int64)))
    c = np.concatenate(cand) if cand else np.empty(0, dtype=np.int64)
    if len(c) == 0:
        return c
    c = c[pid[c] != pid[i]]
    if tier["sx"]:
        c = c[sx[c] == sx[j]]
    wh = tier["wh"]
    if wh == "exact":
        c = c[(W[c] == W[j]) & (H[c] == H[j])]
    elif wh is not None:
        c = c[(np.abs(np.log(W[c] / W[j])) <= wh) & (np.abs(np.log(H[c] / H[j])) <= wh)]
        # preserve the pair-level equality flags of the positive (exact W/H equality with the anchor is a strong cue)
        c = c[((W[c] == W[i]) == (W[j] == W[i])) & ((H[c] == H[i]) == (H[j] == H[i]))]
    return c


def build(d: pd.DataFrame, split: str, design: str = "N2", per_patient: int = 1, seed: str = "p0", positives_df: pd.DataFrame | None = None, col: str = "split") -> tuple[pd.DataFrame, dict]:
    """Return (pairs, report).  pairs columns: triple, label, a, b (image rows), pid_a, pid_b, tier, design, long_gap, fu_gap."""
    pos = positives(d, split, per_patient, seed, col=col) if positives_df is None else positives_df
    sub = d[(d[col] == split) & d["ok"]] if "ok" in d else d[d[col] == split]
    sex, view, age = d["male"].to_numpy(), d["pa"].to_numpy(), d["age"].to_numpy().astype(int)
    W, H = d["W"].to_numpy().astype(float), d["H"].to_numpy().astype(float)
    sx, pid = d["sx"].to_numpy(), d["pid"].to_numpy()
    arr = (sex, view, age, W, H, sx, pid)
    pool_all = sub["row"].to_numpy()
    cells: dict = {}
    for r in pool_all:
        cells.setdefault((sex[r], view[r], age[r]), []).append(r)
    cells = {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}
    tiers = {"N2": TIERS_N2, "N1": TIERS_N1, "N0": ()}[design]
    out, dropped, tier_hist, used = [], 0, {}, set()
    for t, (p, i, j) in enumerate(pos.itertuples(index=False)):
        r = _rng(f"{seed}:{design}:{p}:{i}:{j}")
        k, tier_used = None, -1
        if design == "N0":
            c = pool_all[pid[pool_all] != pid[i]]
            k, tier_used = int(c[r.integers(len(c))]), 0
        else:
            for ti, tier in enumerate(tiers):
                c = _candidates(arr, cells, pool_all, i, j, tier, used)
                if len(c):
                    fresh = c[~np.isin(pid[c], list(used))] if used and len(c) < 5000 else c
                    c = fresh if len(fresh) else c
                    k, tier_used = int(c[r.integers(len(c))]), ti
                    break
        if k is None:
            dropped += 1
            continue
        used.add(pid[k])
        tier_hist[tier_used] = tier_hist.get(tier_used, 0) + 1
        swap = bool(r.integers(2))
        gap = int(age[j] - age[i])
        fu_gap = int(d["fu"].iat[j] - d["fu"].iat[i])
        for lab, partner in ((1, j), (0, k)):
            a, b = (partner, i) if swap else (i, partner)
            out.append((len(out) // 2 if False else t, lab, a, b, int(pid[a]), int(pid[b]), tier_used, design, int(gap >= 1), fu_gap, int(p)))
    pairs = pd.DataFrame(out, columns=["triple", "label", "a", "b", "pid_a", "pid_b", "tier", "design", "long_gap", "fu_gap", "anchor_pid"])
    rep = dict(split=split, design=design, positives_requested=len(pos), dropped=dropped, kept=len(pos) - dropped, tier_hist=tier_hist,
               patients_without_valid_positive=int(pos.attrs.get("patients_without_valid_positive", 0)))
    return pairs, rep


def from_pair_file(path, d: pd.DataFrame) -> pd.DataFrame:
    """PriCheXy-Net's own pair lists (`img1 img2 label`, label 1.0 = same patient); images mapped to metadata rows."""
    a = np.loadtxt(path, dtype=str)
    row = dict(zip(d["img"], d["row"]))
    ra = np.array([row[x] for x in a[:, 0]])
    rb = np.array([row[x] for x in a[:, 1]])
    pid = d["pid"].to_numpy()
    return pd.DataFrame(dict(triple=np.arange(len(a)), label=a[:, 2].astype(float).astype(int), a=ra, b=rb, pid_a=pid[ra], pid_b=pid[rb],
                             tier=0, design="PRICHEXY", long_gap=0, fu_gap=0, anchor_pid=pid[ra]))


def pair_features(pairs: pd.DataFrame, d: pd.DataFrame, noise: dict | None = None, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Metadata features of each pair (no image).  `noise` = {'age_sd': 5, 'sex_flip': .03} simulates an image-estimated demographic attacker."""
    A, B = d.iloc[pairs.a.to_numpy()], d.iloc[pairs.b.to_numpy()]
    ag_a, ag_b = A.age.to_numpy().copy(), B.age.to_numpy().copy()
    m_a, m_b = A.male.to_numpy().copy(), B.male.to_numpy().copy()
    if noise:
        rng = rng or np.random.default_rng(0)
        ag_a += rng.normal(0, noise["age_sd"], len(ag_a)); ag_b += rng.normal(0, noise["age_sd"], len(ag_b))
        m_a = np.where(rng.random(len(m_a)) < noise["sex_flip"], 1 - m_a, m_a); m_b = np.where(rng.random(len(m_b)) < noise["sex_flip"], 1 - m_b, m_b)
    f = pd.DataFrame(dict(age_a=ag_a, age_b=ag_b, dage=ag_b - ag_a, adage=np.abs(ag_b - ag_a), same_sex=(m_a == m_b).astype(float), male_a=m_a, male_b=m_b,
                          pa_a=A.pa.to_numpy(), pa_b=B.pa.to_numpy(), same_view=(A.pa.to_numpy() == B.pa.to_numpy()).astype(float)))
    f["W_a"], f["W_b"], f["H_a"], f["H_b"] = A.W.to_numpy(), B.W.to_numpy(), A.H.to_numpy(), B.H.to_numpy()
    f["sx_a"], f["sx_b"] = A.sx.to_numpy(), B.sx.to_numpy()
    f["dW"], f["dH"] = np.abs(f.W_a - f.W_b), np.abs(f.H_a - f.H_b)
    f["dsx"] = np.abs(f.sx_a - f.sx_b)
    f["eqW"], f["eqH"], f["eqsx"] = (f.dW == 0).astype(float), (f.dH == 0).astype(float), (f.dsx == 0).astype(float)
    return f


FEATURE_SETS = {
    "demo": ["adage", "dage", "same_sex", "same_view"],
    "demo_raw": ["age_a", "age_b", "dage", "adage", "same_sex", "male_a", "male_b", "pa_a", "pa_b", "same_view"],
    "device": ["dW", "dH", "dsx", "eqW", "eqH", "eqsx", "W_a", "W_b", "H_a", "H_b", "sx_a", "sx_b", "same_view"],
    "all": None,
}


def metadata_floor(ptrain: pd.DataFrame, ptest: pd.DataFrame, d: pd.DataFrame, fset: str = "all", noise: dict | None = None, seed: int = 0) -> float:
    """Gradient-boosted metadata-only attacker trained on `ptrain`, AUC on `ptest` (patient-disjoint)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    Xtr, Xte = pair_features(ptrain, d, noise, rng), pair_features(ptest, d, noise, rng)
    cols = FEATURE_SETS[fset] or list(Xtr.columns)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=seed).fit(Xtr[cols], ptrain.label)
    return float(roc_auc_score(ptest.label, m.predict_proba(Xte[cols])[:, 1]))
