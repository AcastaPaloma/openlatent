"""Metrics, chance levels, confidence intervals and the harness's privacy / utility scores.

Every score is computed from per-item predictions (an attacker's or a utility model's averaged clip outputs), so
every number comes with a bootstrap confidence interval over test items.

Metric per target kind:
    multiclass   top-1 accuracy (%)
    multilabel   mean average precision over classes (mAP, %), the cMAP of the privacy literature
    conditional  multilabel secrets whose labels are largely implied by the primary utility label (VP-UCF101 /
                 VP-HMDB51): AP within each utility class where the attribute varies, averaged; asks what the visual
                 data reveals beyond the task label itself

Chance (what an attacker gets with no visual information):
    top-1        accuracy of always predicting the most frequent training class, measured on the test items
    mAP / cond.  expected AP of random scores on the same test labels (Monte Carlo); with few positives per class
                 this sits well above the positive rate, so prevalence is the wrong floor

Baseline (the floor leakage is measured from): max(chance, task-label attacker), where the task-label attacker
predicts the secret from the primary task label alone. A secret the task label already implies is "dataset
leakage" (Wang et al., ICCV 2019), not something the footage leaks; Jayaraman & Evans (CCS 2022) make the same
point for attribute inference against an imputation baseline. When raw footage is not measurably above this
baseline the secret is reported as not measurable, and multilabel secrets switch to the conditional metric.

Scores, all in [0, 1] (reported x100):
    leakage    L = (A_obf - base) / (A_raw - base)   A = the worst case (best attacker) under the same protocol
               0 = nothing beyond chance, 1 = as much as raw data; values outside [0, 1] are kept in the record
               and clipped in the scores. Undefined when raw data itself is near chance (secret not measurable).
    privacy    P = 1 - clip(L, 0, 1); for several secrets and attackers the dataset privacy is the minimum
    utility    U = (u_obf - chance_u) / (u_raw - chance_u), clipped to [0, 1] in the trade-off score
    trade-off  PUS = 2 P U / (P + U) (harmonic mean: high only if both are high)
    frontier   hypervolume of the (U, P) Pareto front of a set of obfuscators, in [0, 1] (reference corner: chance
               utility and no privacy)
    f_lambda   (1 - lambda) U + lambda P, Ilic et al.'s linear trade-off (CVPR 2024), for comparison only

Uncertainty: bootstrap over test recordings (all items of a resampled recording move together), 1000 draws, 95%
percentile interval; attacker seeds are varied on the recommended obfuscator.
"""

import numpy as np

N_BOOT = 1000


def average_precision(scores, labels):
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def top1(probs, y):
    return 100.0 * float((probs.argmax(1) == y).mean())


def mean_ap(probs, y):
    aps = [average_precision(probs[:, j], y[:, j]) for j in range(y.shape[1])]
    return 100.0 * float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan")


def conditional_ap(probs, y, cond):
    """Mean over attributes of the mean AP within each condition class where the attribute varies."""
    per_attr = []
    for j in range(y.shape[1]):
        aps = [average_precision(probs[cond == c, j], y[cond == c, j]) for c in np.unique(cond)
               if 0 < y[cond == c, j].sum() < (cond == c).sum()]
        if aps:
            per_attr.append(np.mean(aps))
    return 100.0 * float(np.mean(per_attr)) if per_attr else float("nan")


def metric(kind, probs, y, cond=None):
    if kind == "multiclass":
        return top1(probs, y)
    if cond is not None:
        return conditional_ap(probs, y, cond)
    return mean_ap(probs, y)


def chance(kind, y_test, y_train=None, cond=None, draws=200, seed=0):
    """Score of an attacker without visual information on these test labels (see module docstring)."""
    if kind == "multiclass":
        ref = y_train if y_train is not None else y_test
        majority = np.bincount(ref).argmax()
        return 100.0 * float((y_test == majority).mean())
    rng = np.random.default_rng(seed)
    return float(np.mean([metric(kind, rng.random(y_test.shape), y_test, cond) for _ in range(draws)]))


def bootstrap(kind, probs, y, cond=None, groups=None, n=N_BOOT, seed=0):
    """(point estimate, 2.5th, 97.5th percentile) of the metric, resampling test recordings (`groups`, one value per
    item) or, without groups, test items, with replacement."""
    point = metric(kind, probs, y, cond)
    rng = np.random.default_rng(seed)
    if groups is not None:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n):
        if groups is not None:
            idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        else:
            idx = rng.integers(0, len(y), len(y))
        v = metric(kind, probs[idx], y[idx], None if cond is None else cond[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan")))
    return point, float(lo), float(hi)


def baseline(chance_level, label_only=None):
    """The floor leakage is measured from: chance, or the task-label attacker if it does better."""
    return chance_level if label_only is None else max(chance_level, label_only)


def leakage(a, base, raw, min_gap=2.0):
    """Baseline-normalised leakage, or None when raw data is within `min_gap` points of the baseline (not measurable)."""
    if raw - base < min_gap:
        return None
    return (a - base) / (raw - base)


def privacy(leak):
    return None if leak is None else 1.0 - min(1.0, max(0.0, leak))


def retention(u, chance_level, raw):
    return (u - chance_level) / (raw - chance_level) if raw - chance_level > 0 else None


def pus(p, u):
    """Privacy-utility score: harmonic mean of privacy and (clipped) utility retention, x100."""
    if p is None or u is None:
        return None
    u = min(1.0, max(0.0, u))
    return 0.0 if p + u == 0 else 100.0 * 2 * p * u / (p + u)


def f_lambda(p, u, lam=0.5):
    """Ilic et al.'s linear trade-off on the harness's normalised scores, x100."""
    if p is None or u is None:
        return None
    return 100.0 * ((1 - lam) * min(1.0, max(0.0, u)) + lam * p)


def pareto(points):
    """Indices of points (u, p) not dominated by another (higher-or-equal on both, higher on one)."""
    keep = []
    for i, (u, p) in enumerate(points):
        if not any((u2 >= u and p2 >= p) and (u2 > u or p2 > p) for j, (u2, p2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def hypervolume(points):
    """Area dominated by the (u, p) points inside [0, 1]^2, reference corner (0, 0)."""
    pts = sorted({(min(1.0, max(0.0, u)), min(1.0, max(0.0, p))) for u, p in points}, reverse=True)
    area, best_p = 0.0, 0.0
    for i, (u, p) in enumerate(pts):
        best_p = max(best_p, p)
        nxt = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        area += (u - nxt) * best_p
    return area


def label_attacker(task_tr, y_tr, task_te, kind, n_out):
    """The task-label attacker: P(secret | task label) estimated on the training items with add-one smoothing ->
    probabilities for the test items.
    task labels: int class (multiclass task) or 0/1 vector (multilabel task: the attacker conditions on the
    exact label set, falling back to the marginal for unseen sets)."""
    key = (lambda v: int(v)) if np.ndim(task_tr[0]) == 0 else (lambda v: tuple(int(x) for x in v))
    groups = {}
    for t, y in zip(task_tr, y_tr):
        groups.setdefault(key(t), []).append(y)
    if kind == "multiclass":
        marg = (np.bincount(y_tr, minlength=n_out) + 1.0) / (len(y_tr) + n_out)
        table = {k: (np.bincount(v, minlength=n_out) + 1.0) / (len(v) + n_out) for k, v in groups.items()}
    else:
        y_tr = np.asarray(y_tr, float)
        marg = (y_tr.sum(0) + 1.0) / (len(y_tr) + 2.0)
        table = {k: (np.asarray(v, float).sum(0) + 1.0) / (len(v) + 2.0) for k, v in groups.items()}
    return np.stack([table.get(key(t), marg) for t in task_te])
