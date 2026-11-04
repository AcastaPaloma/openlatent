"""Deterministic, releasable suite certificate.

Contract
--------
* A suite is DECLARED before any sealed number exists: an ordered tuple of attacker names, the metric, alpha, bootstrap size and seed
  (`SuiteDecl`). Its digest is part of the certificate. `certify` refuses predictions that omit a declared attacker or carry an
  undeclared one (no post-hoc additions, no silent drops).
* Inputs are saved predictions of every declared attacker on two unit-disjoint splits: `val` (used only to pick the reported attacker)
  and `test` (sealed: touched once). Each item belongs to an independent unit (`cluster`: recording, subject, patient, ...).
* Output is a plain dict (JSON-serialisable, no clocks, no paths). Every reported number sits next to its MDE80 (2.8 x SE).

What it states (and what it does not)
-------------------------------------
1. `selected`: the attacker with the best validation metric, scored once on the sealed test, cluster-bootstrap CI. Honest point estimate
   for THAT attacker; it can under-state the best attacker of the suite.
2. `suite_bound`: Bonferroni (Clopper-Pearson on effective counts at alpha/K) and max-statistic bootstrap upper bounds for every declared
   attacker simultaneously; the max over the suite is an upper bound on the best DECLARED attacker's true score. It says nothing about
   attackers outside the suite (suite-conditional).
3. `floor`: the smallest suite bound the design can ever return (observed rate equals chance), i.e. the resolution of the sealed test.
4. `verdict` in {LEAK, BELOW_MARGIN, NOT_RESOLVED}; BELOW_MARGIN uses the Bonferroni bound only (max-t is reported beside it).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import stats as st

VERSION = "s_suite/1"


@dataclass(frozen=True)
class SuiteDecl:
    name: str
    attackers: Tuple[str, ...]
    metric: str = "accuracy"      # 'accuracy' (probs (n,C), top-1) or 'auc' (scores (n,), binary y)
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    margin: float = 0.02          # pre-declared: "below margin" means suite bound <= baseline + margin
    n_classes: Optional[int] = None

    def digest(self) -> str:
        d = asdict(self)
        d["attackers"] = list(self.attackers)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


@dataclass
class SplitPreds:
    y: np.ndarray                      # (n,) integer labels of the secret
    cluster: np.ndarray                # (n,) independent-unit ids
    scores: Dict[str, np.ndarray] = field(default_factory=dict)  # attacker -> probs (n,C) or scores (n,)

    def __post_init__(self):
        self.y = np.asarray(self.y)
        self.cluster = np.asarray(self.cluster)
        n = len(self.y)
        if len(self.cluster) != n:
            raise ValueError("cluster and y differ in length")
        for k, v in self.scores.items():
            if len(v) != n:
                raise ValueError(f"attacker {k}: {len(v)} rows for {n} items")

    def digest(self, names=None) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.y, dtype=np.int64).tobytes())
        h.update("|".join(self.cluster.astype(str)).encode())
        for k in (names if names is not None else sorted(self.scores)):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.scores[k], dtype=np.float32).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------

def _check_suite(decl: SuiteDecl, split: SplitPreds, which: str):
    got, want = set(split.scores), set(decl.attackers)
    if got != want:
        raise ValueError(f"{which}: attackers differ from the declaration; missing {sorted(want - got)}, undeclared {sorted(got - want)}")
    if len(set(decl.attackers)) != len(decl.attackers):
        raise ValueError("duplicate attacker names in declaration")


def item_values(metric: str, y, arr):
    """accuracy: 0/1 correctness per item from probs (n,C); auc: raw score (n,)."""
    arr = np.asarray(arr)
    if metric == "accuracy":
        return (arr.argmax(1) == np.asarray(y)).astype(float)
    if metric == "auc":
        return arr.astype(float).reshape(len(y))
    raise ValueError(metric)


def metric_point(metric, y, arr):
    v = item_values(metric, y, arr)
    return float(v.mean()) if metric == "accuracy" else st.auc_point(y, v)


def _boot_all(decl: SuiteDecl, split: SplitPreds, W, inv, m):
    """-> point (K,), boot (B,K), n_c (m,), S (m,K) [accuracy only]."""
    K = len(decl.attackers)
    if decl.metric == "accuracy":
        C = np.stack([item_values("accuracy", split.y, split.scores[a]) for a in decl.attackers], axis=1)
        S = st.unit_sums(C, inv, m)
        n_c = np.bincount(inv, minlength=m).astype(float)
        return st.ratio_point(S, n_c), st.ratio_boot(S, n_c, W), n_c, S
    point = np.zeros(K)
    boot = np.zeros((W.shape[0], K))
    for k, a in enumerate(decl.attackers):
        v = item_values("auc", split.y, split.scores[a])
        point[k] = st.auc_point(split.y, v)
        boot[:, k] = st.auc_boot(split.y, v, inv, W)
    return point, boot, np.bincount(inv, minlength=m).astype(float), None


def _ucb(decl, p, se, n_items, alpha, n_units):
    if decl.metric == "accuracy":
        return st.ucb_effective(p, se, n_items, alpha, n_units)
    return float(p + st.onesided_z(alpha) * se)


def _chance(decl: SuiteDecl, test: SplitPreds):
    if decl.metric == "auc":
        return dict(uniform=0.5, majority_prior=0.5, used=0.5)
    C = decl.n_classes or int(next(iter(test.scores.values())).shape[1])
    cnt = np.bincount(test.y.astype(int), minlength=C)
    maj = float(cnt.max() / cnt.sum())
    uni = 1.0 / C
    return dict(uniform=uni, majority_prior=maj, used=max(uni, maj))


def certify(decl: SuiteDecl, val: SplitPreds, test: SplitPreds, label_only: Optional[np.ndarray] = None,
            reference: Optional[float] = None) -> dict:
    """Build the certificate. `label_only`: predictions (same format as one attacker) of a model that uses only the public/utility
    label (or nothing); evaluated on the test and used as baseline if above chance. `reference`: ceiling attacker value (e.g. the raw
    release) used to normalise leakage; default 1.0."""
    _check_suite(decl, val, "val")
    _check_suite(decl, test, "test")
    names = list(decl.attackers)
    K = len(names)
    inv, m = st.unit_index(test.cluster)
    n_items = len(test.y)
    W = st.boot_weights(m, decl.n_boot, decl.seed)
    point, boot, n_c, S = _boot_all(decl, test, W, inv, m)
    se = boot.std(0, ddof=1)
    val_point = np.array([metric_point(decl.metric, val.y, val.scores[a]) for a in names])
    ch = _chance(decl, test)
    base = ch["used"]

    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    a1, aK = decl.alpha, decl.alpha / K
    ucb1 = np.array([_ucb(decl, point[k], se[k], n_items, a1, m) for k in range(K)])
    ucbK = np.array([_ucb(decl, point[k], se[k], n_items, aK, m) for k in range(K)])

    # max-statistic (studentised) bootstrap over the declared suite
    ok = se > 1e-12
    if ok.any():
        T = ((boot[:, ok] - point[ok]) / se[ok]).max(1)
        q = float(np.quantile(T, 1 - decl.alpha))
        ucb_maxt = np.where(ok, point + q * se, point)
    else:
        q = 0.0
        ucb_maxt = point.copy()
    corr = np.corrcoef(boot.T) if K > 1 and ok.sum() > 1 else np.ones((1, 1))
    off = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.shape[0] > 1 else np.array([1.0])

    ksel = int(np.argmax(val_point))  # first maximum on ties = declaration order
    per = {}
    for k, a in enumerate(names):
        per[a] = dict(val=val_point[k], test=point[k], se=se[k], ci95=[lo[k], hi[k]], ucb_onesided=ucb1[k], ucb_bonferroni=ucbK[k],
                      ucb_maxt=ucb_maxt[k], mde80=st.mde80(se[k]),
                      design_effect=st.design_effect(se[k], point[k], n_items) if decl.metric == "accuracy" else None)
    kn = int(np.argmax(point))

    # label-only baseline (paired bootstrap against the selected attacker for the excess)
    lab = None
    if label_only is not None:
        lv = item_values(decl.metric, test.y, label_only)
        if decl.metric == "accuracy":
            Sl = st.unit_sums(lv, inv, m)
            lpoint = float(Sl.sum() / n_c.sum())
            lboot = st.ratio_boot(Sl, n_c, W)[:, 0]
        else:
            lpoint = st.auc_point(test.y, lv)
            lboot = st.auc_boot(test.y, lv, inv, W)
        lse = float(lboot.std(ddof=1))
        lab = dict(test=lpoint, se=lse, ci95=list(st.percentile_ci(lboot)), mde80=st.mde80(lse))
        base = max(base, lpoint)
        dsel = boot[:, ksel] - lboot
        dlo = float(np.quantile(dsel, decl.alpha))  # one-sided lower bound on selected minus label-only
        dse = float(dsel.std(ddof=1))
    else:
        dsel = boot[:, ksel] - base
        dlo = float(np.quantile(dsel, decl.alpha))
        dse = float(dsel.std(ddof=1))
    ref = 1.0 if reference is None else float(reference)
    denom = ref - base if ref - base > 1e-9 else float("nan")

    def norm_leak(x):
        return (x - base) / denom

    sel = dict(name=names[ksel], rank_rule="argmax validation metric, ties -> declaration order", val=val_point[ksel], test=point[ksel],
               se=se[ksel], ci95=[lo[ksel], hi[ksel]], ucb_onesided=ucb1[ksel], mde80=st.mde80(se[ksel]),
               excess_over_baseline=point[ksel] - base, excess_lcb_onesided=dlo, excess_mde80=st.mde80(dse),
               normalised_leakage=norm_leak(point[ksel]),
               normalised_leakage_ci95=[norm_leak(lo[ksel]), norm_leak(hi[ksel])],
               normalised_leakage_ucb=norm_leak(ucb1[ksel]))

    ub_bonf = float(ucbK.max())
    ub_maxt = float(ucb_maxt.max())
    floor = dict(ucb_at_baseline_k1=st.zero_leak_floor(base, m, decl.alpha, 1),
                 ucb_at_baseline_suite=st.zero_leak_floor(base, m, decl.alpha, K), n_units=m,
                 note="smallest suite bound this sealed test can return (observed rate == baseline); resolution of the design")
    suite = dict(k=K, alpha=decl.alpha, bonferroni=ub_bonf, maxt=ub_maxt, z_bonferroni=st.onesided_z(aK), q_maxt=q,
                 mean_boot_corr=float(off.mean()), argmax_bonferroni=names[int(np.argmax(ucbK))],
                 excess_bonferroni=ub_bonf - base, excess_maxt=ub_maxt - base,
                 normalised_leakage_ucb_bonferroni=norm_leak(ub_bonf), normalised_leakage_ucb_maxt=norm_leak(ub_maxt),
                 mde80_selected=st.mde80(se[ksel]),
                 note="suite-conditional: bounds the best DECLARED attacker's true score, silent on attackers outside the suite")

    if dlo > 0:
        verdict = "LEAK"
    elif ub_bonf <= base + decl.margin:
        verdict = "BELOW_MARGIN"
    else:
        verdict = "NOT_RESOLVED"

    payload = dict(
        version=VERSION, declaration={**asdict(decl), "attackers": names}, decl_digest=decl.digest(),
        data=dict(n_test_items=n_items, n_test_units=m, n_val_items=len(val.y), n_val_units=st.unit_index(val.cluster)[1],
                  val_digest=val.digest(names), test_digest=test.digest(names)),
        chance=ch, label_only=lab, baseline_used=base, reference=ref,
        attackers=per, naive_max=dict(name=names[kn], test=point[kn], note="max over the suite on the sealed test; comparison only, optimistic"),
        selected=sel, suite_bound=suite, floor=floor,
        verdict=dict(label=verdict, below_margin_maxt=bool(ub_maxt <= base + decl.margin),
                     rule="LEAK if one-sided lower bound of (selected - baseline) > 0; BELOW_MARGIN if the Bonferroni suite bound "
                     f"<= baseline + margin ({decl.margin}); else NOT_RESOLVED. The max-t bound is reported, not used for the verdict"),
        controls=None, valid=True)
    return finalize(payload)


def finalize(payload: dict) -> dict:
    p = dict(payload)
    p.pop("digest", None)
    p = st.r6(p)
    p["digest"] = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return p


def to_json(cert: dict) -> str:
    return json.dumps(cert, sort_keys=True, indent=1)


def attach_controls(cert: dict, controls: Dict[str, dict]) -> dict:
    """controls: name -> certificate computed by the same pipeline on a constant / shuffled-label / black-input release. A control
    passes iff it does not read LEAK. Any failing control invalidates the certificate (SynthGuard-style exclusion)."""
    block = {}
    for name, c in controls.items():
        block[name] = dict(verdict=c["verdict"]["label"], selected_test=c["selected"]["test"], baseline=c["baseline_used"],
                           excess_lcb_onesided=c["selected"]["excess_lcb_onesided"], passes=c["verdict"]["label"] != "LEAK",
                           digest=c["digest"])
    p = {k: v for k, v in cert.items() if k != "digest"}
    p["controls"] = block
    p["valid"] = all(v["passes"] for v in block.values()) if block else True
    return finalize(p)


# ---------------------------------------------------------------------------------------------------------------------------

def paired_difference(decl: SuiteDecl, test_a: SplitPreds, name_a: str, test_b: SplitPreds, name_b: str, n_boot=2000, seed=0) -> dict:
    """Difference in accuracy between attacker `name_a` on release A and `name_b` on release B scored on the SAME units, paired
    cluster bootstrap. The MDE80 is printed next to the difference; `resolved` says whether the 95% CI excludes 0."""
    if decl.metric != "accuracy":
        raise NotImplementedError("paired_difference implemented for accuracy")
    if not (np.array_equal(test_a.y, test_b.y) and np.array_equal(test_a.cluster.astype(str), test_b.cluster.astype(str))):
        raise ValueError("paired comparison requires identical items and units on both releases")
    inv, m = st.unit_index(test_a.cluster)
    ca = item_values("accuracy", test_a.y, test_a.scores[name_a])
    cb = item_values("accuracy", test_b.y, test_b.scores[name_b])
    S = st.unit_sums(ca - cb, inv, m)
    n_c = np.bincount(inv, minlength=m).astype(float)
    W = st.boot_weights(m, n_boot, seed)
    d = float(S.sum() / n_c.sum())
    bs = st.ratio_boot(S, n_c, W)[:, 0]
    se = float(bs.std(ddof=1))
    lo, hi = st.percentile_ci(bs)
    return st.r6(dict(diff=d, se=se, mde80=st.mde80(se), ci95=[lo, hi], resolved=bool(lo > 0 or hi < 0), n_units=m))
