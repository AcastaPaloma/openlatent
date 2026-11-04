"""SC-0 aggregation: unit bootstrap tables, controls, power audit (SC-K0), kill conditions SC-K0..K3.  CPU only.

  python aggregate.py --dataset hlca --runs ~/work/runs/e_floors_hlca_s0 ~/work/runs/e_floors_hlca_s1 ... [--B 1000] [--json out.json] [--md out.md]

Metrics follow results_notes/E.md section 2.4 exactly.  Bootstrap resamples UNITS (HLCA donors; Immune/Pancreas batches = descriptive) and
recomputes every metric from the stored per-cell predictions; the same resamples are shared by all arms and seeds (paired).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATASETS, MIN_HELD, MIN_TRAIN  # noqa: E402

KINDS = ("linear", "mlp", "knn", "lgbm")
PROBES = ("lin", "knn")
import re  # noqa: E402
_IND_RE = re.compile(r"^(raw50|pca\d+|rp\d+|noise_r[\d.]+|dith\d+_r[\d.]+|ldaC|ldaC_pca\d+|leace|leace_pca16)$")
INDUCTIVE = ["raw50", "pca8", "pca16", "pca32", "rp16", "rp32", "noise_r0.5", "noise_r1", "noise_r2", "dith16_r0.5", "dith16_r1",
             "ldaC", "ldaC_pca16", "ldaC_pca32", "leace", "leace_pca16"]      # K3 selection set (full-fit versions; ldaF16 and *_H are not candidates)
REFS = ["T_centre", "T_bwhiten16", "T_harmony", "T_scvi"]
POWER_ARMS = ["pca16", "rp16", "dith16_r0.5", "ldaC_pca16"]


class Data:
    def __init__(self, ds, run_dirs):
        self.ds = ds
        d = CACHE / ds
        self.obs = pd.read_pickle(d / "obs.pkl")
        meta = json.loads((d / "meta.json").read_text())
        self.meta = meta
        self.n = len(self.obs)
        self.b = self.obs["b"].to_numpy()
        self.unit = self.obs["unit"].to_numpy()
        self.fine = self.obs["fine"].to_numpy()
        self.study = self.obs["study"].astype(str).to_numpy()
        self.fine_names = meta["fine_names"]
        self.held_codes = np.array([self.fine_names.index(t) for t in meta["held_types"]])
        name2code = {x: i for i, x in enumerate(meta["batches"])}
        self.folds = [np.array([name2code[x] for x in f]) for f in DATASETS[ds]["folds"]]
        self.nf = len(self.folds)
        self.fold_of = -np.ones(self.n, np.int64)
        self.yb = -np.ones(self.n, np.int64)
        for f, bs in enumerate(self.folds):
            for i, bb in enumerate(bs):
                m = self.b == bb
                self.fold_of[m] = f
                self.yb[m] = i
        assert (self.fold_of >= 0).all()
        self.Bf = np.array([len(x) for x in self.folds])
        self.cls = self.fold_of * 16 + self.yb
        self.seeds = {}
        for r in run_dirs:
            r = Path(r)
            for p in sorted(r.glob("preds_s*.npz")):
                s = int(p.stem.split("_s")[1])
                self.seeds.setdefault(s, []).append(p)
        self.S = {}
        for s, ps in self.seeds.items():
            self.S[s] = self._load_seed(ps)
        self.evalm = np.zeros(self.n, bool)
        s0 = self.S[min(self.S)]
        for f in range(self.nf):
            ty = s0["types"].get(f)
            if ty is not None:
                m = (self.fold_of == f) & np.isin(self.fine, ty)
                self.evalm |= m
        self.fall = sorted(set.intersection(*[v["folds_done"] for v in self.S.values()]))
        self.units = np.unique(self.unit)
        self.unit_idx = np.searchsorted(self.units, self.unit)
        self.study_of_unit = np.array([self.study[self.unit == u][0] for u in self.units])

    def _load_seed(self, paths):
        out = dict(att={}, pr={}, ret={}, types={}, folds_done=set())
        self.dims = getattr(self, "dims", {})
        for p in paths:
            mp = p.with_name(p.name.replace("preds_", "meta_").replace(".npz", ".json"))
            if mp.exists():
                self.dims.update(json.loads(mp.read_text()).get("dims", {}))
            z = np.load(p)
            for k in z.files:
                parts = k.split("|")
                if parts[0] == "held":
                    out["folds_done"].add(int(parts[1]))
                elif parts[0] == "types":
                    out["types"][int(parts[1])] = z[k].astype(np.int64)
                elif parts[0] == "att":
                    _, arm, f, kind = parts
                    out["att"].setdefault((arm, kind), np.full(self.n, -1, np.int8))
                    held = z[f"held|{f}"]
                    out["att"][(arm, kind)][held] = z[k]
                elif parts[0] == "pr":
                    _, arm, f, kind = parts
                    out["pr"].setdefault((arm, kind), np.full(self.n, -1, np.int16))
                    out["pr"][(arm, kind)][z[f"held|{f}"]] = z[k]
                elif parts[0] == "ret":
                    out["ret"].setdefault(parts[1], {})[int(parts[2])] = float(z[k][0])
        return out

    def arms(self):
        s = self.S[min(self.S)]
        return sorted({a for a, _ in s["pr"]})


# ------------------------------------------------------------------------------------------------ metric kernels
def acc_by_kind(D, S, arm, w, fset):
    """pooled (mean over folds in fset) weighted balanced accuracy of each attacker; seed-averaged.  returns {kind: acc}, chance.
    Classes (held-out batches) with zero weight in the resample are dropped from the accuracy and from chance = mean_f 1/(classes present)."""
    res = {k: [] for k in KINDS}
    den = np.bincount(D.cls, weights=w, minlength=D.nf * 16)
    chance = float(np.mean([1.0 / max(int((den[f * 16: f * 16 + D.Bf[f]] > 0).sum()), 1) for f in fset]))
    for s, sd in S.items():
        for k in KINDS:
            key = (arm, k)
            if key not in sd["att"]:
                continue
            p = sd["att"][key]
            num = np.bincount(D.cls, weights=w * (p == D.yb), minlength=D.nf * 16)
            accs = []
            for f in fset:
                sl = slice(f * 16, f * 16 + D.Bf[f])
                ok = den[sl] > 0
                if ok.any():
                    accs.append(np.mean(num[sl][ok] / den[sl][ok]))
            res[k].append(np.mean(accs))
    return {k: float(np.mean(v)) for k, v in res.items() if v}, chance


def f1_macro(D, p, w, fset, classes=None):
    K = len(D.fine_names)
    m = D.evalm & np.isin(D.fold_of, list(fset)) & (p >= 0)
    t, q = D.fine[m], p[m].astype(np.int64)
    C = np.bincount(t * K + q, weights=w[m], minlength=K * K).reshape(K, K)
    tp = np.diag(C)
    fn = C.sum(1) - tp
    fp = C.sum(0) - tp
    sup = C.sum(1) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    sel = sup if classes is None else sup & np.isin(np.arange(K), classes)
    return float(f1[sel].mean()) if sel.any() else np.nan


def f1_chance(D, w, fset, classes=None):
    """macro-F1 of a classifier that predicts uniformly over the K evaluated types: F1_c = 2 pi_c (1/K) / (pi_c + 1/K), mean over (held-out) classes present"""
    K = len(D.fine_names)
    m = D.evalm & np.isin(D.fold_of, list(fset))
    cnt = np.bincount(D.fine[m], weights=w[m], minlength=K)
    pi = cnt / max(cnt.sum(), 1e-12)
    ktypes = max(int((cnt > 0).sum()), 1)
    f1 = 2 * pi * (1.0 / ktypes) / np.maximum(pi + 1.0 / ktypes, 1e-12)
    sel = cnt > 0
    if classes is not None:
        sel = sel & np.isin(np.arange(K), classes)
    return float(f1[sel].mean())


def u_metrics(D, S, arm, w, fset, hold_arm=None):
    """returns (U_fine, U_hold) = max over the two probes of the seed-averaged macro-F1"""
    uf, uh = [], []
    for kind in PROBES:
        a, b = [], []
        for s, sd in S.items():
            if (arm, kind) not in sd["pr"]:
                continue
            a.append(f1_macro(D, sd["pr"][(arm, kind)], w, fset))
            hk = (hold_arm or arm, kind)
            b.append(f1_macro(D, sd["pr"][hk if hk in sd["pr"] else (arm, kind)], w, fset, classes=D.held_codes))
        if a:
            uf.append(np.mean(a))
            uh.append(np.mean(b))
    return float(np.max(uf)), float(np.max(uh))


def hold_name(arm):
    return arm + "_H" if (arm in ("ldaC", "ldaF16") or arm.startswith("ldaC_pca")) and not arm.endswith("_H") else arm


def all_metrics(D, S, arms, w, fset):
    out = {}
    accs = {}
    for arm in arms:
        a, ch = acc_by_kind(D, S, arm, w, fset)
        accs[arm] = a
        uf, uh = u_metrics(D, S, arm, w, fset, hold_arm=hold_name(arm))
        out[arm] = dict(U=uf, Uh=uh, worst=max(a.values()) if a else np.nan, **{f"acc_{k}": v for k, v in a.items()})
    raw = out["raw50"]["worst"]
    for arm in arms:
        out[arm]["L"] = (out[arm]["worst"] - ch) / max(raw - ch, 1e-6)
    # L of the H versions needs their own attacker numbers (they are arms themselves)
    out["_chance"] = ch
    out["_u0"] = f1_chance(D, w, fset)
    out["_u0h"] = f1_chance(D, w, fset, classes=D.held_codes)
    return out


def resamples(D, B, seed=2026, cluster=False):
    rng = np.random.RandomState(seed)
    nu = len(D.units)
    Ws = []
    for _ in range(B):
        if cluster:
            studies = np.unique(D.study_of_unit)
            pick = rng.choice(studies, size=len(studies), replace=True)
            mult = np.zeros(nu)
            for st in pick:
                us = np.where(D.study_of_unit == st)[0]
                mult[us] += np.bincount(rng.choice(us, size=len(us), replace=True), minlength=nu)[us]
        else:
            mult = np.bincount(rng.randint(nu, size=nu), minlength=nu).astype(float)
        Ws.append(mult[D.unit_idx])
    return Ws


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def fmt(pt, v, nd=3):
    lo, hi = ci(v)
    return f"{pt:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# ------------------------------------------------------------------------------------------------ power audit
def power_audit(D, arms, nperm=2000, B=1000, seed=7):
    S = {min(D.S): D.S[min(D.S)]}
    rng = np.random.RandomState(seed)
    ones = np.ones(D.n)
    fset = list(D.fall)
    Ws = resamples(D, B, seed=11)
    rows = []

    def U_of(p_by_kind, w):
        return max(f1_macro(D, p_by_kind[k], w, fset) for k in PROBES)

    sd = S[min(S)]
    preds = {a: {k: sd["pr"][(a, k)] for k in PROBES} for a in arms}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            d0 = U_of(preds[a], ones) - U_of(preds[b], ones)
            boots = np.array([U_of(preds[a], w) - U_of(preds[b], w) for w in Ws])
            se = float(np.nanstd(boots, ddof=1))
            null = []
            for _ in range(nperm):
                swap = (rng.rand(len(D.units)) < 0.5)[D.unit_idx]
                pa = {k: np.where(swap, preds[b][k], preds[a][k]) for k in PROBES}
                pb = {k: np.where(swap, preds[a][k], preds[b][k]) for k in PROBES}
                null.append(U_of(pa, ones) - U_of(pb, ones))
            crit = float(np.percentile(np.abs(null), 95))      # two-sided 5%: 95th percentile of |dU_null|
            rows.append(dict(a=a, b=b, dU=d0, se=se, crit=crit, mde80=crit + 0.8416 * se))
    return rows


# ------------------------------------------------------------------------------------------------ main
def run(D, B, label=""):
    arms = D.arms()
    S = D.S
    fall = list(D.fall)
    ones = np.ones(D.n)
    pt = all_metrics(D, S, arms, ones, fall)
    Ws = resamples(D, B)
    boot = {a: {k: [] for k in ("U", "Uh", "L", "worst")} for a in arms}
    boot_ch, boot_u0, boot_u0h = [], [], []
    for w in Ws:
        m = all_metrics(D, S, arms, w, fall)
        for a in arms:
            for k in ("U", "Uh", "L", "worst"):
                boot[a][k].append(m[a][k])
        boot_ch.append(m["_chance"])
        boot_u0.append(m["_u0"])
        boot_u0h.append(m["_u0h"])
    return dict(arms=arms, pt=pt, boot=boot, Ws=Ws, boot_u0=boot_u0, boot_u0h=boot_u0h, boot_chance=boot_ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    D = Data(a.dataset, a.runs)
    print(f"{a.dataset}: seeds {sorted(D.S)}, cells {D.n}, units {len(D.units)}, folds done per seed "
          f"{ {s: sorted(v['folds_done']) for s, v in D.S.items()} }", flush=True)
    R = run(D, a.B)
    from verdict import verdict_tables
    md, js = verdict_tables(D, R, a)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
