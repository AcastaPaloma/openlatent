#!/usr/bin/env python
"""Protocol-2 aggregation (held-out classes, retention, unseen centres, matched-dimensionality LDA arm).

  python aggregate_p2.py --runs DIR [DIR ...] --out table.md [--B 1000]

For every (encoder, dataset, variant in {holdout, crossed}):
  held-class utility  = balanced accuracy of a linear probe retrained on the released representation (B units), restricted to the classes the
                        adapter never saw, scored on sealed C (and on the benchmark's unseen centres).  Variant 'crossed' (Camelyon: 2 classes) has no held-out classes.
  leak L              = (worst-case retrained centre attacker - floor) / (raw - floor), worst over linear/mlp/knn/tfm (selection families); 'all' adds LightGBM, RFF-SVM.
  retention           = fraction of the 10 raw-space nearest B-neighbours of a C patch that are among its 30 nearest B-neighbours in the released space
                        (xc: gallery restricted to other centres).
Every learned configuration is compared with the LDA arm of the same output dimension (plain LDA if dim <= C_seen-1, else LDA + top residual PCs).
Paired unit bootstrap over the sealed test units C (same resample for all configs).  FOCUS kill test: a learned adapter 'wins' if it beats the matched LDA arm
on held-class utility (i) or retention (ii) with the CI of the difference above 0, while L_worst is within +0.05 of the LDA arm.
Selection of the reported configuration per family uses V only.
"""
import argparse, json
from pathlib import Path

import numpy as np

KC = ("linear", "mlp", "knn", "tfm")
KX = ("lgbm", "rff")
LEARNED = ("vib", "vq", "dann", "adv_retrain", "supcon_xc", "ae")


def family(c):
    return c.split("__")[0]


def ci(b):
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def fmt(pt, b, nd=3):
    lo, hi = ci(b)
    return f"{pt:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


class Units:
    def __init__(self, units, B, seed=0):
        self.u = np.unique(units)
        self.ix = np.searchsorted(self.u, units)
        rng = np.random.RandomState(seed)
        n = len(self.u)
        self.W = np.stack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(B)]).astype(float)

    def bacc(self, truth, pred, mask=None):
        mask = np.ones(len(truth), bool) if mask is None else mask
        t, p, ix = truth[mask], pred, self.ix[mask]
        K = int(t.max()) + 1
        cor, tot = np.zeros((len(self.u), K)), np.zeros((len(self.u), K))
        np.add.at(tot, (ix, t), 1)
        np.add.at(cor, (ix, t), (p == t).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            per = (self.W @ cor) / (self.W @ tot)
        pt = (cor.sum(0) / np.maximum(tot.sum(0), 1))[tot.sum(0) > 0].mean()
        return pt, np.nanmean(per, 1)

    def mean(self, vals):
        s, c = np.zeros(len(self.u)), np.zeros(len(self.u))
        np.add.at(s, self.ix, vals.astype(float))
        np.add.at(c, self.ix, 1)
        return s.sum() / c.sum(), (self.W @ s) / (self.W @ c)


def load(dirs):
    recs = []
    for d in dirs:
        for jl in Path(d).glob("*.jsonl"):
            for line in jl.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); r["_dir"] = str(d); r["_tag"] = jl.stem; recs.append(r)
    best = {}
    for r in recs:
        best[(r["encoder"], r["dataset"], r.get("variant", "crossed"), r["seed"], r["cfg"])] = r
    return list(best.values())


class Cell:
    def __init__(self, enc, ds, var, recs, dirs, B):
        self.enc, self.ds, self.var = enc, ds, var
        self.by = {}
        for r in recs:
            if (r["encoder"], r["dataset"], r.get("variant", "crossed")) == (enc, ds, var):
                self.by.setdefault(r["cfg"], {})[r["seed"]] = r
        suffix = "" if var == "crossed" else f"__{var}"
        arr = None
        for d in dirs:
            f = next(iter(Path(d).glob(f"{enc}__{ds}__seed[0-9]{suffix}.splitarrays.npz")), None)
            if f is not None:
                arr = np.load(f); break
        self.y, self.c, self.u = arr["y"], arr["c"], arr["u"]
        self.iC = arr["idx_C"]
        self.iO = np.concatenate([arr["idx_OOD1"], arr["idx_OOD2"]])
        self.UC, self.UO = Units(self.u[self.iC], B), Units(self.u[self.iO], B, 1)
        self.K = int(self.y.max()) + 1
        self.seen = list(range((self.K + 1) // 2))
        self.held = [k for k in range(self.K) if k not in self.seen]
        self._p = {}

    def p(self, r):
        k = (r["_dir"], r["_tag"], r["cfg"])
        if k not in self._p:
            self._p[k] = np.load(Path(r["_dir"]) / "preds" / f"{r['_tag']}__{r['cfg']}.npz")
        return self._p[k]

    def seeds(self, cfg):
        return sorted(self.by[cfg])

    def centre(self, cfg, kinds):
        pts, bs = [], []
        for s in self.seeds(cfg):
            p = self.p(self.by[cfg][s])
            res = [self.UC.bacc(self.c[self.iC], p[f"centre/{k}/C"].astype(int)) for k in kinds if f"centre/{k}/C" in p.files]
            pts.append(max(r[0] for r in res)); bs.append(np.max(np.stack([r[1] for r in res]), 0))
        return float(np.mean(pts)), np.mean(bs, 0)

    def label(self, cfg, nm, kind="linear", where="C"):
        """nm in {'', 'seen', 'held'}; where in {'C', 'OOD'}"""
        pts, bs = [], []
        U, idx = (self.UC, self.iC) if where == "C" else (self.UO, self.iO)
        for s in self.seeds(cfg):
            p = self.p(self.by[cfg][s])
            key = f"label{nm}/{kind}/{where}"
            if key not in p.files:
                return float("nan"), np.full(len(self.UC.W), np.nan)
            pred = p[key].astype(int)
            if nm == "":
                a, b = U.bacc(self.y[idx], pred)
            else:
                cl = self.seen if nm == "seen" else self.held
                mask = np.isin(self.y[idx], cl)
                t = np.array([cl.index(k) for k in self.y[idx][mask]])
                a, b = self._sub(U, idx, mask, t, pred)
            pts.append(a); bs.append(b)
        return float(np.mean(pts)), np.mean(bs, 0)

    @staticmethod
    def _sub(U, idx, mask, t, pred):
        ix = U.ix[mask]
        K = int(t.max()) + 1
        cor, tot = np.zeros((len(U.u), K)), np.zeros((len(U.u), K))
        np.add.at(tot, (ix, t), 1)
        np.add.at(cor, (ix, t), (pred == t).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            per = (U.W @ cor) / (U.W @ tot)
        return (cor.sum(0) / np.maximum(tot.sum(0), 1))[tot.sum(0) > 0].mean(), np.nanmean(per, 1)

    def ret(self, cfg, key):
        pts, bs = [], []
        for s in self.seeds(cfg):
            p = self.p(self.by[cfg][s])
            if f"ret/{key}" not in p.files:
                return float("nan"), np.full(len(self.UC.W), np.nan)
            a, b = self.UC.mean(p[f"ret/{key}"].astype(float)); pts.append(a); bs.append(b)
        return float(np.mean(pts)), np.mean(bs, 0)

    def vmean(self, cfg, fn):
        v = [fn(r["metrics"]) for r in self.by[cfg].values()]
        return float(np.mean(v))

    def floor(self):
        r = list(self.by["raw"].values())
        return max(np.mean([x["metrics"]["floor/centre_label_only/C"] for x in r]), np.mean([x["metrics"]["chance/centre"] for x in r]))

    def dim(self, cfg):
        return int(list(self.by[cfg].values())[0]["metrics"]["dim"])


def matched_lda(cell, d):
    cands = {int(c.split("__d")[-1]): c for c in cell.by if family(c) == "lda_pca"}
    if "lda" in cell.by and cell.dim("lda") >= d:
        return "lda"
    if d in cands:
        return cands[d]
    ge = sorted(k for k in cands if k >= d)
    return cands[ge[0]] if ge else ("lda" if "lda" in cell.by else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    recs = load(a.runs)
    cells = sorted({(r["encoder"], r["dataset"], r.get("variant", "crossed")) for r in recs if "raw" == r["cfg"] and any(x["cfg"].startswith("lda_pca") for x in recs if x["encoder"] == r["encoder"] and x["dataset"] == r["dataset"])})
    L = []
    wins = []
    for enc, ds, var in cells:
        cell = Cell(enc, ds, var, recs, a.runs, a.B)
        if "raw" not in cell.by:
            continue
        floor = cell.floor()
        raw_c = cell.centre("raw", KC)
        raw_all = cell.centre("raw", KC + KX)
        seeds = cell.seeds("raw")
        held_mode = var != "crossed" and "holdout" in var
        L.append(f"\n### {enc} / {ds} / {var}  (seeds {seeds}{'; PARTIAL, single seed: not yet replicated' if len(seeds) < 3 else ''}; centre chance {1/ (int(cell.c[cell.iC].max())+1):.3f}, floor {floor:.3f}; "
                 f"{'seen classes ' + str(cell.seen) + ', held-out classes ' + str(cell.held) if held_mode else 'all classes seen'})\n")
        tab = {}
        for cfg in sorted(cell.by):
            if cfg.startswith("ctrl_"):
                continue
            wc = cell.centre(cfg, KC)
            wa = cell.centre(cfg, KC + KX)
            Lb = (wc[1] - floor) / np.maximum(raw_c[1] - floor, 1e-6)
            Lp = (wc[0] - floor) / max(raw_c[0] - floor, 1e-6)
            La = (wa[0] - floor) / max(raw_all[0] - floor, 1e-6)
            t = dict(dim=cell.dim(cfg), L=(Lp, Lb), La=La, ret=cell.ret(cfg, "knn_xc"), reta=cell.ret(cfg, "knn_all"),
                     lab=cell.label(cfg, "", "linear", "C"), labO=cell.label(cfg, "", "linear", "OOD"))
            if held_mode:
                t.update(seen=cell.label(cfg, "seen", "linear", "C"), held=cell.label(cfg, "held", "linear", "C"),
                         heldO=cell.label(cfg, "held", "linear", "OOD"), heldm=cell.label(cfg, "held", "mlp", "C"))
            rs = list(cell.by[cfg].values())
            t["rate"] = np.mean([r["extra"]["rate_bits_per_patch"] for r in rs]) if "rate_bits_per_patch" in rs[0]["extra"] else float("nan")
            t["V_held"] = cell.vmean(cfg, lambda m: m.get("labelheld/linear/V", m["label/linear/V"]))
            t["V_L"] = cell.vmean(cfg, lambda m: max(m[f"centre/{k}/V"] for k in KC))
            t["V_ret"] = cell.vmean(cfg, lambda m: m.get("retV/knn_xc", float("nan")))
            tab[cfg] = t
        # --- main table
        hdr = "| config | dim | rate (bits) | "
        hdr += ("seen-class acc | held-class acc [95% CI] | held-class acc, unseen centres | " if held_mode else "label acc | label acc, unseen centres | ")
        hdr += "L_worst [95% CI] | L (+LightGBM, RFF) | retention xc [95% CI] | retention all |"
        L.append(hdr)
        L.append("|" + "---|" * (hdr.count("|") - 1))
        order = sorted(tab, key=lambda c: ({"raw": 0, "centre_whiten": 1, "leace": 2, "lda": 3, "lda_pca": 4}.get(family(c), 5), family(c), tab[c]["dim"], c))
        for c in order:
            t = tab[c]
            if held_mode:
                u_ = f"{t['seen'][0]:.3f} | {fmt(*t['held'][:2])} | {t['heldO'][0]:.3f} | "
            else:
                u_ = f"{t['lab'][0]:.3f} | {t['labO'][0]:.3f} | "
            L.append(f"| {c} | {t['dim']} | {t['rate']:.1f} | {u_}{fmt(t['L'][0], t['L'][1], 2)} | {t['La']:.2f} | {fmt(*t['ret'][:2])} | {t['reta'][0]:.3f} |")
        # --- comparison with the matched LDA arm
        L.append("\nLearned configurations minus the matched-dimensionality LDA arm (sealed C; paired unit bootstrap; positive utility/retention delta = better, positive L delta = more leakage):\n")
        L.append("| learned config | dim | matched LDA arm | Δ " + ("held-class acc" if held_mode else "label acc") + f" [95% CI] | Δ retention xc [95% CI] | Δ L_worst [95% CI] | meets criterion (i)/(ii) at L within +{a.tol:.2f}? |")
        L.append("|---|---|---|---|---|---|---|")
        flags = {}
        for c in order:
            if family(c) not in LEARNED:
                continue
            m_ = matched_lda(cell, tab[c]["dim"])
            if m_ is None or m_ not in tab:
                continue
            ut = "held" if held_mode else "lab"
            du = (tab[c][ut][0] - tab[m_][ut][0], tab[c][ut][1] - tab[m_][ut][1])
            dr = (tab[c]["ret"][0] - tab[m_]["ret"][0], tab[c]["ret"][1] - tab[m_]["ret"][1])
            dL = (tab[c]["L"][0] - tab[m_]["L"][0], tab[c]["L"][1] - tab[m_]["L"][1])
            okL = dL[0] <= a.tol
            wu = ci(du[1])[0] > 0
            wr = ci(dr[1])[0] > 0
            flag = ("(i) utility " if wu else "") + ("(ii) retention " if wr else "")
            flags[c] = (okL and (wu or wr), flag.strip(), m_, du, dr, dL)
            L.append(f"| {c} | {tab[c]['dim']} | {m_} | {fmt(*du)} | {fmt(*dr)} | {fmt(*dL, nd=2)} | {'YES: ' + flag if (okL and (wu or wr)) else ('no (L worse by %.2f)' % dL[0] if not okL else 'no')} |")
        # --- exploratory: same comparison against the best LDA-family / label-free point at <= equal leakage (not dimension matched)
        L.append(f"\nExploratory, leakage-matched: each learned config minus the BEST (by point estimate on C, so optimistic for the baseline) reference with L_worst <= L(config)+{a.tol:.2f}; "
                 "refs = LDA family (lda, lda_pca all d) and label-free floors (pca, randproj, centre_whiten). Positive = learned better.\n")
        L.append("| learned config | L | best LDA-family ref | Δ " + ("held-class acc" if held_mode else "label acc") + " [95% CI] | best label-free ref | Δ " + ("held-class acc" if held_mode else "label acc") + " [95% CI] | best LDA-family ref (retention) | Δ retention xc [95% CI] | best label-free ref (retention) | Δ retention xc [95% CI] |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        ut = "held" if held_mode else "lab"
        frontier_wins = {}
        for c in order:
            if family(c) not in LEARNED:
                continue
            row = [f"| {c} | {tab[c]['L'][0]:.2f} "]
            res = []
            for fams in (("lda", "lda_pca"), ("pca", "randproj", "centre_whiten")):
                refs = [x for x in tab if family(x) in fams and tab[x]["L"][0] <= tab[c]["L"][0] + a.tol]
                for key in (ut, "ret"):
                    if not refs:
                        row.append("| (none at this leakage) | n/a "); res.append(None); continue
                    b = max(refs, key=lambda x: tab[x][key][0])
                    d_ = (tab[c][key][0] - tab[b][key][0], tab[c][key][1] - tab[b][key][1])
                    row.append(f"| {b} | {fmt(*d_)} "); res.append(ci(d_[1])[0] > 0)
            frontier_wins[c] = res
            # reorder columns to header order: LDA util, LF util, LDA ret, LF ret
            cells_ = row[1:]
            order_idx = [0, 1, 4, 5, 2, 3, 6, 7]
            L.append(row[0] + "".join(cells_[i] for i in order_idx) + "|")
        # --- V-selected per family (the official line)
        L.append(f"\nPer family, configuration selected on V (L_V within +{a.tol:.2f} of the matched LDA arm; then max V {'held-class acc' if held_mode else 'retention xc'}):\n")
        L.append("| family | config | matched LDA | Δ " + ("held-class acc" if held_mode else "label acc") + " [95% CI] | Δ retention xc [95% CI] | Δ L_worst | wins? |")
        L.append("|---|---|---|---|---|---|---|")
        for fam in LEARNED:
            cand = []
            for c in tab:
                if family(c) != fam or c not in flags:
                    continue
                m_ = flags[c][2]
                if tab[c]["V_L"] <= tab[m_]["V_L"] + a.tol:
                    cand.append((tab[c]["V_held"] if held_mode else tab[c]["V_ret"], c))
            if not cand:
                L.append(f"| {fam} | (no config within the leakage tolerance on V) | | | | | no |")
                continue
            _, c = max(cand)
            ok, flag, m_, du, dr, dL = flags[c]
            wins.append((enc, ds, var, fam, c, ok, flag))
            L.append(f"| {fam} | {c} | {m_} | {fmt(*du)} | {fmt(*dr)} | {dL[0]:+.2f} | {'YES: ' + flag if ok else 'no'} |")
        # controls
        ct = []
        for c in sorted(cell.by):
            if c.startswith("ctrl_"):
                w = cell.centre(c, KC)
                ct.append(f"{c} {w[0]:.2f}")
        if ct:
            L.append("\nControls (worst of the 4 attackers, centre bacc on C): " + "; ".join(ct))
    L.append("\n### Kill-test tally (FOCUS A, revised): learned families (selected on V) beating the matched LDA arm on (i) held-class utility or (ii) retention with L within tolerance\n")
    L.append("| encoder | dataset | variant | family | config | wins |")
    L.append("|---|---|---|---|---|---|")
    for e, d, v, f, c, ok, fl in wins:
        L.append(f"| {e} | {d} | {v} | {f} | {c} | {'YES: ' + fl if ok else 'no'} |")
    text = "\n".join(L)
    Path(a.out).write_text(text) if a.out != "-" else print(text)


if __name__ == "__main__":
    main()
