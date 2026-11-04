#!/usr/bin/env python
"""Aggregate sweep results: unit-level bootstrap on the sealed test split C, selection on V, trade-off tables.

  python aggregate.py --runs DIR [DIR ...] --out table.md [--B 1000]

Definitions (all on balanced accuracy):
  label   = linear label probe retrained on B-set representations, scored on C.
  worst   = max over the four retrained centre attackers (linear, mlp, knn, tfm) scored on C.
  leak    = (worst - floor) / (worst_raw - floor), floor = max(chance, label-only centre baseline);  0 = as hidden as chance/label-only,
            1 = as leaky as the raw embedding.
Bootstrap: resample the C test units with replacement (same resample for all configs and seeds, so differences are paired),
recompute every metric, take the mean over seeds, report the 2.5/97.5 percentiles.  Seeds are listed separately (SD of point estimates).
Selection ('best config of a family at tolerance delta') uses ONLY V: lowest worst-case centre accuracy among configs whose V label accuracy
is within delta of the raw embedding's.  The reported number is then the C number of that config.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KC = ("linear", "mlp", "knn", "tfm")
KL = ("linear", "mlp", "knn")


def load_runs(dirs):
    recs = []
    for d in dirs:
        for jl in Path(d).glob("*.jsonl"):
            for line in jl.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = str(Path(d))
                    r["_tag"] = jl.stem
                    recs.append(r)
    return recs


class Cell:
    """one (encoder, dataset): sealed-test unit table + bootstrap weights"""

    def __init__(self, enc, ds, recs, dirs, B, seed=0):
        self.enc, self.ds = enc, ds
        self.recs = [r for r in recs if r["encoder"] == enc and r["dataset"] == ds]
        arr = None
        for d in dirs:
            f = next(iter(Path(d).glob(f"{enc}__{ds}__seed*.splitarrays.npz")), None)
            if f is not None:
                arr = np.load(f)
                break
        self.idxC = arr["idx_C"]
        self.y, self.c, self.u = arr["y"][self.idxC], arr["c"][self.idxC], arr["u"][self.idxC]
        self.units = np.unique(self.u)
        self.uix = np.searchsorted(self.units, self.u)
        rng = np.random.RandomState(seed)
        n = len(self.units)
        draws = rng.randint(0, n, size=(B, n))
        self.W = np.stack([np.bincount(d, minlength=n) for d in draws]).astype(np.float64)   # (B, n_units)
        self.B = B
        self.pre = {}      # (tag-dir, cfg) -> preds
        self.by = defaultdict(dict)    # cfg -> seed -> record
        for r in self.recs:
            self.by[r["cfg"]][r["seed"]] = r

    def _bacc_boot(self, truth, pred):
        K = int(truth.max()) + 1
        cor = np.zeros((len(self.units), K))
        tot = np.zeros((len(self.units), K))
        np.add.at(tot, (self.uix, truth), 1)
        np.add.at(cor, (self.uix, truth), (pred == truth).astype(float))
        num, den = self.W @ cor, self.W @ tot            # (B,K)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = num / den
        point = (cor.sum(0) / np.maximum(tot.sum(0), 1))[tot.sum(0) > 0].mean()
        return point, np.nanmean(per, axis=1)

    def preds(self, r):
        key = (r["_dir"], r["_tag"], r["cfg"])
        if key not in self.pre:
            f = Path(r["_dir"]) / "preds" / f"{r['_tag']}__{r['cfg']}.npz"
            self.pre[key] = np.load(f)
        return self.pre[key]

    def metric(self, cfg, what, kinds):
        """returns point (mean over seeds), boot (B,), per-seed points.  what in {'centre','label'}; takes max over kinds for 'worst'."""
        seeds = sorted(self.by[cfg])
        pts, boots = [], []
        for s in seeds:
            r = self.by[cfg][s]
            p = self.preds(r)
            truth = self.c if what == "centre" else self.y
            pk, bk = [], []
            for k in kinds:
                a, b = self._bacc_boot(truth, p[f"{what}/{k}/C"].astype(int))
                pk.append(a); bk.append(b)
            pts.append(max(pk)); boots.append(np.max(np.stack(bk), 0))
        return float(np.mean(pts)), np.mean(np.stack(boots), 0), pts

    def floor(self, cfg="raw"):
        v = [self.by[cfg][s]["metrics"]["floor/centre_label_only/C"] for s in self.by[cfg]]
        ch = [self.by[cfg][s]["metrics"]["chance/centre"] for s in self.by[cfg]]
        return float(max(np.mean(v), np.mean(ch)))


def ci(b):
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def fmt(pt, b, nd=3):
    lo, hi = ci(b)
    return f"{pt:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def vmetrics(cell, cfg):
    """V-split (selection) numbers, mean over seeds"""
    rs = list(cell.by[cfg].values())
    lab = np.mean([r["metrics"]["label/linear/V"] for r in rs])
    worst = np.mean([max(r["metrics"][f"centre/{k}/V"] for k in KC) for r in rs])
    return lab, worst


def family(cfg):
    return cfg.split("__")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    recs = load_runs(a.runs)
    cells = sorted({(r["encoder"], r["dataset"]) for r in recs})
    lines = []
    csv = ["encoder,dataset,cfg,family,n_seeds,label_lin,label_lin_lo,label_lin_hi,label_mlp,worst_centre,worst_lo,worst_hi,leak_norm,leak_lo,leak_hi,"
           "label_V,worst_V,ood_label_lin,onA_linear,rate_bits"]
    verdict = []
    verdict2 = []
    for enc, ds in cells:
        cell = Cell(enc, ds, recs, a.runs, a.B)
        cfgs = sorted(cell.by)
        if "raw" not in cell.by:
            continue
        seeds_raw = sorted(cell.by["raw"])
        chance = float(np.mean([cell.by["raw"][s]["metrics"]["chance/centre"] for s in seeds_raw]))
        floor = cell.floor()
        raw_lab = cell.metric("raw", "label", ("linear",))
        raw_worst = cell.metric("raw", "centre", KC)
        lab_v_raw, worst_v_raw = vmetrics(cell, "raw")
        table = {}
        for cfg in cfgs:
            if cfg.startswith("ctrl_"):
                continue
            lab = cell.metric(cfg, "label", ("linear",))
            labm = cell.metric(cfg, "label", ("mlp",))
            wc = cell.metric(cfg, "centre", KC)
            leak_b = (wc[1] - floor) / max(raw_worst[1] - floor, 1e-6)
            leak = (wc[0] - floor) / max(raw_worst[0] - floor, 1e-6)
            lv, wv = vmetrics(cell, cfg)
            rs = list(cell.by[cfg].values())
            ood = np.mean([r["metrics"]["label/linear/OOD"] for r in rs])
            ona = np.mean([r["metrics"]["centre_onA/linear/C"] for r in rs])
            rate = np.mean([r["extra"]["rate_bits_per_patch"] for r in rs]) if "rate_bits_per_patch" in rs[0]["extra"] else float("nan")
            table[cfg] = dict(lab=lab, labm=labm, wc=wc, leak=(leak, leak_b), lv=lv, wv=wv, ood=ood, ona=ona, rate=rate, n=len(rs))
            lo, hi = ci(lab[1]); wlo, whi = ci(wc[1]); llo, lhi = ci(leak_b)
            csv.append(f"{enc},{ds},{cfg},{family(cfg)},{len(rs)},{lab[0]:.4f},{lo:.4f},{hi:.4f},{labm[0]:.4f},{wc[0]:.4f},{wlo:.4f},{whi:.4f},"
                       f"{leak:.4f},{llo:.4f},{lhi:.4f},{lv:.4f},{wv:.4f},{ood:.4f},{ona:.4f},{rate:.2f}")
        # controls
        ctrl = []
        for cfg in cfgs:
            if cfg.startswith("ctrl_"):
                wc = cell.metric(cfg, "centre", KC)
                ctrl.append((cfg, wc, sorted(cell.by[cfg])))
        ctrl = sorted(ctrl, key=lambda x: x[0])
        lines.append(f"\n### {enc} / {ds}  (seeds {seeds_raw}; chance {chance:.3f}; label-only floor {floor:.3f}; raw label acc {fmt(*raw_lab[:2])}; raw worst-case centre acc {fmt(*raw_worst[:2])})\n")
        if ctrl:
            lines.append("Controls (worst-case over the 4 attackers, centre balanced accuracy on C; must be at chance):  " +
                         "; ".join(f"{c} {fmt(w[0], w[1])}" for c, w, _ in ctrl))
        # family-level selection on V
        lines.append("\nSelected per family on V (label within delta of raw), reported on sealed C:\n")
        lines.append("| delta | family | config | label acc (linear) | worst-case centre acc | leak (0=chance, 1=raw) | label acc on unseen centres | A-trained linear attacker | rate bits |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        fams = sorted({family(c) for c in table})
        sel_all = {}
        for dl in a.deltas:
            for fam in fams:
                cand = [(t["wv"], c) for c, t in table.items() if family(c) == fam and t["lv"] >= lab_v_raw - dl]
                if not cand:
                    continue
                wv, c = min(cand)
                t = table[c]
                sel_all[(dl, fam)] = c
                lines.append(f"| {dl:.2f} | {fam} | {c} | {fmt(*t['lab'][:2])} | {fmt(*t['wc'][:2])} | {fmt(t['leak'][0], t['leak'][1], 2)} | {t['ood']:.3f} | {t['ona']:.3f} | {t['rate']:.1f} |")
        # kill criterion: best learned (selected on V) vs LEACE, paired bootstrap
        learned = ("vib", "adv_retrain", "dann", "vq", "supcon_xc")
        for dl in a.deltas:
            if (dl, "leace") not in sel_all:
                continue
            cl = sel_all[(dl, "leace")]
            best = None
            for fam in learned:
                if (dl, fam) in sel_all:
                    c = sel_all[(dl, fam)]
                    if best is None or table[c]["wv"] < table[best]["wv"]:
                        best = c
            if best is None:
                continue
            d_pt = table[best]["wc"][0] - table[cl]["wc"][0]
            d_b = table[best]["wc"][1] - table[cl]["wc"][1]
            lo, hi = ci(d_b)
            dlab = table[best]["lab"][0] - table[cl]["lab"][0]
            verdict.append((enc, ds, dl, best, cl, d_pt, lo, hi, dlab))
            bn = [sel_all[(dl, f)] for f in ("vib", "vq") if (dl, f) in sel_all]
            ad = [sel_all[(dl, f)] for f in ("adv_retrain", "dann", "supcon_xc") if (dl, f) in sel_all]
            if bn and ad:
                b0 = min(bn, key=lambda c: table[c]["wv"])
                a0 = min(ad, key=lambda c: table[c]["wv"])
                db = table[a0]["wc"][1] - table[b0]["wc"][1]
                lo2, hi2 = ci(db)
                verdict2.append((enc, ds, dl, a0, b0, table[a0]["wc"][0] - table[b0]["wc"][0], lo2, hi2, table[a0]["lab"][0] - table[b0]["lab"][0]))
        lines.append("\nFull ladder (C numbers; 95% CI from unit bootstrap):\n")
        lines.append("| config | seeds | label acc (linear) | label acc (mlp) | worst-case centre acc | leak | V label | V worst | unseen-centre label | rate bits |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for cfg, t in table.items():
            lines.append(f"| {cfg} | {t['n']} | {fmt(*t['lab'][:2])} | {t['labm'][0]:.3f} | {fmt(*t['wc'][:2])} | {fmt(t['leak'][0], t['leak'][1], 2)} | {t['lv']:.3f} | {t['wv']:.3f} | {t['ood']:.3f} | {t['rate']:.1f} |")
    lines.append("\n### Kill criterion A: best learned adapter (selected on V) minus LEACE (selected on V) at equal label accuracy, worst-case centre accuracy on C\n")
    lines.append("| encoder | dataset | delta | best learned | LEACE cfg | diff (learned - LEACE) [95% CI] | label diff | learned beats LEACE (CI < 0)? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for enc, ds, dl, best, cl, d, lo, hi, dlab in verdict:
        lines.append(f"| {enc} | {ds} | {dl:.2f} | {best} | {cl} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {dlab:+.3f} | {'YES' if hi < 0 else 'no'} |")
    lines.append("\n### Mandatory control: best adversarial/contrastive adapter minus best NO-adversary bottleneck (vib, vq), both selected on V\n")
    lines.append("| encoder | dataset | delta | best adversarial | best bottleneck | diff worst-case centre acc [95% CI] | label diff | adversary helps (CI < 0)? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for enc, ds, dl, a0, b0, d, lo, hi, dlab in verdict2:
        lines.append(f"| {enc} | {ds} | {dl:.2f} | {a0} | {b0} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {dlab:+.3f} | {'YES' if hi < 0 else 'no'} |")
    text = "\n".join(lines)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).write_text(text)
    if a.csv:
        Path(a.csv).write_text("\n".join(csv))


if __name__ == "__main__":
    main()
