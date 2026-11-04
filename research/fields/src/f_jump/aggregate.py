"""JS-1 aggregator: tables with intervals, power audit and the JK0..JK3 verdict from the stored per-compound AP vectors and per-well attacker predictions.

  python aggregate.py --scenario 4 --dirs d_s0,d_s1,d_s2 --out results/f_jump/s4 [--B 1000]

Pre-registration: results_notes/F.md 2.3-2.5.  Units: compounds (utility), plates (leakage).  Selection (R, F*, label-only best, F_L) uses V only; H is read once.
Decisions use point estimates (as registered); intervals are conditional on the selection.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

CHANCE = 0.2
K = 5
KINDS = ("linear", "mlp", "knn", "lgbm")
FAM_IND = ("floor", "labelonly", "dither")


# ------------------------------------------------------------------------------------------ leakage
class Leak:
    """per-plate sufficient statistics of the attackers on one eval set; bootstrap over plates."""

    def __init__(self, z, sname, arms, kinds):
        self.y = z[f"y|{sname}"].astype(np.int64)
        pl = z[f"plate|{sname}"]
        self.pl_ids, self.pl = np.unique(pl, return_inverse=True)
        self.nP = len(self.pl_ids)
        self.stats = {}
        for arm in arms:
            for k in kinds:
                key = f"att|{sname}|{arm}|{k}"
                if key not in z.files:
                    continue
                p = z[key].astype(np.int64)
                ok = p >= 0
                corr = np.zeros((self.nP, K))
                cnt = np.zeros((self.nP, K))
                np.add.at(cnt, (self.pl[ok], self.y[ok]), 1)
                np.add.at(corr, (self.pl[ok & (p == self.y)], self.y[ok & (p == self.y)]), 1)
                self.stats[(arm, k)] = (corr, cnt)

    def acc(self, arm, k, w=None):
        corr, cnt = self.stats[(arm, k)]
        if w is None:
            c, n = corr.sum(0), cnt.sum(0)
        else:
            c, n = w @ corr, w @ cnt
        pres = n > 0
        return float(np.mean(c[pres] / n[pres]))

    def acc_boot(self, arm, k, W):
        corr, cnt = self.stats[(arm, k)]
        c, n = W @ corr, W @ cnt                                  # (B, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = c / n
        return np.nanmean(np.where(n > 0, r, np.nan), axis=1)

    def worst(self, arm, kinds, w=None):
        return max(self.acc(arm, k, w) for k in kinds if (arm, k) in self.stats)

    def worst_boot(self, arm, kinds, W):
        return np.max(np.stack([self.acc_boot(arm, k, W) for k in kinds if (arm, k) in self.stats]), axis=0)


def lnorm(w_arm, w_raw, min_excess=0.10):
    ex = np.asarray(w_raw) - CHANCE
    out = (np.asarray(w_arm) - CHANCE) / np.where(ex < min_excess, np.nan, ex)
    return out


def ci(x, q=(2.5, 97.5)):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, q[0])), float(np.percentile(x, q[1]))]


# ------------------------------------------------------------------------------------------ one split
class Split:
    def __init__(self, d, salt, B, seed, s5, label_prefix):
        self.z = np.load(Path(d) / f"preds_{salt}.npz")
        self.meta = json.loads((Path(d) / f"meta_{salt}.json").read_text())
        self.salt, self.s5 = salt, s5
        self.arms = [a for a in self.meta["arms"]]
        self.fam = self.meta["families"]
        self.kinds = [k for k in KINDS if k in self.meta["kinds"]]
        rng = np.random.default_rng(seed)
        self.leak = {s: Leak(self.z, s, self.arms, self.kinds) for s in ("V", "H")}
        self.W = {s: np.stack([np.bincount(rng.integers(0, self.leak[s].nP, self.leak[s].nP), minlength=self.leak[s].nP) for _ in range(B)]).astype(np.float64) for s in ("V", "H")}
        nC = {s: len(self.z[f"cid|{s}|xs"]) for s in ("V", "H")}
        self.nC = nC
        self.cnt = {s: rng.multinomial(nC[s], np.ones(nC[s]) / nC[s], size=B).astype(np.float32) for s in ("V", "H")}
        self.B = B
        self.label_prefix = label_prefix

    def has_att(self, arm):
        return (arm, self.kinds[0]) in self.leak["H"].stats

    def apc(self, s, arm, mode="xs"):
        key = f"apc|{s}|{arm}|{mode}"
        return self.z[key].astype(np.float64) if key in self.z.files else None

    def U(self, s, arm, mode="xs"):
        a = self.apc(s, arm, mode)
        return None if a is None else float(a.mean())

    def U_boot(self, s, arm, mode="xs"):
        a = self.apc(s, arm, mode).astype(np.float32)
        return (self.cnt[s] @ a) / self.nC[s]

    def dU_boot(self, s, a, b):
        d = (self.apc(s, a) - self.apc(s, b)).astype(np.float32)
        return (self.cnt[s] @ d) / self.nC[s]

    def L(self, s, arm):
        lk = self.leak[s]
        return float((lk.worst(arm, self.kinds) - CHANCE) / max(lk.worst("raw", self.kinds) - CHANCE, 1e-9))

    def worst(self, s, arm):
        return self.leak[s].worst(arm, self.kinds)

    def L_boot(self, s, arm):
        lk = self.leak[s]
        return lnorm(lk.worst_boot(arm, self.kinds, self.W[s]), lk.worst_boot("raw", self.kinds, self.W[s]))

    def dL_boot(self, s, a, b):
        return self.L_boot(s, a) - self.L_boot(s, b)

    def names(self, fams):
        return [a for a in self.arms if self.fam[a] in fams]

    # ---- selection and the JK3 / JK2 ingredients
    def pick_ref(self, how="U"):
        refs = [a for a in self.names(("ref",)) if self.U("V", a) is not None and self.has_att(a)]
        if not refs:
            return None
        if how == "U":
            return max(refs, key=lambda a: self.U("V", a))
        return min(refs, key=lambda a: self.L("V", a))

    def jk3(self, R):
        out = dict(R=R, U_V_R=self.U("V", R), L_V_R=self.L("V", R), U_H_R=self.U("H", R), L_H_R=self.L("H", R))
        inds = [a for a in self.names(FAM_IND) if self.has_att(a) and self.U("V", a) is not None]
        # U-axis
        cand = [a for a in inds if self.L("V", a) <= out["L_V_R"] + 0.05]
        if cand:
            F = max(cand, key=lambda a: self.U("V", a))
            dU = self.U("H", R) - self.U("H", F)
            b = self.dU_boot("H", R, F)
            out.update(F_star=F, U_H_F=self.U("H", F), L_H_F=self.L("H", F), dU_H=dU, dU_H_ci=ci(b), headroom_U=bool(dU > 0.03))
        else:
            out.update(F_star=None, headroom_U=True, note_U="no inductive floor reaches L_V(R)+0.05")
        # L-axis
        cand2 = [a for a in inds if self.U("V", a) >= out["U_V_R"] - 0.01]
        if cand2:
            FL = min(cand2, key=lambda a: self.L("V", a))
            dL = self.L("H", FL) - self.L("H", R)
            b = self.dL_boot("H", FL, R)
            out.update(F_L=FL, U_H_FL=self.U("H", FL), L_H_FL=self.L("H", FL), dL_H=dL, dL_H_ci=ci(b), headroom_L=bool(dL >= 0.15))
        else:
            out.update(F_L=None, headroom_L=True, note_L="no inductive floor has U_V >= U_V(R)-0.01")
        out["headroom"] = bool(out["headroom_U"] or out["headroom_L"])
        out["fires"] = not out["headroom"]
        return out

    def jk2(self, R):
        lab = [a for a in self.names(("labelonly",)) if self.has_att(a)]
        res = dict(R=R, U_H_R=self.U("H", R))
        cons = [a for a in lab if self.L("V", a) <= 0.25]
        res["constrained_candidates"] = len(cons)
        if cons:
            Bst = max(cons, key=lambda a: self.U("V", a))
            res.update(best=Bst, U_V=self.U("V", Bst), L_V=self.L("V", Bst), U_H=self.U("H", Bst), L_H=self.L("H", Bst),
                       dU_vs_R=self.U("H", Bst) - self.U("H", R), dU_ci=ci(self.dU_boot("H", Bst, R)), L_H_ci=ci(self.L_boot("H", Bst)))
            res["fires"] = bool(res["L_H"] <= 0.25 and res["U_H"] >= self.U("H", R) - 0.03)
        else:
            res["fires"] = False
            res["best"] = None
        lit = max(lab, key=lambda a: self.U("V", a))
        res["literal"] = dict(best=lit, U_H=self.U("H", lit), L_H=self.L("H", lit), L_V=self.L("V", lit),
                              fires=bool(self.L("H", lit) <= 0.25 and self.U("H", lit) >= self.U("H", R) - 0.03))
        incl = [a for a in self.names(("labelonly", "dither")) if a.startswith(self.label_prefix)] + lab
        consd = [a for a in set(incl) if self.L("V", a) <= 0.25]
        if consd:
            Bd = max(consd, key=lambda a: self.U("V", a))
            res["with_dither"] = dict(best=Bd, U_H=self.U("H", Bd), L_H=self.L("H", Bd), fires=bool(self.L("H", Bd) <= 0.25 and self.U("H", Bd) >= self.U("H", R) - 0.03))
        return res


# ------------------------------------------------------------------------------------------ power audit
def mde80(d, B=1000, perms=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    bm = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)]) if n < 5000 else np.array([d[rng.integers(0, n, n)].mean() for _ in range(300)])
    se = float(bm.std())
    nm = []
    for _ in range(0, perms, 250):
        S = rng.choice([-1.0, 1.0], size=(250, n))
        nm.append(np.abs(S @ d / n))
    c = float(np.quantile(np.concatenate(nm), 0.95))
    return dict(se=se, crit=c, mde80=c + 0.8416 * se, d_mean=float(d.mean()))


def audit(sp, pairs_arms):
    out = {}
    for a, b in itertools.combinations(pairs_arms, 2):
        if sp.apc("H", a) is None or sp.apc("H", b) is None:
            continue
        out[f"{a} vs {b}"] = mde80(sp.apc("H", a) - sp.apc("H", b))
    med = float(np.median([v["mde80"] for v in out.values()])) if out else float("nan")
    return dict(pairs=out, median_mde80=med, n_compounds=sp.nC["H"])


# ------------------------------------------------------------------------------------------ report
def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fci(x, c, nd=3):
    return f"{fmt(x, nd)} [{fmt(c[0], nd)}, {fmt(c[1], nd)}]"


def arm_table(sp):
    rows = ["| arm | family | d | U_V | U_H xs [95% CI] | U_H nonrep | worst att H [CI] | L_V | L_H [CI] | lin / mlp / knn / lgbm (H) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in sp.arms:
        f = sp.fam[a]
        if not sp.has_att(a):
            uH = sp.U("H", a)
            rows.append(f"| {a} | {f} | {sp.meta['dims'][a]} | {fmt(sp.U('V', a))} | {fmt(uH)} | {fmt(sp.U('H', a, 'nonrep'))} | n/a | n/a | n/a | n/a |")
            continue
        uH = sp.U("H", a)
        uci = ci(sp.U_boot("H", a)) if uH is not None else [float("nan")] * 2
        w = sp.worst("H", a)
        wci = ci(sp.leak["H"].worst_boot(a, sp.kinds, sp.W["H"]))
        lci = ci(sp.L_boot("H", a))
        accs = " / ".join(fmt(sp.leak["H"].acc(a, k), 2) for k in sp.kinds)
        rows.append(f"| {a} | {f} | {sp.meta['dims'][a]} | {fmt(sp.U('V', a))} | {fci(uH, uci) if uH is not None else 'n/a'} | {fmt(sp.U('H', a, 'nonrep'))} | {fci(w, wci, 2)} | {fmt(sp.L('V', a), 2)} | {fci(sp.L('H', a), lci, 2)} | {accs} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--salts", default="s0,s1,s2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    s5 = a.scenario == 5
    lab = "ldaT2" if s5 else "ldaT"
    dirs = a.dirs.split(",")
    salts = a.salts.split(",")
    splits = [Split(d, s, a.B, 100 + i, s5, lab) for i, (d, s) in enumerate(zip(dirs, salts))]
    pairs_arms = ["pca64", "dpca64_r0.5", "rp64", ("ldaP_64" if s5 else "ldaT_64"), (f"{lab}_max"), "raw"]
    res = dict(scenario=a.scenario, salts=salts, splits={})
    md = [f"# JS-1 aggregate, scenario {a.scenario} (B={a.B}); units: compounds (U), plates (L). Point estimates drive the registered decisions; intervals are conditional on V-selection.\n"]
    v_jk1, v_jk1b, v_jk2, v_jk3, v_mde = [], [], [], [], []
    for sp in splits:
        r = dict(meta=sp.meta["n_compounds"] | {"wells_H": sp.meta["n_wells"]["H"], "wells_V": sp.meta["n_wells"]["V"]})
        lk = sp.leak["H"]
        raw_w = sp.worst("H", "raw")
        r["JK1"] = dict(raw_worst=raw_w, raw_worst_ci=ci(lk.worst_boot("raw", sp.kinds, sp.W["H"])), threshold=max(min(3 * CHANCE, 0.85), CHANCE + 0.15),
                        controls={c: sp.worst("H", c) for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_source") if (c, sp.kinds[0]) in lk.stats},
                        U_shuffle_compound=sp.U("H", "ctrl_shuffle_compound"), U_noise16=sp.U("H", "ctrl_noise16"))
        base = raw_w >= r["JK1"]["threshold"] and all(v <= CHANCE + 0.03 for v in r["JK1"]["controls"].values())
        us, un = r["JK1"]["U_shuffle_compound"], r["JK1"]["U_noise16"]
        v0 = base and (us is None or un is None or abs(us - un) <= 0.01)                  # v0 text: two-sided
        j0 = base and (us is None or un is None or us <= un + 0.03)                        # JUMP0 text / deviation D1: one-sided
        r["JK1"].update(pass_v0_literal=bool(v0), pass_jump0_text=bool(j0), substantive_pass=bool(base))
        r["JK1"]["pass"] = bool(v0)
        v_jk1.append(v0)
        v_jk1b.append(j0)
        R = sp.pick_ref("U")
        if R is not None:
            r["R"] = R
            r["JK3"] = sp.jk3(R)
            r["JK3_per_ref"] = {x: sp.jk3(x) for x in sp.names(("ref",)) if sp.U("V", x) is not None and sp.has_att(x)}
            Rl = sp.pick_ref("L")
            r["JK3_lowestL_ref"] = sp.jk3(Rl)
            r["JK2"] = sp.jk2(R)
            v_jk3.append(r["JK3"]["fires"])
            v_jk2.append(r["JK2"]["fires"])
        r["power"] = audit(sp, [x for x in pairs_arms if x in sp.arms])
        v_mde.append(r["power"]["median_mde80"])
        res["splits"][sp.salt] = r
        md.append(f"## split {sp.salt}: {sp.meta['n_compounds']} compounds (V/H), wells H {sp.meta['n_wells']['H']}, V {sp.meta['n_wells']['V']}, fit {sp.meta['n_wells']['fit']}\n")
        md.append(arm_table(sp) + "\n")
        md.append(f"**JK1**: raw worst-of-4 {fci(raw_w, r['JK1']['raw_worst_ci'], 3)} (needs >= {r['JK1']['threshold']:.2f}); controls {json.dumps({k: round(v, 3) for k, v in r['JK1']['controls'].items()})}; "
                  f"U(shuffled compound) {fmt(r['JK1']['U_shuffle_compound'])} vs U(noise16) {fmt(r['JK1']['U_noise16'])} -> pass v0-literal={r['JK1']['pass_v0_literal']}, pass JUMP0-text (D1, post-hoc sensitivity)={r['JK1']['pass_jump0_text']}\n")
        if R is not None:
            j = r["JK3"]
            md.append(f"**JK3** (R = {R}: U_V {fmt(j['U_V_R'])}, U_H {fmt(j['U_H_R'])}, L_V {fmt(j['L_V_R'], 2)}, L_H {fmt(j['L_H_R'], 2)}): U-axis F* = {j.get('F_star')} "
                      f"(U_H {fmt(j.get('U_H_F'))}, L_H {fmt(j.get('L_H_F'), 2)}), dU = {fci(j.get('dU_H'), j.get('dU_H_ci', [float('nan')] * 2)) if 'dU_H' in j else 'n/a'} -> headroom_U={j['headroom_U']}; "
                      f"L-axis F_L = {j.get('F_L')} (U_H {fmt(j.get('U_H_FL'))}, L_H {fmt(j.get('L_H_FL'), 2)}), dL = {fci(j.get('dL_H'), j.get('dL_H_ci', [float('nan')] * 2), 2) if 'dL_H' in j else 'n/a'} -> headroom_L={j['headroom_L']}; fires={j['fires']}\n")
            md.append("per-reference: " + "; ".join(f"{x}: U_V {fmt(v['U_V_R'])}, L_V {fmt(v['L_V_R'], 2)}, F* {v.get('F_star')}, dU_H {fmt(v.get('dU_H'))}, headroom_U={v['headroom_U']}, headroom_L={v['headroom_L']}, fires={v['fires']}" for x, v in r["JK3_per_ref"].items()) + "\n")
            k2 = r["JK2"]
            md.append(f"**JK2** (constrained V-selection, L_V <= 0.25): best = {k2.get('best')} " + (f"U_H {fmt(k2['U_H'])}, L_H {fci(k2['L_H'], k2['L_H_ci'], 2)}, dU vs R {fci(k2['dU_vs_R'], k2['dU_ci'])}" if k2.get('best') else f"(no label-only arm with L_V <= 0.25; {k2['constrained_candidates']} candidates)") +
                      f" -> fires={k2['fires']}; literal reading: best-by-U_V = {k2['literal']['best']} (U_H {fmt(k2['literal']['U_H'])}, L_H {fmt(k2['literal']['L_H'], 2)}) fires={k2['literal']['fires']}; with dithered LDA: {k2.get('with_dither')}\n")
        md.append(f"**power (JK0c)**: median floor-vs-floor MDE80 = {fmt(r['power']['median_mde80'], 4)} over {len(r['power']['pairs'])} pairs; n_compounds_H = {r['power']['n_compounds']}\n")
    # verdict
    ns = len(splits)
    res["verdict"] = dict(
        JK1_v0_literal_pass_splits=int(sum(v_jk1)), JK1_v0_literal_pass=bool(all(v_jk1)),
        JK1_jump0_text_pass_splits=int(sum(v_jk1b)), JK1_jump0_text_pass=bool(all(v_jk1b)),
        JK2_fires_splits=int(sum(v_jk2)), JK2_fires=bool(sum(v_jk2) >= 2) if ns == 3 else None,
        JK3_fires_splits=int(sum(v_jk3)), JK3_fires=bool(sum(v_jk3) >= 2) if ns == 3 else None,
        median_mde80_per_split=v_mde, JK0c_pass=bool(np.median(v_mde) <= 0.02))
    md.append("## Verdict table\n")
    md.append("```\n" + json.dumps(res["verdict"], indent=1) + "\n```\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(str(a.out) + ".md").write_text("\n".join(md))
    print("\n".join(md[-3:]))


if __name__ == "__main__":
    main()
