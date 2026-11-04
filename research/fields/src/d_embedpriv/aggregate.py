"""Aggregate Focus D runs: selection on V, numbers on C, paired bootstrap over C documents (pre-registered rule in results_notes/D.md).

  python aggregate.py DIR [DIR ...] [--taus 0.97,0.90] [--stage headroom|kill] [--margin 0.1] [--rare-only]

DIR = a job output folder with records.jsonl + units.npz (+ units_index.json).  Records from all folders are merged by rid = "arm|s<seed>".
Prints markdown tables.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ATT = ("ridge", "mlp", "knn")
TIERS = ("rare", "mid")
FLOORS = ("randproj", "pca", "noise", "leace", "sup_lin")
CONTROLS = ("bott_mlp",)


def load(dirs):
    recs, units = {}, {}
    for d in dirs:
        d = Path(d)
        for line in open(d / "records.jsonl"):
            r = json.loads(line)
            if "error" in r:
                continue
            recs[r["rid"]] = r
        z = np.load(d / "units.npz")
        for k in z.files:
            units[k] = z[k]
    return recs, units


def worst(rec, split, tier):
    return max(rec["attackers"][a][split][tier] for a in ATT)


def rp_units(units, rid, split, tier):
    """[n_att, n_docs] per-document R-precision."""
    return np.stack([units[f"{rid}|rp_{split}_{tier}_{a}"] for a in ATT])


def boot_worst(U, idx):
    """Worst-case-over-attackers mean R-precision for each bootstrap resample. U [n_att, n]; idx [B, n] -> [B]."""
    return np.nanmax(np.stack([np.nanmean(u[idx], axis=1) for u in U]), axis=0)


class Agg:
    def __init__(self, recs, units, B=1000):
        self.recs, self.units, self.B = recs, units, B
        self.rng = np.random.default_rng(0)
        any_c = next(k for k in units if "rp_C_rare_ridge" in k)
        self.n_c = len(units[any_c])
        self.idx_c = self.rng.integers(0, self.n_c, size=(B, self.n_c))
        any_q = next(k for k in units if k.endswith("|ndcg_test"))
        self.n_q = len(units[any_q])
        self.idx_q = self.rng.integers(0, self.n_q, size=(B, self.n_q))
        self.seeds = sorted({r["seed"] for r in recs.values()})

    # --- references (per seed)
    def ref(self, seed, tier):
        raw = self.recs.get(f"raw|s{seed}")
        const = self.recs.get(f"const|s{seed}")
        if raw is None or const is None:
            return None
        return dict(raw=raw, const=const)

    def L_boot(self, rid, seed, tier, split="C"):
        """Bootstrap distribution of normalised leakage for a record on split (shared resample indices => paired across arms)."""
        r = self.ref(seed, tier)
        idx = self.idx_c if split == "C" else None
        Rarm = boot_worst(rp_units(self.units, rid, split, tier), idx)
        Rraw = boot_worst(rp_units(self.units, f"raw|s{seed}", split, tier), idx)
        Rpri = boot_worst(rp_units(self.units, f"const|s{seed}", split, tier), idx)
        return (Rarm - Rpri) / np.maximum(Rraw - Rpri, 1e-6), Rarm, Rraw, Rpri

    def retention(self, rec, split="test"):
        return rec[f"ndcg_{split}"] / rec["ndcg_raw"][split]

    def ret_boot(self, rid, seed):
        a = self.units[f"{rid}|ndcg_test"]
        b = self.units[f"raw|s{seed}|ndcg_test"]
        return a[self.idx_q].mean(1) / b[self.idx_q].mean(1)

    # --- selection
    def family_recs(self, family, seed):
        return [r for r in self.recs.values() if r["family"] == family and r["seed"] == seed and not r["spec"].get("mask") and not r["spec"].get("shuffle")]

    def select(self, family, seed, tau):
        """Lowest V worst-case rare R-precision among configs with V retention >= tau (ties: smaller dim, smaller lambda)."""
        ok = [r for r in self.family_recs(family, seed) if self.retention(r, "dev") >= tau]
        if not ok:
            return None
        return min(ok, key=lambda r: (round(worst(r, "V", "rare"), 4), r["dim"], r["spec"].get("lam", 0)))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def fmt(x, lo=None, hi=None, nd=3):
    return f"{x:.{nd}f}" + (f" [{lo:.{nd}f}, {hi:.{nd}f}]" if lo is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--taus", default="0.97,0.90")
    ap.add_argument("--margin", type=float, default=None, help="frozen margin m (normalised leakage); enables the kill-rule evaluation")
    ap.add_argument("--families", default="")
    a = ap.parse_args()
    recs, units = load(a.dirs)
    A = Agg(recs, units)
    taus = [float(t) for t in a.taus.split(",")]
    seeds = A.seeds
    orc = [r for r in recs.values() if r["spec"].get("mask") == "rare"]
    tauO = None
    if orc:  # H2 when the oracle cannot reach tau=0.97: compare floors at the oracle's own (dev) retention
        tauO = float(np.floor(np.mean([A.retention(r, "dev") for r in orc]) * 1000) / 1000)
        if tauO < min(taus):
            taus.append(tauO)
            print(f"(oracle_mask_rare reaches dev retention {tauO:.3f} < tau: added as an oracle-matched operating point for H2)\n")
    print(f"records {len(recs)}, seeds {seeds}, C docs {A.n_c}, test queries {A.n_q}\n")

    # ---- controls / references
    print("### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)\n")
    print("| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |")
    print("|---|---|---|---|---|---|---|")
    preds = {
        "const": lambda r: r["arm"] == "const",
        "raw": lambda r: r["arm"] == "raw" and not r["spec"].get("mask") and not r["spec"].get("shuffle"),
        "raw|shuffled pairs": lambda r: r["arm"] == "raw" and r["spec"].get("shuffle"),
        "oracle_mask_rare": lambda r: r["spec"].get("mask") == "rare",
        "oracle_mask_all": lambda r: r["spec"].get("mask") == "all",
    }
    for arm, pred in preds.items():
        for tier in TIERS:
            xs, per, nd, rt, rd, Ls = [], [], [], [], [], []
            for s in seeds:
                cand = [r for r in recs.values() if r["seed"] == s and pred(r)]
                if not cand:
                    continue
                r = cand[0]
                U = rp_units(units, r["rid"], "C", tier)
                xs.append(boot_worst(U, A.idx_c))
                per.append([np.nanmean(u) for u in U])
                nd.append(r["ndcg_test"])
                rt.append(A.retention(r))
                rd.append(A.retention(r, "dev"))
                if A.ref(s, tier) is not None:
                    Ls.append(A.L_boot(r["rid"], s, tier)[0])
            if not xs:
                continue
            bx = np.mean(xs, axis=0)
            lo, hi = ci(bx)
            pm = np.mean(per, axis=0)
            Lt = fmt(np.mean(Ls, axis=0).mean(), *ci(np.mean(Ls, axis=0)), nd=2) if Ls else "n/a"
            print(f"| {arm} | {tier} | {fmt(bx.mean(), lo, hi)} | {pm[0]:.3f}/{pm[1]:.3f}/{pm[2]:.3f} | {Lt} | {np.mean(nd):.4f} | {np.mean(rd):.3f} / {np.mean(rt):.3f} |")
    print()

    # ---- curve table (all configs)
    fams = [f for f in (a.families.split(",") if a.families else (*FLOORS, *CONTROLS, "adv")) if any(r["family"] == f for r in recs.values())]
    print("### All configs (mean over seeds; V = selection split, C = sealed). R = worst-of-3 rare-tier R-precision\n")
    print("| arm | dim | seeds | retention dev | retention test | R_V rare | R_C rare | R_C mid |")
    print("|---|---|---|---|---|---|---|---|")
    byarm = defaultdict(list)
    for r in recs.values():
        if r["family"] in fams and not r["spec"].get("mask") and not r["spec"].get("shuffle"):
            byarm[r["arm"]].append(r)
    for arm in sorted(byarm, key=lambda x: (byarm[x][0]["family"], -byarm[x][0]["dim"], x)):
        rs = byarm[arm]
        print(f"| {arm} | {rs[0]['dim']} | {len(rs)} | {np.mean([A.retention(r, 'dev') for r in rs]):.3f} | {np.mean([A.retention(r) for r in rs]):.3f} | "
              f"{np.mean([worst(r, 'V', 'rare') for r in rs]):.3f} | {np.mean([worst(r, 'C', 'rare') for r in rs]):.3f} | {np.mean([worst(r, 'C', 'mid') for r in rs]):.3f} |")
    print()

    # ---- selection at each tau
    if orc:
        headroom(A, seeds, [t for t in taus if t != tauO], tauO)
    for tau in taus:
        print(f"### Selected on V at retention >= {tau} (config per family and seed; numbers on sealed C)\n")
        print("| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C rare worst [95% CI] | L_rare [95% CI] | L_mid [95% CI] | mean V L_rare |")
        print("|---|---|---|---|---|---|---|---|")
        summ = {}
        for fam in fams:
            sel = {s: A.select(fam, s, tau) for s in seeds}
            ok = {s: r for s, r in sel.items() if r is not None}
            if not ok or any(A.ref(s, "rare") is None for s in ok):
                print(f"| {fam} | {len(ok)}/{len(seeds)} | cannot reach tau | | | | | |")
                continue
            Lr = np.mean([A.L_boot(r["rid"], s, "rare")[0] for s, r in ok.items()], axis=0)
            Lm = np.mean([A.L_boot(r["rid"], s, "mid")[0] for s, r in ok.items()], axis=0)
            Rr = np.mean([A.L_boot(r["rid"], s, "rare")[1] for s, r in ok.items()], axis=0)
            Rt = np.mean([A.ret_boot(r["rid"], s) for s, r in ok.items()], axis=0)
            # V leakage point estimate (for strongest-floor choice)
            LV = []
            for s, r in ok.items():
                ref = A.ref(s, "rare")
                rp, rr, rc = worst(r, "V", "rare"), worst(ref["raw"], "V", "rare"), worst(ref["const"], "V", "rare")
                LV.append((rp - rc) / max(rr - rc, 1e-6))
            if len(ok) == len(seeds):  # a family is a candidate for F* only if it reaches tau in every seed (clarification added after seeing noise r=0.5 feasible in 1 of 3 seeds)
                summ[fam] = dict(L=Lr.mean(), LV=np.mean(LV), n=len(ok))
            cfgs = ", ".join(f"s{s}:{r['arm'].split('_', 1)[-1]}" for s, r in ok.items())
            print(f"| {fam} | {len(ok)}/{len(seeds)} | {cfgs} | {fmt(Rt.mean(), *ci(Rt))} | {fmt(Rr.mean(), *ci(Rr))} | {fmt(Lr.mean(), *ci(Lr))} | {fmt(Lm.mean(), *ci(Lm))} | {np.mean(LV):.3f} |")
        floors = {f: v for f, v in summ.items() if f in FLOORS}
        if floors:
            fstar = min(floors, key=lambda f: floors[f]["LV"])
            print(f"\nStrongest floor F*({tau}) (lowest seed-mean V leakage): **{fstar}** (V L_rare {floors[fstar]['LV']:.3f}; C L_rare {floors[fstar]['L']:.3f})\n")
            if a.margin is not None and "adv" in fams and tau == taus[0]:
                kill_rule(A, fstar, tau, a.margin, seeds)


def headroom(A, seeds, taus, tauO):
    """Pre-registered H1 (measurability) and H2 (room: strongest floor vs the masked-text oracle at the oracle's own retention)."""
    print("### Headroom check (pre-registered H1, H2; floors + oracle only)\n")
    d = []
    for s in seeds:
        raw, con = f"raw|s{s}", f"const|s{s}"
        if raw not in A.recs or con not in A.recs:
            continue
        Rr = boot_worst(rp_units(A.units, raw, "C", "rare"), A.idx_c)
        Rp = boot_worst(rp_units(A.units, con, "C", "rare"), A.idx_c)
        d.append(Rr - Rp)
    d = np.mean(d, axis=0)
    print(f"H1 measurability: R_raw - R_prior (rare tier, worst of 3 attackers) = {d.mean():.3f} [{ci(d)[0]:.3f}, {ci(d)[1]:.3f}] (rule: >= 0.05 and CI excludes 0): "
          f"{'PASS' if ci(d)[0] > 0 and d.mean() >= 0.05 else 'FAIL'}\n")
    orc = {s: next((r for r in A.recs.values() if r["seed"] == s and r["spec"].get("mask") == "rare"), None) for s in seeds}
    if any(v is None for v in orc.values()):
        return
    Lo = np.mean([A.L_boot(r["rid"], s, "rare")[0] for s, r in orc.items()], axis=0)
    ret_o = np.mean([A.retention(r) for r in orc.values()])
    print(f"oracle_mask_rare: L_rare = {Lo.mean():.3f} [{ci(Lo)[0]:.3f}, {ci(Lo)[1]:.3f}], test retention {ret_o:.3f} (< 0.97: not a valid reference at tau = 0.97, so H2 is judged at the oracle's own dev retention tau_O = {tauO:.3f})\n")
    print("| tau | strongest floor F* (by V) | F* configs | F* test retention | L_rare(F*) [95% CI] | L_rare(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2 (>= 0.10) |")
    print("|---|---|---|---|---|---|---|---|")
    for tau in [t for t in taus if t >= 0.9] + [tauO]:
        best = None
        for fam in FLOORS:
            sel = {s: A.select(fam, s, tau) for s in seeds}
            if any(v is None for v in sel.values()):
                continue
            LV = np.mean([(worst(r, "V", "rare") - worst(A.ref(s, "rare")["const"], "V", "rare")) /
                          max(worst(A.ref(s, "rare")["raw"], "V", "rare") - worst(A.ref(s, "rare")["const"], "V", "rare"), 1e-6) for s, r in sel.items()])
            if best is None or LV < best[0]:
                best = (LV, fam, sel)
        if best is None:
            print(f"| {tau:.3f} | none feasible | | | | | | |")
            continue
        _, fam, sel = best
        Lf = np.mean([A.L_boot(r["rid"], s, "rare")[0] for s, r in sel.items()], axis=0)
        hd = Lf - Lo
        rt = np.mean([A.retention(r) for r in sel.values()])
        flag = "" if tau == tauO else " (oracle not valid at this tau; reference only)"
        verdict = ("PASS" if hd.mean() >= 0.10 else "FAIL") if tau == tauO else "n/a"
        print(f"| {tau:.3f}{flag} | {fam} | {', '.join(f's{s_}:' + r_['arm'] for s_, r_ in sel.items())} | {rt:.3f} | {Lf.mean():.3f} [{ci(Lf)[0]:.3f}, {ci(Lf)[1]:.3f}] | {Lo.mean():.3f} | {hd.mean():.3f} [{ci(hd)[0]:.3f}, {ci(hd)[1]:.3f}] | {verdict} |")
    print()


def kill_rule(A, fstar, tau, m, seeds):
    print(f"#### Kill rule at tau={tau}, margin m={m} (pre-registered K1-K4), F*={fstar}\n")
    print("| seed | adv config | F* config | adv test ret. | F* test ret. | L(F*)-L(adv) [95% CI] | L(bott)-L(adv) [95% CI] | mid: L(adv)-L(F*) |")
    print("|---|---|---|---|---|---|---|---|")
    k1 = k2 = k3 = k4 = 0
    for s in seeds:
        ra, rf, rb = A.select("adv", s, tau), A.select(fstar, s, tau), A.select("bott_mlp", s, tau)
        if ra is None or rf is None or A.ref(s, "rare") is None:
            print(f"| {s} | n/a | | | | | | |")
            continue
        La = A.L_boot(ra["rid"], s, "rare")[0]
        Lf = A.L_boot(rf["rid"], s, "rare")[0]
        d1 = Lf - La
        d3 = (A.L_boot(rb["rid"], s, "rare")[0] - La) if rb is not None else None
        mid = A.L_boot(ra["rid"], s, "mid")[0].mean() - A.L_boot(rf["rid"], s, "mid")[0].mean()
        ret_a, ret_f = A.retention(ra), A.retention(rf)
        d3s = None
        p1 = ci(d1)[0] >= m
        p2 = ret_a >= ret_f - 0.01
        p3 = d3 is not None and ci(d3)[0] >= m / 2
        p4 = mid <= 0.05
        d3s = 'n/a' if d3 is None else f"{d3.mean():.3f} [{ci(d3)[0]:.3f}, {ci(d3)[1]:.3f}] ({'ok' if p3 else 'fail'})"
        k1 += p1 and p2
        k3 += bool(p1 and p2 and p3)
        k4 += p4
        print(f"| {s} | {ra['arm']} | {rf['arm']} | {ret_a:.3f} | {ret_f:.3f} | {d1.mean():.3f} [{ci(d1)[0]:.3f}, {ci(d1)[1]:.3f}] "
              f"({'ok' if p1 else 'fail'}; ret {'ok' if p2 else 'fail'}) | {d3s} | {mid:.3f} ({'ok' if p4 else 'fail'}) |")
    print(f"\nK1+K2 (>= m with equal utility) in {k1}/{len(seeds)} seeds; K1+K2+K3 in {k3}/{len(seeds)}; K4 in {k4}/{len(seeds)}. "
          f"PASS requires K1+K2+K3 in >= 2 seeds and K4 in those seeds.\n")


if __name__ == "__main__":
    main()
