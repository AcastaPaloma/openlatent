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
P, S2, OM, OM2 = "rare", "mid", "rare", "all"  # primary tier, second tier, oracle mask names (set from the records in main)
FLOORS = ("randproj", "pca", "noise", "leace", "sup_lin")
CONTROLS = ("bott_mlp",)


def load(dirs, legacy_adv=False):
    """legacy_adv=False drops the original `adv` fits (conf_norm absent: all 24 collapsed to the uniform-softmax retrieval loss, audit F-01)."""
    recs, units = {}, {}
    for d in dirs:
        d = Path(d)
        for line in open(d / "records.jsonl"):
            r = json.loads(line)
            if "error" in r:
                continue
            if r.get("family") == "adv" and not r["spec"].get("conf_norm") and not legacy_adv:
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
        any_c = next(k for k in units if f"rp_C_{P}_ridge" in k)
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
        return [r for r in self.recs.values() if r["family"] == family and r["seed"] == seed and not r["spec"].get("mask") and not r["spec"].get("shuffle")
                and not r.get("collapsed")]

    def select(self, family, seed, tau):
        """Lowest V worst-case rare R-precision among configs with V retention >= tau (ties: smaller dim, smaller lambda)."""
        ok = [r for r in self.family_recs(family, seed) if self.retention(r, "dev") >= tau]
        if not ok:
            return None
        return min(ok, key=lambda r: (round(worst(r, "V", P), 4), r["dim"], r["spec"].get("lam", 0)))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def fmt(x, lo=None, hi=None, nd=3):
    return f"{x:.{nd}f}" + (f" [{lo:.{nd}f}, {hi:.{nd}f}]" if lo is not None else "")


def main():
    global P, S2, OM, OM2
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--taus", default="0.97,0.90")
    ap.add_argument("--margin", type=float, default=None, help="frozen margin m (normalised leakage); enables the kill-rule evaluation")
    ap.add_argument("--margins", default="", help="per-tau frozen margins, e.g. 0.90:0.0775,0.97:0.10 (D2)")
    ap.add_argument("--mde", action="store_true", help="print the paired-bootstrap SD and MDE80 of F* vs the bott_mlp control at each tau, then continue")
    ap.add_argument("--legacy-adv", action="store_true", help="keep the original collapsed adv fits")
    ap.add_argument("--families", default="")
    a = ap.parse_args()
    recs, units = load(a.dirs, a.legacy_adv)
    margins = {float(x.split(":")[0]): float(x.split(":")[1]) for x in a.margins.split(",") if x}
    if a.margin is not None and not margins:
        margins = {0.97: a.margin}
    tiers = list(next(iter(recs.values()))["attackers"]["ridge"]["V"].keys())
    P, S2 = tiers[0], tiers[1]
    if P == "n1":  # digits / entity scheme (H2-009)
        OM, OM2 = "digits", "n2"
    A = Agg(recs, units)
    taus = [float(t) for t in a.taus.split(",")]
    seeds = A.seeds
    orc = [r for r in recs.values() if r["spec"].get("mask") == OM]
    tauO = None
    if orc:  # H2 when the oracle cannot reach tau=0.97: compare floors at the oracle's own (dev) retention
        tauO = float(np.floor(np.mean([A.retention(r, "dev") for r in orc]) * 1000) / 1000)
        if min(tauO, np.mean([A.retention(r) for r in orc])) >= 0.97:
            tauO = None  # oracle reaches 0.97 on dev and test: H2 is judged at tau = 0.97
        elif tauO < min(taus):
            taus.append(tauO)
            print(f"(oracle_mask_{OM} reaches dev retention {tauO:.3f} < tau: added as an oracle-matched operating point for H2)\n")
    print(f"records {len(recs)}, seeds {seeds}, C docs {A.n_c}, test queries {A.n_q}\n")

    # ---- controls / references
    print("### References and controls (sealed C docs; R-precision of the worst of ridge/mlp/knn; mean over seeds; bootstrap over C docs)\n")
    print("| arm | tier | R_worst [95% CI] | per attacker (ridge/mlp/knn) | L = (R-R_prior)/(R_raw-R_prior) [95% CI] | nDCG@10 test | retention (dev / test) |")
    print("|---|---|---|---|---|---|---|")
    preds = {
        "const": lambda r: r["arm"] == "const",
        "raw": lambda r: r["arm"] == "raw" and not r["spec"].get("mask") and not r["spec"].get("shuffle"),
        "raw|shuffled pairs": lambda r: r["arm"] == "raw|shuffle",
        f"oracle_mask_{OM}": lambda r: r["spec"].get("mask") == OM,
        f"oracle_mask_{OM2}": lambda r: r["spec"].get("mask") == OM2,
    }
    for arm, pred in preds.items():
        for tier in (P, S2):
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
    print("### All configs (mean over seeds; V = selection split, C = sealed). R = worst-of-3 primary-tier R-precision\n")
    print(f"| arm | dim | seeds | retention dev | retention test | R_V {P} | R_C {P} | R_C {S2} |")
    print("|---|---|---|---|---|---|---|---|")
    byarm = defaultdict(list)
    for r in recs.values():
        if r["family"] in fams and not r["spec"].get("mask") and not r["spec"].get("shuffle"):
            byarm[r["arm"]].append(r)
    for arm in sorted(byarm, key=lambda x: (byarm[x][0]["family"], -byarm[x][0]["dim"], x)):
        rs = byarm[arm]
        print(f"| {arm} | {rs[0]['dim']} | {len(rs)} | {np.mean([A.retention(r, 'dev') for r in rs]):.3f} | {np.mean([A.retention(r) for r in rs]):.3f} | "
              f"{np.mean([worst(r, 'V', P) for r in rs]):.3f} | {np.mean([worst(r, 'C', P) for r in rs]):.3f} | {np.mean([worst(r, 'C', S2) for r in rs]):.3f} |")
    print()

    # ---- selection at each tau
    collapse_table(A)
    if orc:
        headroom(A, seeds, [t for t in taus if t != tauO], tauO)
    for tau in taus:
        print(f"### Selected on V at retention >= {tau} (config per family and seed; numbers on sealed C)\n")
        print(f"| family | seeds with a feasible config | configs (per seed) | test retention [95% CI] | R_C {P} worst [95% CI] | L_{P} [95% CI] | L_{S2} [95% CI] | mean V L_{P} |")
        print("|---|---|---|---|---|---|---|---|")
        summ = {}
        for fam in fams:
            sel = {s: A.select(fam, s, tau) for s in seeds}
            ok = {s: r for s, r in sel.items() if r is not None}
            if not ok or any(A.ref(s, P) is None for s in ok):
                print(f"| {fam} | {len(ok)}/{len(seeds)} | cannot reach tau | | | | | |")
                continue
            Lr = np.mean([A.L_boot(r["rid"], s, P)[0] for s, r in ok.items()], axis=0)
            Lm = np.mean([A.L_boot(r["rid"], s, S2)[0] for s, r in ok.items()], axis=0)
            Rr = np.mean([A.L_boot(r["rid"], s, P)[1] for s, r in ok.items()], axis=0)
            Rt = np.mean([A.ret_boot(r["rid"], s) for s, r in ok.items()], axis=0)
            # V leakage point estimate (for strongest-floor choice)
            LV = []
            for s, r in ok.items():
                ref = A.ref(s, P)
                rp, rr, rc = worst(r, "V", P), worst(ref["raw"], "V", P), worst(ref["const"], "V", P)
                LV.append((rp - rc) / max(rr - rc, 1e-6))
            if len(ok) == len(seeds):  # a family is a candidate for F* only if it reaches tau in every seed (clarification added after seeing noise r=0.5 feasible in 1 of 3 seeds)
                summ[fam] = dict(L=Lr.mean(), LV=np.mean(LV), n=len(ok))
            cfgs = ", ".join(f"s{s}:{r['arm'].split('_', 1)[-1]}" for s, r in ok.items())
            print(f"| {fam} | {len(ok)}/{len(seeds)} | {cfgs} | {fmt(Rt.mean(), *ci(Rt))} | {fmt(Rr.mean(), *ci(Rr))} | {fmt(Lr.mean(), *ci(Lr))} | {fmt(Lm.mean(), *ci(Lm))} | {np.mean(LV):.3f} |")
        floors = {f: v for f, v in summ.items() if f in FLOORS}
        if floors:
            fstar = min(floors, key=lambda f: floors[f]["LV"])
            print(f"\nStrongest floor F*({tau}) (lowest seed-mean V leakage): **{fstar}** (V L_{P} {floors[fstar]['LV']:.3f}; C L_{P} {floors[fstar]['L']:.3f})\n")
            if tau in margins and "adv" in fams:
                kill_rule(A, fstar, tau, margins[tau], seeds)
            if a.mde:
                mde_print(A, fstar, tau, seeds)


def headroom(A, seeds, taus, tauO):
    """Pre-registered H1 (measurability) and H2 (room: strongest floor vs the masked-text oracle at the oracle's own retention)."""
    print("### Headroom check (pre-registered H1, H2; floors + oracle only)\n")
    d = []
    for s in seeds:
        raw, con = f"raw|s{s}", f"const|s{s}"
        if raw not in A.recs or con not in A.recs:
            continue
        Rr = boot_worst(rp_units(A.units, raw, "C", P), A.idx_c)
        Rp = boot_worst(rp_units(A.units, con, "C", P), A.idx_c)
        d.append(Rr - Rp)
    d = np.mean(d, axis=0)
    print(f"H1 measurability: R_raw - R_prior ({P} tier, worst of 3 attackers) = {d.mean():.3f} [{ci(d)[0]:.3f}, {ci(d)[1]:.3f}] (rule: >= 0.05 and CI excludes 0): "
          f"{'PASS' if ci(d)[0] > 0 and d.mean() >= 0.05 else 'FAIL'}\n")
    orc = {s: next((r for r in A.recs.values() if r["seed"] == s and r["spec"].get("mask") == OM), None) for s in seeds}
    if any(v is None for v in orc.values()):
        return
    Lo = np.mean([A.L_boot(r["rid"], s, P)[0] for s, r in orc.items()], axis=0)
    ret_o = np.mean([A.retention(r) for r in orc.values()])
    ret_od = np.mean([A.retention(r, "dev") for r in orc.values()])
    gate = tauO if tauO is not None else 0.97
    print(f"oracle_mask_{OM}: L_{P} = {Lo.mean():.3f} [{ci(Lo)[0]:.3f}, {ci(Lo)[1]:.3f}], retention dev {ret_od:.3f} / test {ret_o:.3f}; H2(a) (both >= 0.97): {'PASS' if min(ret_od, ret_o) >= 0.97 else 'FAIL'}"
          + ("" if tauO is None else f" (not a valid reference at tau = 0.97, so (b) is judged at the oracle's own dev retention tau_O = {tauO:.3f})") + "\n")
    print(f"| tau | strongest floor F* (by V) | F* configs | F* test retention | L_{P}(F*) [95% CI] | L_{P}(oracle) | headroom = L(F*) - L(oracle) [95% CI] | H2(b) (>= 0.10, lower bound > 0) |")
    print("|---|---|---|---|---|---|---|---|")
    for tau in [t for t in taus if t >= 0.9] + ([tauO] if tauO is not None else []):
        best = None
        for fam in FLOORS:
            sel = {s: A.select(fam, s, tau) for s in seeds}
            if any(v is None for v in sel.values()):
                continue
            LV = np.mean([(worst(r, "V", P) - worst(A.ref(s, P)["const"], "V", P)) /
                          max(worst(A.ref(s, P)["raw"], "V", P) - worst(A.ref(s, P)["const"], "V", P), 1e-6) for s, r in sel.items()])
            if best is None or LV < best[0]:
                best = (LV, fam, sel)
        if best is None:
            print(f"| {tau:.3f} | none feasible | | | | | | |")
            continue
        _, fam, sel = best
        Lf = np.mean([A.L_boot(r["rid"], s, P)[0] for s, r in sel.items()], axis=0)
        hd = Lf - Lo
        rt = np.mean([A.retention(r) for r in sel.values()])
        flag = "" if tau == gate else (" (fallback if no floor reaches 0.97 in every seed)" if tauO is None else " (oracle not valid at this tau; reference only)")
        verdict = ("PASS" if hd.mean() >= 0.10 and ci(hd)[0] > 0 else "FAIL") if tau == gate else "n/a"
        print(f"| {tau:.3f}{flag} | {fam} | {', '.join(f's{s_}:' + r_['arm'] for s_, r_ in sel.items())} | {rt:.3f} | {Lf.mean():.3f} [{ci(Lf)[0]:.3f}, {ci(Lf)[1]:.3f}] | {Lo.mean():.3f} | {hd.mean():.3f} [{ci(hd)[0]:.3f}, {ci(hd)[1]:.3f}] | {verdict} |")
    print()


def mde_print(A, fstar, tau, seeds):
    """Paired-bootstrap SD of L(F*) - L(bott_mlp control) at tau (seed-averaged, shared resamples) and MDE80 = 2.8 x SD (alpha 0.05 two-sided, power 0.8).
    Computed from floors + the (valid) no-adversary control only, i.e. before any repaired-adversary number exists."""
    ds = []
    for s in seeds:
        rf, rb = A.select(fstar, s, tau), A.select("bott_mlp", s, tau)
        if rf is None or rb is None or A.ref(s, P) is None:
            print(f"MDE tau={tau}: no pair for seed {s} (bott_mlp feasible: {rb is not None})\n")
            return
        ds.append(A.L_boot(rf["rid"], s, P)[0] - A.L_boot(rb["rid"], s, P)[0])
    d = np.mean(ds, axis=0)
    sd = d.std()
    print(f"**MDE at tau={tau}** (F*={fstar} vs bott_mlp, {len(seeds)} seeds): paired L difference {d.mean():.3f}, bootstrap SD {sd:.4f}, MDE80 = 2.8 x SD = **{2.8 * sd:.3f}**\n")


def collapse_table(A):
    adv = sorted((r for r in A.recs.values() if r["family"] == "adv"), key=lambda r: (r["spec"].get("lam", 0), r["dim"], r["seed"]))
    if not adv:
        return
    print("### Collapse check of the repaired adversarial fits (reject if final retrieval loss > ln N - 0.5 = 7.415)\n")
    print("| arm | seed | final retrieval loss | confusion / initial (last log) | collapsed | dev retention | R_V primary |")
    print("|---|---|---|---|---|---|---|")
    for r in adv:
        cr = r["train_log"][-1].get("conf_rel")
        print(f"| {r['arm']} | {r['seed']} | {r['final_ret']:.2f} | {'n/a' if cr is None else f'{cr:.2f}'} | {'YES' if r.get('collapsed') else 'no'} | {A.retention(r, 'dev'):.3f} | {worst(r, 'V', P):.3f} |")
    n = sum(1 for r in adv if r.get("collapsed"))
    print(f"\n{n} of {len(adv)} adversarial fits collapsed and are excluded from selection.\n")


def kill_rule(A, fstar, tau, m, seeds):
    print(f"#### Kill rule at tau={tau}, margin m={m} (pre-registered K1-K4), F*={fstar}\n")
    print("| seed | adv config | F* config | adv test ret. | F* test ret. | L(F*)-L(adv) [95% CI] | L(bott)-L(adv) [95% CI] | mid: L(adv)-L(F*) |")
    print("|---|---|---|---|---|---|---|---|")
    k1 = k2 = k3 = k4 = 0
    ok_seeds = []
    for s in seeds:
        ra, rf, rb = A.select("adv", s, tau), A.select(fstar, s, tau), A.select("bott_mlp", s, tau)
        if ra is None or rf is None or A.ref(s, P) is None:
            print(f"| {s} | n/a | | | | | | |")
            continue
        La = A.L_boot(ra["rid"], s, P)[0]
        Lf = A.L_boot(rf["rid"], s, P)[0]
        d1 = Lf - La
        d3 = (A.L_boot(rb["rid"], s, P)[0] - La) if rb is not None else None
        mid = A.L_boot(ra["rid"], s, S2)[0].mean() - A.L_boot(rf["rid"], s, S2)[0].mean()
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
        ok_seeds.append(bool(p1 and p2 and p3 and p4))
        print(f"| {s} | {ra['arm']} | {rf['arm']} | {ret_a:.3f} | {ret_f:.3f} | {d1.mean():.3f} [{ci(d1)[0]:.3f}, {ci(d1)[1]:.3f}] "
              f"({'ok' if p1 else 'fail'}; ret {'ok' if p2 else 'fail'}) | {d3s} | {mid:.3f} ({'ok' if p4 else 'fail'}) |")
    need = len(seeds) if len(seeds) < 3 else 2
    print(f"\nK1+K2 (>= m with equal utility) in {k1}/{len(seeds)} seeds; K1+K2+K3 in {k3}/{len(seeds)}; K4 in {k4}/{len(seeds)}. "
          f"PASS requires K1+K2+K3+K4 in >= {need} of {len(seeds)} seeds (all of them when only 2 seeds exist): **{'PASS' if sum(ok_seeds) >= need else 'FAIL'}** "
          f"({sum(ok_seeds)} seeds pass all four).\n")


if __name__ == "__main__":
    main()
