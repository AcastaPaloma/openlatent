"""Focus D3 aggregation: cluster-bootstrap tables (numbers only).  RUN ON THE VM (per-document arrays must not leave scratch):

  python pii_aggregate.py RUNDIR [RUNDIR ...] [--taus 0.97,0.90] [--margins 0.90:0.08,0.97:0.10] [--mde] > ~/work/runs/d3_agg/<name>.md

Prints markdown with aggregates only.  Bootstrap unit = template cluster (B = 1000, shared resamples across arms => paired).
Normalised leakage L = (R - R_prior)/(R_raw - R_prior) with the denominator held at its bootstrap mean; gap-closing share G = (R - R_ref)/(1 - R_ref),
R_ref = max(R_prior, R_B1) (D.md 8.2).  Acknowledgement: AI4Privacy pii-masking-300k, academic non-commercial use.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ATT = ("ridge", "mlp", "knn")
FLOORS = ("randproj", "pca", "noise", "leace", "sup_lin")
P, S2 = "val", "typ"
B = 1000


class Agg:
    def __init__(self, dirs):
        self.recs, self.units = {}, {}
        for d in dirs:
            d = Path(d)
            for line in open(d / "records.jsonl"):
                r = json.loads(line)
                if "error" not in r:
                    self.recs[r["rid"]] = r
            z = np.load(d / "units.npz")
            for k in z.files:
                self.units[k] = z[k]
            self.index = np.load(d / "units_index.npz")
        self.seeds = sorted({r["seed"] for r in self.recs.values()})
        rng = np.random.default_rng(0)
        self.W = {}
        for sp_, key in (("V", "cluster_V"), ("C", "cluster_C")):
            cl = self.index[key]
            u, inv = np.unique(cl, return_inverse=True)
            draw = rng.integers(0, len(u), size=(B, len(u)))
            Wc = np.stack([np.bincount(row, minlength=len(u)) for row in draw]).astype(np.float32)
            self.W[sp_] = Wc[:, inv]
            self.n = {**getattr(self, "n", {}), sp_: (len(cl), len(u))}

    def bm(self, u, split):
        m = ~np.isnan(u)
        W = self.W[split][:, m]
        return (W @ u[m]) / W.sum(1)

    def R(self, rid, split, tier):
        return np.max([self.bm(self.units[f"{rid}|rp_{split}_{tier}_{a}"], split) for a in ATT], axis=0)

    def pt(self, rid, split, tier):
        return max(self.recs[rid]["attackers"][a][split][tier] for a in ATT)

    def ret_boot(self, rid, seed, split="C"):
        return self.bm(self.units[f"{rid}|nd_{split}"], split) / self.bm(self.units[f"raw|s{seed}|nd_{split}"], split)

    def ret_pt(self, rec, split):
        return rec[f"ndcg_{split}"] / rec["ndcg_raw"][split]

    def ref(self, seed):
        return f"raw|s{seed}", f"const|s{seed}"

    def L(self, rid, seed, tier, split="C"):
        raw, con = self.ref(seed)
        Ra, Rr, Rc = self.R(rid, split, tier), self.R(raw, split, tier), self.R(con, split, tier)
        den = max(float(np.mean(Rr - Rc)), 1e-6)
        return (Ra - Rc) / den

    def G(self, rid, seed, tier, split="C", b1=True):
        raw, con = self.ref(seed)
        Rc = self.R(con, split, tier)
        ref = np.maximum(Rc, self.R(f"feat_b1|s{seed}", split, tier)) if b1 and f"feat_b1|s{seed}" in self.recs else Rc
        return (self.R(rid, split, tier) - ref) / (1 - ref)

    def fam(self, family, seed):
        return [r for r in self.recs.values() if r["family"] == family and r["seed"] == seed and not r["spec"].get("mask") and not r["spec"].get("shuffle")
                and not r.get("collapsed")]

    def select(self, family, seed, tau):
        ok = [r for r in self.fam(family, seed) if self.ret_pt(r, "V") >= tau]
        if not ok:
            return None
        return min(ok, key=lambda r: (round(self.pt(r["rid"], "V", P), 4), r["dim"], r["spec"].get("lam", 0)))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def f(x, nd=3):
    return f"{np.mean(x):.{nd}f} [{ci(x)[0]:.{nd}f}, {ci(x)[1]:.{nd}f}]"


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--taus", default="0.97,0.90")
    ap.add_argument("--margins", default="")
    ap.add_argument("--mde", action="store_true")
    ap.add_argument("--primary", default="val", help="primary tier: val (registered) or name / place (post hoc D3b)")
    a = ap.parse_args()
    P = a.primary
    A = Agg(a.dirs)
    seeds, taus = A.seeds, [float(t) for t in a.taus.split(",")]
    margins = {float(x.split(":")[0]): float(x.split(":")[1]) for x in a.margins.split(",") if x}
    print(f"records {len(A.recs)}, seeds {seeds}; V kept {A.n['V'][0]} docs / {A.n['V'][1]} clusters, C kept {A.n['C'][0]} docs / {A.n['C'][1]} clusters\n")
    raw_nd = {s: A.recs[f"raw|s{s}"]["ndcg_raw"] for s in seeds if f"raw|s{s}" in A.recs}

    # ---------------- controls / references
    print("### References and controls (sealed C, cluster bootstrap, mean of seeds; worst-of-3 attackers)\n")
    print("| arm | R_val [CI] | R_name | R_place | R_type | L_val [CI] | G_val vs (prior, B1) [CI] | nDCG@10 C | retention C [CI] |")
    print("|---|---|---|---|---|---|---|---|---|")
    names = ["const", "raw|shuffle", "feat_b1", "feat_b1g", "raw", "raw|mask_oracle"]
    for nm in names:
        rids = [f"{nm}|s{s}" for s in seeds if f"{nm}|s{s}" in A.recs]
        if not rids:
            continue
        Rv = np.mean([A.R(r, "C", P) for r in rids], axis=0)
        tiers = [np.mean([A.R(r, "C", t) for r in rids], axis=0).mean() for t in ("name", "place", S2)]
        s_list = [int(r.split("|s")[-1]) for r in rids]
        Lv = f(np.mean([A.L(r, s, P) for r, s in zip(rids, s_list)], axis=0), 2) if all(f"const|s{s}" in A.recs and f"raw|s{s}" in A.recs for s in s_list) else "n/a"
        Gv = f(np.mean([A.G(r, s, P) for r, s in zip(rids, s_list)], axis=0), 3) if all(f"const|s{s}" in A.recs for s in s_list) else "n/a"
        nd = np.mean([A.recs[r].get("ndcg_C", np.nan) for r in rids])
        rt = f(np.mean([A.ret_boot(r, s) for r, s in zip(rids, s_list)], axis=0), 3) if "ndcg_C" in A.recs[rids[0]] else "n/a"
        print(f"| {nm} | {f(Rv)} | {tiers[0]:.3f} | {tiers[1]:.3f} | {tiers[2]:.3f} | {Lv} | {Gv} | {nd:.4f} | {rt} |")
    rs = [A.recs[f"recsplit|s{s}"] for s in seeds if f"recsplit|s{s}" in A.recs]
    if rs:
        w = lambda r: max(r["attackers"][a]["C"]["val"] for a in r["attackers"])
        print(f"\nRecord-split vs cluster-split demonstration (raw vectors, worst-of-3 R_val): record-level split {np.mean([w(r) for r in rs]):.3f}; cluster split (purged C) "
              f"{np.mean([A.pt(f'raw|s{s}', 'C', P) for s in seeds]):.3f}; per attacker record split "
              + ", ".join(f"{a}={np.mean([r['attackers'][a]['C']['val'] for r in rs]):.3f}" for a in ATT)
              + "; cluster split " + ", ".join(f"{a}={np.mean([A.recs[f'raw|s{s}']['attackers'][a]['C'][P] for s in seeds]):.3f}" for a in ATT) + "\n")

    # ---------------- H1
    if all(f"raw|s{s}" in A.recs and f"const|s{s}" in A.recs and f"feat_b1|s{s}" in A.recs for s in seeds):
        print("### H1 (measurability): gap-closing share G of the RAW embedding over max(prior, B1) on C, S2 value tokens\n")
        Gm = np.mean([A.G(f"raw|s{s}", s, P) for s in seeds], axis=0)
        ok = Gm.mean() >= 0.10 and ci(Gm)[0] >= 0.05
        print(f"G_raw = {f(Gm)} (rule: >= 0.10 with lower bound >= 0.05): **{'PASS' if ok else 'FAIL'}**")
        for t in ("name", "place", S2):
            Gt = np.mean([A.G(f"raw|s{s}", s, t) for s in seeds], axis=0)
            Rg = np.mean([A.pt(f"raw|s{s}", "C", t) for s in seeds])
            Rb = np.mean([A.pt(f"feat_b1|s{s}", "C", t) for s in seeds])
            Rp = np.mean([A.pt(f"const|s{s}", "C", t) for s in seeds])
            print(f"* tier {t} (descriptive): R_raw {Rg:.3f}, R_B1 {Rb:.3f}, R_prior {Rp:.3f}, G = {f(Gt)}")
        Rg1 = np.mean([A.pt(f"raw|s{s}", "C", P) for s in seeds]); Rb1g = np.mean([A.pt(f"feat_b1g|s{s}", "C", P) for s in seeds]) if f"feat_b1g|s0" in A.recs else float("nan")
        print(f"* B1g (context only, types unknown) R_val {Rb1g:.3f}; shuffled-pairs control R_val {np.mean([A.pt(f'raw|shuffle|s{s}', 'C', P) for s in seeds]):.3f} vs prior {np.mean([A.pt(f'const|s{s}', 'C', P) for s in seeds]):.3f}\n")

    # ---------------- selection tables
    fams = [f_ for f_ in (*FLOORS, "bott_mlp", "adv") if any(r["family"] == f_ for r in A.recs.values())]
    if fams:
        print("### All configs (mean over seeds; retention on V / C queries; R worst-of-3 value-token R-precision)\n")
        print("| arm | dim | seeds | collapsed | retention V | retention C | R_V | R_C | R_C type |")
        print("|---|---|---|---|---|---|---|---|---|")
        by = {}
        for r in A.recs.values():
            if r["family"] in fams and not r["spec"].get("mask"):
                by.setdefault(r["arm"], []).append(r)
        for arm in sorted(by, key=lambda x: (by[x][0]["family"], -by[x][0]["dim"], x)):
            rs_ = by[arm]
            print(f"| {arm} | {rs_[0]['dim']} | {len(rs_)} | {sum(bool(r.get('collapsed')) for r in rs_)} | {np.mean([A.ret_pt(r, 'V') for r in rs_]):.3f} | "
                  f"{np.mean([A.ret_pt(r, 'C') for r in rs_]):.3f} | {np.mean([A.pt(r['rid'], 'V', P) for r in rs_]):.3f} | {np.mean([A.pt(r['rid'], 'C', P) for r in rs_]):.3f} | "
                  f"{np.mean([A.pt(r['rid'], 'C', S2) for r in rs_]):.3f} |")
        print()
    orc_ok = all(f"raw|mask_oracle|s{s}" in A.recs for s in seeds)
    if orc_ok:
        Lo = np.mean([A.L(f"raw|mask_oracle|s{s}", s, P) for s in seeds], axis=0)
        rdev = np.mean([A.ret_pt(A.recs[f"raw|mask_oracle|s{s}"], "V") for s in seeds]); rtest = np.mean([A.ret_pt(A.recs[f"raw|mask_oracle|s{s}"], "C") for s in seeds])
        print(f"### Oracle (PII spans deleted before embedding): L_val = {f(Lo)}, retention V {rdev:.3f} / C {rtest:.3f}; H2(a) (both >= 0.97): **{'PASS' if min(rdev, rtest) >= 0.97 else 'FAIL'}**\n")
    for tau in taus:
        print(f"### Selected on V at retention >= {tau} (config per family and seed; numbers on sealed C)\n")
        print("| family | feasible seeds | configs | retention C [CI] | R_C val [CI] | L_val [CI] | L_type [CI] | mean V L_val |")
        print("|---|---|---|---|---|---|---|---|")
        summ = {}
        for fam in fams:
            sel = {s: A.select(fam, s, tau) for s in seeds}
            okk = {s: r for s, r in sel.items() if r is not None}
            if not okk:
                print(f"| {fam} | 0/{len(seeds)} | cannot reach tau | | | | | |")
                continue
            Lr = np.mean([A.L(r["rid"], s, P) for s, r in okk.items()], axis=0)
            Lt = np.mean([A.L(r["rid"], s, S2) for s, r in okk.items()], axis=0)
            Rr = np.mean([A.R(r["rid"], "C", P) for r in okk.values()], axis=0)
            Rt = np.mean([A.ret_boot(r["rid"], s) for s, r in okk.items()], axis=0)
            LV = np.mean([(A.pt(r["rid"], "V", P) - A.pt(f"const|s{s}", "V", P)) / max(A.pt(f"raw|s{s}", "V", P) - A.pt(f"const|s{s}", "V", P), 1e-6) for s, r in okk.items()])
            if len(okk) == len(seeds):
                summ[fam] = dict(LV=LV, L=Lr.mean())
            print(f"| {fam} | {len(okk)}/{len(seeds)} | {', '.join(f's{s}:' + r['arm'] for s, r in okk.items())} | {f(Rt)} | {f(Rr)} | {f(Lr, 2)} | {f(Lt, 2)} | {LV:.3f} |")
        floors = {k: v for k, v in summ.items() if k in FLOORS}
        if floors:
            fstar = min(floors, key=lambda k: floors[k]["LV"])
            print(f"\nStrongest floor F\\*({tau}) (lowest seed-mean V leakage, feasible in every seed): **{fstar}** (V L {floors[fstar]['LV']:.3f}; C L {floors[fstar]['L']:.3f})")
            if orc_ok:
                sel = {s: A.select(fstar, s, tau) for s in seeds}
                Lf = np.mean([A.L(r["rid"], s, P) for s, r in sel.items()], axis=0)
                hd = Lf - Lo
                print(f"H2(b) headroom at tau {tau} = L(F\\*) - L(oracle) = {f(hd)} (rule: >= 0.10, lower bound > 0): **{'PASS' if hd.mean() >= 0.10 and ci(hd)[0] > 0 else 'FAIL'}**")
            if a.mde and "bott_mlp" in fams:
                try:
                    ds = [A.L(A.select(fstar, s, tau)["rid"], s, P) - A.L(A.select("bott_mlp", s, tau)["rid"], s, P) for s in seeds]
                    d = np.mean(ds, axis=0)
                    print(f"MDE tau {tau}: F\\* vs bott_mlp paired L difference {d.mean():.3f}, bootstrap SD {d.std():.4f}, MDE80 = **{2.8 * d.std():.3f}**")
                except Exception as e_:
                    print(f"MDE tau {tau}: unavailable ({type(e_).__name__})")
            if tau in margins and "adv" in fams:
                kill(A, fstar, tau, margins[tau], seeds)
        print()

    # ---------------- engagement table
    adv = sorted((r for r in A.recs.values() if r["family"] == "adv"), key=lambda r: (r["spec"].get("adv_tier") or "", r["spec"].get("lam", 0), r["dim"], r["seed"]))
    if adv:
        print(f"### Engagement check of the adversarial arm (primary tier {P}; engaged iff E1 AND E2, not collapsed)\n")
        print("| arm | seed | collapsed | final retrieval loss | retention V | E1 L_head (held-out Dval) | head in-sample R | memorisation floor in / held-out R | E2 L_fresh on V | L_fresh ridge/mlp/knn | engaged |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in adv:
            e = r.get("engagement", {})
            raw, con = A.ref(r["seed"])
            et = e.get(P, {})
            den = max(A.pt(raw, "V", P) - A.pt(con, "V", P), 1e-6)
            lf = {a_: (r["attackers"][a_]["V"][P] - A.pt(con, "V", P)) / den for a_ in ATT}
            L2 = (A.pt(r["rid"], "V", P) - A.pt(con, "V", P)) / den
            E1 = et.get("L_head", np.nan) <= 0.5
            E2 = L2 <= 0.5
            mf = e.get("memfloor", {}).get(P, {})
            print(f"| {r['arm']} | {r['seed']} | {'YES' if r.get('collapsed') else 'no'} | {r['final_ret']:.2f} | {A.ret_pt(r, 'V'):.3f} | {et.get('L_head', float('nan')):.2f} | "
                  f"{et.get('R_head_insample', float('nan')):.3f} | {mf.get('insample', float('nan')):.3f} / {mf.get('heldout', float('nan')):.3f} | {L2:.2f} | "
                  f"{lf['ridge']:.2f}/{lf['mlp']:.2f}/{lf['knn']:.2f} | {'YES' if (E1 and E2 and not r.get('collapsed')) else 'no'} |")
        e0 = next((r["engagement"] for r in adv if "engagement" in r and P in r["engagement"]), None)
        if e0:
            print(f"\nYardsticks on Dval (n={e0['n_dval']}, tier {P}): frequency prior R = {e0[P]['R_prior_dval']:.3f}, fresh MLP on RAW defender vectors R = {e0[P]['R_mlp_raw_dval']:.3f}.\n")


def kill(A, fstar, tau, m, seeds):
    print(f"\n#### K1-K4 at tau={tau}, margin m={m}, F\\*={fstar}\n")
    print("| seed | adv config | F\\* config | adv test ret. | F\\* test ret. | L(F\\*)-L(adv) [CI] | L(bott)-L(adv) [CI] | type-tier: L(adv)-L(F\\*) |")
    print("|---|---|---|---|---|---|---|---|")
    ok_s = []
    for s in seeds:
        ra, rf, rb = A.select("adv", s, tau), A.select(fstar, s, tau), A.select("bott_mlp", s, tau)
        if ra is None or rf is None:
            print(f"| {s} | n/a | | | | | | |")
            ok_s.append(False)
            continue
        La, Lf = A.L(ra["rid"], s, P), A.L(rf["rid"], s, P)
        d1 = Lf - La
        d3 = (A.L(rb["rid"], s, P) - La) if rb is not None else None
        mid = A.L(ra["rid"], s, S2).mean() - A.L(rf["rid"], s, S2).mean()
        p1, p2 = ci(d1)[0] >= m, A.ret_pt(ra, "C") >= A.ret_pt(rf, "C") - 0.01
        p3 = d3 is not None and ci(d3)[0] >= m / 2
        p4 = mid <= 0.05
        ok_s.append(bool(p1 and p2 and p3 and p4))
        print(f"| {s} | {ra['arm']} | {rf['arm']} | {A.ret_pt(ra, 'C'):.3f} | {A.ret_pt(rf, 'C'):.3f} | {f(d1)} ({'ok' if p1 else 'fail'}; ret {'ok' if p2 else 'fail'}) | "
              f"{'n/a' if d3 is None else f(d3) + (' (ok)' if p3 else ' (fail)')} | {mid:.3f} ({'ok' if p4 else 'fail'}) |")
    print(f"\nK1-K4 in {sum(ok_s)}/{len(seeds)} seeds (required: all seeds): **{'PASS' if all(ok_s) else 'FAIL'}**\n")


if __name__ == "__main__":
    main()
