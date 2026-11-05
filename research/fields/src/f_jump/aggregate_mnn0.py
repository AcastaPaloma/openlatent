"""H3-005 stage 0 aggregator (results_notes/F.md section 22).  CPU only.

  python aggregate_mnn0.py --runs <run_s0> <run_s1> <run_s2> [--json out.json] [--md out.md] [--B 1000]

Reads meta_s{0,1,2}.json (MNN statistics) and preds_s0.npz (AP vectors of mlp_a / mlp_b1 / mlp_b3 / orc_q1 / orc_q3).  Applies the section-22 gates at the
original (1% retention at 0.80 hidden precision; oracle ceiling >= 0.010) and tightened (2% at 0.85; >= 0.020) thresholds.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def boot(a, b, B=1000, seed=0):
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    bs = np.array([d[rng.randint(n, size=n)].mean() for _ in range(B)])
    return float(d.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    meta, preds = {}, {}
    for r in a.runs:
        for p in Path(r).glob("meta_s*.json"):
            m = json.loads(p.read_text())
            if m.get("mnn0"):
                meta[m["salt"]] = m
                pn = p.with_name(p.name.replace("meta_", "preds_").replace(".json", ".npz"))
                if pn.exists():
                    preds[m["salt"]] = np.load(pn)
    salts = sorted(meta)
    out = dict(splits={}, original={}, tightened={})
    lines = ["# H3-005 stage 0: cross-source MNN pseudo-pairs (JUMP S5, view A)", ""]
    thr = {"original": dict(ret=0.01, prec=0.80, oracle=0.010), "tightened": dict(ret=0.02, prec=0.85, oracle=0.020)}
    p1_ok = True
    g1 = {k: {} for k in thr}
    g2 = {k: {} for k in thr}
    for s in salts:
        m = meta[s]
        st = m["mnn0_stats"]
        out["splits"][s] = dict(pool=m["mnn0_pool"], labels=m["mnn0_labels"], stats=st, train={k: dict(loss_first20=v["loss_first20"], loss_last200=v["loss_last200"]) for k, v in m["mnn0_train"].items()})
        labs = ", ".join("%s: %d" % (k_, v_["n_labelled_prod_wells"]) for k_, v_ in m["mnn0_labels"].items())
        lines += ["## split %s: pool %d production wells; labelled production wells {%s}" % (s, m["mnn0_pool"]["n_production_wells"], labs), "",
                  "| embedding | mutual pairs | tau_cal (lab pairs in prefix) | G1 retention / hidden precision | G2 retention (tau*) | G2 at 0.85 retention | P1 random precision | P2 shuffled-source retention / precision / true cross-source share |", "|---|---|---|---|---|---|---|---|"]
        for key, r in st.items():
            P2 = r["P2_shuffled_source_at_tau_cal"]
            lines.append(f"| {key} | {r['n_mutual']} | {r['tau_cal']:.3f} ({r['k_cal']}) | {r['G1']['retention']:.4f} / {r['G1']['precision']:.3f} | {r['G2']['retention']:.4f} ({r['tau_oracle']:.3f}) | {r['G2_p85']['retention']:.4f} | "
                         f"{r['P1_random']['precision']:.4f} | {P2['retention']:.4f} / {P2['precision']:.3f} / {P2.get('true_cross_source_share', float('nan')):.2f} |")
            if r["P1_random"]["precision"] > 0.02:
                p1_ok = False
        lines.append("")
        for k, t in thr.items():
            ok1, ok2 = False, False
            for R in ("a", "b1"):
                r = st.get(f"{R}|adapter")
                if r is None:
                    continue
                if r["G1"]["n_pairs"] > 0 and r["G1"]["precision"] >= t["prec"] and r["G1"]["retention"] >= t["ret"]:
                    ok1 = True
                g2r = r["G2"] if t["prec"] == 0.80 else r["G2_p85"]
                if g2r["n_pairs"] > 0 and g2r["retention"] >= t["ret"]:
                    ok2 = True
            g1[k][s], g2[k][s] = ok1, ok2
    # oracle ceiling (split s0)
    cei = {}
    if "s0" in preds:
        z = preds["s0"]
        U = {n: z[f"apc|H|{n}|xs"] for n in ("mlp_a_d8", "mlp_b1_d8", "mlp_b3_d8", "orc_q1", "orc_q3") if f"apc|H|{n}|xs" in z.files}
        cei["U_H"] = {n: float(np.nanmean(v)) for n, v in U.items()}
        for nm, (x, y) in {"orc_q1 - b1": ("orc_q1", "mlp_b1_d8"), "orc_q3 - b1": ("orc_q3", "mlp_b1_d8"), "b3 - b1": ("mlp_b3_d8", "mlp_b1_d8"), "orc_q3 - b3": ("orc_q3", "mlp_b3_d8"), "orc_q1 - b3": ("orc_q1", "mlp_b3_d8")}.items():
            if x in U and y in U:
                ok = ~(np.isnan(U[x]) | np.isnan(U[y]))
                d, ci = boot(U[x][ok], U[y][ok], a.B)
                cei[nm] = dict(dU=d, ci=ci)
        lines += ["## oracle-pair ceiling (split s0, U_H = mean cross-source replicate AP over H compounds, P0)", "", f"U_H: {json.dumps({k: round(v, 4) for k, v in cei['U_H'].items()})}", ""]
        for nm in ("orc_q1 - b1", "orc_q3 - b1", "b3 - b1", "orc_q1 - b3", "orc_q3 - b3"):
            if nm in cei:
                lines.append(f"* {nm}: {cei[nm]['dU']:+.4f} [{cei[nm]['ci'][0]:+.4f}, {cei[nm]['ci'][1]:+.4f}]")
        lines.append("")
    verd = {}
    for k, t in thr.items():
        g3 = bool("orc_q1 - b1" in cei and cei["orc_q1 - b1"]["dU"] >= t["oracle"] and cei["orc_q1 - b1"]["ci"][0] > 0)
        n1, n2 = sum(g1[k].values()), sum(g2[k].values())
        if not p1_ok:
            v = "INVALID (random-pair precision > 0.02)"
        elif n2 < 2:
            v = "FAIL-PAIRS (G2: fewer than 2 of 3 splits reach the retention at the precision even with the hidden-label threshold)"
        elif not g3:
            v = "FAIL-CEILING (perfect pairs at the gate size are worth less than the margin)"
        elif n1 >= 2:
            v = "STAGE 1 JUSTIFIED"
        else:
            v = "FAIL-CALIBRATION (clean pairs exist and perfect pairs help, but the threshold cannot be set from the labelled part)"
        verd[k] = dict(verdict=v, G1_splits=g1[k], G2_splits=g2[k], G3=g3, thresholds=t)
        lines.append(f"**{k} verdict: {v}**  (G1 in {n1}/{len(salts)} splits, G2 in {n2}/{len(salts)}, G3 {g3})")
    out["verdicts"], out["ceiling"], out["P1_ok"] = verd, cei, p1_ok
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    if a.md:
        Path(a.md).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
