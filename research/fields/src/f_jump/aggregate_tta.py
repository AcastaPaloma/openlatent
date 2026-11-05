"""H3-004 pilot aggregator (results_notes/F.md section 16): test-time per-source recentring of the released vector.

  python aggregate_tta.py --run dir --out $SCR/tta [--B 1000]
Variants are named '<base>|<op>', op in none, T1, T2, T3, T5, Pglobal, Pshuf.  Non-inferiority: dU_self >= -0.010 (gains reported).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregate as AG

OPS = ["T1", "T2", "T3", "T5"]
PLACEBO = ["Pglobal", "Pshuf"]
ADAPT = ["mlp_b10_d8", "mlp_b10_d64", "mlp_d8", "mlp_d64"]
FLOORS_FOR = {"mlp_b10_d8": "ldaM_b10_8", "mlp_b10_d64": "ldaM_b10_8", "mlp_d8": "ldaM_b100_8", "mlp_d64": "ldaM_b100_8"}
FLOORS = ["ldaM_b10_8", "ldaM_b100_8", "ldaP_8", "ldaP_16", "dpca16_r1"]
MATCH = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    d = Path(a.run)
    P = {k: v for k, v in np.load(d / "preds_s0.npz").items()}
    meta = json.loads((d / "meta_s0.json").read_text())
    sp = AG.Split(d, "s0", a.B, 1000, True, "ldaT2")
    rng = np.random.default_rng(1001)
    counts = {}
    for p in ("xs", "Pabc"):
        nC = len(P[f"cid|H|{p}"])
        counts[p] = rng.multinomial(nC, np.ones(nC) / nC, size=a.B).astype(np.float32)
    key = lambda x, s, p: f"apc|{s}|{x}|{p}"  # noqa: E731
    U = lambda x, s, p: float(P[key(x, s, p)].mean())  # noqa: E731
    res = dict(verdict_original="NO-GO at the S4 gate (JK1 v0 literal); stage 1 partial survival; F4 PARTIAL (F.md 5, 12.6, 13)", base={})
    md = ["# H3-004 pilot aggregate (pre-registration: results_notes/F.md section 16); B=%d\n" % a.B]
    bases = [b for b in meta["base_arms"]] + FLOORS + ["ctrl_noise16"]
    bases = [b for b in dict.fromkeys(bases) if f"{b}|none" in meta["arms"]]
    # ---- table
    rows = ["| base | op | U_H P0 | U_H Pabc | L_V | L_H [CI] | dL_H vs none | dU_H(self) P0 [CI] | dU_H(self) Pabc |", "|---|---|---|---|---|---|---|---|---|"]
    for b in bases:
        e = {}
        for op in ["none"] + OPS + PLACEBO:
            v = f"{b}|{op}"
            if v not in meta["arms"]:
                continue
            ent = dict(U_H=U(v, "H", "xs"), U_V=U(v, "V", "xs"), L_H=sp.L("H", v), L_V=sp.L("V", v), worst_H=sp.worst("H", v))
            if f"apc|H|{v}|Pabc" in P:
                ent["U_H_Pabc"], ent["U_V_Pabc"] = U(v, "H", "Pabc"), U(v, "V", "Pabc")
            e[op] = ent
        for op in [o for o in e if o != "none"]:
            v0, v1 = f"{b}|none", f"{b}|{op}"
            dv = P[key(v1, "H", "xs")].astype(np.float64) - P[key(v0, "H", "xs")].astype(np.float64)
            bb = (counts["xs"] @ dv.astype(np.float32)) / counts["xs"].shape[1]
            e[op]["dU_self"], e[op]["dU_self_ci"] = float(dv.mean()), AG.ci(bb)
            e[op]["dU_self_V"] = e[op]["U_V"] - e["none"]["U_V"]
            e[op]["dL_H"], e[op]["dL_V"] = e["none"]["L_H"] - e[op]["L_H"], e["none"]["L_V"] - e[op]["L_V"]
            e[op]["dL_H_ci"] = AG.ci(-sp.dL_boot("H", v1, v0))
            if "U_H_Pabc" in e[op] and "U_H_Pabc" in e["none"]:
                dv2 = P[key(v1, "H", "Pabc")].astype(np.float64) - P[key(v0, "H", "Pabc")].astype(np.float64)
                e[op]["dU_self_Pabc"] = float(dv2.mean())
                e[op]["dU_self_Pabc_ci"] = AG.ci((counts["Pabc"] @ dv2.astype(np.float32)) / counts["Pabc"].shape[1])
        res["base"][b] = e
        for op in e:
            x = e[op]
            rows.append(f"| {b} | {op} | {AG.fmt(x['U_H'])} | {AG.fmt(x.get('U_H_Pabc'))} | {AG.fmt(x['L_V'], 2)} | {AG.fmt(x['L_H'], 2)} {x.get('dL_H_ci', '') if op == 'none' else ''} | {AG.fmt(x.get('dL_H'), 3)} {x.get('dL_H_ci', '') if op != 'none' else ''} | "
                        f"{AG.fci(x.get('dU_self'), x.get('dU_self_ci', [float('nan')] * 2), 4) if op != 'none' else '-'} | {AG.fmt(x.get('dU_self_Pabc'), 4) if op != 'none' else '-'} |")
    md.append("\n".join(rows) + "\n")
    # ---- controls (INVALID gates)
    ctrl = {}
    ctrl["placebo_max_dL_H"] = float(max(res["base"][b][o]["dL_H"] for b in bases for o in PLACEBO if o in res["base"][b]))
    nz = res["base"]["ctrl_noise16"]
    ctrl["noise16"] = {o: dict(dU=nz[o]["dU_self"], worst_H=nz[o]["worst_H"]) for o in OPS if o in nz}
    ctrl["noise16_ok"] = bool(all(abs(v["dU"]) <= 0.005 and v["worst_H"] <= 0.23 for v in ctrl["noise16"].values()))
    ctrl["placebo_ok"] = bool(ctrl["placebo_max_dL_H"] < 0.02)
    ctrl["valid"] = bool(ctrl["noise16_ok"] and ctrl["placebo_ok"])
    res["controls"] = ctrl
    # ---- pilot decision: V-selected operator per adapter arm, H read once
    dec = {}
    for b in ADAPT:
        if b not in res["base"]:
            continue
        e = res["base"][b]
        cand = [o for o in OPS if o in e and e[o]["dU_self_V"] >= -0.010]
        if not cand:
            dec[b] = dict(op=None)
            continue
        o = max(cand, key=lambda o_: e[o_]["dL_V"])
        x = e[o]
        dec[b] = dict(op=o, dL_H=x["dL_H"], dL_H_ci=x["dL_H_ci"], dU_self_P0=x["dU_self"], dU_self_P0_ci=x["dU_self_ci"], dU_self_Pabc=x.get("dU_self_Pabc"),
                      passes=bool(x["dL_H"] >= 0.05 and x["dU_self"] >= -0.010 and x.get("dU_self_Pabc", -1) >= -0.010))
    res["decision"] = dec
    # ---- same-treatment comparison: adapter+op vs floors+op at matched leakage (V-selected floor), and vs the untreated envelope
    st = {}
    for b in ADAPT:
        if b not in dec or dec[b]["op"] is None:
            continue
        o = dec[b]["op"]
        x = f"{b}|{o}"
        for tag, ops_f in (("same_treatment", o), ("untreated_envelope", "none")):
            cand = [f"{f}|{ops_f}" for f in FLOORS if f"{f}|{ops_f}" in meta["arms"] and sp.L("V", f"{f}|{ops_f}") <= sp.L("V", x) + MATCH]
            if not cand:
                continue
            F = max(cand, key=lambda f_: U(f_, "V", "xs"))
            dv = P[key(x, "H", "xs")].astype(np.float64) - P[key(F, "H", "xs")].astype(np.float64)
            bb = (counts["xs"] @ dv.astype(np.float32)) / counts["xs"].shape[1]
            ent = dict(adapter=x, floor=F, U_H_adapter=U(x, "H", "xs"), U_H_floor=U(F, "H", "xs"), L_H_adapter=sp.L("H", x), L_H_floor=sp.L("H", F), dU=float(dv.mean()), ci=AG.ci(bb))
            if f"apc|H|{x}|Pabc" in P and f"apc|H|{F}|Pabc" in P:
                dv2 = P[key(x, "H", "Pabc")].astype(np.float64) - P[key(F, "H", "Pabc")].astype(np.float64)
                ent["dU_Pabc"] = float(dv2.mean())
            st[f"{b}|{tag}"] = ent
    res["comparison"] = st
    # T5 vs best transductive operator
    t5 = {b: dict(L_T5=res["base"][b]["T5"]["L_H"], L_best_transductive=min(res["base"][b][o]["L_H"] for o in ("T1", "T2", "T3"))) for b in ADAPT if b in res["base"]}
    for v in t5.values():
        v["within_0.03"] = bool(abs(v["L_T5"] - v["L_best_transductive"]) <= 0.03)
    res["T5_vs_transductive"] = t5
    passes = [b for b, x in dec.items() if x.get("passes")]
    res["verdict_tightened"] = "INVALID" if not ctrl["valid"] else ("PILOT PASSES (extend to s1/s2 and b1/b3)" if passes else "FALSIFIED / NOT PASSED (no adapter arm lowers L_H by >= 0.05 at non-inferior utility under both scorings)")
    res["passing_arms"] = passes
    md.append("## Decision\n\n```\n" + json.dumps(dict(verdict_original=res["verdict_original"], verdict_tightened=res["verdict_tightened"], controls=ctrl, decision=dec, comparison=st, T5_vs_transductive=t5), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
