"""H3-007 stage 1 aggregator (results_notes/F.md section 12).  How much of the F3 gain survives each shortcut control?

  python aggregate_s1.py --s1 d0,d1,d2 --merged $HOME/work/f3_regen/merge --salts s0,s1,s2 --out $SCR/s1 [--B 1000]
`--merged` holds merged3_<salt>/ (S5 + F2 + F3 outputs: stored leakage L and the stored U_H used for the reproduction check).  Writes <out>.json/.md (scratch path).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregate as AG

DELTA_U = 0.010
MATCH = 0.05
PROTOS = ["P0", "Pa", "Pb", "Pc", "Pd", "Pabc"]
PNAME = {"P0": "original", "Pa": "position-restricted positives", "Pb": "position-matched negatives", "Pc": "near-duplicate pruning", "Pd": "source-level negatives", "Pabc": "all controls (a+b+c)"}
FLOORS = ["raw", "pca16", "dpca16_r1", "dpca16_r2", "ldaT2_8", "ldaP_4", "ldaP_8", "ldaP_16"]
LF = ["raw", "pca16", "dpca16_r1", "dpca16_r2"]
ADAPT = {"a": ["mlp_a_d8", "mlp_a_d8_r0.5"], "b10": ["mlp_b10_d8", "mlp_b10_d8_r0.5"], "b100": ["mlp_d8", "mlp_d8_r0.5"]}
LDAM = {"a": "ldaM_a_8", "b10": "ldaM_b10_8", "b100": "ldaM_b100_8"}
POS = ["pos_P1_3number", "pos_P2_bucket8", "pos_P3_onehot", "pos_P4_shuffled"]


def boot_mean(counts, d):
    return (counts @ d.astype(np.float32)) / counts.shape[1]


def plate_cluster(P, a, b, proto, B, rng):
    ax, ay = P[f"apq|H|{a}|{proto}"].astype(np.float64), P[f"apq|H|{b}|{proto}"].astype(np.float64)
    ok = ~np.isnan(ax) & ~np.isnan(ay)
    ids, inv = np.unique(P["plate|H"], return_inverse=True)
    ds = np.bincount(inv[ok], weights=(ax - ay)[ok], minlength=len(ids))
    n = np.bincount(inv[ok], minlength=len(ids)).astype(np.float64)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        bs.append((w @ ds) / (w @ n))
    return AG.ci(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--salts", default="s0,s1,s2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    res = dict(margin=DELTA_U, protocols=PNAME, splits={})
    md = [f"# H3-007 stage 1 aggregate (pre-registration: results_notes/F.md section 12); units: H compounds, plates; B={a.B}\n"]
    surv = {R: {p: [] for p in PROTOS} for R in ("a", "b10", "b100")}
    valid_all = {p: [] for p in PROTOS}
    repro = []
    for i, (d, salt) in enumerate(zip(a.s1.split(","), a.salts.split(","))):
        P = {k: v for k, v in np.load(Path(d) / f"preds_{salt}.npz").items()}
        meta = json.loads((Path(d) / f"meta_{salt}.json").read_text())
        sp = AG.Split(Path(a.merged) / f"merged3_{salt}", salt, 20, 700 + i, True, "ldaT2")          # stored leakage and stored U_H
        rng = np.random.default_rng(800 + i)
        r = dict(pruned_compounds=meta["s1_pruned_compounds"], train={k: dict(first=v["loss_first20"], last=v["loss_last200"]) for k, v in meta["s1_train"].items()})
        arms = meta["arms"]
        # ---- reproduction check (P0 vs the stored xs U_H)
        rep = {}
        for x in FLOORS + list(LDAM.values()) + sum(ADAPT.values(), []) + ["ctrl_noise16"]:
            if x in arms and sp.U("H", x) is not None:
                rep[x] = dict(stage1=float(P[f"apc|H|{x}|P0"].mean()), stored=sp.U("H", x))
                rep[x]["ok"] = bool(abs(rep[x]["stage1"] - rep[x]["stored"]) <= 0.005)
        r["reproduction"] = rep
        repro.append(all(v["ok"] for v in rep.values()))
        U = lambda s, x, p: float(P[f"apc|{s}|{x}|{p}"].mean())  # noqa: E731
        # ---- controls beside every protocol
        ctrl = {}
        for p in PROTOS:
            un = U("H", "ctrl_noise16", p)
            c = dict(noise16=un, **{x: U("H", x, p) for x in POS})
            c["P4_vs_noise_ok"] = bool(abs(c["pos_P4_shuffled"] - un) <= 0.01)
            c["P3_ok"] = bool(c["pos_P3_onehot"] <= un + (0.01 if p == "Pa" else 0.02)) if p in ("Pa", "Pb", "Pabc") else None
            c["valid"] = bool(c["P4_vs_noise_ok"] and (c["P3_ok"] is not False))
            ctrl[p] = c
            valid_all[p].append(c["valid"])
        r["controls"] = ctrl
        # ---- bootstrap count matrices per protocol (shared across arms)
        counts = {}
        for p in PROTOS:
            nC = len(P[f"cid|H|{p}"])
            counts[p] = rng.multinomial(nC, np.ones(nC) / nC, size=a.B).astype(np.float32)
        r["regimes"] = {}
        for R in ("a", "b10", "b100"):
            out_R = {}
            E_full = FLOORS + list(LDAM.values())
            E_match = LF + [LDAM[R]]
            for p in PROTOS:
                entry = {}
                for envname, env in (("full", E_full), ("match", E_match)):
                    best = None
                    for x in ADAPT[R]:
                        Lx = sp.L("V", x)
                        cand = [f for f in env if sp.L("V", f) <= Lx + MATCH]
                        if not cand:
                            continue
                        F = max(cand, key=lambda f: U("V", f, p))
                        gapV = U("V", x, p) - U("V", F, p)
                        if best is None or gapV > best[0]:
                            best = (gapV, x, F)
                    if best is None:
                        continue
                    _, x, F = best
                    dv = P[f"apc|H|{x}|{p}"].astype(np.float64) - P[f"apc|H|{F}|{p}"].astype(np.float64)
                    bb = boot_mean(counts[p], dv)
                    entry[envname] = dict(adapter=x, floor=F, U_H_adapter=U("H", x, p), U_H_floor=U("H", F, p), L_H_adapter=sp.L("H", x), L_H_floor=sp.L("H", F), dU_H=float(dv.mean()), ci=AG.ci(bb),
                                          plate_ci=plate_cluster(P, x, F, p, min(a.B, 500), rng))
                out_R[p] = entry
            r["regimes"][R] = out_R
            for p in PROTOS:
                base = out_R["P0"]["full" if R != "a" else "match"]["dU_H"]
                this = out_R[p]["full" if R != "a" else "match"]["dU_H"]
                surv[R][p].append(dict(dU=this, base=base, S=this / base if base else float("nan")))
        res["splits"][salt] = r
        # ---- markdown
        md.append(f"## split {salt}: pruned (near-duplicate) H compounds {meta['s1_pruned_compounds']}; reproduction of stored U_H within 0.005: {repro[-1]}\n")
        rows = ["| protocol | noise16 | pos P1 (3-number) | pos P3 (exact) | pos P4 (shuffled) | valid | b10 dU_H vs E_full [CI] (plate CI) | S | b100 dU_H vs E_full [CI] (plate CI) | S | a dU_H vs E_match [CI] | S |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in PROTOS:
            c = ctrl[p]

            def cell(R, env):
                e = r["regimes"][R][p][env]
                s_ = surv[R][p][-1]["S"]
                return f"{AG.fci(e['dU_H'], e['ci'], 4)} ({AG.fmt(e['plate_ci'][0], 4)}, {AG.fmt(e['plate_ci'][1], 4)}) | {AG.fmt(s_, 2)}"
            rows.append(f"| {p} ({PNAME[p]}) | {AG.fmt(c['noise16'])} | {AG.fmt(c['pos_P1_3number'])} | {AG.fmt(c['pos_P3_onehot'])} | {AG.fmt(c['pos_P4_shuffled'])} | {c['valid']} | {cell('b10', 'full')} | {cell('b100', 'full')} | {cell('a', 'match')} |")
        md.append("\n".join(rows) + "\n")
        rows = ["| protocol | regime | adapter | U_H adapter [L_H] | floor | U_H floor [L_H] | vs E_match: dU_H [CI] | same-label LDA ldaM at matched leakage |", "|---|---|---|---|---|---|---|---|"]
        for R in ("b10", "b100"):
            for p in ("P0", "Pb", "Pabc"):
                e = r["regimes"][R][p]["full"]
                m = r["regimes"][R][p]["match"]
                rows.append(f"| {p} | {R} | {e['adapter']} | {AG.fmt(e['U_H_adapter'])} [{AG.fmt(e['L_H_adapter'], 2)}] | {e['floor']} | {AG.fmt(e['U_H_floor'])} [{AG.fmt(e['L_H_floor'], 2)}] | {AG.fci(m['dU_H'], m['ci'], 4)} ({m['floor']}) | |")
        md.append("\n".join(rows) + "\n")
        rows = ["| arm | " + " | ".join(PROTOS) + " |", "|---|" + "---|" * len(PROTOS)]
        for x in POS + ["ctrl_noise16"] + FLOORS + list(LDAM.values()) + sum(ADAPT.values(), []):
            if x in arms:
                rows.append(f"| {x} | " + " | ".join(AG.fmt(U("H", x, p)) for p in PROTOS) + " |")
        md.append("U_H by arm and protocol (split " + salt + "):\n\n" + "\n".join(rows) + "\n")
        res["splits"][salt]["U_H_table"] = {x: {p: U("H", x, p) for p in PROTOS} for x in arms}
    # ---- verdict
    ns = len(res["splits"])
    ge2 = (lambda xs: sum(xs) >= 2) if ns >= 2 else (lambda xs: sum(xs) >= 1)
    get = lambda R, p: [x["dU"] for x in surv[R][p]]  # noqa: E731
    v = {}
    v["valid_protocols"] = {p: bool(all(valid_all[p])) for p in PROTOS}
    v["reproduction_ok_all"] = bool(all(repro))
    v["collapse_i_Pb_b10_below_0.010_in_>=2_splits"] = bool(ge2([x < 0.010 for x in get("b10", "Pb")]))
    v["collapse_ii_Pc_removes_>=50%_of_b100"] = bool(ge2([x < 0.5 * b["base"] for x, b in zip(get("b100", "Pc"), surv["b100"]["Pc"])]))
    v["collapse_iii_Pd_reverses_sign_b10"] = bool(ge2([x < 0 for x in get("b10", "Pd")]))
    v["not_supported_Pa_b10_below_0.010_in_>=2_splits"] = bool(ge2([x < 0.010 for x in get("b10", "Pa")]))
    v["not_supported_Pabc_b10_below_0.010_in_>=2_splits"] = bool(ge2([x < 0.010 for x in get("b10", "Pabc")]))
    v["survives_all"] = bool(all(all(x >= 0.015 for x in get("b10", p)) and all(x >= 0.095 for x in get("b100", p)) for p in ("Pa", "Pb", "Pc", "Pd", "Pabc")))
    v["collapse"] = bool(v["collapse_i_Pb_b10_below_0.010_in_>=2_splits"] or v["collapse_ii_Pc_removes_>=50%_of_b100"] or v["collapse_iii_Pd_reverses_sign_b10"])
    v["survival_fraction_mean_over_splits"] = {R: {p: float(np.nanmean([x["S"] for x in surv[R][p]])) for p in PROTOS} for R in surv}
    v["dU_H_mean_over_splits"] = {R: {p: float(np.mean([x["dU"] for x in surv[R][p]])) for p in PROTOS} for R in surv}
    res["verdict"] = v
    md.append("## Verdict\n\n```\n" + json.dumps(v, indent=1) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
