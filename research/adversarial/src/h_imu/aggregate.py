"""Aggregation for R-IMU: subject-cluster bootstrap, floors envelope, outcome labels (H.md 2.4, 2.8, 2.9, H-A1).
python aggregate.py --pred_dirs DIR1,DIR2 --out OUT.json [--learned msda,rep] [--control pd_pub_s0,...] [--folds O]
Per-arm statistics are computed from the saved window-level scores (preds__<arm>__<family>.npz); nothing here retrains a model.
"""
import os, sys, json, glob, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import datasets as D

MAIN_TAB = ["linear", "mlp", "knn", "lgbm", "rf"]; MAIN_NN = ["cnn1d", "mlp_flat"]; HELD = ["svm_rbf", "transformer"]
CONTROLS = {"ctrl_const", "ctrl_noise16", "shuf_raw"}; ORACLES = {"oracle_act", "oracle_id"}
LDA_ONLY = {"lda1", "lda2", "lda3", "lda3_r0.5", "lda3_r1", "lda3_r2"}
MDE_SET = ["lda3", "lda3_r1", "pcaphi8", "rp16", "noise_s1", "quant_b3"]
NSUB = 24
CHANCE = 1 / 24


class Store:
    def __init__(self, S):
        self.S = S; self.cache = {}
        self.tkey = C.trial_key(S)

    def fold_groups(self, te_idx):
        k = ("g", te_idx.tobytes()[:64], len(te_idx))
        if k not in self.cache:
            tk = self.tkey[te_idx]; order = np.argsort(tk, kind="stable"); tks = tk[order]
            starts = np.r_[0, np.where(np.diff(tks) != 0)[0] + 1]
            self.cache[k] = (order, starts, tks[starts], np.diff(np.r_[starts, len(tks)]))
        return self.cache[k]

    def arm_arrays(self, arm_files, folds):
        """arm_files: list of npz paths (tab and/or nn). Returns dict attacker -> dict of per-subject arrays (n_id,), summed over `folds`."""
        out = {}
        for fn in arm_files:
            d = np.load(fn)
            for f in folds:
                if f"f{f}_te_idx" not in d.files: continue
                te = d[f"f{f}_te_idx"].astype(int); order, starts, tks, cnts = self.fold_groups(te)
                sub_t = (tks // 1000).astype(int)
                sub_w = self.S["sub"][te]; act_w = self.S["act"][te]
                for key in d.files:
                    if not key.startswith(f"f{f}_") or key.endswith("te_idx"): continue
                    att, tg = key[len(f"f{f}_"):].split("__")
                    P = d[key].astype(np.float32)  # (S, n_te, C)
                    rec = out.setdefault(att, {}).setdefault(tg, dict(corr_t=np.zeros(NSUB), n_t=np.zeros(NSUB), corr_w=np.zeros(NSUB), n_w=np.zeros(NSUB)))
                    if tg == "id":
                        cw = np.mean([(p.argmax(1) == sub_w) for p in P], axis=0)  # seed-averaged window correctness
                        rec["corr_w"] += np.bincount(sub_w, weights=cw, minlength=NSUB); rec["n_w"] += np.bincount(sub_w, minlength=NSUB)
                        ct = np.zeros(len(starts))
                        for p in P:
                            pm = np.add.reduceat(p[order], starts, axis=0) / cnts[:, None]
                            ct += (pm.argmax(1) == sub_t)
                        ct /= len(P)
                        rec["corr_t"] += np.bincount(sub_t, weights=ct, minlength=NSUB); rec["n_t"] += np.bincount(sub_t, minlength=NSUB)
                    elif tg == "act":
                        cw = np.mean([(p.argmax(1) == act_w) for p in P], axis=0)
                        rec["corr_w"] += np.bincount(sub_w, weights=cw, minlength=NSUB); rec["n_w"] += np.bincount(sub_w, minlength=NSUB)
        return out


def boot_counts(B, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, NSUB, size=(B, NSUB))
    return np.stack([np.bincount(r, minlength=NSUB) for r in idx]).astype(np.float64)  # (B,24) multiplicities


def stat_from(rec_by_att, names, tgt, kind, Wt):
    """returns (point, boot[B], per-attacker point dict) of max over attackers `names` of the pooled accuracy."""
    pts, bts = {}, []
    for a in names:
        if a not in rec_by_att or tgt not in rec_by_att[a]: continue
        r = rec_by_att[a][tgt]; c, n = (r["corr_t"], r["n_t"]) if kind == "t" else (r["corr_w"], r["n_w"])
        pts[a] = float(c.sum() / n.sum()); bts.append((Wt @ c) / (Wt @ n))
    if not bts: return None, None, {}
    bts = np.stack(bts); best = max(pts, key=pts.get)
    return pts[best], bts.max(0), pts


def frontier_L_at_U(Ls, Us, u):
    """min leakage at utility >= u (piecewise-linear on the Pareto frontier; clamped to the best floor if u exceeds all floors)."""
    o = np.argsort(-Us); Ls, Us = Ls[o], Us[o]
    # frontier: as utility decreases, keep points with strictly lower leakage than all higher-utility points
    fl, fu, best = [], [], np.inf
    for l, uu in zip(Ls, Us):
        if l < best: fl.append(l); fu.append(uu); best = l
    fl, fu = np.array(fl[::-1]), np.array(fu[::-1])  # ascending leakage -> ascending utility
    if u >= fu[-1]: return fl[-1]
    if u <= fu[0]: return fl[0]
    return float(np.interp(u, fu, fl))


def frontier_U_at_L(Ls, Us, l):
    o = np.argsort(Ls); Ls, Us = Ls[o], Us[o]
    fl, fu, best = [], [], -np.inf
    for ll, u in zip(Ls, Us):
        if u > best: fl.append(ll); fu.append(u); best = u
    fl, fu = np.array(fl), np.array(fu)
    if l <= fl[0]: return fu[0]
    if l >= fl[-1]: return fu[-1]
    return float(np.interp(l, fl, fu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dirs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--learned", default="msda,rep"); ap.add_argument("--control", default="pd_pub_s0,pd_pub_s1,pd_pub_s2,pd_plain_s0,pd_w1_1.8_s0,pd_w2_0.5_s0")
    ap.add_argument("--folds", default="O"); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--set", default="astar", choices=["astar", "sealed", "wisdm"])
    ap.add_argument("--mL_floor", type=float, default=0.15); ap.add_argument("--mU_floor", type=float, default=0.02, help="pre-registered margin floors (raise-only): m = max(floor, empirical MDE80)")
    ap.add_argument("--exclude", default="", help="comma-separated arms removed from the floor set (used for the frozen-list reading, verdict_original)")
    a = ap.parse_args()
    folds = a.folds.split(",")
    global NSUB, CHANCE
    S = D.get_set(a.set); NSUB = int(S["n_id"]); CHANCE = 1.0 / NSUB; st = Store(S)
    files = {}
    for d in a.pred_dirs.split(","):
        for fn in glob.glob(os.path.join(os.path.expanduser(d), "preds__*__*.npz")):
            _, arm, fam = os.path.basename(fn)[:-4].split("__"); files.setdefault(arm, []).append(fn)
    Wt = boot_counts(a.B)
    R = {}
    for arm, fl in sorted(files.items()):
        rec = st.arm_arrays(fl, folds)
        has_nn = any(os.path.basename(f).endswith("__nn.npz") for f in fl)
        main_names = MAIN_TAB + (MAIN_NN if has_nn else [])
        L, Lb, Lpts = stat_from(rec, main_names, "id", "t", Wt)
        Lw, Lwb, Lwpts = stat_from(rec, main_names, "id", "w", Wt)
        U, Ub, Upts = stat_from(rec, main_names, "act", "w", Wt)
        H, Hb, Hpts = stat_from(rec, HELD, "id", "t", Wt)
        if L is None: continue
        R[arm] = dict(L=L, Lb=Lb, L_win=Lw, U=U, Ub=Ub, L_att=Lpts, L_win_att=Lwpts, U_att=Upts, L_held=Hpts, has_nn=has_nn, n_trials=float(next(iter(rec.values()))["id"]["n_t"].sum()))
    # ---------- margins from the floor pairs (fold set `folds`)
    mde = {}
    pairs = [(x, y) for x, y in itertools.combinations(MDE_SET, 2) if x in R and y in R]
    for tgt, key in [("L", "Lb"), ("U", "Ub")]:
        ses = [float(np.std(R[x][key] - R[y][key])) for x, y in pairs if R[x].get(key) is not None and R[y].get(key) is not None]
        mde[tgt] = dict(median_se=float(np.median(ses)) if ses else None, mde80=float(2.556 * np.median(ses)) if ses else None, n_pairs=len(ses))
    m_L = max(a.mL_floor, mde["L"]["mde80"] or 0); m_U = max(a.mU_floor, mde["U"]["mde80"] or 0)
    # ---------- floors set and gates
    excl = set(a.exclude.split(",")) - {""}
    floors = [x for x in R if x not in CONTROLS | ORACLES | excl and not x.startswith(("pd_", "msda", "rep", "aae_re")) and R[x]["U"] is not None]
    gates = {}
    if "raw" in R:
        gates["H1_raw_L>=0.40"] = bool(R["raw"]["L"] >= 0.40); gates["raw_L"] = R["raw"]["L"]; gates["raw_U"] = R["raw"]["U"]; gates["H3_raw_U>=0.90"] = bool((R["raw"]["U"] or 0) >= 0.90)
    for c in CONTROLS:
        if c in R:
            se = float(np.std(R[c]["Lb"])); tol = CHANCE + max(0.06, 1.96 * se); gates[f"H2_{c}"] = dict(L=R[c]["L"], tol=tol, ok=bool(R[c]["L"] <= tol))
    if "oracle_act" in R: gates["H4_oracle_act"] = dict(L=R["oracle_act"]["L"], U=R["oracle_act"]["U"], ok=bool(R["oracle_act"]["L"] <= CHANCE + 0.08 and (R["oracle_act"]["U"] or 0) >= 0.99))
    K1 = bool(mde["L"]["mde80"] is not None and mde["L"]["mde80"] > 0.15)
    # ---------- outcomes
    def outcome(arm, floor_set):
        Ls_b = np.stack([R[f]["Lb"] for f in floor_set], 1); Us_b = np.stack([R[f]["Ub"] for f in floor_set], 1)
        La_b, Ua_b = R[arm]["Lb"], R[arm]["Ub"]
        dL = np.array([frontier_L_at_U(Ls_b[b], Us_b[b], Ua_b[b]) - La_b[b] for b in range(len(La_b))])
        dU = np.array([Ua_b[b] - frontier_U_at_L(Ls_b[b], Us_b[b], La_b[b]) for b in range(len(La_b))])
        Ls0 = np.array([R[f]["L"] for f in floor_set]); Us0 = np.array([R[f]["U"] for f in floor_set])
        pL = frontier_L_at_U(Ls0, Us0, R[arm]["U"]) - R[arm]["L"]; pU = R[arm]["U"] - frontier_U_at_L(Ls0, Us0, R[arm]["L"])
        lab = {}
        for nm, p, bs, m in [("L", pL, dL, m_L), ("U", pU, dU, m_U)]:
            lo, hi = np.percentile(bs, [5, 95])
            lab[nm] = dict(delta=float(p), lo5=float(lo), hi95=float(hi), margin=float(m), label="WIN" if (p >= m and lo > 0) else ("MATCH" if hi < m else "UNRESOLVED"))
        labs = [lab["L"]["label"], lab["U"]["label"]]
        lab["defence_label"] = "WIN" if "WIN" in labs else ("MATCH" if labs == ["MATCH", "MATCH"] else "UNRESOLVED")
        return lab
    out_arms = {}
    learned = [x for x in a.learned.split(",") if x in R]; control = [x for x in a.control.split(",") if x in R]
    for arm in learned + control:
        if R[arm]["U"] is None: continue
        out_arms[arm] = dict(full_envelope=outcome(arm, floors), lda_only=outcome(arm, [f for f in floors if f in LDA_ONLY | {"raw"}]) if any(f in LDA_ONLY for f in floors) else None)
    labs = [out_arms[x]["full_envelope"]["defence_label"] for x in learned if x in out_arms]
    gate_ok = all(v if isinstance(v, bool) else v.get("ok", True) for k, v in gates.items() if k.startswith(("H1", "H2", "H3", "H4")))
    if not learned or not labs: verdict_t = "NOT_RUN"
    elif K1: verdict_t = "NO-GO-UNRESOLVED (K1 power)"
    elif not gate_ok: verdict_t = "NO-GO (K0 instrument gates)"
    elif "WIN" in labs: verdict_t = "GO"
    elif all(l == "MATCH" for l in labs): verdict_t = "NO-GO-MATCHED"
    else: verdict_t = "NO-GO-UNRESOLVED"
    res = dict(floor_set=sorted(floors), excluded=sorted(excl), folds=folds, B=a.B, margins=dict(m_L=m_L, m_U=m_U, mde=mde), K1_power_fires=K1, gates=gates, gates_ok=gate_ok, n_distinct_learned_defences=3, k2_set=learned,
               verdict_tightened=verdict_t, verdict_original="see verdict_original_pd (v0 rule on PD arms)", outcomes=out_arms,
               arms={x: dict(L=R[x]["L"], L_ci=[float(v) for v in np.percentile(R[x]["Lb"], [2.5, 97.5])], U=R[x]["U"], U_ci=[float(v) for v in np.percentile(R[x]["Ub"], [2.5, 97.5])], L_win=R[x]["L_win"],
                             L_att=R[x]["L_att"], U_att=R[x]["U_att"], L_held=R[x]["L_held"], has_nn=R[x]["has_nn"], n_trials=R[x]["n_trials"]) for x in R})
    json.dump(res, open(a.out, "w"), indent=1)
    # compact print
    print(f"margins m_L={m_L:.3f} m_U={m_U:.3f}  MDE80_L={mde['L']['mde80']}  K1={K1}  gates_ok={gate_ok}")
    print(f"{'arm':14s} {'L':>6s} {'L95':>13s} {'Uact':>6s} {'Lwin':>6s}  best-attacker  held")
    for x in sorted(R, key=lambda z: (R[z]['L'])):
        r = R[x]; ba = max(r["L_att"], key=r["L_att"].get) if r["L_att"] else "-"
        print(f"{x:14s} {r['L']:6.3f} [{np.percentile(r['Lb'],2.5):.3f},{np.percentile(r['Lb'],97.5):.3f}] {(r['U'] or 0):6.3f} {r['L_win']:6.3f}  {ba:8s} {max(r['L_held'].values()) if r['L_held'] else float('nan'):.3f}")
    for x, o in out_arms.items():
        print(x, {k: (o['full_envelope'][k]['label'], round(o['full_envelope'][k]['delta'], 3)) for k in ("L", "U")}, o["full_envelope"]["defence_label"])
    print("verdict_tightened:", verdict_t)


if __name__ == "__main__":
    main()
