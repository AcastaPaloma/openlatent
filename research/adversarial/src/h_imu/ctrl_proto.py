"""H3-017 (H.md 7.2): same-session adjacency, leave-activity-out and static/dynamic transfer for MotionSense identity (six activities, 360 trials).
  python ctrl_proto.py run --arm raw --out DIR [--threads 4]       # one arm over all protocols; writes proto_<arm>.npz
  python ctrl_proto.py astar --arm msda --out DIR                  # shipped/first-pass releases under leave-one-dynamic-activity-out on the first-pass A* set
  python ctrl_proto.py merge --dir DIR --out OUT.json              # bootstrap, claims C1-C9, falsifiers, verdict_original / verdict_tightened
The frozen evaluator files are only imported (suite._fit_predict_tab, suite.build_release); nothing is edited."""
import os, sys, json, glob, argparse, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import datasets as D
import suite as SU

ACT6 = ["dws", "ups", "wlk", "jog", "sit", "std"]
TR6 = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [7, 8, 15], "jog": [9, 16], "sit": [5, 13], "std": [6, 14]}
STATIC = {4, 5}
ATTACKERS = ["linear", "knn", "lgbm", "rf", "mlp"]
NS = 24


def build6(cache):
    if os.path.exists(cache):
        d = np.load(cache); return {k: d[k] for k in d.files}
    mean, std = C.pd_train_stats(C.load_trials())
    X, G, SUB, ACT, TRI, ST = [], [], [], [], [], []
    for ai, a in enumerate(ACT6):
        for t in TR6[a]:
            for s in range(24):
                df = pd.read_csv(os.path.join(C.DATA_ROOT, "A_DeviceMotion_data", f"{a}_{t}", f"sub_{s + 1}.csv"))
                rot = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].values; grav = df[["gravity.x", "gravity.y", "gravity.z"]].values
                acc = df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].values + grav
                z = ((np.stack([np.linalg.norm(rot, axis=1), np.linalg.norm(acc, axis=1)], 1) - mean) / std).T
                for i in range(0, z.shape[1] - 128, 64):
                    X.append(z[:, i:i + 128]); G.append(grav[i:i + 128].mean(0)); SUB.append(s); ACT.append(ai); TRI.append(t); ST.append(i)
    S = dict(X=np.asarray(X, np.float32), grav=np.asarray(G, np.float32), sub=np.asarray(SUB), act=np.asarray(ACT), trial=np.asarray(TRI), start=np.asarray(ST))
    np.savez_compressed(cache, **S); return S


def protocols(S):
    """name -> list of (train_mask, test_mask)"""
    rank = np.zeros(len(S["X"]), int)
    for ai, a in enumerate(ACT6):
        for ti, t in enumerate(TR6[a]): rank[(S["act"] == ai) & (S["trial"] == t)] = ti
    stat = np.isin(S["act"], list(STATIC))
    P = {"P-int": [(rank != f, rank == f) for f in range(3)],
         "P-block": [(S["trial"] <= 9, S["trial"] >= 11)],
         "P-LAO": [(S["act"] != a, S["act"] == a) for a in range(6)],
         "P-S2D": [(stat, ~stat)], "P-D2S": [(~stat, stat)]}
    return P


def feats_for(arm, S, PHI, tr, fold):
    if arm == "gravity":
        return S["grav"]
    S2 = dict(S); S2["is_train_window"] = np.ones(len(S["X"]), bool)
    R, kind = SU.build_release(arm, S2, PHI, tr, fold, None)
    return C.phi(R) if kind == "win" else R


def collect(P_acc, key, probs_by_att, te_idx, S):
    sub = S["sub"][te_idx]; act = S["act"][te_idx]; tk = C.trial_key(S)[te_idx]
    order = np.argsort(tk, kind="stable"); tks = tk[order]; starts = np.r_[0, np.where(np.diff(tks) != 0)[0] + 1]; cnt = np.diff(np.r_[starts, len(tks)])
    sub_t = (tks[starts] // 1000).astype(int); act_t = ((tks[starts] % 1000) // 100).astype(int)
    for att, P in probs_by_att.items():
        pm = np.add.reduceat(P[order], starts, axis=0) / cnt[:, None]; ok_t = (pm.argmax(1) == sub_t).astype(float)
        ok_w = (P.argmax(1) == sub).astype(float)
        strata = [("all", np.ones(len(sub_t), bool), np.ones(len(sub), bool)), ("static", np.isin(act_t, list(STATIC)), np.isin(act, list(STATIC))),
                  ("dynamic", ~np.isin(act_t, list(STATIC)), ~np.isin(act, list(STATIC)))] + [("act%d" % a_, act_t == a_, act == a_) for a_ in range(6)]
        for stratum, mt, mw in strata:
            rec = P_acc.setdefault((key, stratum, att), np.zeros((4, NS)))
            rec[0] += np.bincount(sub_t[mt], weights=ok_t[mt], minlength=NS); rec[1] += np.bincount(sub_t[mt], minlength=NS)
            rec[2] += np.bincount(sub[mw], weights=ok_w[mw], minlength=NS); rec[3] += np.bincount(sub[mw], minlength=NS)


def run(arm, out, threads):
    os.makedirs(out, exist_ok=True); t0 = time.time()
    SU.N_ACT = 6
    S = build6(os.path.join(out, "windows6.npz")); PHI = C.phi(S["X"])
    P_acc = {}
    for pname, folds in protocols(S).items():
        if arm == "gravity" and pname not in ("P-int", "P-block"): continue
        if pname in ("P-S2D", "P-D2S") and arm not in ("raw", "demean", "ds8", "gravity"): continue
        for fi, (tr, te) in enumerate(folds):
            F = feats_for(arm, S, PHI, tr, f"{pname}{fi}")
            y = S["sub"]; probs = {}
            for att in ATTACKERS:
                Ps = [SU._fit_predict_tab(att, F[tr], y[tr], F[te], NS, sd, threads) for sd in ([0, 1, 2] if att == "mlp" else [0])]
                probs[att] = np.mean(Ps, 0)
            collect(P_acc, pname, probs, np.where(te)[0], S)
        print(f"[{time.time()-t0:.0f}s] {arm} {pname} done", flush=True)
    np.savez_compressed(os.path.join(out, f"proto_{arm}.npz"), **{"|".join(k): v for k, v in P_acc.items()})


def run_astar(arm, out, threads):
    """shipped / first-pass releases under leave-one-DYNAMIC-activity-out on the first-pass A* set (descriptive; shipped checkpoints saw the held-out activity)"""
    os.makedirs(out, exist_ok=True); t0 = time.time()
    S = D.get_set("astar"); PHI = C.phi(S["X"]); P_acc = {}
    pdf = {"msda": os.path.expanduser("~/work/h_imu/out/aae_rel/msda_s0.npz"), "rep": os.path.expanduser("~/work/h_imu/out/aae_rel/rep_s0.npz"),
           "pd_pub_s0": os.path.expanduser("~/work/runs/h_pd_pub_s0/pd_pub_s0.npz")}.get(arm)
    for a in range(4):
        te = S["act"] == a; tr = (~te) & S["is_train_window"]
        R, kind = SU.build_release(arm if arm in ("msda", "rep") else ("pd_pub" if arm == "pd_pub_s0" else arm), S, PHI, tr, "O", pdf)
        F = C.phi(R) if kind == "win" else R; y = S["sub"]; probs = {}
        for att in ATTACKERS:
            Ps = [SU._fit_predict_tab(att, F[tr], y[tr], F[te], NS, sd, threads) for sd in ([0, 1, 2] if att == "mlp" else [0])]
            probs[att] = np.mean(Ps, 0)
        collect(P_acc, "LAOdyn", probs, np.where(te)[0], S)
        print(f"[{time.time()-t0:.0f}s] {arm} LAOdyn act {a} done", flush=True)
    np.savez_compressed(os.path.join(out, f"astar_{arm}.npz"), **{"|".join(k): v for k, v in P_acc.items()})


def boot_counts(B, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, NS, size=(B, NS))
    return np.stack([np.bincount(r, minlength=NS) for r in idx]).astype(np.float64)


def stat(recs, Wt, level="t"):
    """recs: {att: array(4,NS)} -> (point = max over attackers, boot (B,), per-attacker)"""
    c, n = (0, 1) if level == "t" else (2, 3)
    pts = {a: float(r[c].sum() / max(r[n].sum(), 1)) for a, r in recs.items()}
    bs = np.stack([(Wt @ r[c]) / np.maximum(Wt @ r[n], 1e-9) for r in recs.values()]).max(0)
    return max(pts.values()), bs, pts


def merge(d, out):
    Wt = boot_counts(2000); T = {}
    for fn in glob.glob(os.path.join(d, "proto_*.npz")) + glob.glob(os.path.join(d, "astar_*.npz")):
        arm = os.path.basename(fn).split(".")[0].split("_", 1)[1]; z = np.load(fn)
        for k in z.files:
            prot, stratum, att = k.split("|"); T.setdefault((arm, prot, stratum), {})[att] = z[k]
    R = {}
    for (arm, prot, stratum), recs in T.items():
        p, bs, pts = stat(recs, Wt, "t"); pw, bw, _ = stat(recs, Wt, "w")
        R.setdefault(arm, {}).setdefault(prot, {})[stratum] = dict(L=p, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), L_win=pw, per_attacker=pts, boot=bs)
    def g(arm, prot, st="all"): return R.get(arm, {}).get(prot, {}).get(st)
    claims = {}
    def claim(name, desc, p_ok, t_ok, vals): claims[name] = dict(desc=desc, original="PASS" if p_ok else "FAIL", tightened="PASS" if t_ok else ("FAIL" if not p_ok else "UNRESOLVED"), values=vals)
    r = lambda prot, st="all": g("raw", prot, st)
    if r("P-int"): claim("C1", "raw P-int >= 0.95", r("P-int")["L"] >= 0.95, r("P-int")["lo"] >= 0.95, dict(L=r("P-int")["L"], ci=[r("P-int")["lo"], r("P-int")["hi"]]))
    if r("P-block"): claim("C2", "raw P-block in [0.85, 0.97]", 0.85 <= r("P-block")["L"] <= 0.97, r("P-block")["lo"] >= 0.85 and r("P-block")["hi"] <= 0.97, dict(L=r("P-block")["L"], ci=[r("P-block")["lo"], r("P-block")["hi"]]))
    if r("P-LAO"): claim("C3", "raw P-LAO <= 0.60", r("P-LAO")["L"] <= 0.60, r("P-LAO")["hi"] <= 0.60, dict(L=r("P-LAO")["L"], ci=[r("P-LAO")["lo"], r("P-LAO")["hi"]], per_activity={a: round(g("raw", "P-LAO", a)["L"], 3) for a in [f"act{i}" for i in range(6)] if g("raw", "P-LAO", a)}))
    if r("P-S2D"): claim("C4", "raw P-S2D <= 0.25", r("P-S2D")["L"] <= 0.25, r("P-S2D")["hi"] <= 0.25, dict(L=r("P-S2D")["L"], ci=[r("P-S2D")["lo"], r("P-S2D")["hi"]]))
    if r("P-D2S"): claim("C5", "raw P-D2S <= 0.25", r("P-D2S")["L"] <= 0.25, r("P-D2S")["hi"] <= 0.25, dict(L=r("P-D2S")["L"], ci=[r("P-D2S")["lo"], r("P-D2S")["hi"]]))
    c6 = {}
    for arm in ("lda3", "act60_r1", "predlabel"):
        if all(g(arm, p) for p in ("P-int", "P-block", "P-LAO")):
            Ls = np.stack([g(arm, p)["boot"] for p in ("P-int", "P-block", "P-LAO")]); rng_b = Ls.max(0) - Ls.min(0)
            pt = [g(arm, p)["L"] for p in ("P-int", "P-block", "P-LAO")]; c6[arm] = dict(values=[round(x, 3) for x in pt], range=round(max(pt) - min(pt), 3), range_hi95=round(float(np.percentile(rng_b, 95)), 3))
    if c6: claim("C6", "lda3 / act60_r1 / predlabel change <= 0.05 across P-int, P-block, P-LAO", all(v["range"] <= 0.05 for v in c6.values()), all(v["range_hi95"] <= 0.05 for v in c6.values()), c6)
    c7 = {a: dict(L=g(a, "LAOdyn")["L"], ci=[g(a, "LAOdyn")["lo"], g(a, "LAOdyn")["hi"]]) for a in ("msda", "rep", "pd_pub_s0") if g(a, "LAOdyn")}
    if "msda" in c7 and "rep" in c7: claim("C7", "msda and rep under leave-one-dynamic-activity-out <= 0.40 (shipped checkpoints saw the held-out activity)", c7["msda"]["L"] <= 0.40 and c7["rep"]["L"] <= 0.40, c7["msda"]["ci"][1] <= 0.40 and c7["rep"]["ci"][1] <= 0.40, c7)
    c8 = {}
    for prot, arms_ in (("LAOdyn", ["lda3", "act60_r1", "predlabel", "lda3_r1"]),):
        if g("msda", prot) and g("rep", prot):
            for a in arms_:
                if g(a, prot): c8[a] = dict(L=round(g(a, prot)["L"], 3), below_msda=bool(g(a, prot)["L"] < g("msda", prot)["L"]), below_rep=bool(g(a, prot)["L"] < g("rep", prot)["L"]))
    if c8: claim("C8", "high-U floors below msda / rep (leave-one-dynamic-activity-out, first-pass A* set)", all(v["below_msda"] and v["below_rep"] for v in c8.values()), all(v["below_msda"] and v["below_rep"] for v in c8.values()), c8)
    if g("gravity", "P-int", "static") and r("P-int", "static"):
        ratio = g("gravity", "P-int", "static")["L"] / max(r("P-int", "static")["L"], 1e-9)
        claim("C9", "gravity-direction attacker >= 50% of the full suite's static accuracy (P-int)", ratio >= 0.5, g("gravity", "P-int", "static")["lo"] / max(r("P-int", "static")["hi"], 1e-9) >= 0.5,
              dict(gravity_static=g("gravity", "P-int", "static")["L"], raw_static=r("P-int", "static")["L"], ratio=round(ratio, 3), gravity_block_static=(g("gravity", "P-block", "static") or {}).get("L")))
    fals = {}
    if r("P-LAO") and r("P-S2D"):
        fals["durable_biometric"] = dict(original=bool(r("P-LAO")["L"] >= 0.85 and r("P-S2D")["L"] >= 0.5), tightened=bool(r("P-LAO")["lo"] >= 0.85 and r("P-S2D")["lo"] >= 0.5))
        fals["same_activity_only"] = dict(original=bool(r("P-LAO")["L"] <= 0.15), tightened=bool(r("P-LAO")["hi"] <= 0.15))
    prot_table = {arm: {p: {s: dict(L=round(v["L"], 4), ci=[round(v["lo"], 4), round(v["hi"], 4)], L_win=round(v["L_win"], 4), best=max(v["per_attacker"], key=v["per_attacker"].get)) for s, v in sd.items() if s in ("all", "static", "dynamic")} for p, sd in pd_.items()} for arm, pd_ in R.items()}
    po = [v["original"] for v in claims.values()]; pt_ = [v["tightened"] for v in claims.values()]
    vo = "DURABLE-BIOMETRIC FALSIFIER FIRED" if fals.get("durable_biometric", {}).get("original") else ("SAME-ACTIVITY-ONLY FALSIFIER FIRED" if fals.get("same_activity_only", {}).get("original") else ("ORDERING AS PREDICTED" if all(x == "PASS" for x in po) else "PARTIAL: %d of %d claims pass" % (po.count("PASS"), len(po))))
    vt = "DURABLE-BIOMETRIC FALSIFIER FIRED" if fals.get("durable_biometric", {}).get("tightened") else ("SAME-ACTIVITY-ONLY FALSIFIER FIRED" if fals.get("same_activity_only", {}).get("tightened") else ("ORDERING AS PREDICTED" if all(x == "PASS" for x in pt_) else "PARTIAL/UNRESOLVED: %d of %d claims pass with bounds" % (pt_.count("PASS"), len(pt_))))
    res = dict(control="H3-017", protocols="P-int (3 rotating folds by trial rank), P-block (1-9 vs 11-16), P-LAO (6 activities), P-S2D, P-D2S; LAOdyn = shipped / first-pass releases", table=prot_table, claims=claims, falsifiers=fals, verdict_original=vo, verdict_tightened=vt,
               note="same-session closed-set wording; cannot change a floor-vs-defender label; 1D-CNN dropped (CPU only)")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("verdict_original", "verdict_tightened", "falsifiers")}, indent=1))
    for arm in sorted(prot_table):
        print(arm, {p: prot_table[arm][p].get("all", {}).get("L") for p in prot_table[arm]}, "static:", {p: prot_table[arm][p].get("static", {}).get("L") for p in prot_table[arm] if "static" in prot_table[arm][p]})
    for k, v in claims.items(): print(k, v["desc"], "|", v["original"], v["tightened"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "astar", "merge"]); ap.add_argument("--arm", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--dir", default=None); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "run": run(a.arm, a.out, a.threads)
    elif a.mode == "astar": run_astar(a.arm, a.out, a.threads)
    else: merge(a.dir, a.out)
