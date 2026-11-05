"""JS-1 evaluation job: floors, references and controls on one scenario and one split salt.  Writes per-compound AP vectors and per-well attacker
predictions (no profiles, no embeddings).

  python evaluate.py --scenario 4 --salt s0 --out $OUT [--arms a,b,c] [--no_refs]

Per arm and for both eval sets (V = validation compounds, H = held-out compounds): cross-source retrieval AP (xs, primary) and the
benchmark-style non-replicate AP (nonrep), plus four source attackers (linear, mlp, knn, lgbm) retrained on the released embedding on a
plate-disjoint half and scored on the other half (cross-fit).  Fit set / roles / split rule: results_notes/F.md 2.1.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap as AP  # noqa: E402
import arms as AR  # noqa: E402
import common as C  # noqa: E402
import adapter as AD  # noqa: E402

_spec = importlib.util.spec_from_file_location("e_attackers", str(HERE.parent / "e_singlecell" / "attackers.py"))
EA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EA)
KINDS = ("linear", "mlp", "knn", "lgbm")
TRAIN_CAP = 25000


def universe(meta, s5):
    """boolean masks / role arrays.  Returns dict with comp (str), roles per well ('' if not in the compound universe)."""
    jcp = meta["Metadata_JCP2022"].to_numpy()
    ptype = meta["Metadata_PlateType"].to_numpy()
    pert = meta["PertType"].to_numpy()
    src = meta["Metadata_Source"].to_numpy()
    trt = pert == "trt"
    js = pd.Series(jcp)
    t2_ids = set(jcp[(ptype == "TARGET2") & trt])
    exc = js.isin(C.EXCLUDE_FROM_MAP).to_numpy()
    if s5:
        inu = (ptype == "COMPOUND") & trt & ~js.isin(t2_ids).to_numpy() & ~exc
    else:
        inu = (ptype == "TARGET2") & trt & ~exc
    # >= 2 sources
    df = pd.DataFrame({"j": jcp[inu], "s": src[inu]})
    ns = df.groupby("j")["s"].nunique()
    inu &= js.isin(set(ns[ns >= 2].index)).to_numpy()
    return dict(inu=inu, jcp=jcp, ptype=ptype, pert=pert, src=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--no_refs", action="store_true")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--no_attackers", action="store_true", help="AP only (debug)")
    ap.add_argument("--f4", action="store_true", help="F4 job (results_notes/F.md section 10): budget grid 0/1/3/10/30/100%, leave-source-out via --hold_source, feature view via --view")
    ap.add_argument("--view", default="all", choices=["all", "objects"], help="objects = drop the Image_* features (F4 view B)")
    ap.add_argument("--f3", action="store_true", help="F3 job (results_notes/F.md section 8): label-budget curve, matched LDA, AE control, permuted-label memorisation floor")
    ap.add_argument("--f2", action="store_true", help="F2 job (results_notes/F.md section 6): small-d LDA, oracle, MLP adapters, same-embedding nulls")
    ap.add_argument("--hold_source", default="", help="P-S (descriptive): fit without this source's wells; AP queries restricted to this source")
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s5 = a.scenario == 5
    tag = f"s{a.scenario}"
    d = C.cache_dir(tag)
    t0 = time.time()
    X = np.load(d / "X.npy", mmap_mode="r")
    cols = None
    if a.view == "objects":
        cols = np.array([not f.startswith("Image_") for f in (d / "features.txt").read_text().split("\n")])
        print(f"view objects: {int(cols.sum())} of {len(cols)} features", flush=True)

    def getX(idx):
        A = np.asarray(X[idx])
        return A if cols is None else A[:, cols]
    meta = pd.read_parquet(d / "meta.parquet")
    U = universe(meta, s5)
    jcp, src, pert, ptype = U["jcp"], U["src"], U["pert"], U["ptype"]
    plate, well = meta["Metadata_Plate"].to_numpy(), meta["Metadata_Well"].to_numpy()
    uni_ids = np.unique(jcp[U["inu"]])
    role_of = dict(zip(uni_ids, C.split_role(uni_ids, a.salt)))
    role = np.array([role_of.get(j, "") if i else "" for j, i in zip(jcp, U["inu"])])
    Hm, Vm, Tm = role == "H", role == "V", role == "T"
    print(f"{tag} {a.salt}: universe {len(uni_ids)} compounds; H {len(np.unique(jcp[Hm]))} V {len(np.unique(jcp[Vm]))} T {len(np.unique(jcp[Tm]))}; wells H {Hm.sum()} V {Vm.sum()}", flush=True)
    # Fit set and label vectors
    if s5:
        fitm = Tm | (ptype == "TARGET2") | (pert == "negcon")
        t2w = ptype == "TARGET2"
        lab_class = -np.ones(len(meta), np.int64)
        lab_class[t2w] = pd.factorize(jcp[t2w])[0]
        lab_prod = -np.ones(len(meta), np.int64)
        lab_prod[Tm] = pd.factorize(jcp[Tm])[0]
    else:
        fitm = Tm | (pert == "poscon") | (pert == "negcon")
        lab_class = -np.ones(len(meta), np.int64)
        lab_class[fitm] = pd.factorize(jcp[fitm])[0]
        lab_prod = None
    if a.hold_source:
        fitm = fitm & (src != a.hold_source)
    fit_idx = np.where(fitm)[0]
    Xfit = AR.T(getX(fit_idx))
    fac = AR.ArmFactory(Xfit, lab_class[fit_idx], None if lab_prod is None else lab_prod[fit_idx], a.salt, s5)
    # eval sets
    sets = {}
    for nm, m in (("V", Vm), ("H", Hm)):
        idx = np.where(m)[0]
        comp = pd.factorize(jcp[idx])[0]
        sets[nm] = dict(idx=idx, comp=comp, src=pd.Categorical(src[idx], categories=C.SOURCES5).codes.astype(np.int64), plate=plate[idx], well=well[idx],
                        pert=np.array(["trt"] * len(idx)), half=C.plate_halves(plate[idx], src[idx], a.salt))
        sets[nm]["plate_code"] = pd.factorize(plate[idx])[0]
    allidx = np.concatenate([sets["V"]["idx"], sets["H"]["idx"]])
    nV = len(sets["V"]["idx"])
    Xall = AR.T(getX(allidx))
    P = {}
    meta_out = dict(scenario=a.scenario, salt=a.salt, arms=[], families={}, dims={}, timings={}, kinds=a.kinds.split(","),
                    n_compounds=dict(V=int(len(np.unique(sets["V"]["comp"]))), H=int(len(np.unique(sets["H"]["comp"])))),
                    n_wells=dict(V=int(len(sets["V"]["idx"])), H=int(len(sets["H"]["idx"])), fit=int(len(fit_idx))),
                    commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    for nm, S in sets.items():
        P[f"y|{nm}"] = S["src"].astype(np.int8)
        P[f"plate|{nm}"] = S["plate_code"].astype(np.int32)
        P[f"comp|{nm}"] = S["comp"].astype(np.int32)
        P[f"idx|{nm}"] = S["idx"].astype(np.int64)
        P[f"half|{nm}"] = S["half"]
    kinds = a.kinds.split(",")
    pos_cache = {}
    rng_cap = np.random.RandomState(AR.seed_of(a.salt, "cap") % (2 ** 31))

    def do_ap(name, E, sname, comp_override=None, modes=("xs", "nonrep"), prefix="apc"):
        S = sets[sname]
        comp = S["comp"] if comp_override is None else comp_override
        for mode in modes:
            if comp_override is None:
                if (sname, mode) not in pos_cache:
                    pos_cache[(sname, mode)] = AP.build_pos(comp, S["src"], S["well"], mode)
                pos = pos_cache[(sname, mode)]
            else:
                pos = None
            qm = None if not a.hold_source else (S["src"] == C.SOURCES5.index(a.hold_source))
            apq = AP.ap_queries(E, S["plate"], comp, S["src"], S["well"], S["pert"], mode=mode, pos=pos, qmask=qm)
            ids, mean_ap = AP.per_compound(apq, comp)
            P[f"{prefix}|{sname}|{name}|{mode}"] = mean_ap.astype(np.float32)
            if prefix != "apc":
                continue
            if s5 and mode == "xs":
                P[f"apq|{sname}|{name}"] = apq.astype(np.float32)          # per-query AP (NaN = dropped): plate-cluster sensitivity
            if f"cid|{sname}|{mode}" not in P:
                P[f"cid|{sname}|{mode}"] = ids.astype(np.int32)

    def do_att(name, E, sname, shuffle=False):
        S = sets[sname]
        half, y, pc = S["half"], S["src"], S["plate_code"]
        res = {k: np.full(len(y), -1, np.int8) for k in kinds}
        for dd in (0, 1):
            tr = np.where(half == dd)[0]
            te = np.where(half != dd)[0]
            if len(tr) > TRAIN_CAP:
                tr = np.sort(rng_cap.choice(tr, TRAIN_CAP, replace=False))
            if len(te) > 100000:
                te = np.sort(rng_cap.choice(te, 100000, replace=False))
            pr = EA.run_attackers(E[tr], y[tr], pc[tr], E[te], len(C.SOURCES5), kinds=tuple(kinds), seed=0, shuffle=shuffle)
            for k in kinds:
                res[k][te] = pr[k].astype(np.int8)
        for k in kinds:
            P[f"att|{sname}|{name}|{k}"] = res[k]

    def run_arm(name, family, Eall, ap_ok=True, null=False, att=True):
        t1 = time.time()
        Eall_n = Eall.detach().cpu().numpy() if torch.is_tensor(Eall) else Eall
        for sname, sl in (("V", slice(0, nV)), ("H", slice(nV, None))):
            E = Eall_n[sl]
            if ap_ok:
                do_ap(name, E, sname, modes=("xs",) if (a.f2 or a.f3 or a.f4) else ("xs", "nonrep"))
                if null:
                    if ("null", sname) not in pos_cache:
                        perm = np.random.RandomState(AR.seed_of(a.salt, "null", sname) % (2 ** 31)).permutation(len(E))
                        pos_cache[("null", sname)] = (sets[sname]["comp"][perm], AP.build_pos(sets[sname]["comp"][perm], sets[sname]["src"], sets[sname]["well"], "xs"))
                    cn, pn = pos_cache[("null", sname)]
                    qm = None if not a.hold_source else (sets[sname]["src"] == C.SOURCES5.index(a.hold_source))
                    apq = AP.ap_queries(E, sets[sname]["plate"], cn, sets[sname]["src"], sets[sname]["well"], sets[sname]["pert"], mode="xs", pos=pn, qmask=qm)
                    P[f"apcnull|{sname}|{name}|xs"] = AP.per_compound(apq, cn)[1].astype(np.float32)
            if att and not a.no_attackers:
                do_att(name, E, sname)
        meta_out["arms"].append(name)
        meta_out["families"][name] = family
        meta_out["dims"][name] = int(Eall_n.shape[1])
        meta_out["timings"][name] = round(time.time() - t1, 1)
        np.savez_compressed(out / f"preds_{a.salt}.npz", **P)
        (out / f"meta_{a.salt}.json").write_text(json.dumps(meta_out, indent=1))
        print(f"  arm {name:22s} d={Eall_n.shape[1]:5d} {time.time() - t1:6.1f}s  (elapsed {time.time() - t0:.0f}s)", flush=True)

    want = a.arms.split(",") if a.arms else None
    if a.f2:
        run_f2(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, src, out)
        print("done", flush=True)
        return
    if a.f4:
        run_f4(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, ptype, well, src, out)
        print("done", flush=True)
        return
    if a.f3:
        run_f3(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, ptype, well, src, out)
        print("done", flush=True)
        return
    # inductive floors
    for name, family in fac.names():
        if want and name not in want:
            continue
        if name == "ctrl_noise16":
            E = torch.randn(Xall.shape[0], 16, generator=torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "ctrl")), device=AR.DEV)
        else:
            E = fac.build(name, Xall)
        run_arm(name, family, E)
        del E
        torch.cuda.empty_cache()
    # controls
    if not want or "ctrl_const" in want:
        run_arm("ctrl_const", "ctrl", torch.ones(Xall.shape[0], 1, device=AR.DEV), ap_ok=False)
    if not want or "ctrl_shuffle" in want:
        Eall_n = Xall.cpu().numpy()
        t1 = time.time()
        for sname, sl in (("V", slice(0, nV)), ("H", slice(nV, None))):
            E = Eall_n[sl]
            perm = np.random.RandomState(AR.seed_of(a.salt, "shufcomp") % (2 ** 31)).permutation(len(E))
            do_ap("ctrl_shuffle_compound", E, sname, comp_override=sets[sname]["comp"][perm])
            if not a.no_attackers:
                do_att("ctrl_shuffle_source", E, sname, shuffle=True)
        for nm in ("ctrl_shuffle_compound", "ctrl_shuffle_source"):
            meta_out["arms"].append(nm)
            meta_out["families"][nm] = "ctrl"
            meta_out["dims"][nm] = int(Eall_n.shape[1])
        np.savez_compressed(out / f"preds_{a.salt}.npz", **P)
        (out / f"meta_{a.salt}.json").write_text(json.dumps(meta_out, indent=1))
        print(f"  controls shuffle {time.time() - t1:.0f}s", flush=True)
    # transductive references
    if not a.no_refs:
        rdir = d / "refs"
        for nm in ("T_centre", "T_swhiten64", "T_combat", "T_harmony"):
            if want and nm not in want:
                continue
            f = rdir / f"{nm}.npy"
            if not f.exists():
                print(f"  reference {nm}: file missing, skipped", flush=True)
                continue
            R = np.load(f, mmap_mode="r")[allidx]
            run_arm(nm, "ref", np.asarray(R))
        if not want or "T_sphering" in want:
            eps = AR.log_uniform_eps()
            dm = np.where(pert == "negcon")[0]
            pars = AR.sphering_fit(AR.T(getX(dm)), eps)
            best = (-1.0, None, None)
            sel = {}
            for i, par in enumerate(pars):
                E = AR.sphering_apply(Xall, par).detach().cpu().numpy()
                apq = AP.ap_queries(E[:nV], sets["V"]["plate"], sets["V"]["comp"], sets["V"]["src"], sets["V"]["well"], sets["V"]["pert"], mode="xs", pos=pos_cache.get(("V", "xs")))
                u = float(np.nanmean(AP.per_compound(apq, sets["V"]["comp"])[1]))
                sel[f"{eps[i]:.6g}"] = u
                if u > best[0]:
                    best = (u, i, E)
                print(f"    sphering eps {eps[i]:9.4g} U_V {u:.4f}", flush=True)
            meta_out["sphering_eps_UV"] = sel
            meta_out["sphering_eps_chosen"] = float(eps[best[1]])
            run_arm("T_sphering", "ref", best[2])
    print("done", flush=True)


def run_f2(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, src, out):
    """F2 (results_notes/F.md section 6): small-d LDA (6 arms), compound-supervised in-sample oracle (2), ONE supervised nonlinear adapter family
    (SupCon MLP, d in {8,16,32,64} x output dither {0,0.5,1}), controls; every arm also gets a same-embedding shuffled-label AP null."""
    import json as _json
    t0 = time.time()
    only = a.arms.split(",") if a.arms else None
    ok = lambda nm: only is None or nm in only or nm.split("_")[0] in only  # noqa: E731
    meta_out["f2"] = True
    meta_out["f2_train"] = {}
    # (c) controls
    E = torch.randn(Xall.shape[0], 16, generator=torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "ctrl")), device=AR.DEV)
    if ok("ctrl_noise16"):
        run_arm("ctrl_noise16", "ctrl", E, null=True)
    # (a) small-d LDA
    for lab in ("ldaT2", "ldaP"):
        for dd in (2, 4, 8):
            name = f"{lab}_{dd}"
            if ok(name):
                run_arm(name, "labelonly", fac.build(name, Xall), null=True)
    # oracle: PCA-256 -> LDA fitted on the H compounds' OWN labels, scored in-sample (upper reference, not a method)
    Z = fac.pca(Xall, 256)
    muo, Wo = AR.lda_fit(Z[nV:], sets["H"]["comp"], AR.SHRINK)
    for dd in (16, 64):
        if ok(f"oracleLDA_{dd}") or ok("oracleLDA"):
            run_arm(f"oracleLDA_{dd}", "oracle", AR.lda_apply(Z, muo, Wo, dd), null=False)
    del Z
    # (b) the supervised nonlinear adapter: labels = T' production compounds (>= 2 Fit wells) + Target2 compounds (DMSO excluded)
    lp, lc, jf = lab_prod[fit_idx], lab_class[fit_idx], jcp[fit_idx]
    cnts = np.bincount(lp[lp >= 0])
    lp2 = np.where((lp >= 0) & (cnts[np.maximum(lp, 0)] >= 2), lp, -1)
    lab_final = -np.ones(len(fit_idx), np.int64)
    m = lp2 >= 0
    lab_final[m] = pd.factorize(lp2[m])[0]
    n_prod = int(lab_final.max()) + 1
    t2m = (lc >= 0) & (lp < 0) & (jf != "DMSO")
    lab_final[t2m] = n_prod + pd.factorize(lc[t2m])[0]
    meta_out["f2_labels"] = dict(n_prod_classes=n_prod, n_t2_classes=int(len(np.unique(lab_final[t2m]))), n_labeled_wells=int((lab_final >= 0).sum()))
    srcf = pd.Categorical(src[fit_idx], categories=C.SOURCES5).codes.astype(np.int64)
    rs = np.random.RandomState(AR.seed_of(a.salt, "probe") % (2 ** 31))
    sub = rs.choice(len(fit_idx), 20000, replace=False)
    for dd in (8, 16, 32, 64):
        if not ok(f"mlp_d{dd}"):
            continue
        t1 = time.time()
        net, info = AD.train_supcon(Xfit, lab_final, n_prod, dd, AR.seed_of(a.salt, "mlp", dd) % (2 ** 31), steps=int(os.environ.get("F2_STEPS", AD.STEPS)))
        Ef = AD.embed(net, Xfit)
        info["eff_rank"] = AD.effective_rank(Ef[torch.as_tensor(sub, device=AR.DEV)])
        # in-sample linear source probe (logging only; no adversary is used)
        Es = Ef[torch.as_tensor(sub, device=AR.DEV)]
        Es = (Es - Es.mean(0, keepdim=True)) / (Es.std(0, keepdim=True) + 1e-6)
        ys = torch.as_tensor(srcf[sub], device=AR.DEV)
        pr = EA.fit_linear(Es, ys, len(C.SOURCES5), 1e-2)(Es).argmax(1).cpu().numpy()
        info["insample_source_probe_bacc"] = float(EA.bacc(srcf[sub], pr))
        info["trained"] = bool(info["loss_last200"] <= 0.8 * info["loss_first20"] and info["eff_rank"] >= dd / 4 and not info["nan"])
        info["train_seconds"] = round(time.time() - t1, 1)
        meta_out["f2_train"][f"d{dd}"] = info
        print(f"  mlp d={dd}: loss {info['loss_first20']:.3f} -> {info['loss_last200']:.3f}, eff_rank {info['eff_rank']:.1f}, probe {info['insample_source_probe_bacc']:.3f}, trained={info['trained']} ({info['train_seconds']}s)", flush=True)
        Eall = AD.embed(net, Xall)
        sdv = Ef.std(0, keepdim=True)
        run_arm(f"mlp_d{dd}", "adapter", Eall, null=True)
        for r in (0.5, 1):
            gen = torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "mlpnoise", dd, r))
            run_arm(f"mlp_d{dd}_r{r:g}", "adapter", Eall + r * sdv * torch.randn(Eall.shape, generator=gen, device=AR.DEV), null=True)
        del net, Ef, Eall
        torch.cuda.empty_cache()
    (out / f"meta_{a.salt}.json").write_text(_json.dumps(meta_out, indent=1))


def run_f3(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, ptype, well, src, out):
    """F3 (results_notes/F.md section 8): label-budget curve for the SupCon MLP adapter.  Regimes: a (Target2 ids), b1/b10 (Target2 + 1% / 10% of T' compounds),
    c (replicate pairs from Metadata_Well position on Target2 plates, no ids); b100 = the F2 arms (not retrained).  Matched LDA on the same labels, label-free AE,
    permuted-label memorisation floors (d = 64)."""
    import hashlib
    import json as _json
    only = a.arms.split(",") if a.arms else None
    ok = lambda nm: only is None or nm in only or nm.split("_")[0] in only  # noqa: E731
    steps = int(os.environ.get("F2_STEPS", AD.STEPS))
    meta_out["f3"] = True
    meta_out["f3_train"] = {}
    meta_out["regime"] = {}
    lp, lc, jf, wf, pf = lab_prod[fit_idx], lab_class[fit_idx], jcp[fit_idx], well[fit_idx], ptype[fit_idx]
    cnts = np.bincount(lp[lp >= 0])
    lp2 = np.where((lp >= 0) & (cnts[np.maximum(lp, 0)] >= 2), lp, -1)
    m_prod = lp2 >= 0
    cls_j = pd.DataFrame({"c": lp2[m_prod], "j": jf[m_prod]}).drop_duplicates("c").set_index("c")["j"]
    t2m = (lc >= 0) & (lp < 0) & (jf != "DMSO")

    def frac_mask(f):
        if f >= 1.0:
            return m_prod
        keep = {c: int(hashlib.md5((str(j) + a.salt + "lab").encode()).hexdigest(), 16) % 10000 < int(round(f * 10000)) for c, j in cls_j.items()}
        sel = np.zeros(int(cls_j.index.max()) + 1, bool)
        for c, k in keep.items():
            sel[c] = k
        return m_prod & sel[np.maximum(lp2, 0)]

    def make_labels(R):
        lab = -np.ones(len(fit_idx), np.int64)
        if R == "c":
            m = pf == "TARGET2"
            lab[m] = pd.factorize(wf[m])[0]
            return lab, 0
        m = {"a": np.zeros(len(fit_idx), bool), "b1": frac_mask(0.01), "b10": frac_mask(0.10), "b100": frac_mask(1.0)}[R]
        n_prod = 0
        if m.any():
            lab[m] = pd.factorize(lp2[m])[0]
            n_prod = int(lab[m].max()) + 1
        lab[t2m] = n_prod + pd.factorize(lc[t2m])[0]
        return lab, n_prod

    labels = {R: make_labels(R) for R in ("a", "b1", "b10", "b100", "c")}
    meta_out["f3_labels"] = {R: dict(n_prod_classes=int(npd), n_classes=int(len(np.unique(l[l >= 0]))), n_labeled_wells=int((l >= 0).sum())) for R, (l, npd) in labels.items()}
    print("  label sets:", meta_out["f3_labels"], flush=True)
    # controls
    E = torch.randn(Xall.shape[0], 16, generator=torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "ctrl")), device=AR.DEV)
    if ok("ctrl_noise16"):
        run_arm("ctrl_noise16", "ctrl", E, null=True)
    # matched LDA (PCA-256 -> shrinkage LDA on exactly the regime's labelled wells)
    Zfit, Zall = fac.pca(Xfit, 256), fac.pca(Xall, 256)
    for R in ("a", "b1", "b10", "b100", "c"):
        lab, _ = labels[R]
        idx = torch.as_tensor(np.where(lab >= 0)[0], device=AR.DEV)
        if not any(ok(f"ldaM_{R}_{dd}") for dd in (8, 64)):
            continue
        mu, W = AR.lda_fit(Zfit[idx], lab[lab >= 0], AR.SHRINK)
        for dd in (8, 64):
            nm = f"ldaM_{R}_{dd}"
            if ok(nm):
                run_arm(nm, "matched", AR.lda_apply(Zall, mu, W, dd), null=True)
                meta_out["regime"][nm] = R
    del Zfit, Zall
    torch.cuda.empty_cache()
    srcf = pd.Categorical(src[fit_idx], categories=C.SOURCES5).codes.astype(np.int64)
    sub = np.random.RandomState(AR.seed_of(a.salt, "probe") % (2 ** 31)).choice(len(fit_idx), 20000, replace=False)
    subt = torch.as_tensor(sub, device=AR.DEV)

    def probe_and_rank(Ef):
        Es = Ef[subt]
        rk = AD.effective_rank(Es)
        Es = (Es - Es.mean(0, keepdim=True)) / (Es.std(0, keepdim=True) + 1e-6)
        pr = EA.fit_linear(Es, torch.as_tensor(srcf[sub], device=AR.DEV), len(C.SOURCES5), 1e-2)(Es).argmax(1).cpu().numpy()
        return rk, float(EA.bacc(srcf[sub], pr))

    rng_perm = np.random.RandomState(AR.seed_of(a.salt, "perm") % (2 ** 31))
    for R in ("a", "b1", "b10", "c", "b100"):
        lab, npd = labels[R]
        kw = dict(n_prod_batch=192 if npd > 0 else 0, n_t2_batch=32 if npd > 0 else 192, k_t2=4)
        for dd in (8, 64):
            if R == "b100" or not ok(f"mlp_{R}_d{dd}"):
                continue
            t1 = time.time()
            net, info = AD.train_supcon(Xfit, lab, npd, dd, AR.seed_of(a.salt, "f3", R, dd) % (2 ** 31), steps=steps, **kw)
            Ef = AD.embed(net, Xfit)
            info["eff_rank"], info["insample_source_probe_bacc"] = probe_and_rank(Ef)
            info["regime"], info["train_seconds"] = R, round(time.time() - t1, 1)
            meta_out["f3_train"][f"{R}_d{dd}"] = info
            print(f"  {R} d={dd}: loss {info['loss_first20']:.3f} -> {info['loss_last200']:.3f}, eff_rank {info['eff_rank']:.1f}, probe {info['insample_source_probe_bacc']:.3f} ({info['train_seconds']}s)", flush=True)
            Eall = AD.embed(net, Xall)
            sdv = Ef.std(0, keepdim=True)
            run_arm(f"mlp_{R}_d{dd}", "adapter", Eall, null=True)
            meta_out["regime"][f"mlp_{R}_d{dd}"] = R
            gen = torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "f3noise", R, dd))
            run_arm(f"mlp_{R}_d{dd}_r0.5", "adapter", Eall + 0.5 * sdv * torch.randn(Eall.shape, generator=gen, device=AR.DEV), null=True)
            meta_out["regime"][f"mlp_{R}_d{dd}_r0.5"] = R
            del net, Ef, Eall
            torch.cuda.empty_cache()
        # memorisation floor: same d = 64 adapter, labels permuted among the labelled wells
        if ok(f"perm_{R}_d64") or ok("perm"):
            t1 = time.time()
            labp = lab.copy()
            idx = np.where(lab >= 0)[0]
            labp[idx] = lab[idx][rng_perm.permutation(len(idx))]
            net, info = AD.train_supcon(Xfit, labp, npd, 64, AR.seed_of(a.salt, "f3perm", R) % (2 ** 31), steps=steps, **kw)
            info["regime"], info["train_seconds"] = R, round(time.time() - t1, 1)
            meta_out["f3_train"][f"perm_{R}_d64"] = info
            print(f"  perm {R} d=64: loss {info['loss_first20']:.3f} -> {info['loss_last200']:.3f} ({info['train_seconds']}s)", flush=True)
            run_arm(f"perm_{R}_d64", "permctrl", AD.embed(net, Xall), null=True, att=False)
            meta_out["regime"][f"perm_{R}_d64"] = R
            del net
            torch.cuda.empty_cache()
    # label-free rate/dither control: bottleneck autoencoder
    for dd in (8, 64):
        if not ok(f"ae_d{dd}"):
            continue
        t1 = time.time()
        net, info = AD.train_ae(Xfit, dd, AR.seed_of(a.salt, "f3ae", dd) % (2 ** 31), steps=steps)
        with torch.no_grad():
            Ef = torch.cat([net(Xfit[i:i + 65536]) for i in range(0, Xfit.shape[0], 65536)])
            Eall = torch.cat([net(Xall[i:i + 65536]) for i in range(0, Xall.shape[0], 65536)])
        info["eff_rank"], info["insample_source_probe_bacc"] = probe_and_rank(Ef)
        info["regime"], info["train_seconds"] = "ae", round(time.time() - t1, 1)
        meta_out["f3_train"][f"ae_d{dd}"] = info
        print(f"  ae d={dd}: recon {info['loss_first20']:.3f} -> {info['loss_last200']:.3f}, eff_rank {info['eff_rank']:.1f} ({info['train_seconds']}s)", flush=True)
        sdv = Ef.std(0, keepdim=True)
        run_arm(f"ae_d{dd}", "adapter_lf", Eall, null=True)
        gen = torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "f3aenoise", dd))
        run_arm(f"ae_d{dd}_r0.5", "adapter_lf", Eall + 0.5 * sdv * torch.randn(Eall.shape, generator=gen, device=AR.DEV), null=True)
        meta_out["regime"][f"ae_d{dd}"] = meta_out["regime"][f"ae_d{dd}_r0.5"] = "ae"
        del net, Ef, Eall
        torch.cuda.empty_cache()
    (out / f"meta_{a.salt}.json").write_text(_json.dumps(meta_out, indent=1))


def run_f4(a, fac, Xfit, Xall, nV, sets, run_arm, meta_out, P, fit_idx, lab_class, lab_prod, jcp, ptype, well, src, out):
    """F4 (results_notes/F.md section 10): budget grid a / b1 / b3 / b10 / b30 / b100 for the d = 8 SupCon adapter, trimmed floors, matched LDA, permuted-label floors.
    Leave-source-out and the feature view are handled by the caller (--hold_source, --view)."""
    import hashlib
    import json as _json
    steps = int(os.environ.get("F2_STEPS", AD.STEPS))
    meta_out["f4"] = True
    meta_out["f4_train"] = {}
    meta_out["regime"] = {}
    meta_out["f4_hold_source"], meta_out["f4_view"], meta_out["f4_dim"] = a.hold_source, a.view, int(Xfit.shape[1])
    lp, lc, jf = lab_prod[fit_idx], lab_class[fit_idx], jcp[fit_idx]
    cnts = np.bincount(lp[lp >= 0]) if (lp >= 0).any() else np.zeros(1, int)
    lp2 = np.where((lp >= 0) & (cnts[np.maximum(lp, 0)] >= 2), lp, -1)
    m_prod = lp2 >= 0
    cls_j = pd.DataFrame({"c": lp2[m_prod], "j": jf[m_prod]}).drop_duplicates("c").set_index("c")["j"]
    t2m = (lc >= 0) & (lp < 0) & (jf != "DMSO")
    bud = [("a", 0.0), ("b1", 0.01), ("b3", 0.03), ("b10", 0.10), ("b30", 0.30), ("b100", 1.0)]

    def frac_mask(f):
        if f <= 0:
            return np.zeros(len(fit_idx), bool)
        if f >= 1.0:
            return m_prod
        sel = np.zeros(int(cls_j.index.max()) + 1, bool)
        for c, j in cls_j.items():
            sel[c] = int(hashlib.md5((str(j) + a.salt + "lab").encode()).hexdigest(), 16) % 10000 < int(round(f * 10000))
        return m_prod & sel[np.maximum(lp2, 0)]

    labels = {}
    for R, f in bud:
        lab = -np.ones(len(fit_idx), np.int64)
        m = frac_mask(f)
        n_prod = 0
        if m.any():
            lab[m] = pd.factorize(lp2[m])[0]
            n_prod = int(lab[m].max()) + 1
        lab[t2m] = n_prod + pd.factorize(lc[t2m])[0]
        labels[R] = (lab, n_prod)
    meta_out["f4_labels"] = {R: dict(n_prod_classes=int(npd), n_classes=int(len(np.unique(l[l >= 0]))), n_labeled_wells=int((l >= 0).sum())) for R, (l, npd) in labels.items()}
    meta_out["f4_n_fit"] = int(len(fit_idx))
    print("  label sets:", meta_out["f4_labels"], flush=True)
    # controls and floors
    E = torch.randn(Xall.shape[0], 16, generator=torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "ctrl")), device=AR.DEV)
    run_arm("ctrl_noise16", "ctrl", E, null=True)
    for nm, fam in (("raw", "floor"), ("pca16", "floor"), ("dpca16_r1", "dither"), ("dpca16_r2", "dither"), ("dpca16_r4", "dither"),
                    ("ldaT2_8", "labelonly"), ("ldaT2_16", "labelonly"), ("ldaP_4", "labelonly"), ("ldaP_8", "labelonly"), ("ldaP_16", "labelonly")):
        run_arm(nm, fam, fac.build(nm, Xall))
        torch.cuda.empty_cache()
    # matched LDA
    Zfit, Zall = fac.pca(Xfit, 256), fac.pca(Xall, 256)
    for R, _ in bud:
        lab, _n = labels[R]
        idx = torch.as_tensor(np.where(lab >= 0)[0], device=AR.DEV)
        mu, W = AR.lda_fit(Zfit[idx], lab[lab >= 0], AR.SHRINK)
        nm = f"ldaM_{R}_8"
        run_arm(nm, "matched", AR.lda_apply(Zall, mu, W, 8), null=True)
        meta_out["regime"][nm] = R
    del Zfit, Zall
    torch.cuda.empty_cache()
    srcf = pd.Categorical(src[fit_idx], categories=C.SOURCES5).codes.astype(np.int64)
    sub = np.random.RandomState(AR.seed_of(a.salt, "probe") % (2 ** 31)).choice(len(fit_idx), min(20000, len(fit_idx)), replace=False)
    subt = torch.as_tensor(sub, device=AR.DEV)
    tag = (a.hold_source or "pc") + "|" + a.view
    rng_perm = np.random.RandomState(AR.seed_of(a.salt, "perm4", tag) % (2 ** 31))
    for R, _f in bud:
        lab, npd = labels[R]
        kw = dict(n_prod_batch=192 if npd > 0 else 0, n_t2_batch=32 if npd > 0 else 192, k_t2=4)
        t1 = time.time()
        net, info = AD.train_supcon(Xfit, lab, npd, 8, AR.seed_of(a.salt, "f4", tag, R) % (2 ** 31), steps=steps, **kw)
        Ef = AD.embed(net, Xfit)
        Es = Ef[subt]
        info["eff_rank"] = AD.effective_rank(Es)
        Ess = (Es - Es.mean(0, keepdim=True)) / (Es.std(0, keepdim=True) + 1e-6)
        pr = EA.fit_linear(Ess, torch.as_tensor(srcf[sub], device=AR.DEV), len(C.SOURCES5), 1e-2)(Ess).argmax(1).cpu().numpy()
        info["insample_source_probe_bacc"] = float(EA.bacc(srcf[sub], pr))
        info["regime"], info["train_seconds"] = R, round(time.time() - t1, 1)
        meta_out["f4_train"][f"{R}_d8"] = info
        print(f"  {R} d=8: loss {info['loss_first20']:.3f} -> {info['loss_last200']:.3f}, eff_rank {info['eff_rank']:.1f}, probe {info['insample_source_probe_bacc']:.3f} ({info['train_seconds']}s)", flush=True)
        Eall = AD.embed(net, Xall)
        sdv = Ef.std(0, keepdim=True)
        run_arm(f"mlp_{R}_d8", "adapter", Eall, null=True)
        gen = torch.Generator(device=AR.DEV).manual_seed(AR.seed_of(a.salt, "f4noise", tag, R))
        run_arm(f"mlp_{R}_d8_r0.5", "adapter", Eall + 0.5 * sdv * torch.randn(Eall.shape, generator=gen, device=AR.DEV), null=True)
        meta_out["regime"][f"mlp_{R}_d8"] = meta_out["regime"][f"mlp_{R}_d8_r0.5"] = R
        del net, Ef, Eall
        torch.cuda.empty_cache()
        # memorisation floor: d = 64 adapter on the same labelled wells with permuted labels
        t1 = time.time()
        labp = lab.copy()
        ii = np.where(lab >= 0)[0]
        labp[ii] = lab[ii][rng_perm.permutation(len(ii))]
        net, info = AD.train_supcon(Xfit, labp, npd, 64, AR.seed_of(a.salt, "f4perm", tag, R) % (2 ** 31), steps=steps, **kw)
        info["regime"], info["train_seconds"] = R, round(time.time() - t1, 1)
        meta_out["f4_train"][f"perm_{R}_d64"] = info
        run_arm(f"perm_{R}_d64", "permctrl", AD.embed(net, Xall), null=True, att=False)
        meta_out["regime"][f"perm_{R}_d64"] = R
        del net
        torch.cuda.empty_cache()
    (out / f"meta_{a.salt}.json").write_text(_json.dumps(meta_out, indent=1))


if __name__ == "__main__":
    main()
