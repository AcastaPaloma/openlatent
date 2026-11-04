"""Evaluate one representation with the retrained attacker suite; returns metrics + stored predictions.

Tasks
  centre    attackers trained on B (E(z) of attacker-train units) -> predict V, C.  Secret = medical centre.
  label     probes trained on B -> predict V, C and all unseen-centre (OOD) patches.  Utility = tissue/diagnosis.
  oodcentre attackers trained on OOD1 (half of the unseen centres' units) -> predict OOD2 (units disjoint).
  centre_onA  diagnostic: linear attacker trained on the adapter-train units A (detects adapter overfitting).
Floors (label-only baseline): predict the centre from the TRUE label with the attacker-train joint frequencies.
All scores are balanced accuracy.
"""
import time

import numpy as np

from .attackers import run_suite, bacc, EXTRA

KINDS_CENTRE = ("linear", "mlp", "knn", "tfm")
KINDS_LABEL = ("linear", "mlp", "knn")
KINDS_OOD = ("linear", "mlp", "knn")


def label_only_floor(c_tr, y_tr, c_te, y_te):
    n_c = int(max(c_tr.max(), c_te.max())) + 1
    n_y = int(max(y_tr.max(), y_te.max())) + 1
    cnt = np.zeros((n_y, n_c))
    for yy, cc in zip(y_tr, c_tr):
        cnt[yy, cc] += 1
    pred = cnt.argmax(1)[y_te]
    return bacc(c_te, pred)


def knn_retention(Zq, Zg, Rq, Rg, cq, cg, k_raw=10, k_rep=30):
    """Retention of non-label neighbourhood structure.  Queries = test units C, gallery = attacker-train units B (disjoint units).
    For each query the k_raw nearest gallery patches in the RAW embedding space (euclidean, B-standardised) are the reference; the score is the
    fraction of them found among the k_rep nearest gallery patches in the released representation (B-standardised).  `all` uses the whole gallery,
    `xc` only gallery patches from OTHER centres (cross-centre structure, which a centre-invariant representation should keep).  Chance ~ k_rep/|gallery|."""
    import torch
    from .attackers import DEV, _t, Std

    def prep(Q, G):
        s = Std().fit(_t(G))
        return s(_t(Q)), s(_t(G))
    zq, zg = prep(Zq, Zg)
    rq, rg = prep(Rq, Rg)
    cq_t, cg_t = _t(cq, torch.long), _t(cg, torch.long)
    out_all, out_xc = [], []
    for i in range(0, zq.shape[0], 1024):
        dz = torch.cdist(zq[i:i + 1024], zg)
        dr = torch.cdist(rq[i:i + 1024], rg)
        for dd_z, dd_r, out in ((dz, dr, out_all), (dz.clone(), dr.clone(), out_xc)):
            if out is out_xc:
                same = cq_t[i:i + 1024, None] == cg_t[None, :]
                dd_z[same] = float("inf"); dd_r[same] = float("inf")
            ref = dd_r.topk(k_raw, largest=False).indices
            got = dd_z.topk(k_rep, largest=False).indices
            hit = (ref[:, :, None] == got[:, None, :]).any(-1).float().mean(1)
            out.append(hit.cpu())
    return torch.cat(out_all).numpy(), torch.cat(out_xc).numpy()


def remap(c):
    u, inv = np.unique(c, return_inverse=True)
    return inv, len(u)


def evaluate(rep, P, sp, seed, kinds_centre=KINDS_CENTRE + EXTRA, secret_shuffle=False, shuffle_rep=0, do_ood=True, do_onA=True, holdout=False, patchsplit=False):
    t0 = time.time()
    m, preds, timing = {}, {}, {}
    Kc, Ky = int(P.n_id_centres), len(P.classes)
    yB, cB, uB = P.y[sp["B"]], P.c[sp["B"]], P.u[sp["B"]]
    zood = np.concatenate([rep["OOD1"], rep["OOD2"]])
    ev = {"V": rep["V"], "C": rep["C"]}
    t = time.time()
    r = run_suite(rep["B"], cB, uB, ev, Kc, kinds=kinds_centre, seed=seed, secret_shuffle=secret_shuffle, shuffle_rep=shuffle_rep)
    timing["centre"] = time.time() - t
    for k, d in r.items():
        for s, p in d.items():
            preds[f"centre/{k}/{s}"] = p.astype(np.uint8)
            m[f"centre/{k}/{s}"] = bacc(P.c[sp[s]], p)
    ev_l = dict(ev, OOD=zood)
    t = time.time()
    r = run_suite(rep["B"], yB, uB, ev_l, Ky, kinds=KINDS_LABEL, seed=seed)
    timing["label"] = time.time() - t
    ood_idx = np.concatenate([sp["OOD1"], sp["OOD2"]])
    for k, d in r.items():
        for s, p in d.items():
            tgt = P.y[ood_idx] if s == "OOD" else P.y[sp[s]]
            preds[f"label/{k}/{s}"] = p.astype(np.uint8)
            m[f"label/{k}/{s}"] = bacc(tgt, p)
    ra, rx = knn_retention(rep["C"], rep["B"], P.X[sp["C"]], P.X[sp["B"]], P.c[sp["C"]], cB)
    preds["ret/knn_all"], preds["ret/knn_xc"] = ra.astype(np.float16), rx.astype(np.float16)
    m["ret/knn_all"], m["ret/knn_xc"] = float(ra.mean()), float(rx.mean())
    rav, rxv = knn_retention(rep["V"], rep["B"], P.X[sp["V"]], P.X[sp["B"]], P.c[sp["V"]], cB)
    m["retV/knn_all"], m["retV/knn_xc"] = float(rav.mean()), float(rxv.mean())
    if holdout:      # utility on classes the adapter never saw (and on the seen ones), probes trained on B restricted to those classes
        from .splits import seen_held_classes
        seen, held = seen_held_classes(P)
        for nm, cl in (("seen", seen), ("held", held)):
            mb = np.isin(yB, cl)
            remap_c = {k: i for i, k in enumerate(cl)}
            ytr = np.array([remap_c[k] for k in yB[mb]])
            evs = {}
            ood_idx_h = np.concatenate([sp["OOD1"], sp["OOD2"]])
            for s in ("V", "C"):
                ms = np.isin(P.y[sp[s]], cl)
                evs[s] = rep[s][ms]
            mo = np.isin(P.y[ood_idx_h], cl)
            evs["OOD"] = zood[mo]
            r = run_suite(rep["B"][mb], ytr, uB[mb], evs, len(cl), kinds=KINDS_LABEL, seed=seed)
            for k, d in r.items():
                for s, p in d.items():
                    yy = P.y[ood_idx_h] if s == "OOD" else P.y[sp[s]]
                    ms = np.isin(yy, cl)
                    tgt = np.array([remap_c[k2] for k2 in yy[ms]])
                    preds[f"label{nm}/{k}/{s}"] = p.astype(np.uint8)
                    m[f"label{nm}/{k}/{s}"] = bacc(tgt, p)
    if do_ood:
        cO, Ko = remap(P.c[ood_idx])
        n1 = len(sp["OOD1"])
        t = time.time()
        r = run_suite(rep["OOD1"], cO[:n1], P.u[sp["OOD1"]], {"OOD2": rep["OOD2"]}, Ko, kinds=KINDS_OOD, seed=seed)
        timing["oodcentre"] = time.time() - t
        for k, d in r.items():
            p = d["OOD2"]
            preds[f"oodcentre/{k}/OOD2"] = p.astype(np.uint8)
            m[f"oodcentre/{k}/OOD2"] = bacc(cO[n1:], p)
        m["floor/oodcentre_label_only"] = label_only_floor(cO[:n1], P.y[sp["OOD1"]], cO[n1:], P.y[sp["OOD2"]])
        m["chance/oodcentre"] = 1.0 / Ko
    if do_onA:
        r = run_suite(rep["A"], P.c[sp["A"]], P.u[sp["A"]], {"C": rep["C"]}, Kc, kinds=("linear",), seed=seed)
        m["centre_onA/linear/C"] = bacc(P.c[sp["C"]], r["linear"]["C"])
    if patchsplit:   # slide-fingerprint diagnostic (H-A-001 sub-control): attackers trained and tested on patches of the SAME test units
        rng = np.random.RandomState(4242 + seed)
        idx = rng.permutation(len(sp["C"]))
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        r = run_suite(rep["C"][h1], P.c[sp["C"]][h1], P.u[sp["C"]][h1], {"C2": rep["C"][h2]}, Kc, kinds=KINDS_LABEL, seed=seed)
        for k, d in r.items():
            m[f"centre_patch/{k}/C2"] = bacc(P.c[sp["C"]][h2], d["C2"])
    m["floor/centre_label_only/C"] = label_only_floor(cB, yB, P.c[sp["C"]], P.y[sp["C"]])
    m["floor/centre_label_only/V"] = label_only_floor(cB, yB, P.c[sp["V"]], P.y[sp["V"]])
    m["chance/centre"] = 1.0 / Kc
    m["chance/label"] = 1.0 / Ky
    m["dim"] = int(rep["C"].shape[1])
    m["seconds_eval"] = round(time.time() - t0, 1)
    m["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return m, preds
