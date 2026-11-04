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

from .attackers import run_suite, bacc

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


def remap(c):
    u, inv = np.unique(c, return_inverse=True)
    return inv, len(u)


def evaluate(rep, P, sp, seed, kinds_centre=KINDS_CENTRE, secret_shuffle=False, do_ood=True, do_onA=True):
    t0 = time.time()
    m, preds, timing = {}, {}, {}
    Kc, Ky = int(P.n_id_centres), len(P.classes)
    yB, cB, uB = P.y[sp["B"]], P.c[sp["B"]], P.u[sp["B"]]
    zood = np.concatenate([rep["OOD1"], rep["OOD2"]])
    ev = {"V": rep["V"], "C": rep["C"]}
    t = time.time()
    r = run_suite(rep["B"], cB, uB, ev, Kc, kinds=kinds_centre, seed=seed, secret_shuffle=secret_shuffle)
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
    m["floor/centre_label_only/C"] = label_only_floor(cB, yB, P.c[sp["C"]], P.y[sp["C"]])
    m["floor/centre_label_only/V"] = label_only_floor(cB, yB, P.c[sp["V"]], P.y[sp["V"]])
    m["chance/centre"] = 1.0 / Kc
    m["chance/label"] = 1.0 / Ky
    m["dim"] = int(rep["C"].shape[1])
    m["seconds_eval"] = round(time.time() - t0, 1)
    m["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return m, preds
