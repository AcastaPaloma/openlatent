"""Replicate-retrieval average precision.

(1) `ap_queries`: GPU implementation of copairs' per-query AP (positives vs negatives ranked together by cosine similarity,
    AP = mean over positives of precision at the positive's rank; strict ">" for ties), validated against copairs 0.4.1 (see validate.py).
(2) `benchmark_stats`: the benchmark's own metric code (metrics/map.py of carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection, BSD-3) calling copairs.

Modes (positives / negatives), all with cosine similarity:
  xs      pos = same compound, DIFFERENT source ; neg = same plate, other compound, treatment wells           (primary U, JS-1)
  nonrep  pos = same compound, any other well   ; neg = same plate, other compound, treatment wells           (benchmark "non-replicates")
  negcon  pos = same compound, different well position ; neg = DMSO wells of the same plate                  (benchmark "controls")
"""
import numpy as np
import pandas as pd
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pos(comp, source, well, mode):
    n = len(comp)
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    pos = [None] * n
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        if mode == "xs":
            M = source[idx][:, None] != source[idx][None, :]
        elif mode == "nonrep":
            M = ~np.eye(len(idx), dtype=bool)
        elif mode == "negcon":
            M = well[idx][:, None] != well[idx][None, :]
        else:
            raise ValueError(mode)
        for r, q in enumerate(idx):
            pos[q] = idx[M[r]]
    return pos


def ap_queries(E, plate, comp, source, well, pert, mode="xs", chunk=512, pos=None):
    """E: (n, d) array of the eval universe; comp/source: int arrays; plate/well: any hashable arrays; pert in {'trt','negcon','poscon'}.
    Queries = 'trt' wells with >= 1 positive and >= 1 negative.  Returns per-query AP (NaN for dropped queries)."""
    n = len(comp)
    Et = torch.as_tensor(np.ascontiguousarray(E), dtype=torch.float32, device=DEV)
    Et = torch.nn.functional.normalize(Et, dim=1)
    if pos is None:
        pos = build_pos(comp, source, well, mode)
    ap = np.full(n, np.nan, np.float64)
    plate = np.asarray(plate)
    order = np.argsort(plate, kind="stable")
    ps = plate[order]
    bounds = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    trt = pert == "trt"
    for a, b in zip(bounds[:-1], bounds[1:]):
        wp = order[a:b]
        q = np.array([i for i in wp[trt[wp]] if pos[i] is not None and len(pos[i]) > 0], dtype=np.int64)
        if len(q) == 0:
            continue
        cand = wp[trt[wp]] if mode in ("xs", "nonrep") else wp[pert[wp] == "negcon"]
        if len(cand) == 0:
            continue
        for s in range(0, len(q), chunk):
            qs = q[s:s + chunk]
            Pm = max(len(pos[i]) for i in qs)
            P = np.full((len(qs), Pm), -1, np.int64)
            for r, i in enumerate(qs):
                P[r, :len(pos[i])] = pos[i]
            qt = torch.as_tensor(qs, device=DEV)
            ct = torch.as_tensor(cand, device=DEV)
            Pt = torch.as_tensor(P, device=DEV)
            valid = Pt >= 0
            Sn = Et[qt] @ Et[ct].T                                          # (nq, nc)
            negok = torch.as_tensor(comp[qs][:, None] != comp[cand][None, :], device=DEV)
            Sn = torch.where(negok, Sn, torch.full_like(Sn, -float("inf")))
            Sp = torch.einsum("qd,qpd->qp", Et[qt], Et[Pt.clamp(min=0)])
            Sp = torch.where(valid, Sp, torch.full_like(Sp, -float("inf")))
            nneg = negok.sum(1)
            gt_neg = (Sn.unsqueeze(1) > Sp.unsqueeze(2)).sum(-1)               # (nq, Pm)
            gt_pos = ((Sp.unsqueeze(1) > Sp.unsqueeze(2)) & valid.unsqueeze(1)).sum(-1)
            prec = (1.0 + gt_pos) / (1.0 + gt_pos + gt_neg)
            val = (prec * valid).sum(1) / valid.sum(1)
            ok = (nneg > 0).cpu().numpy()
            v = val.double().cpu().numpy()
            ap[qs[ok]] = v[ok]
    return ap


def per_compound(ap, comp):
    """mean over queries of each compound -> (compound ids, mean AP); compounds without a valid query are dropped."""
    m = ~np.isnan(ap)
    ids, inv = np.unique(comp[m], return_inverse=True)
    s = np.bincount(inv, weights=ap[m], minlength=len(ids))
    c = np.bincount(inv, minlength=len(ids))
    return ids, s / c


# ----------------------------------------------------------------------------------- the benchmark's own metric code (copairs 0.4.1)
def _index(meta, plate_types, ignore_dmso=False):
    index = meta["Metadata_PlateType"].isin(plate_types)
    index &= (meta["Metadata_PertType"] != "poscon")
    valid = meta.loc[index, "Metadata_JCP2022"].value_counts()
    valid = valid[valid > 1].index
    index &= meta["Metadata_JCP2022"].isin(valid)
    index &= (meta["Metadata_JCP2022"] != "JCP2022_033954")
    if ignore_dmso:
        index &= (meta["Metadata_JCP2022"] != "DMSO")
    return index.values


def benchmark_stats(meta, X, plate_types, null_size=10000, seed=0):
    """Arevalo metrics/map.py: returns {'negcon': {...}, 'nonrep': {...}} with the three candidate 'mAP' statistics each.
    meta needs Metadata_PlateType/PertType/JCP2022/Plate/Well; X float array aligned with meta."""
    import copairs.map as cp
    out = {}
    for kind in ("negcon", "nonrep"):
        ix = _index(meta, plate_types, ignore_dmso=(kind == "nonrep"))
        m = meta[ix].copy().reset_index(drop=True)
        v = X[ix]
        if kind == "negcon":
            neg = (m["Metadata_JCP2022"] == "DMSO").to_numpy()
            ids = m["Metadata_JCP2022"].astype("category")
            ids = ids.cat.add_categories([f"DMSO_{i}" for i in range(int(neg.sum()))])
            ids[neg] = [f"DMSO_{i}" for i in range(int(neg.sum()))]
            m["Metadata_JCP2022"] = ids
            res = cp.average_precision(m, v, pos_sameby=["Metadata_JCP2022"], pos_diffby=["Metadata_Well"], neg_sameby=["Metadata_Plate"],
                                       neg_diffby=["Metadata_PertType", "Metadata_JCP2022"], batch_size=20000)
            res = res.query('Metadata_PertType!="negcon"')
        else:
            res = cp.average_precision(m, v, pos_sameby=["Metadata_JCP2022"], pos_diffby=[], neg_sameby=["Metadata_Plate"],
                                       neg_diffby=["Metadata_JCP2022"], batch_size=20000)
        res = res.reset_index(drop=True)
        ms = cp.mean_average_precision(res, "Metadata_JCP2022", threshold=0.05, null_size=null_size, seed=seed)
        out[kind] = dict(mean_average_precision=float(ms["mean_average_precision"].dropna().mean()), fraction_below_p=float(ms["below_p"].mean()),
                         fraction_below_corrected_p=float(ms["below_corrected_p"].mean()), n_compounds=int(len(ms)), n_queries=int(len(res)))
    return out


def copairs_xs_ap(meta, X):
    """copairs per-query AP for the primary protocol (cross-source positives, same-plate other-compound negatives); used only to validate ap_queries."""
    import copairs.map as cp
    m = meta.reset_index(drop=True)
    res = cp.average_precision(m, X, pos_sameby=["Metadata_JCP2022"], pos_diffby=["Metadata_Source"], neg_sameby=["Metadata_Plate"],
                               neg_diffby=["Metadata_JCP2022"], batch_size=20000)
    return res
