"""H3-007 stage 0 (CPU, metadata only): well-position share of positive pairs (A0), chemical near-duplicates between H and T' (B0), position-only releases.
Pre-registered thresholds: results_notes/F.md section 11 (addendum A-H3-007).

  CUDA_VISIBLE_DEVICES= python ctrl_position.py --stage 0 --out $SCRATCH/f5_ctrl_position_meta
Nothing is written to a tracked path; copy the JSON into results/f_jump/ after checking.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap as AP  # noqa: E402
import common as C  # noqa: E402
import evaluate as EV  # noqa: E402

import nd_worker as NW  # noqa: E402


def a0(meta_u, ids_h):
    """share of cross-source positive well pairs (same compound) that share Metadata_Well / row / column; chance from the empirical position distribution."""
    d = meta_u[meta_u["j"].isin(ids_h)]
    N = d.groupby("j").size()
    n_js = d.groupby(["j", "s"]).size()
    S2 = (n_js ** 2).groupby(level=0).sum()
    total = float(((N ** 2 - S2.reindex(N.index)) / 2).sum())

    def same(keycols):
        m = d.groupby(["j"] + keycols).size()
        c = d.groupby(["j"] + keycols + ["s"]).size()
        s2 = (c ** 2).groupby(level=list(range(len(keycols) + 1))).sum()
        return float(((m ** 2 - s2.reindex(m.index)) / 2).sum())
    out = dict(n_compounds=int(len(N)), n_positive_pairs=total, same_well=same(["w"]) / total, same_row=same(["row"]) / total, same_col=same(["col"]) / total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = dict(stage=a.stage, thresholds=dict(A0=0.05, B0=0.02, position_only_open=0.02, position_only_closed=0.01))
    d = C.cache_dir("s5")
    meta = pd.read_parquet(d / "meta.parquet")
    U = EV.universe(meta, True)
    inu = U["inu"]
    jcp, src = U["jcp"], U["src"]
    well = meta["Metadata_Well"].to_numpy()
    plate = meta["Metadata_Plate"].to_numpy()
    ids = np.unique(jcp[inu])
    row = np.array([ord(w[0]) - 65 for w in well])
    col = np.array([int(w[1:]) - 1 for w in well])
    u = pd.DataFrame({"j": jcp[inu], "s": src[inu], "w": well[inu], "row": row[inu], "col": col[inu]})
    p = u["w"].value_counts(normalize=True)
    res["chance_same_well"] = float((p ** 2).sum())
    res["chance_same_row"] = float((u["row"].value_counts(normalize=True) ** 2).sum())
    res["chance_same_col"] = float((u["col"].value_counts(normalize=True) ** 2).sum())
    res["n_positions"] = int(u["w"].nunique())
    roles = {salt: dict(zip(ids, C.split_role(ids, salt))) for salt in C.SALTS}
    # ---------------- A0
    res["A0"] = {}
    for salt in C.SALTS:
        ids_h = [j for j in ids if roles[salt][j] == "H"]
        res["A0"][salt] = a0(u, set(ids_h))
        print(f"A0 {salt}: {res['A0'][salt]}  ({time.time() - t0:.0f}s)", flush=True)
    allp = a0(u, set(ids))
    res["A0"]["all_production"] = allp
    # per source pair (all production compounds)
    sp = {}
    for s1 in C.SOURCES5:
        for s2 in C.SOURCES5:
            if s1 < s2:
                d2 = u[u["s"].isin([s1, s2])]
                m = d2.groupby(["j", "s"]).size().unstack(fill_value=0)
                m = m[(m > 0).all(axis=1)]
                tot = float((m[s1] * m[s2]).sum())
                c = d2[d2["j"].isin(m.index)].groupby(["j", "w", "s"]).size().unstack(fill_value=0)
                if s1 in c and s2 in c and tot > 0:
                    sp[f"{s1}|{s2}"] = dict(pairs=tot, same_well=float((c[s1] * c[s2]).sum() / tot))
    res["A0_by_source_pair"] = sp
    # ---------------- B0
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    comp = pd.read_csv(C.META / "compound.csv.gz").drop_duplicates("Metadata_JCP2022").set_index("Metadata_JCP2022")
    in_tab = np.array([j in comp.index for j in ids])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keys, unscored = [], [], 0
    for j in ids:
        fp, k = None, None
        if j in comp.index:
            r = comp.loc[j]
            k = r["Metadata_InChIKey"][:14] if isinstance(r["Metadata_InChIKey"], str) else None
            mol = Chem.MolFromSmiles(r["Metadata_SMILES"]) if isinstance(r["Metadata_SMILES"], str) else None
            if mol is None and isinstance(r["Metadata_InChI"], str):
                mol = Chem.MolFromInchi(r["Metadata_InChI"])
            if mol is not None:
                fp = gen.GetFingerprint(mol)
        if fp is None:
            unscored += 1
        fps.append(fp)
        keys.append(k)
    print(f"fingerprints done, unscored {unscored} of {len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    res["B0_inputs"] = dict(n_compounds=len(ids), not_in_table=int((~in_tab).sum()), unscored=int(unscored))
    n = len(ids)
    import scipy.sparse as sps
    import torch
    torch.set_num_threads(a.procs)
    rows_, cols_ = [], []
    for i, fp in enumerate(fps):
        if fp is not None:
            on = list(fp.GetOnBits())
            rows_ += [i] * len(on)
            cols_ += on
    A = sps.csr_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)), shape=(n, 2048))
    cnt = torch.as_tensor(np.asarray(A.sum(1)).ravel(), dtype=torch.float32)
    AT = torch.as_tensor(A.T.toarray())                                    # (2048, n)
    edges_t = set()
    for a0_ in range(0, n, 400):
        blk = torch.as_tensor(A[a0_:a0_ + 400].toarray())
        inter = blk @ AT
        sim = inter / (cnt[a0_:a0_ + 400, None] + cnt[None, :] - inter).clamp(min=1)
        sim[:, : a0_ + 1] = 0                                              # strictly upper triangle only (global index j > i)
        for r in range(sim.shape[0]):
            sim[r, : a0_ + r + 1] = 0
        ii, jj = torch.where(sim >= 0.8)
        for x, y in zip(ii.tolist(), jj.tolist()):
            if cnt[a0_ + x] > 0 and cnt[y] > 0:
                edges_t.add((a0_ + x, y))
    print(f"Tanimoto >= 0.8 pairs: {len(edges_t)} ({time.time() - t0:.0f}s)", flush=True)
    kg = pd.Series(keys).dropna()
    edges_k = set()
    for _, g in kg.groupby(kg):
        if len(g) > 1:
            g = list(g.index)
            for x in range(len(g)):
                for y in range(x + 1, len(g)):
                    edges_k.add((g[x], g[y]))
    res["B0_pairs"] = dict(tanimoto_ge_0p8=len(edges_t), same_connectivity_block=len(edges_k), union=len(edges_t | edges_k))
    idx = {j: i for i, j in enumerate(ids)}
    res["B0"] = {}
    for salt in C.SALTS:
        r = np.array([roles[salt][j] for j in ids])
        out_s = {}
        for nm, E in (("tanimoto", edges_t), ("connectivity", edges_k), ("either", edges_t | edges_k)):
            for tag, train in (("T", {"T"}), ("T_or_V", {"T", "V"})):
                exp = set()
                for (i, j) in E:
                    if r[i] == "H" and r[j] in train:
                        exp.add(i)
                    if r[j] == "H" and r[i] in train:
                        exp.add(j)
                out_s[f"{nm}|vs_{tag}"] = dict(n_H_exposed=len(exp), share_of_H=len(exp) / int((r == "H").sum()))
        out_s["n_H"] = int((r == "H").sum())
        res["B0"][salt] = out_s
        print(f"B0 {salt}: either|vs_T {out_s['either|vs_T']}", flush=True)
    # ---------------- position-only releases on split s0 H wells
    salt = "s0"
    hm = np.array([roles[salt].get(j) == "H" if i else False for j, i in zip(jcp, inu)])
    hidx = np.where(hm)[0]
    comp_h = pd.factorize(jcp[hidx])[0]
    src_h = pd.Categorical(src[hidx], categories=C.SOURCES5).codes.astype(np.int64)
    pl_h, wl_h = plate[hidx], well[hidx]
    r_h, c_h = row[hidx], col[hidx]
    pert_h = np.array(["trt"] * len(hidx))
    edge = ((r_h == 0) | (r_h == 15) | (c_h == 0) | (c_h == 23)).astype(np.float32)
    codes = {}
    codes["P1_3number"] = np.stack([r_h, c_h, edge], 1).astype(np.float32)
    oh = np.zeros((len(hidx), 8), np.float32)
    oh[np.arange(len(hidx)), r_h // 4] = 1
    oh[np.arange(len(hidx)), 4 + c_h // 6] = 1
    codes["P2_onehot_bucket8"] = oh
    pos_id = pd.factorize(wl_h)[0]
    P3 = np.zeros((len(hidx), 384), np.float32)
    P3[np.arange(len(hidx)), pos_id] = 1
    codes["P3_onehot_exact384"] = P3
    rng = np.random.RandomState(0)
    pos_sh = pos_id.copy()
    for pl in np.unique(pl_h):
        m = np.where(pl_h == pl)[0]
        pos_sh[m] = pos_id[m][rng.permutation(len(m))]
    P4 = np.zeros_like(P3)
    P4[np.arange(len(hidx)), pos_sh] = 1
    codes["P4_shuffled_within_plate"] = P4
    codes["ref_iid_noise16"] = rng.randn(len(hidx), 16).astype(np.float32)
    pos_cache = AP.build_pos(comp_h, src_h, wl_h, "xs")
    res["position_only"] = {}
    for nm, E in codes.items():
        E = E + 1e-3 * np.random.RandomState(1).randn(*E.shape).astype(np.float32)        # random tie-breaking
        apq = AP.ap_queries(E, pl_h, comp_h, src_h, wl_h, pert_h, mode="xs", pos=pos_cache)
        _, mean_ap = AP.per_compound(apq, comp_h)
        res["position_only"][nm] = dict(U_H=float(mean_ap.mean()), n_compounds=int(len(mean_ap)))
        print(f"position-only {nm}: U_H {mean_ap.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    res["U_ctrl_noise16_s0_from_F3"] = 0.045934073404539484
    # ---------------- verdict (pre-registered rule, section 11.3)
    a0s = [res["A0"][s]["same_well"] for s in C.SALTS]
    b0s = [res["B0"][s]["either|vs_T"]["share_of_H"] for s in C.SALTS]
    un = res["U_ctrl_noise16_s0_from_F3"]
    pu = {k: v["U_H"] for k, v in res["position_only"].items() if k.startswith("P1") or k.startswith("P2") or k.startswith("P3")}
    open_A = bool(max(a0s) >= 0.05)
    open_B = bool(max(b0s) >= 0.02)
    open_P = bool(any(v >= un + 0.02 for v in pu.values()))
    closed_P = bool(all(v <= un + 0.01 for v in pu.values()))
    res["verdict"] = dict(A0_max=max(a0s), B0_max=max(b0s), A_open=open_A, B_open=open_B, position_only_open=open_P, position_only_closed=closed_P,
                          stage0="CLOSED" if (not open_A and not open_B and closed_P) else "OPEN")
    res["seconds"] = round(time.time() - t0, 1)
    (out / "h3_007_stage0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    main()
