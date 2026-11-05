"""T2 utility (T.md 7.3): U_T2 = the DenseNet-121 classifier of train_pcx.py (ImageNet init, trained on RAW S-role images) FINE-TUNED 2 epochs on the DEFENDED images of S-role
patients only, then scored on the defended images of ALL generator-unseen patients; the T / V images of every rotation are scored from this single fine-tune (rotation-free).
For release `raw` with --labels14 the 14-number code floor is also evaluated per rotation (pair attackers on the codes).  JSON only in $OUT (hashed patient ids).
   python -m t_cxr.run_utility_t2 --release rw2_mu0.01 --rots 0,1,2 --clf-init W/classifier.pth ..."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import heads as H, meta, pairs, releases as R, stats as S, utility as U
from .meta import PRI, hid
from .run_suite import planted_bank, split_pairs
from .run_utility import findings_matrix
from .train_pcx import dense121


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--rots", default="0,1,2"); ap.add_argument("--labels14", action="store_true")
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", default=None); ap.add_argument("--salt-file", default=None)
    ap.add_argument("--exclude", default=None); ap.add_argument("--gen-dir", default=None); ap.add_argument("--pcxr-dir", default=None); ap.add_argument("--flow-stats", default=None)
    ap.add_argument("--clf-init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--ladder-s", type=int, default=0); ap.add_argument("--design", default="N2", choices=["N2", "N3", "N3s"]); ap.add_argument("--skip-utility-json", action="store_true"); ap.add_argument("--r5", action="store_true"); ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--res", type=int, default=64); ap.add_argument("--no-pretrain", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); dev = a.device; rots = [int(x) for x in a.rots.split(",")]
    if not a.synthetic and not a.exclude:
        raise SystemExit("--exclude exclude_dups.json is required for real runs (T.md A2)")
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    d = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in PRI], d)
    if a.synthetic:
        salt = "PROVISIONAL-selftest"; rng = np.random.default_rng(0)
        keep = set(rng.choice(d.pid.unique(), size=a.synthetic, replace=False)); keep |= set(list(seen)[: a.synthetic // 3])
        d = d[d.pid.isin(keep)].reset_index(drop=True); d["row"] = np.arange(len(d)); bank = planted_bank(d, a.res, 1.0)
        seen = seen & keep
    else:
        salt = Path(a.salt_file).read_text().strip(); bank = np.load(a.bank, mmap_mode="r")
    droles = {r: meta.assign_roles(d, seen, salt, r) for r in rots}
    isS = d["pid"].isin(seen).to_numpy()
    rowsS = d.index[isS & d.ok].to_numpy(); rowsU = d.index[~isS & d.ok].to_numpy()
    Y = findings_matrix(d, csv)
    flow = torch.load(a.flow_stats) if a.flow_stats else None
    pca = R.fit_pca(bank, droles[rots[0]].loc[(droles[rots[0]].role == "A") & d.ok, "row"].to_numpy()[:10000], size=min(64, a.res)) if a.release.startswith("pca_") else None
    rel = R.Releases(dev, gen_dir=a.gen_dir, flow_stats=flow, pca=pca, pcxr_dir=a.pcxr_dir)
    ids = d["img"].to_numpy()
    mat = lambda idx: rel.materialize(a.release, bank, idx, ids[idx].tolist())
    Xs, Xu = mat(rowsS), mat(rowsU)
    clf = dense121(not a.no_pretrain)
    clf.load_state_dict(torch.load(a.clf_init, map_location="cpu")); clf = clf.to(dev)
    model = U.finetune(clf, Xs, Y[rowsS], epochs=a.epochs, lr=1e-4, bs=32, seed=0, device=dev, size=a.size, log=lambda *x: print(*x, flush=True))
    P = U.predict(model, Xu, dev, size=a.size)
    pos_u = {r: i for i, r in enumerate(rowsU)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for r in (rots if not a.skip_utility_json else []):
        dr = droles[r]
        def for_role(role):
            rows = dr.index[(dr.role == role) & dr.ok].to_numpy()
            return np.array([pos_u[i] for i in rows]), rows
        iv, rv = for_role("V"); it, rt = for_role("T")
        rec = lambda idx, rows: dict(patient=[hid(p, salt) for p in d["pid"].to_numpy()[rows]], Y=Y[rows].tolist(), P=np.round(P[idx], 4).tolist(), P_fixed=np.round(P[idx], 4).tolist())
        uv, _ = U.mean_auc(Y[rv], P[iv]); ut, per = U.mean_auc(Y[rt], P[it])
        (out / f"utility_{a.release}_rot{r}.json").write_text(json.dumps(dict(release=a.release, rot=r, definition="T2: fine-tuned 2 epochs on defended S images, scored on unseen", synthetic=bool(a.synthetic),
                                                                        U_val=uv, U_test=ut, per_label_test=np.round(per, 4).tolist(), V=rec(iv, rv), T=rec(it, rt), seconds=round(time.time() - t0, 1))))
        print(f"utility {a.release} rot{r}: U_val {uv:.3f} U_test {ut:.3f}", flush=True)
    if a.labels14 and a.release == "raw":
        labcode = (Y.astype(np.int64) * (1 << np.arange(14))).sum(1)
        sfx = (("_ladder" + ("all" if a.ladder_s < 0 else str(a.ladder_s))) if a.ladder_s else "") + (("_" + a.design.lower()) if a.design != "N2" else "") + ("_r5" if a.r5 else "")
        for r in rots:
            dr = droles[r]
            pA, _ = pairs.build(dr, "A", a.design, 2, "tr0", col="role", labcode=labcode); pV, _ = pairs.build(dr, "V", a.design, 1, "v0", col="role", labcode=labcode); pT, _ = pairs.build(dr, "T", a.design, 1, "te0", col="role", labcode=labcode)
            if a.ladder_s:                                   # T4a: S-role extra training pairs (their codes are in-sample for the S-trained classifier: stated)
                posS = pairs.positives(dr, "S", 2, "lad", col="role")
                order = posS.pid.drop_duplicates().map(lambda p: meta.unit(p, f"{salt}:ladder")).sort_values().index
                keep_pid = set(posS.pid.drop_duplicates().loc[order][: (None if a.ladder_s < 0 else a.ladder_s)])
                posS = posS[posS.pid.isin(keep_pid)].reset_index(drop=True)
                pS, _ = pairs.build(dr, "S", "N2", 2, "lad", positives_df=posS, col="role")
                pS["triple"] = pS["triple"] + 10**6
                pA = pd.concat([pA, pS], ignore_index=True)
            need = np.unique(np.r_[pA.a, pA.b, pV.a, pV.b, pT.a, pT.b]); pos = {x: i for i, x in enumerate(need)}
            C = U.predict(model, mat(need), dev, size=a.size)
            parts = dict(tr=split_pairs(pA, pos), va=split_pairs(pV, pos), te=split_pairs(pT, pos))
            o = (H.fit_f3_xl if a.r5 else H.fit_f3)(C, parts["tr"], C, parts["va"], C, parts["te"])
            l14 = out / f"labels14_rot{r}{sfx}"; l14.mkdir(exist_ok=True)
            cl_te = pT.anchor_pid.map(lambda p: hid(p, salt)).to_numpy(); cl_va = pV.anchor_pid.map(lambda p: hid(p, salt)).to_numpy()
            for nm, (sv, st) in o.items():
                (l14 / f"preds_{nm}.json").write_text(json.dumps(dict(pair_id=list(range(len(st))), cluster=cl_te.tolist(), y=parts["te"]["y"].tolist(), score_test=np.round(st, 5).tolist(),
                                                                    score_val=np.round(sv, 5).tolist(), cluster_val=cl_va.tolist(), y_val=parts["va"]["y"].tolist())))
                print(f"  labels14{sfx} rot{r} {nm}: test {S.auc(parts['te']['y'], st):.3f}", flush=True)
            (l14 / "summary.json").write_text(json.dumps(dict(release="labels14", rot=r, design=a.design, ladder_s=a.ladder_s, n_triples_te=int(len(pT) // 2))))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
