"""Utility driver: fine-tune the paper's DenseNet-121 on the DEFENDED images of A- and S-role patients, score V and T images (hashed patient ids), and
(for release `raw` with --labels14) run the `labels14` CODE FLOOR: the 14 outputs of the raw-fine-tuned classifier are the released object; pair attackers on the codes.

  python -m t_cxr.run_utility --release rw2_mu0.01 --rot 0 --out $OUT ...
Outputs (JSON only): utility_<release>_rot<r>.json = {V,T}: hashed patient id per image, Y (14 findings), P (fine-tuned classifier), P_fixed (the paper's fixed classifier);
labels14_rot<r>/{summary.json, preds_*.json} when --labels14.  Nothing else is written to $OUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import heads as H, meta, pairs, releases as R, stats as S, utility as U
from .meta import PRI, hid
from .run_suite import planted_bank, split_pairs


def findings_matrix(d: pd.DataFrame, csv: str) -> np.ndarray:
    raw = pd.read_csv(csv); raw.columns = [c.strip() for c in raw.columns]
    lab = dict(zip(raw["Image Index"], raw["Finding Labels"].astype(str)))
    fl = d["img"].map(lab).str.split("|")
    return np.array([[f in row for f in meta.FINDINGS] for row in fl], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--rot", type=int, default=0); ap.add_argument("--labels14", action="store_true")
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", default=None); ap.add_argument("--salt-file", default=None)
    ap.add_argument("--gen-dir", required=True); ap.add_argument("--flow-stats", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--exclude", default=None); ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--feat", default="dinov2", choices=["dinov2", "random"]); ap.add_argument("--tiny", action="store_true", help="tests only: tiny classifier, no weights")
    a = ap.parse_args()
    t0 = time.time(); dev = a.device
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    if not a.synthetic and not a.exclude:
        raise SystemExit("--exclude exclude_dups.json is required for real runs (T.md A2)")
    d = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in PRI], d)
    if a.synthetic:
        salt = "PROVISIONAL-selftest"; rng = np.random.default_rng(0)
        keep = set(rng.choice(d.pid.unique(), size=a.synthetic, replace=False))
        d = d[d.pid.isin(keep)].reset_index(drop=True); d["row"] = np.arange(len(d)); bank = planted_bank(d, a.res, 1.0)
    else:
        salt = Path(a.salt_file).read_text().strip(); bank = np.load(a.bank, mmap_mode="r")
    d = meta.assign_roles(d, seen, salt, a.rot)
    Y = findings_matrix(d, csv)
    flow = torch.load(a.flow_stats) if a.flow_stats else None
    pca = R.fit_pca(bank, d.loc[(d.role == "A") & d.ok, "row"].to_numpy()[:10000], size=min(64, a.res)) if a.release.startswith("pca_") else None
    rel = R.Releases(dev, gen_dir=a.gen_dir, flow_stats=flow, pca=pca)
    rows = {r: d.index[(d.role == r) & d.ok].to_numpy() for r in "SAVT"}
    ids = d["img"].to_numpy()
    mat = lambda idx: rel.materialize(a.release, bank, idx, ids[idx].tolist())
    tr_idx = np.r_[rows["A"], rows["S"]]
    Xtr, Xv, Xt = mat(tr_idx), mat(rows["V"]), mat(rows["T"])
    if a.tiny:
        import torch.nn as nn
        base = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 14), nn.Sigmoid()); size = 32
        fixed = base
    else:
        base = U.load_classifier(f"{a.gen_dir}/pretrained_classifier.pth", dev); size = 224
        fixed = U.load_classifier(f"{a.gen_dir}/pretrained_classifier.pth", dev)
    Pf_t = U.predict(fixed, Xt, dev, size=size); Pf_v = U.predict(fixed, Xv, dev, size=size)
    model = U.finetune(base, Xtr, Y[tr_idx], epochs=a.epochs, device=dev, size=size, log=lambda *x: print(*x, flush=True))
    P_v, P_t = U.predict(model, Xv, dev, size=size), U.predict(model, Xt, dev, size=size)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = lambda idx, P, Pf: dict(patient=[hid(p, salt) for p in d["pid"].to_numpy()[idx]], Y=Y[idx].tolist(), P=np.round(P, 4).tolist(), P_fixed=np.round(Pf, 4).tolist())
    uv, _ = U.mean_auc(Y[rows["V"]], P_v); ut, per = U.mean_auc(Y[rows["T"]], P_t); uf, _ = U.mean_auc(Y[rows["T"]], Pf_t)
    (out / f"utility_{a.release}_rot{a.rot}.json").write_text(json.dumps(dict(release=a.release, rot=a.rot, synthetic=bool(a.synthetic), U_val=uv, U_test=ut, U_fixed_test=uf, per_label_test=np.round(per, 4).tolist(),
                                                                      V=rec(rows["V"], P_v, Pf_v), T=rec(rows["T"], P_t, Pf_t), seconds=round(time.time() - t0, 1))))
    print(f"utility {a.release}: U_val {uv:.3f} U_test {ut:.3f} U_fixed_test {uf:.3f}", flush=True)
    if a.labels14 and a.release == "raw":
        # CODE FLOOR: released object = the 14 outputs of the raw-fine-tuned classifier; attackers see codes of both elements
        need_rows = {}
        pA, _ = pairs.build(d, "A", "N2", 2, "tr0", col="role"); pV, _ = pairs.build(d, "V", "N2", 1, "v0", col="role"); pT, _ = pairs.build(d, "T", "N2", 1, "te0", col="role")
        need = np.unique(np.r_[pA.a, pA.b, pV.a, pV.b, pT.a, pT.b]); pos_of = {r: i for i, r in enumerate(need)}
        Xn = mat(need)
        C = U.predict(model, Xn, dev, size=size)                                   # (n,14) codes
        parts = dict(tr=split_pairs(pA, pos_of), va=split_pairs(pV, pos_of), te=split_pairs(pT, pos_of))
        o = H.fit_f3(C, parts["tr"], C, parts["va"], C, parts["te"])
        l14 = out / f"labels14_rot{a.rot}"; l14.mkdir(exist_ok=True)
        cl_te = pT.anchor_pid.map(lambda p: hid(p, salt)).to_numpy(); cl_va = pV.anchor_pid.map(lambda p: hid(p, salt)).to_numpy()
        summ = dict(release="labels14", rot=a.rot, synthetic=bool(a.synthetic), attackers={})
        for nm, (sv, st) in o.items():
            pb = S.PatientBootstrap(cl_te, B=1000, seed=0)
            r = dict(val_auc=S.auc(parts["va"]["y"], sv), test=S.summarize(S.auc(parts["te"]["y"], st), S.auc_boot(parts["te"]["y"], st, pb.codes(cl_te), pb)))
            summ["attackers"][nm] = r
            (l14 / f"preds_{nm}.json").write_text(json.dumps(dict(pair_id=list(range(len(st))), cluster=cl_te.tolist(), y=parts["te"]["y"].tolist(), score_test=np.round(st, 5).tolist(),
                                                                score_val=np.round(sv, 5).tolist(), cluster_val=cl_va.tolist(), y_val=parts["va"]["y"].tolist())))
            print(f"  labels14 {nm}: val {r['val_auc']:.3f} test {r['test']['point']:.3f}", flush=True)
        (l14 / "summary.json").write_text(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
