"""R0 (T.md 7.4): the paper's own protocol with the RELEASED weights, on PriCheXy's own pair lists and test patients (descriptive, no kill).
Verifier = Siamese ResNet-50, ImageNet init, Adam 1e-4, batch 32, early stopping patience 5 on the validation pairs, retrained for every run.
cond `mismatch` (the paper): train and validate on (defended, defended) pairs, test (defended, REAL).  cond `match`: train, validate and test on (defended, REAL).
Release `raw` or `pcx_mu0.01` (released generator).  Outputs per-run test AUC, mean +- std over runs, the AUC of the run-averaged score with a patient-cluster bootstrap
(1,307 patients behind the 5,000 test pairs).  JSON only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import attackers as A, meta, pairs, releases as R, stats as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--cond", default="mismatch", choices=["mismatch", "match"]); ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--bank", required=True); ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--max-epochs", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time(); dev = a.device
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt")       # burned data, no exclusion: faithful replication
    bank = np.load(a.bank, mmap_mode="r")
    P = {k: pairs.from_pair_file(f"{a.pri}/image_pairs_{k}.txt", d) for k in ("training_10000", "validation_2000", "testing_5000")}
    need = np.unique(np.concatenate([np.r_[p.a, p.b] for p in P.values()])); pos = {r: i for i, r in enumerate(need)}
    ids = d["img"].to_numpy()[need].tolist()
    rel = R.Releases(dev, gen_dir=a.gen_dir)
    Xr = rel.materialize("raw", bank, need, ids)
    Xd = Xr if a.release == "raw" else rel.materialize(a.release, bank, need, ids)
    mk = lambda p: dict(ia=np.array([pos[r] for r in p.a]), ib=np.array([pos[r] for r in p.b]), y=p.label.to_numpy().astype(np.float32))
    tr, va, te = (mk(P[k]) for k in ("training_10000", "validation_2000", "testing_5000"))
    xb_train = Xd if a.cond == "mismatch" else Xr
    scores, aucs = [], []
    for run in range(a.runs):
        o = A.train_siamese(dict(xa=Xd, xb=xb_train, **tr), dict(xa=Xd, xb=xb_train, **va), dict(xa=Xd, xb=Xr, **te), arch="resnet50", pretrained=True, seed=run, max_epochs=a.max_epochs, bs=32, patience=5,
                            device=dev, amp=dev == "cuda", log=lambda *x: None)
        scores.append(o["test_score"]); aucs.append(S.auc(te["y"], o["test_score"]))
        print(f"run {run}: test AUC {aucs[-1]:.4f} ({o['epochs']} epochs)", flush=True)
    cl = np.array([meta.hid(p, "r0") for p in P["testing_5000"].pid_a])
    pb = S.PatientBootstrap(np.unique(np.r_[P["testing_5000"].pid_a, P["testing_5000"].pid_b]), B=1000, seed=0)
    cl_pairs = np.array([meta.hid(p, "r0") for p in P["testing_5000"].pid_a])
    pb = S.PatientBootstrap(np.unique(cl_pairs), B=1000, seed=0)
    ens = np.mean(scores, 0)
    res = dict(release=a.release, cond=a.cond, runs=a.runs, test_auc_runs=aucs, mean=float(np.mean(aucs)), std=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
               ensemble=S.summarize(S.auc(te["y"], ens), S.auc_boot(te["y"], ens, pb.codes(cl_pairs), pb)), n_test_pairs=len(te["y"]), n_patients_anchor=int(len(np.unique(cl_pairs))),
               paper=dict(raw=[81.8, 0.6], pcx_mu0_01=[57.7, 4.0]), seconds=round(time.time() - t0, 1))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"r0_{a.release}_{a.cond}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "test_auc_runs"}, default=float))


if __name__ == "__main__":
    main()
