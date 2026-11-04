"""E2 job: inductive learned encoders on HLCA.  One seed per job.  Writes per-cell predictions in the SC-0 format (arm names `e2_<config>`) plus training-only checks.

  python evaluate_e2.py --seed 0 --out $OUT [--configs a,b] [--folds 0,1,2] [--beta_scale 1.0] [--diag]
--diag: train fold 0 / seed 0 only and print training KL, reconstruction R^2 and the in-sample adversary accuracy; NO held-out cell is scored.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attackers as AT  # noqa: E402
import inductive as IN  # noqa: E402
from common import CACHE, MIN_HELD, MIN_TRAIN, SC  # noqa: E402
from evaluate import get_z0, halves, retention  # noqa: E402


def adapter_check(codes_tr, btr, utr, seed, max_n=15000):
    """training donors split in two halves; fresh linear + MLP attackers (class-balanced) on one half's codes, scored on the other half. returns (acc_linear, acc_mlp, chance)"""
    rng = np.random.RandomState(300 + seed)
    us = np.unique(utr)
    rng.shuffle(us)
    half_of = {u: i % 2 for i, u in enumerate(us)}
    h = np.array([half_of[u] for u in utr])
    B = int(btr.max()) + 1
    res = {"linear": [], "mlp": []}
    for d in (0, 1):
        a = np.where(h == d)[0]
        t = np.where(h != d)[0]
        a = rng.choice(a, size=min(max_n, len(a)), replace=False)
        t = rng.choice(t, size=min(max_n, len(t)), replace=False)
        pr = AT.run_attackers(codes_tr[a], btr[a], utr[a], codes_tr[t], B, kinds=("linear", "mlp"), seed=seed)
        for k in res:
            res[k].append(AT.bacc(btr[t], pr[k]))
    present = np.unique(btr)
    return float(np.mean(res["linear"])), float(np.mean(res["mlp"])), 1.0 / len(present)


def insample_acc(codes_tr, btr, utr, seed, max_n=15000):
    """memorisation measure: the SAME attacker code trained AND scored on the same cells (in-sample). returns (acc_linear, acc_mlp)"""
    rng = np.random.RandomState(400 + seed)
    a = rng.choice(len(codes_tr), size=min(max_n, len(codes_tr)), replace=False)
    B = int(btr.max()) + 1
    pr = AT.run_attackers(codes_tr[a], btr[a], utr[a], codes_tr[a], B, kinds=("linear", "mlp"), seed=seed)
    return AT.bacc(btr[a], pr["linear"]), AT.bacc(btr[a], pr["mlp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--configs", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--beta_scale", type=float, default=1.0)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--kinds", default=",".join(AT.KINDS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SC(a.dataset)
    cfgs = a.configs.split(",") if a.configs else (IN.DIAG if a.diag else IN.CONFIGS)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else ([0] if a.diag else list(range(len(sc.folds))))
    kinds = a.kinds.split(",")
    P, meta = {}, dict(dataset=a.dataset, seed=a.seed, folds=folds, arms=[], dims={}, checks={}, timings={}, beta_scale=a.beta_scale, kinds=kinds, e2=True,
                       commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    for f in folds:
        t0 = time.time()
        T, Hm = sc.fold_masks(f)
        held = np.where(Hm)[0]
        Z0 = get_z0(sc, f, T)
        Xs = IN.hvg_standardised(sc.X, T)
        tr_idx = np.where(T)[0]
        bcode = {int(b): i for i, b in enumerate(np.unique(sc.b[T]))}
        btr = np.array([bcode[int(b)] for b in sc.b[tr_idx]])
        B = len(bcode)
        utr = sc.unit[tr_idx]
        r2_pca16 = float(Z0[T][:, :16].var(0).sum() / Xs[T].var(0).sum())
        # memorisation floor (E.md 6.10 amendment E2-A2): in-sample and donor-disjoint adversary accuracy on noise latents and on dithered PCA-16 latents of the same size
        rngf = np.random.RandomState(500 + a.seed)
        noise_codes = rngf.randn(len(tr_idx), 16).astype(np.float32)
        sdz = Z0[T][:, :16].std(0)
        dith_codes = (Z0[tr_idx][:, :16] + 0.5 * sdz * rngf.randn(len(tr_idx), 16)).astype(np.float32)
        fl = {}
        for nm, cd in (("noise", noise_codes), ("dith_pca16", dith_codes)):
            ins = insample_acc(cd, btr, utr, a.seed)
            out_ = adapter_check(cd, btr, utr, a.seed)
            fl[nm] = dict(in_linear=ins[0], in_mlp=ins[1], out_linear=out_[0], out_mlp=out_[1], chance_train=out_[2])
        meta.setdefault("floors", {})[str(f)] = fl
        print(f"  memorisation floor fold {f}: " + "; ".join(f"{k} in-sample lin {v['in_linear']:.3f} mlp {v['in_mlp']:.3f}, donor-disjoint lin {v['out_linear']:.3f} mlp {v['out_mlp']:.3f}" for k, v in fl.items()), flush=True)
        print(f"fold {f}: train {len(tr_idx)} cells, B_train {B}, PCA-16 R2 {r2_pca16:.3f}, HVG/standardise {time.time() - t0:.0f}s", flush=True)
        # probe set-up (identical to evaluate.py)
        ntr = np.bincount(sc.fine[T & (sc.fine >= 0)], minlength=len(sc.fine_names))
        nhe = np.bincount(sc.fine[Hm & (sc.fine >= 0)], minlength=len(sc.fine_names))
        types = np.where((ntr >= MIN_TRAIN) & (nhe >= MIN_HELD))[0]
        loc = -np.ones(len(sc.fine_names), np.int64)
        loc[types] = np.arange(len(types))
        trm = T & (sc.fine >= 0) & (loc[np.maximum(sc.fine, 0)] >= 0)
        ptr = np.where(trm)[0]
        ytr = loc[sc.fine[ptr]]
        K = len(types)
        bl = {int(b): i for i, b in enumerate(sc.folds[f])}
        yb = np.array([bl[int(b)] for b in sc.b[held]])
        ub = sc.unit[held]
        hf = halves(sc, held, a.seed)
        P[f"held|{f}"] = held.astype(np.int32)
        P[f"half|{f}"] = hf
        P[f"types|{f}"] = types.astype(np.int16)
        Kb = len(bl)
        for name in cfgs:
            t1 = time.time()
            cfg = IN.parse(name)
            net, log = IN.train_encoder(Xs[tr_idx], btr, B, cfg, a.seed, beta_scale=a.beta_scale)
            kl, r2 = IN.train_stats(net, Xs[tr_idx], btr, cfg, seed=a.seed)
            codes = IN.release(net, Xs, a.seed)
            acc_l, acc_m, ch = adapter_check(codes[tr_idx], btr, utr, a.seed)
            ins_l, ins_m = insample_acc(codes[tr_idx], btr, utr, a.seed)
            floor_in = max(fl["noise"]["in_linear"], fl["noise"]["in_mlp"])
            chk = dict(kl=kl, r2=r2, r2_pca16=r2_pca16, acc_linear=acc_l, acc_mlp=acc_m, chance_train=ch, in_linear=ins_l, in_mlp=ins_m, floor_in_noise=floor_in,
                       adapter_ok=bool(max(acc_l, acc_m) <= ch + 0.05 and max(ins_l, ins_m) <= floor_in + 0.05), collapsed=bool(kl < 1.0 or r2 < 0.5 * r2_pca16),
                       last_adv_ce=(float(np.mean(log["adv_ce"][-10:])) if log["adv_ce"] else None), train_s=round(time.time() - t1, 1))
            meta["checks"][f"{f}|e2_{name}"] = chk
            print(f"  fold {f} {name:16s} KL {kl:7.1f} R2 {r2:.3f} (pca16 {r2_pca16:.3f}) in-sample adv lin {acc_l:.3f} mlp {acc_m:.3f} (chance {ch:.3f}) "
                  f"in-sample lin {ins_l:.3f} mlp {ins_m:.3f} (noise floor {floor_in:.3f}) adapter_ok={chk['adapter_ok']} collapsed={chk['collapsed']} [{time.time() - t1:.0f}s]", flush=True)
            if a.diag:
                continue
            arm = f"e2_{name}"
            pr = AT.run_probes(codes[ptr], ytr, codes[held], K, seed=a.seed)
            for k, v in pr.items():
                P[f"pr|{arm}|{f}|{k}"] = types[v].astype(np.int16)
            res = {k: np.zeros(len(held), np.int8) for k in kinds}
            for d in (0, 1):
                tr_, te_ = hf == d, hf != d
                p2 = AT.run_attackers(codes[held[tr_]], yb[tr_], ub[tr_], codes[held[te_]], Kb, kinds=kinds, seed=a.seed)
                for k in kinds:
                    res[k][te_] = p2[k]
            for k in kinds:
                P[f"att|{arm}|{f}|{k}"] = res[k]
            P[f"ret|{arm}|{f}"] = np.array([retention(Z0, codes, T, held, f)], np.float32)
            if arm not in meta["arms"]:
                meta["arms"].append(arm)
            meta["dims"][arm] = int(codes.shape[1])
            meta["timings"][f"{f}|{arm}"] = round(time.time() - t1, 1)
        print(f"fold {f} done in {time.time() - t0:.0f}s", flush=True)
        if not a.diag:
            np.savez_compressed(out / f"preds_s{a.seed}.npz", **P)
        (out / f"meta_s{a.seed}.json").write_text(json.dumps(meta, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
