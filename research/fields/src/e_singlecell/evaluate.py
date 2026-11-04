"""SC-0 evaluation job: floors, references and controls on one dataset and one seed.  Writes per-cell predictions only (no embeddings, no expression).

  python evaluate.py --dataset immune --seed 0 --out $OUT [--arms a,b,c] [--folds 0,1,2]

Per fold: encoder 0 (cached, shared across seeds) -> arms -> {probes trained on training batches, scored on held-out batches;
batch attackers cross-fit on the two halves of the held-out cells; raw-neighbourhood retention}.
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
import arms as A  # noqa: E402
import attackers as AT  # noqa: E402
from common import CACHE, MIN_HELD, MIN_TRAIN, SC  # noqa: E402


def get_z0(sc, f, T):
    p = CACHE / sc.ds / f"z0_f{f}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    Z, _ = A.encoder0(sc.X, T)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp.npy")
    np.save(tmp, Z)
    os.replace(tmp, p)
    print(f"  encoder0 fold {f}: {time.time() - t0:.0f}s", flush=True)
    return Z


def halves(sc, held, seed):
    """0/1 half label per held-out cell.  Donor-disjoint (donors alternate within each held-out batch) when donors exist, else random cells within batch."""
    h = np.zeros(len(held), np.int8)
    rng = np.random.RandomState(900 + seed)
    for bb in np.unique(sc.b[held]):
        m = np.where(sc.b[held] == bb)[0]
        if sc.has_donor:
            us = np.unique(sc.unit[held][m])
            us = us[rng.permutation(len(us))]
            half_of = {u: i % 2 for i, u in enumerate(us)}
            h[m] = [half_of[u] for u in sc.unit[held][m]]
        else:
            p = rng.permutation(len(m))
            h[m[p[: len(m) // 2]]] = 0
            h[m[p[len(m) // 2:]]] = 1
    return h


def retention(Zraw, arm, T, held, f, k_raw=10, k_rep=30):
    import torch
    rng = np.random.RandomState(500 + f)
    tr = np.where(T)[0]
    tr = rng.choice(tr, size=min(30000, len(tr)), replace=False)
    he = rng.choice(held, size=min(3000, len(held)), replace=False)

    def topk(M, k):
        M = torch.as_tensor(M, dtype=torch.float32, device=AT.DEV)
        out = []
        for i in range(0, len(he), 1000):
            out.append(torch.cdist(M[len(tr) + i: len(tr) + i + 1000], M[: len(tr)]).topk(k, largest=False).indices.cpu().numpy())
        return np.concatenate(out)

    mu, sd = arm[T].mean(0), arm[T].std(0)
    sd[sd < 1e-6] = 1.0
    S = np.concatenate([arm[tr], arm[he]])
    S = (S - mu) / sd
    R = np.concatenate([Zraw[tr], Zraw[he]])
    nr, nq = topk(R, k_raw), topk(S, k_rep)
    return float(np.mean([len(set(a) & set(b)) / k_raw for a, b in zip(nr, nq)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--kinds", default=",".join(AT.KINDS))
    ap.add_argument("--no_retention", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SC(a.dataset)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(len(sc.folds)))
    only = a.arms.split(",") if a.arms else None
    kinds = a.kinds.split(",")
    scvi_p = CACHE / sc.ds / "scvi_latent.npy"
    scvi = np.load(scvi_p) if scvi_p.exists() else None
    P, meta = {}, dict(dataset=a.dataset, seed=a.seed, folds=folds, arms=[], timings={}, kinds=kinds,
                       commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                       scvi_available=scvi is not None)
    for f in folds:
        t0 = time.time()
        T, Hm = sc.fold_masks(f)
        held = np.where(Hm)[0]
        Z = get_z0(sc, f, T)
        arms = A.build_arms(Z, T, sc, a.seed, sc.folds[f], scvi=scvi, only=only)
        if only is not None and "raw50" not in only:
            arms["raw50"] = Z
        t_arms = time.time() - t0
        # probe label set of this fold
        ntr = np.bincount(sc.fine[T & (sc.fine >= 0)], minlength=len(sc.fine_names))
        nhe = np.bincount(sc.fine[Hm & (sc.fine >= 0)], minlength=len(sc.fine_names))
        types = np.where((ntr >= MIN_TRAIN) & (nhe >= MIN_HELD))[0]
        loc = -np.ones(len(sc.fine_names), np.int64)
        loc[types] = np.arange(len(types))
        trm = T & (loc[np.maximum(sc.fine, 0)] >= 0) & (sc.fine >= 0)
        tr_idx = np.where(trm)[0]
        ytr = loc[sc.fine[tr_idx]]
        K = len(types)
        # attacker setup
        bl = {int(b): i for i, b in enumerate(sc.folds[f])}
        yb = np.array([bl[int(b)] for b in sc.b[held]])
        ub = sc.unit[held]
        hf = halves(sc, held, a.seed)
        P[f"held|{f}"] = held.astype(np.int32)
        P[f"half|{f}"] = hf
        P[f"types|{f}"] = types.astype(np.int16)
        Kb = len(bl)

        def do_att(name, X, shuffle=False):
            res = {k: np.zeros(len(held), np.int8) for k in kinds}
            for d in (0, 1):
                tr_, te_ = hf == d, hf != d
                pr = AT.run_attackers(X[held[tr_]], yb[tr_], ub[tr_], X[held[te_]], Kb, kinds=kinds, seed=a.seed, shuffle=shuffle)
                for k in kinds:
                    res[k][te_] = pr[k]
            for k in kinds:
                P[f"att|{name}|{f}|{k}"] = res[k]

        def do_probe(name, X, ytr_use):
            pr = AT.run_probes(X[tr_idx], ytr_use, X[held], K, seed=a.seed)
            for k, v in pr.items():
                P[f"pr|{name}|{f}|{k}"] = types[v].astype(np.int16)

        for name, X in arms.items():
            t1 = time.time()
            do_probe(name, X, ytr)
            if not name.startswith("onehot") or True:
                do_att(name, X)
            if not a.no_retention and not name.startswith(("ctrl", "onehot")):
                P[f"ret|{name}|{f}"] = np.array([retention(Z, X, T, held, f)], np.float32)
            meta["timings"][f"{f}|{name}"] = round(time.time() - t1, 1)
            if name not in meta["arms"]:
                meta["arms"].append(name)
            print(f"  fold {f} arm {name:14s} d={X.shape[1]:3d} {time.time() - t1:5.1f}s", flush=True)
        if only is None or "raw50" in (only or []):
            do_att("ctrl_shuffle_batch", arms["raw50"], shuffle=True)
            do_probe("ctrl_shuffle_labels", arms["raw50"], np.random.RandomState(31 + a.seed).permutation(ytr))
            for nm in ("ctrl_shuffle_batch", "ctrl_shuffle_labels"):
                if nm not in meta["arms"]:
                    meta["arms"].append(nm)
        print(f"fold {f} done: arms {t_arms:.0f}s, total {time.time() - t0:.0f}s, probe types {K}, held {len(held)}", flush=True)
        np.savez_compressed(out / f"preds_s{a.seed}.npz", **P)
        (out / f"meta_s{a.seed}.json").write_text(json.dumps(meta, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
