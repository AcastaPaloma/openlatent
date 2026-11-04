#!/usr/bin/env python
"""One job = one (encoder, dataset, seed): evaluate every configuration of the grid.  Results JSONL + per-config predictions
go to $OUT (small; never embeddings).  Resumable: configs already in the JSONL are skipped.

   python research/fields/src/a_pathorob/run_sweep.py --encoder phikonv2_clsmean --dataset tcga --seed 0 --grid main
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a_pathorob import common as C, splits as S, adapters as AD, suite as SU


def cfg_id(method, hp):
    return method + ("__" + "_".join(f"{k}{v}" for k, v in sorted(hp.items())) if hp else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="main", help="main | controls | all | comma list of method names")
    ap.add_argument("--smoke", action="store_true", help="small epochs / subset of configs for a quick check")
    a = ap.parse_args()
    out = Path(os.environ.get("OUT", "./out"))
    (out / "preds").mkdir(parents=True, exist_ok=True)
    tag = f"{a.encoder}__{a.dataset}__seed{a.seed}"
    jl = out / f"{tag}.jsonl"
    done = set()
    if jl.exists():
        done = {json.loads(l)["cfg"] for l in jl.read_text().splitlines() if l.strip()}
    P = S.load_pdata(a.encoder, a.dataset)
    sp = S.make_splits(P, a.seed)
    info = dict(encoder=a.encoder, dataset=a.dataset, seed=a.seed, commit=C.git_commit(), device=str(AD.DEV),
                n_patches=int(len(P.y)), dim=int(P.X.shape[1]), classes=P.classes, centres=P.centres,
                units={k: int(len(set(P.u[v].tolist()))) for k, v in sp.items()},
                patches={k: int(len(v)) for k, v in sp.items()}, torch=torch.__version__,
                pathorob_commit=C.git_commit(C.PATHOROB_REPO))
    (out / f"{tag}.split.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(out / f"{tag}.splitarrays.npz", **{f"idx_{k}": v for k, v in sp.items()}, y=P.y, c=P.c, u=P.u)
    print(json.dumps(info), flush=True)
    grid = []
    if a.grid in ("main", "all"):
        grid += AD.GRID
    if a.grid in ("controls", "all"):
        grid += AD.CONTROLS
    if a.grid not in ("main", "all", "controls"):
        want = set(a.grid.split(","))
        grid = [g for g in AD.GRID + AD.CONTROLS if g[0] in want]
    if a.smoke:      # first config of every method
        seen, g2 = set(), []
        for g in grid:
            if g[0] not in seen:
                seen.add(g[0]); g2.append(g)
        grid = g2
    for method, hp in grid:
        cid = cfg_id(method, hp)
        if cid in done:
            continue
        t0 = time.time()
        try:
            rep, extra = AD.build(method, hp, P, sp, a.seed)
            tfit = time.time() - t0
            m, preds = SU.evaluate(rep, P, sp, a.seed, secret_shuffle=(method == "ctrl_shuffle"))
            rec = dict(cfg=cid, method=method, hp=hp, seed=a.seed, encoder=a.encoder, dataset=a.dataset, metrics=m, extra=extra,
                       seconds_fit=round(tfit, 1))
            np.savez_compressed(out / "preds" / f"{tag}__{cid}.npz", **preds)
            with open(jl, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            lc = max(m[f"centre/{k}/C"] for k in SU.KINDS_CENTRE)
            print(f"[{tag}] {cid:48s} fit {tfit:5.1f}s eval {m['seconds_eval']:5.1f}s  label_lin={m['label/linear/C']:.3f} "
                  f"worst_centre={lc:.3f} (chance {m['chance/centre']:.2f})", flush=True)
        except Exception:
            print(f"[{tag}] {cid} FAILED\n{traceback.format_exc()}", flush=True)
            with open(out / f"{tag}.failures.txt", "a") as fh:
                fh.write(f"{cid}\n{traceback.format_exc()}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
