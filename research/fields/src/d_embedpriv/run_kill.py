"""Focus D job runner: evaluate a list of arms x seeds and write records + per-unit arrays.

  python run_kill.py --name d_floors_gtr_s0 --grid floors --seeds 0 [--embedder gtr_base] [--out $OUT]
  python run_kill.py --name x --specs '[{"family":"pca","d":64}]' --seeds 0

Per (arm, seed) it: fits/trains the arm on the defender side only, releases all NFCorpus doc vectors (+ queries), computes nDCG@10 on dev/test queries
(per query), retrains the attacker suite on the released vectors of the attacker pool (NFCorpus A + SciFact EA) and scores V and C docs
(R-precision per document per tier per attacker).  Output: records.jsonl (summary means) and units.npz (per-unit arrays for paired bootstrap).
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from arms import make_arm
from attackers import run_suite
from common import CACHE, World, ndcg10_per_query, rprec_per_doc
from grids import grid


class Ctx:
    def __init__(self, embedder, scheme="words"):
        self.W = World(scheme)
        W = self.W
        c = CACHE / embedder
        self.dir = c
        self.X_nf = np.load(c / "nf_docs.npy")
        self.X_q = np.load(c / "nf_queries.npy")
        self.X_sf = np.load(c / "sf_docs.npy")
        self._mask = {}
        self.idx = {s: W.idx(s, "nf") for s in "DAVC"}
        self.sidx = {s: W.idx(s, "sf") for s in ("ED", "EA")}
        # retrieval-loss data (outside C)
        self.nc_idx = np.where(W.nf_split != "C")[0]
        pos = {W.nf_ids[i]: j for j, i in enumerate(self.nc_idx)}
        rows, T = [], []
        qtr = W.qrels["train"]
        for i, q in enumerate(W.q_ids):
            if q in qtr:
                t = np.zeros(len(self.nc_idx), np.float32)
                for d, s in qtr[q].items():
                    if d in pos and s > 0:
                        t[pos[d]] = s
                if t.sum() > 0:
                    rows.append(i)
                    T.append(t / t.sum())
        self.trq_idx = np.array(rows)
        self.T_train = np.stack(T)

    def defender_docs(self):
        return np.concatenate([self.X_nf[self.idx["D"]], self.X_sf[self.sidx["ED"]]])

    def defender_bags(self):
        import scipy.sparse as sp
        return sp.vstack([self.W.nf_bags[self.idx["D"]], self.W.sf_bags[self.sidx["ED"]]]).tocsr()

    def docs_variant(self, mask):
        if not mask:
            return self.X_nf, self.X_sf
        if mask not in self._mask:
            self._mask[mask] = (np.load(self.dir / f"nf_docs_mask_{mask}.npy"), np.load(self.dir / f"sf_docs_mask_{mask}.npy"))
        return self._mask[mask]


def evaluate(ctx, spec, seed, boots=None):
    W = ctx.W
    t0 = time.time()
    arm = make_arm(spec).fit(ctx, seed)
    t_fit = time.time() - t0
    rng = np.random.default_rng(5000 + seed)
    Xnf, Xsf = ctx.docs_variant(spec.get("mask"))
    Znf = arm.docs(Xnf, rng)
    Zq = arm.queries(ctx.X_q)
    out, units = {}, {}
    for key in ("dev", "test"):
        qs, nd = ndcg10_per_query(Zq, Znf, W.q_ids, W.nf_ids, W.qrels[key])
        out[f"ndcg_{key}"] = float(nd.mean())
        units[f"ndcg_{key}"] = nd.astype(np.float32)
    # attacker pool: NFCorpus A + SciFact EA, released through the same defence (fresh noise draws; 4 copies if stochastic)
    copies = 4 if arm.stochastic else 1
    Zs, Ys = [], []
    for c in range(copies):
        r = np.random.default_rng(7000 + 100 * seed + c)
        Zs.append(np.concatenate([arm.docs(Xnf[ctx.idx["A"]], r), arm.docs(Xsf[ctx.sidx["EA"]], r)]))
        Ys.append(sp.vstack([W.nf_bags[ctx.idx["A"]], W.sf_bags[ctx.sidx["EA"]]]).tocsr())
    Zpool = torch.as_tensor(np.concatenate(Zs))
    Ypool = sp.vstack(Ys).tocsr()
    if spec.get("shuffle"):  # control: break the (vector, bag) pairing inside the attacker's training data
        perm = np.random.default_rng(99 + seed).permutation(Ypool.shape[0])
        Ypool = Ypool[perm]
    Yp = torch.as_tensor(Ypool.toarray())
    grp = np.tile(np.arange(Zs[0].shape[0]), copies)
    evals = {"V": torch.as_tensor(Znf[ctx.idx["V"]]), "C": torch.as_tensor(Znf[ctx.idx["C"]])}
    t1 = time.time()
    sc = run_suite(Zpool, Yp, evals, seed=seed, grp=grp)
    t_att = time.time() - t1
    Yv, Yc = W.nf_bags[ctx.idx["V"]], W.nf_bags[ctx.idx["C"]]
    att = {}
    for a, d in sc.items():
        att[a] = {}
        for split, Y in (("V", Yv), ("C", Yc)):
            att[a][split] = {}
            for tier in W.tiers:
                rp = rprec_per_doc(d[split], Y, W.tier_cols[tier])
                att[a][split][tier] = float(np.nanmean(rp))
                units[f"rp_{split}_{tier}_{a}"] = rp.astype(np.float32)
    full = arm.name + (f"|mask_{spec['mask']}" if spec.get("mask") else "") + ("|shuffle" if spec.get("shuffle") else "")
    rec = dict(arm=full, family=spec["family"], spec=spec, seed=seed, **out, attackers=att,
               t_fit=round(t_fit, 1), t_attack=round(t_att, 1), n_pool=int(Zpool.shape[0]), dim=int(Znf.shape[1]))
    if hasattr(arm, "log"):
        rec["train_log"] = arm.log
        rec["collapsed"] = arm.collapsed
        rec["final_ret"] = arm.final_ret
    if hasattr(arm, "rank"):
        rec["leace_rank"] = arm.rank
    return rec, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--embedder", default="gtr_base")
    ap.add_argument("--grid", default="")
    ap.add_argument("--specs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--scheme", default="words")
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(a.specs) if a.specs else grid(a.grid)
    seeds = [int(s) for s in a.seeds.split(",")]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ctx = Ctx(a.embedder, a.scheme)
    # raw reference utility for retention (cheap, no attackers)
    raw = {}
    for key in ("dev", "test"):
        _, nd = ndcg10_per_query(ctx.X_q, ctx.X_nf, ctx.W.q_ids, ctx.W.nf_ids, ctx.W.qrels[key])
        raw[key] = float(nd.mean())
    meta = dict(name=a.name, embedder=a.embedder, scheme=a.scheme, tiers=list(ctx.W.tiers), commit=commit, grid=a.grid, seeds=seeds, n_specs=len(specs),
                ndcg_raw=raw, n_vocab=len(ctx.W.vocab), split_sizes={s: int(len(v)) for s, v in ctx.idx.items()},
                device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    units_all = {}
    with open(out / "records.jsonl", "a") as fh:
        for seed in seeds:
            for spec in specs:
                t = time.time()
                try:
                    rec, units = evaluate(ctx, spec, seed)
                except Exception as e:  # keep going; the failure is recorded
                    import traceback
                    traceback.print_exc()
                    fh.write(json.dumps(dict(arm=str(spec), seed=seed, error=str(e)[:300])) + "\n")
                    fh.flush()
                    continue
                rec["commit"] = commit
                rec["ndcg_raw"] = raw
                rid = f"{rec['arm']}|s{seed}"
                rec["rid"] = rid
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                for k, v in units.items():
                    units_all[f"{rid}|{k}"] = v
                np.savez_compressed(out / "units.npz", **units_all)
                pt = ctx.W.tiers[0]
                aw = {s: max(rec["attackers"][x][s][pt] for x in rec["attackers"]) for s in ("V", "C")}
                print(f"[{rid}] dim={rec['dim']} ndcg dev {rec['ndcg_dev']:.4f} test {rec['ndcg_test']:.4f} (raw {raw['dev']:.4f}/{raw['test']:.4f}) "
                      f"{pt} worst V {aw['V']:.3f} C {aw['C']:.3f} | fit {rec['t_fit']}s att {rec['t_attack']}s", flush=True)
    # qid order of the per-query arrays, for the aggregator
    qd = [q for q in ctx.W.qrels["dev"] if q in set(ctx.W.q_ids)]
    qt = [q for q in ctx.W.qrels["test"] if q in set(ctx.W.q_ids)]
    (out / "units_index.json").write_text(json.dumps(dict(dev_q=qd, test_q=qt, V_docs=[ctx.W.nf_ids[i] for i in ctx.idx["V"]],
                                                           C_docs=[ctx.W.nf_ids[i] for i in ctx.idx["C"]])))


if __name__ == "__main__":
    main()
