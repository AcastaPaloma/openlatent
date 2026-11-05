"""Focus D3 runner: PII-Masking-300K as the secret, GTR-base released vectors, known-item retrieval (non-PII excerpt queries) as utility.

LICENCE / HANDLING (docs/data_licences/WHITELIST.md): academic, non-commercial, AI4Privacy-acknowledged.  Reads only the scratch cache; writes
  $OUT/summary.json   aggregates only (means per attacker/tier/arm, no per-document values)  -> uploaded (hf_allow json/log/md)
  $OUT/records.jsonl, $OUT/units.npz, $OUT/units_index.npz   per-document arrays and cluster ids: scratch only, excluded by hf_allow
Prints only arm names and aggregate numbers.  <= 8 threads.

  python pii_run.py --name d_pii_s1_s0 --grid p_ctrl --seeds 0 --out $OUT
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.sparse as sp
import torch

torch.set_num_threads(8)
from arms import Arm, make_arm
from attackers import MLP, run_suite
from common import rprec_per_doc
from grids import grid

CACHE3 = Path(os.environ.get("D3_CACHE", "~/work/cache/d3_pii")).expanduser()
TIERS = ("val", "name", "place")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_C = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, 12)))])  # _C[k] = sum_{pos<=k} 1/log2(pos+1)


class Ctx3:
    def __init__(self, embedder="gtr_base"):
        M = np.load(CACHE3 / "meta.npz", allow_pickle=True)
        self.split, self.keep, self.dval, self.cluster = M["split"], M["keep"], M["dval"], M["cluster"]
        d = CACHE3 / embedder
        self.X_nf, self.X_or = np.load(d / "docs.npy"), np.load(d / "docs_oracle.npy")
        self.EX0, self.EX1 = np.load(d / "ex0.npy"), np.load(d / "ex1.npy")
        self.B = sp.load_npz(CACHE3 / "bags.npz").tocsr()
        self.Yt = sp.load_npz(CACHE3 / "types.npz").tocsr()
        self.feat = {"b1": M["b1"], "b1g": M["b1g"]}
        s = self.split
        self.idx = {"D": np.where(s == 0)[0], "A": np.where(s == 1)[0], "V": np.where((s == 2) & self.keep)[0], "C": np.where((s == 3) & self.keep)[0]}
        self.dfit = np.where((s == 0) & ~self.dval)[0]
        self.dvalx = np.where(self.dval)[0]
        tiers = M["tiers"]
        self.tier_cols = {"val": np.arange(self.B.shape[1]), "name": np.where(tiers == "name")[0], "place": np.where(tiers == "place")[0]}
        self.tier_cols["typ"] = np.arange(self.Yt.shape[1])
        # attributes used by arms.py / train.py (defender side only: D minus Dval)
        self.nc_idx, self.trq_idx, self.X_q = self.dfit, self.dfit, self.EX1
        self.T_train = np.eye(len(self.dfit), dtype=np.float32)
        self._ref_cache = {}

    def defender_docs(self):
        return self.X_nf[self.dfit]

    def defender_bags(self):
        return self.B[self.dfit]


class FeatArm(Arm):
    """Non-embedding reference feature (LSA of TF-IDF of the masked text), attacked by the same suite; no retrieval utility."""
    def __init__(self, name, Z):
        self.name, self.Z = name, Z


def known_item(Zq, Zd):
    """nDCG@10 per query, single relevant item (the document i for query i), exact expectation under random tie-breaking."""
    n = len(Zq)
    out = np.zeros(n, np.float32)
    for s in range(0, n, 1000):
        S = Zq[s: s + 1000] @ Zd.T
        own = S[np.arange(len(S)), np.arange(s, s + len(S))]
        g = (S > own[:, None]).sum(1)
        t = (S == own[:, None]).sum(1)
        out[s: s + len(S)] = (_C[np.minimum(g + t, 10)] - _C[np.minimum(g, 10)]) / t
    return out


def mean_rp(scores, Y, cols):
    return float(np.nanmean(rprec_per_doc(scores, Y, cols)))


def ref_levels(ctx, seed):
    """Engagement yardsticks on Dval: frequency prior and a fresh MLP trained on RAW defender vectors, for the pooled and the name tier."""
    if seed not in ctx._ref_cache:
        Bd, Bv = ctx.B[ctx.dfit], ctx.B[ctx.dvalx]
        prior = np.tile(np.asarray(Bd.mean(0)).ravel()[None, :], (len(ctx.dvalx), 1))
        m = MLP().fit(torch.as_tensor(ctx.X_nf[ctx.dfit]), torch.as_tensor(Bd.toarray()), seed=seed)
        sc = m.score(torch.as_tensor(ctx.X_nf[ctx.dvalx]))
        ctx._ref_cache[seed] = {t: dict(prior=mean_rp(prior, Bv, ctx.tier_cols[t]), mlp_raw=mean_rp(sc, Bv, ctx.tier_cols[t])) for t in ("val", "name")}
    return ctx._ref_cache[seed]


def engagement(ctx, arm, seed):
    """E1: in-loop head scored on held-out Dval (pooled and name tier); E3: memorisation floor on Gaussian latents of the same shape (in-sample vs held-out)."""
    ref = ref_levels(ctx, seed)
    Bd, Bv = ctx.B[ctx.dfit], ctx.B[ctx.dvalx]
    sub = np.random.default_rng(seed).choice(len(ctx.dfit), 1500, replace=False)
    with torch.no_grad():
        head_v = arm.g(torch.as_tensor(arm.docs(ctx.X_nf[ctx.dvalx], None)).to(DEV)).cpu().numpy()
        head_d = arm.g(torch.as_tensor(arm.docs(ctx.X_nf[ctx.dfit][sub], None)).to(DEV)).cpu().numpy()
    out = dict(n_dval=int(len(ctx.dvalx)), adv_tier=getattr(arm, "adv_tier", None))
    for t in ("val", "name"):
        cols = ctx.tier_cols[t]
        if arm.adv_tier:  # head outputs only the adversary tier's columns; map tier t into that block (only meaningful for t == adv_tier)
            if t != arm.adv_tier:
                continue
            hv, hd, Bvt, Bdt, cc = head_v, head_d, Bv[:, ctx.tier_cols[t]], Bd[sub][:, ctx.tier_cols[t]], np.arange(len(cols))
        else:
            hv, hd, Bvt, Bdt, cc = head_v, head_d, Bv, Bd[sub], cols
        r_head, r_in = mean_rp(hv, Bvt, cc), mean_rp(hd, Bdt, cc)
        L = (r_head - ref[t]["prior"]) / max(ref[t]["mlp_raw"] - ref[t]["prior"], 1e-6)
        out[t] = dict(R_head_heldout=r_head, R_head_insample=r_in, R_prior_dval=ref[t]["prior"], R_mlp_raw_dval=ref[t]["mlp_raw"], L_head=L)
    g = torch.Generator().manual_seed(seed)
    Zn = torch.randn(len(ctx.dfit), arm.d, generator=g)
    mem = MLP().fit(Zn, torch.as_tensor(Bd.toarray()), seed=seed)
    out["memfloor"] = {t: dict(insample=mean_rp(mem.score(Zn[sub]), Bd[sub], ctx.tier_cols[t]),
                               heldout=mean_rp(mem.score(torch.randn(len(ctx.dvalx), arm.d, generator=g)), Bv, ctx.tier_cols[t])) for t in ("val", "name")}
    return out


def attack(ctx, Zpool, Ypool_val, Ypool_typ, evals, seed, grp):
    out, units = {}, {}
    for kind, Y in (("val", Ypool_val), ("typ", Ypool_typ)):
        sc = run_suite(Zpool, torch.as_tensor(Y), evals, seed=seed, grp=grp)
        for a, d in sc.items():
            for split in ("V", "C"):
                idx = ctx.idx[split]
                tiers = TIERS if kind == "val" else ("typ",)
                for tier in tiers:
                    Yd = ctx.B[idx] if kind == "val" else ctx.Yt[idx]
                    rp = rprec_per_doc(d[split], Yd, ctx.tier_cols[tier])
                    out.setdefault(a, {}).setdefault(split, {})[tier] = float(np.nanmean(rp))
                    units[f"rp_{split}_{tier}_{a}"] = rp.astype(np.float32)
    return out, units


def evaluate(ctx, spec, seed, raw):
    t0 = time.time()
    fam = spec["family"]
    if fam == "recsplit":
        return recsplit_demo(ctx, seed)
    if fam == "feat":
        arm = FeatArm(spec["feat"], ctx.feat[spec["feat"]])
        Z = arm.Z
    else:
        arm = make_arm(spec).fit(ctx, seed)
    t_fit = time.time() - t0
    rng = np.random.default_rng(5000 + seed)
    Xd = ctx.X_or if spec.get("mask") else ctx.X_nf
    out, units = {}, {}
    if fam != "feat":
        Z = arm.docs(Xd, rng)
        Zq = arm.queries(ctx.EX0)
        for g in ("V", "C"):
            ix = ctx.idx[g]
            nd = known_item(Zq[ix], Z[ix])
            units[f"nd_{g}"] = nd
            out[f"ndcg_{g}"] = float(nd.mean())
    copies = 4 if getattr(arm, "stochastic", False) else 1
    A = ctx.idx["A"]
    perm = np.random.default_rng(99 + seed).permutation(len(A)) if spec.get("shuffle") else np.arange(len(A))
    Zs = []
    for c in range(copies):
        r = np.random.default_rng(7000 + 100 * seed + c)
        Zs.append(Z[A] if fam == "feat" else arm.docs(Xd[A], r))
    Zpool = torch.as_tensor(np.concatenate(Zs))
    Yv = np.tile(ctx.B[A][perm].toarray(), (copies, 1))
    Yt = np.tile(ctx.Yt[A][perm].toarray(), (copies, 1))
    grp = np.tile(np.arange(len(A)), copies)
    evals = {"V": torch.as_tensor(Z[ctx.idx["V"]]), "C": torch.as_tensor(Z[ctx.idx["C"]])}
    t1 = time.time()
    att, u2 = attack(ctx, Zpool, Yv, Yt, evals, seed, grp)
    units.update(u2)
    name = (arm.name if fam != "feat" else f"feat_{spec['feat']}") + (f"|mask_{spec['mask']}" if spec.get("mask") else "") + ("|shuffle" if spec.get("shuffle") else "")
    rec = dict(arm=name, family=fam, spec=spec, seed=seed, **out, attackers=att, t_fit=round(t_fit, 1), t_attack=round(time.time() - t1, 1),
               n_pool=int(Zpool.shape[0]), dim=int(Z.shape[1]), n_V=int(len(ctx.idx["V"])), n_C=int(len(ctx.idx["C"])))
    if hasattr(arm, "log"):
        rec.update(train_log=arm.log, collapsed=arm.collapsed, final_ret=arm.final_ret)
    if fam == "adv" and getattr(arm, "g", None) is not None:
        rec["engagement"] = engagement(ctx, arm, seed)
    return rec, units


def recsplit_demo(ctx, seed):
    """Template-leakage demonstration: raw vectors, attackers trained/evaluated under a RECORD-level random split vs the cluster split."""
    rng = np.random.default_rng(321 + seed)
    N = len(ctx.split)
    nA, nC = len(ctx.idx["A"]), len(ctx.idx["C"])
    perm = rng.permutation(N)
    A2, C2 = perm[:nA], perm[nA: nA + nC]
    sc = run_suite(torch.as_tensor(ctx.X_nf[A2]), torch.as_tensor(ctx.B[A2].toarray()), {"C": torch.as_tensor(ctx.X_nf[C2])}, seed=seed)
    att = {a: {"C": {"val": mean_rp(d["C"], ctx.B[C2], ctx.tier_cols["val"])}} for a, d in sc.items()}
    return dict(arm="recsplit", family="recsplit", spec={"family": "recsplit"}, seed=seed, attackers=att, n_pool=int(nA), n_C=int(nC)), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--embedder", default="gtr_base")
    ap.add_argument("--grid", default="")
    ap.add_argument("--specs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(a.specs) if a.specs else grid(a.grid)
    seeds = [int(s) for s in a.seeds.split(",")]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ctx = Ctx3(a.embedder)
    raw = {g: float(known_item(ctx.EX0[ctx.idx[g]], ctx.X_nf[ctx.idx[g]]).mean()) for g in ("V", "C")}
    meta = dict(name=a.name, embedder=a.embedder, commit=commit, grid=a.grid, seeds=seeds, ndcg_raw=raw,
                sizes={k: int(len(v)) for k, v in ctx.idx.items()}, n_clusters={g: int(len(set(ctx.cluster[ctx.idx[g]]))) for g in ("V", "C")},
                n_dfit=int(len(ctx.dfit)), n_dval=int(len(ctx.dvalx)), vocab=int(ctx.B.shape[1]), device=str(DEV),
                note="aggregates only; per-document arrays stay in scratch (units.npz, not uploaded)")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    np.savez(out / "units_index.npz", cluster_V=ctx.cluster[ctx.idx["V"]], cluster_C=ctx.cluster[ctx.idx["C"]])
    units_all, recs = {}, []
    with open(out / "records.jsonl", "a") as fh:
        for seed in seeds:
            for spec in specs:
                try:
                    rec, units = evaluate(ctx, spec, seed, raw)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    rec, units = dict(arm=str(spec.get("family")), seed=seed, error=type(e).__name__), {}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    continue
                rec["commit"], rec["ndcg_raw"] = commit, raw
                rid = f"{rec['arm']}|s{seed}"
                rec["rid"] = rid
                recs.append(rec)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                for k, v in units.items():
                    units_all[f"{rid}|{k}"] = v
                np.savez_compressed(out / "units.npz", **units_all)
                (out / "summary.json").write_text(json.dumps(dict(meta=meta, records=recs), indent=1))
                aw = max(rec["attackers"][x]["C"]["val"] for x in rec["attackers"]) if "C" in next(iter(rec["attackers"].values())) else float("nan")
                print(f"[{rid}] dim={rec.get('dim')} ndcg V {rec.get('ndcg_V', float('nan')):.4f} C {rec.get('ndcg_C', float('nan')):.4f} "
                      f"(raw {raw['V']:.4f}/{raw['C']:.4f}) val-worst C {aw:.4f}" + (f" collapsed={rec.get('collapsed')}" if "collapsed" in rec else "")
                      + (" L_head(" + rec['engagement']['adv_tier'] + ")=" + format(rec['engagement'][rec['engagement']['adv_tier']]['L_head'], '.2f') if "engagement" in rec and rec['engagement']['adv_tier'] else (f" L_head(val)={rec['engagement']['val']['L_head']:.2f} L_head(name)={rec['engagement']['name']['L_head']:.2f}" if "engagement" in rec else "")), flush=True)


if __name__ == "__main__":
    main()
