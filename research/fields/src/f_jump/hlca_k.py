"""Reduced H3-003: supervised SupCon adapter vs same-label LDA on HLCA at label budgets f in {1,3,10,30,100}% (regime R2: finest labels, K ~ 60).

  python hlca_k.py --folds 0 --out $OUT [--dataset hlca] [--seed 0] [--steps 6000] [--budgets 1,3,10,30,100]

Pre-registration: results_notes/F.md section 18 (commit 71a802f, before any number).  Imports E's harness (e_singlecell: SC, build_arms, attackers, halves)
unchanged and writes E-format per-cell predictions (preds_s{seed}.npz, meta_s{seed}.json) so that E's Data / all_metrics / resamples aggregate it.
Per fold: floors (E's registered list) + ctrl_noise16 + same-label LDA on f% of the labelled training cells + SupCon adapter (r in {0,0.5}) + permuted-label
adapter (probes only), all at d = 16.  Probes (utility) are trained on ALL labelled training-fold cells exactly as in E; f only limits the supervision of the map.
Nothing but per-cell predictions leaves the job: no embeddings, no expression.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# f_jump and e_singlecell both have common.py / arms.py / evaluate.py / aggregate.py: make sure E's are the ones imported here
sys.path = [p for p in sys.path if p and Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "e_singlecell"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arms as A  # noqa: E402  (E)
import attackers as AT  # noqa: E402  (E)
import evaluate as EV  # noqa: E402  (E)
from common import MIN_HELD, MIN_TRAIN, SC  # noqa: E402  (E)

_spec = importlib.util.spec_from_file_location("fj_adapter", HERE / "adapter.py")
FJ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FJ)                      # supcon, effective_rank (F2 / F3 code, unchanged)

DEV = AT.DEV
D_OUT = 16
FLOORS = ["raw50", "pca16", "rp16", "noise_r0.5", "noise_r1", "dith16_r0.5", "dith16_r1", "ldaC_pca16", "ldaF16", "ctrl_noise16"]
LDA_BUDGETS = (1, 3, 10, 30)          # f = 100 is E's ldaF16 (identical function, all labelled training cells)
N_CLS, PER = 64, 8                    # class-balanced batch: 64 classes x up to 8 cells


def hash01(i):
    return int(hashlib.md5(f"h3003|{i}".encode()).hexdigest()[:8], 16) / 2 ** 32


class Net(nn.Module):
    def __init__(self, D, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.f = nn.Sequential(nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.f((x - self.mu) / self.sd), dim=1)


def train_adapter(Xl, yl, seed, steps):
    """Xl (m, D) float tensor on DEV (the labelled cells only), yl int labels.  SupCon (tau 0.1), class-balanced batches, AdamW 1e-3 cosine, wd 1e-4."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    yl = np.asarray(yl)
    order = np.argsort(yl, kind="stable")
    cl, start, cnt = np.unique(yl[order], return_index=True, return_counts=True)
    elig = np.where(cnt >= 2)[0]                                   # a class needs >= 2 labelled cells to give a positive pair
    mu, sd = Xl.mean(0, keepdim=True), Xl.std(0, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    net = Net(Xl.shape[1], D_OUT, mu, sd).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    order_t = torch.as_tensor(order, device=DEV)
    start_t, cnt_t = torch.as_tensor(start, device=DEV), torch.as_tensor(cnt, device=DEV)
    elig_t = torch.as_tensor(elig, device=DEV)
    nb = min(N_CLS, len(elig))
    losses, bsz = [], []
    net.train()
    for _ in range(steps):
        pick = elig_t[torch.randperm(len(elig), device=DEV, generator=g)[:nb]]
        n = cnt_t[pick]
        r = torch.minimum((torch.rand(nb, PER, device=DEV, generator=g) * n[:, None]).long(), (n - 1)[:, None])
        gi, _ = (start_t[pick][:, None] + r).sort(dim=1)               # distinct cells only (duplicates would be fake positives)
        keep = torch.ones_like(gi, dtype=torch.bool)
        keep[:, 1:] = gi[:, 1:] != gi[:, :-1]
        cells = order_t[gi[keep]]
        y = pick[:, None].expand_as(gi)[keep]
        loss = FJ.supcon(net(Xl[cells]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.detach())
        bsz.append(int(cells.shape[0]))
    losses = torch.stack(losses).cpu().numpy()
    net.eval()
    return net, dict(loss_first20=float(losses[:20].mean()), loss_last200=float(losses[-200:].mean()), nan=bool(np.isnan(losses).any()), steps=steps,
                     n_classes_sampled=int(len(elig)), n_classes_labelled=int(len(cl)), batch_mean=float(np.mean(bsz)),
                     loss_curve=[float(losses[i:i + 500].mean()) for i in range(0, steps, 500)])


@torch.no_grad()
def embed_all(net, Z, bs=65536):
    return torch.cat([net(Z[i:i + bs]) for i in range(0, Z.shape[0], bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--folds", default="")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--budgets", default="1,3,10,30,100")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SC(a.dataset)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(len(sc.folds)))
    budgets = [int(x) for x in a.budgets.split(",")]
    kinds = list(AT.KINDS)
    P = {}
    meta = dict(dataset=a.dataset, seed=a.seed, folds=folds, arms=[], timings={}, kinds=kinds, budgets=budgets, steps=a.steps, adapters={}, dims={},
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                prereg="results_notes/F.md section 18, commit 71a802f")
    for f in folds:
        t0 = time.time()
        T, Hm = sc.fold_masks(f)
        held = np.where(Hm)[0]
        Z = EV.get_z0(sc, f, T)
        arms = A.build_arms(Z, T, sc, a.seed, sc.folds[f], scvi=None, only=FLOORS)
        ntr = np.bincount(sc.fine[T & (sc.fine >= 0)], minlength=len(sc.fine_names))
        nhe = np.bincount(sc.fine[Hm & (sc.fine >= 0)], minlength=len(sc.fine_names))
        types = np.where((ntr >= MIN_TRAIN) & (nhe >= MIN_HELD))[0]
        loc = -np.ones(len(sc.fine_names), np.int64)
        loc[types] = np.arange(len(types))
        trm = T & (loc[np.maximum(sc.fine, 0)] >= 0) & (sc.fine >= 0)
        tr_idx = np.where(trm)[0]
        ytr = loc[sc.fine[tr_idx]]
        K = len(types)
        bl = {int(b): i for i, b in enumerate(sc.folds[f])}
        yb = np.array([bl[int(b)] for b in sc.b[held]])
        ub = sc.unit[held]
        hf = EV.halves(sc, held, a.seed)
        P[f"held|{f}"] = held.astype(np.int32)
        P[f"half|{f}"] = hf
        P[f"types|{f}"] = types.astype(np.int16)
        Kb = len(bl)

        def do_att(name, X):
            res = {k: np.zeros(len(held), np.int8) for k in kinds}
            for d in (0, 1):
                tr_, te_ = hf == d, hf != d
                pr = AT.run_attackers(X[held[tr_]], yb[tr_], ub[tr_], X[held[te_]], Kb, kinds=kinds, seed=a.seed)
                for k in kinds:
                    res[k][te_] = pr[k]
            for k in kinds:
                P[f"att|{name}|{f}|{k}"] = res[k]

        def do_probe(name, X):
            pr = AT.run_probes(X[tr_idx], ytr, X[held], K, seed=a.seed)
            for k, v in pr.items():
                P[f"pr|{name}|{f}|{k}"] = types[v].astype(np.int16)

        def run_arm(name, X, attackers=True):
            t1 = time.time()
            X = np.ascontiguousarray(X, dtype=np.float32)
            do_probe(name, X)
            if attackers:
                do_att(name, X)
            meta["timings"][f"{f}|{name}"] = round(time.time() - t1, 1)
            meta["dims"][name] = int(X.shape[1])
            if name not in meta["arms"]:
                meta["arms"].append(name)
            print(f"  fold {f} arm {name:16s} d={X.shape[1]:3d} {time.time() - t1:5.1f}s", flush=True)

        def save():
            np.savez_compressed(out / f"preds_s{a.seed}.npz", **P)
            (out / f"meta_s{a.seed}.json").write_text(json.dumps(meta, indent=1))

        # 1. controls and the registered inductive floors (E's own construction)
        for name in ["ctrl_noise16"] + [x for x in FLOORS if x != "ctrl_noise16"]:
            run_arm(name, arms[name])
        save()
        # 2. nested labelled sets: md5 of the cell index, labelled training-fold cells with a supported finest label
        pool = np.where(T & (sc.fine >= 0))[0]
        u = np.array([hash01(i) for i in pool])
        sets = {b: pool[u < b / 100.0] for b in budgets}
        meta.setdefault("n_labelled", {})[str(f)] = {str(b): int(len(s)) for b, s in sets.items()}
        # 3. same-label LDA (linear comparator), E's lda_pad with at most 16 discriminant directions
        for b in LDA_BUDGETS:
            if b not in budgets:
                continue
            idx = sets[b]
            Tm = np.zeros(len(sc.fine), bool)
            Tm[idx] = True
            yl = np.where(Tm, sc.fine, -1)
            run_arm(f"ldaM_f{b}", A.lda_pad(Z, Tm, yl, None, max_dims=16))
            save()
        # 4. adapters (r = 0, 0.5) and the permuted-label memorisation floor (probes only)
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=DEV)
        zero_held = torch.as_tensor(held, device=DEV)
        for b in budgets:
            idx = sets[b]
            idx_t = torch.as_tensor(idx, device=DEV)
            ylab = sc.fine[idx]
            sd_ = a.seed * 7919 + f * 101 + b
            t1 = time.time()
            net, info = train_adapter(Zt[idx_t], ylab, sd_, a.steps)
            E0 = embed_all(net, Zt)
            info["eff_rank_held"] = FJ.effective_rank(E0[zero_held])
            info["n_labelled"] = int(len(idx))
            sdv = E0[idx_t].std(0, keepdim=True)
            gen = torch.Generator(device=DEV).manual_seed(sd_ + 1)
            noise = torch.randn(E0.shape, generator=gen, device=DEV)
            for r in (0.0, 0.5):
                run_arm(f"adp_f{b}_r{r:g}", (E0 + r * sdv * noise).cpu().numpy())
            info["train_s"] = round(time.time() - t1, 1)
            meta["adapters"][f"{f}|adp_f{b}"] = info
            netp, infop = train_adapter(Zt[idx_t], np.random.RandomState(sd_ + 50000).permutation(ylab), sd_ + 50000, a.steps)
            Ep = embed_all(netp, Zt)
            infop["eff_rank_held"] = FJ.effective_rank(Ep[zero_held])
            meta["adapters"][f"{f}|perm_f{b}"] = infop
            run_arm(f"perm_f{b}_r0", Ep.cpu().numpy(), attackers=False)
            print(f"  fold {f} budget {b}%: n_lab {len(idx)} loss {info['loss_first20']:.3f}->{info['loss_last200']:.3f}  "
                  f"perm {infop['loss_first20']:.3f}->{infop['loss_last200']:.3f}  rank {info['eff_rank_held']:.1f}", flush=True)
            save()
        print(f"fold {f} done: {time.time() - t0:.0f}s, probe types {K}, held {len(held)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
