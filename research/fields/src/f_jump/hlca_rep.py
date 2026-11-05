"""Controlled replication of the HLCA supervised-adapter result (results_notes/F.md section 19; pre-registration committed before any number).

  python hlca_rep.py --seed 1 --folds 0 --stage full --budgets 1,10,100 --out $OUT            # everything
  python hlca_rep.py --seed 0 --folds 0 --stage new  --budgets 1,3,10,30,100 --out $OUT       # only the NEW label floors (seed 0 re-uses section 18)
  python hlca_rep.py --seed 0 --folds 0 --stage full --encoder B --budgets 10 --out $OUT      # second input encoder (PCA-100 of 4000 HVG)

New label-using floors, no source-removal term, trained on exactly the same labelled cells as the SupCon adapter:
  ce_f{b}_r{0,0.5}  : the adapter's trunk 50 -> 512 -> 256 -> 16 (unnormalised bottleneck) + linear head, plain cross-entropy, same sampler / steps / optimiser
  ldaX_f{b}         : LDA with all discriminant directions (<= 50 dims) on the f% labelled cells; extra 'mlp' probe (descriptive)
Writes E-format per-cell predictions only (preds_s{seed}.npz, meta_s{seed}.json); no embeddings, no expression.
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
sys.path = [p for p in sys.path if p and Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "e_singlecell"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arms as A  # noqa: E402  (E)
import attackers as AT  # noqa: E402  (E)
import evaluate as EV  # noqa: E402  (E)
from common import CACHE, MIN_HELD, MIN_TRAIN, SC  # noqa: E402  (E)


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


K = _load("fj_k", "hlca_k.py")        # section-18 code, unchanged: train_adapter, embed_all, FLOORS, hash01, N_CLS, PER
FJ = K.FJ                              # adapter.py: supcon, effective_rank
DEV = AT.DEV
D_OUT = K.D_OUT


def hash_u(i, seed):
    if seed == 0:
        return K.hash01(i)                                   # the section-18 label draw
    return int(hashlib.md5(f"h3003|s{seed}|{i}".encode()).hexdigest()[:8], 16) / 2 ** 32


class NetCE(nn.Module):
    def __init__(self, D, d, nK, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.f = nn.Sequential(nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d))
        self.head = nn.Linear(d, nK)

    def forward(self, x):                                    # released vector: unnormalised 16-d bottleneck
        return self.f((x - self.mu) / self.sd)


def train_ce(Xl, yl, seed, steps, nK):
    """Same sampler (64 classes x <= 8 distinct cells), steps, optimiser and schedule as K.train_adapter; plain cross-entropy through a linear head."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    yl = np.asarray(yl)
    order = np.argsort(yl, kind="stable")
    cl, start, cnt = np.unique(yl[order], return_index=True, return_counts=True)
    elig = np.where(cnt >= 2)[0]
    mu, sd = Xl.mean(0, keepdim=True), Xl.std(0, keepdim=True)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    net = NetCE(Xl.shape[1], D_OUT, nK, mu, sd).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    order_t = torch.as_tensor(order, device=DEV)
    start_t, cnt_t = torch.as_tensor(start, device=DEV), torch.as_tensor(cnt, device=DEV)
    cl_t = torch.as_tensor(cl, device=DEV)
    elig_t = torch.as_tensor(elig, device=DEV)
    nb = min(K.N_CLS, len(elig))
    losses, bsz = [], []
    net.train()
    for _ in range(steps):
        pick = elig_t[torch.randperm(len(elig), device=DEV, generator=g)[:nb]]
        n = cnt_t[pick]
        r = torch.minimum((torch.rand(nb, K.PER, device=DEV, generator=g) * n[:, None]).long(), (n - 1)[:, None])
        gi, _ = (start_t[pick][:, None] + r).sort(dim=1)
        keep = torch.ones_like(gi, dtype=torch.bool)
        keep[:, 1:] = gi[:, 1:] != gi[:, :-1]
        cells = order_t[gi[keep]]
        y = cl_t[pick][:, None].expand_as(gi)[keep]
        loss = F.cross_entropy(net.head(net(Xl[cells])), y)
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


def get_zB(sc, f, T):
    p = CACHE / sc.ds / f"zB_f{f}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    Z, _ = A.encoder0(sc.X, T, n_hvg=4000, n_pc=100)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp.npy")
    np.save(tmp, Z)
    os.replace(tmp, p)
    print(f"  encoder B fold {f}: {time.time() - t0:.0f}s", flush=True)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--folds", default="")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--budgets", default="1,10,100")
    ap.add_argument("--stage", choices=["full", "new"], default="full")
    ap.add_argument("--encoder", choices=["A", "B"], default="A")
    ap.add_argument("--perm", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SC(a.dataset)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(len(sc.folds)))
    budgets = [int(x) for x in a.budgets.split(",")]
    kinds = list(AT.KINDS)
    P = {}
    meta = dict(dataset=a.dataset, seed=a.seed, folds=folds, arms=[], timings={}, kinds=kinds, budgets=budgets, steps=a.steps, adapters={}, dims={},
                stage=a.stage, encoder=a.encoder, label_draw="md5(h3003|i)" if a.seed == 0 else f"md5(h3003|s{a.seed}|i)",
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                prereg="results_notes/F.md section 19, commit bc1f308")
    for f in folds:
        t0 = time.time()
        T, Hm = sc.fold_masks(f)
        held = np.where(Hm)[0]
        Z = EV.get_z0(sc, f, T) if a.encoder == "A" else get_zB(sc, f, T)
        ntr = np.bincount(sc.fine[T & (sc.fine >= 0)], minlength=len(sc.fine_names))
        nhe = np.bincount(sc.fine[Hm & (sc.fine >= 0)], minlength=len(sc.fine_names))
        types = np.where((ntr >= MIN_TRAIN) & (nhe >= MIN_HELD))[0]
        loc = -np.ones(len(sc.fine_names), np.int64)
        loc[types] = np.arange(len(types))
        trm = T & (loc[np.maximum(sc.fine, 0)] >= 0) & (sc.fine >= 0)
        tr_idx = np.where(trm)[0]
        ytr = loc[sc.fine[tr_idx]]
        Kc = len(types)
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

        def do_probe(name, X, mlp=False):
            pr = AT.run_probes(X[tr_idx], ytr, X[held], Kc, seed=a.seed)
            for k, v in pr.items():
                P[f"pr|{name}|{f}|{k}"] = types[v].astype(np.int16)
            if mlp:                                          # descriptive extra readout (not part of U)
                Xt, Xe = AT._t(X[tr_idx]), AT._t(X[held])
                st = AT.Std().fit(Xt)
                torch.manual_seed(a.seed)
                pred = AT.fit_mlp(st(Xt), AT._t(ytr, torch.long), Kc, 15, a.seed)
                P[f"pr|{name}|{f}|mlp"] = types[pred(st(Xe)).argmax(1).cpu().numpy()].astype(np.int16)

        def run_arm(name, X, attackers=True, mlp=False):
            t1 = time.time()
            X = np.ascontiguousarray(X, dtype=np.float32)
            do_probe(name, X, mlp=mlp)
            if attackers:
                do_att(name, X)
            meta["timings"][f"{f}|{name}"] = round(time.time() - t1, 1)
            meta["dims"][name] = int(X.shape[1])
            if name not in meta["arms"]:
                meta["arms"].append(name)
            print(f"  s{a.seed} fold {f} arm {name:16s} d={X.shape[1]:3d} {time.time() - t1:5.1f}s", flush=True)

        def save():
            np.savez_compressed(out / f"preds_s{a.seed}.npz", **P)
            (out / f"meta_s{a.seed}.json").write_text(json.dumps(meta, indent=1))

        if a.stage == "full":
            arms = A.build_arms(Z, T, sc, a.seed, sc.folds[f], scvi=None, only=K.FLOORS)
            for name in ["ctrl_noise16"] + [x for x in K.FLOORS if x != "ctrl_noise16"]:
                run_arm(name, arms[name])
            save()
        pool = np.where(T & (sc.fine >= 0))[0]
        u = np.array([hash_u(i, a.seed) for i in pool])
        sets = {b: pool[u < b / 100.0] for b in budgets}
        meta.setdefault("n_labelled", {})[str(f)] = {str(b): int(len(s)) for b, s in sets.items()}

        def mask_labels(idx):
            Tm = np.zeros(len(sc.fine), bool)
            Tm[idx] = True
            return Tm, np.where(Tm, sc.fine, -1)

        for b in budgets:
            Tm, yl = mask_labels(sets[b])
            if a.stage == "full" and b < 100:
                run_arm(f"ldaM_f{b}", A.lda_pad(Z, Tm, yl, None, max_dims=16))
            run_arm(f"ldaX_f{b}", A.lda_pad(Z, Tm, yl, None), mlp=True)
            save()
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=DEV)
        held_t = torch.as_tensor(held, device=DEV)
        for b in budgets:
            idx = sets[b]
            idx_t = torch.as_tensor(idx, device=DEV)
            ylab = sc.fine[idx]
            sd_ = a.seed * 7919 + f * 101 + b
            if a.stage == "full":                            # SupCon adapter: exactly the section-18 code path
                net, info = K.train_adapter(Zt[idx_t], ylab, sd_, a.steps)
                E0 = K.embed_all(net, Zt)
                info["eff_rank_held"] = FJ.effective_rank(E0[held_t])
                info["n_labelled"] = int(len(idx))
                sdv = E0[idx_t].std(0, keepdim=True)
                noise = torch.randn(E0.shape, generator=torch.Generator(device=DEV).manual_seed(sd_ + 1), device=DEV)
                for r in (0.0, 0.5):
                    run_arm(f"adp_f{b}_r{r:g}", (E0 + r * sdv * noise).cpu().numpy())
                meta["adapters"][f"{f}|adp_f{b}"] = info
                if a.perm:
                    netp, infop = K.train_adapter(Zt[idx_t], np.random.RandomState(sd_ + 50000).permutation(ylab), sd_ + 50000, a.steps)
                    Ep = K.embed_all(netp, Zt)
                    infop["eff_rank_held"] = FJ.effective_rank(Ep[held_t])
                    meta["adapters"][f"{f}|perm_f{b}"] = infop
                    run_arm(f"perm_f{b}_r0", Ep.cpu().numpy(), attackers=False)
            # CE-MLP floor, same labelled cells, same trunk / sampler / steps
            t1 = time.time()
            netc, infoc = train_ce(Zt[idx_t], ylab, sd_ + 77000, a.steps, len(sc.fine_names))
            with torch.no_grad():
                C0 = torch.cat([netc(Zt[i:i + 65536]) for i in range(0, Zt.shape[0], 65536)])
            infoc["eff_rank_held"] = FJ.effective_rank(C0[held_t])
            infoc["n_labelled"] = int(len(idx))
            sdc = C0[idx_t].std(0, keepdim=True)
            noisec = torch.randn(C0.shape, generator=torch.Generator(device=DEV).manual_seed(sd_ + 2), device=DEV)
            for r in (0.0, 0.5):
                run_arm(f"ce_f{b}_r{r:g}", (C0 + r * sdc * noisec).cpu().numpy())
            infoc["train_s"] = round(time.time() - t1, 1)
            meta["adapters"][f"{f}|ce_f{b}"] = infoc
            print(f"  s{a.seed} fold {f} budget {b}%: n_lab {len(idx)} CE loss {infoc['loss_first20']:.3f}->{infoc['loss_last200']:.3f} rank {infoc['eff_rank_held']:.1f}"
                  + (f" | SupCon loss {info['loss_first20']:.3f}->{info['loss_last200']:.3f} rank {info['eff_rank_held']:.1f}" if a.stage == "full" else ""), flush=True)
            save()
        print(f"fold {f} done: {time.time() - t0:.0f}s, probe types {Kc}, held {len(held)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
