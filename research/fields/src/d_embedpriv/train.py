"""Learned released-vector maps for Focus D.

All three share the retrieval objective and training budget, so any difference is attributable to the one thing that changes:
  (D2 repair, 2026-11-04 ~19:00Z: with conf_norm=True the confusion term is divided by its value at step 0, so lambda is a relative weight; every fit is checked for collapse,
   i.e. final retrieval loss > ln(N docs) - 0.5 = the uniform-softmax value, and flagged `collapsed`.  The original `adv` arm (conf_norm=False, lambda 0.3/3) collapsed in all 24 fits.)
  sup_lin    linear map 768->d (PCA-d init), trained with the retrieval loss only            [task-supervised linear floor at matched d]
  bott_mlp   linear + MLP-residual map 768->d (PCA-d init), retrieval loss only             [rate bottleneck, NO adversary: mandatory control]
  adv        same map as bott_mlp + a token-bag adversary g(z) that is re-trained against the current map; the map is trained to make
             g's predictions equal the word-frequency prior  (confusion loss  KL(pi || sigmoid(g(f(x)))) summed over words, nats/doc)
             g is re-initialised every `reinit` steps (+ warm-up) so f cannot over-fit one decoder.  Adversary data: defender pool only
             (NFCorpus D docs + SciFact ED docs); the attacker's pool (A, EA) and V, C docs are never touched.

Retrieval loss: listwise cross-entropy of softmax(cos(f(q), f(d)) / tau) against graded train qrels, over all NFCorpus docs outside the sealed
split C (C docs, dev and test queries never enter any fit).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arms import Arm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FNet(nn.Module):
    def __init__(self, d_in, d, mlp, P0, mu, hidden=1024):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.lin = nn.Linear(d_in, d, bias=False)
        self.lin.weight.data = torch.as_tensor(P0.T, dtype=torch.float32).clone()
        self.mlp = None
        if mlp:
            self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        x = x - self.mu
        z = self.lin(x)
        if self.mlp is not None:
            z = z + self.mlp(x)
        return F.normalize(z, dim=-1)


def _adv_net(d, V, prior, hidden=1024):
    net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, V))
    with torch.no_grad():
        net[-1].bias.copy_(torch.logit(prior.clamp(1e-5, 1 - 1e-5)))
        net[-1].weight.mul_(0.1)
    return net.to(DEV)


class Learned(Arm):
    def __init__(self, kind, d, lam=0.0, steps=1500, bs=256, lr=1e-3, tau=0.05, reinit=250, warm=150, k_adv=2, conf_norm=False):
        self.kind, self.d, self.lam, self.conf_norm = kind, d, lam, conf_norm
        self.steps, self.bs, self.lr, self.tau, self.reinit, self.warm, self.k_adv = steps, bs, lr, tau, reinit, warm, k_adv
        self.name = f"{kind}_d{d}" + ((f"_n{lam:g}" if conf_norm else f"_l{lam:g}") if kind == "adv" else "")
        self.collapsed, self.final_ret = False, None
        self.log = []

    def fit(self, ctx, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        Xdef = ctx.defender_docs()
        mu = Xdef.mean(0)
        _, _, Vt = np.linalg.svd(Xdef - mu, full_matrices=False)
        f = FNet(Xdef.shape[1], self.d, self.kind != "sup_lin", Vt[: self.d].T, mu).to(DEV)
        opt = torch.optim.AdamW(f.parameters(), lr=self.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        # retrieval data (outside C)
        nc = ctx.nc_idx
        Xd = torch.as_tensor(ctx.X_nf[nc], device=DEV)
        Xq = torch.as_tensor(ctx.X_q[ctx.trq_idx], device=DEV)
        T = torch.as_tensor(ctx.T_train, device=DEV)  # [n_trq, n_nc] normalised graded relevance
        adv = self.kind == "adv"
        if adv:
            Xa = torch.as_tensor(Xdef, device=DEV)
            Ya = torch.as_tensor(ctx.defender_bags().toarray(), device=DEV).float()
            pi = Ya.mean(0).clamp(1e-5, 1 - 1e-5)
            Hpi = (-(pi * pi.log() + (1 - pi) * (1 - pi).log())).sum()
            g = None
            conf0 = None
        for step in range(self.steps):
            if adv:
                if step % self.reinit == 0:
                    g = _adv_net(self.d, Ya.shape[1], pi)
                    og = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-2)
                    for _ in range(self.warm):
                        self._adv_step(f, g, og, Xa, Ya)
                for _ in range(self.k_adv):
                    self._adv_step(f, g, og, Xa, Ya)
            b = torch.randint(0, len(Xq), (self.bs,), device=DEV)
            zq = f(Xq[b])
            zd = f(Xd)
            loss_ret = -(T[b] * F.log_softmax(zq @ zd.T / self.tau, dim=1)).sum(1).mean()
            loss = loss_ret
            conf = torch.zeros(())
            if adv:
                ia = torch.randint(0, len(Xa), (self.bs,), device=DEV)
                lg = g(f(Xa[ia]))
                conf = (F.binary_cross_entropy_with_logits(lg, pi.expand_as(lg), reduction="none").sum(1) - Hpi).mean()
                if self.conf_norm:
                    if conf0 is None:  # scale of the confusion term at step 0 (adversary already warmed up on the initial map), detached
                        conf0 = max(float(conf.detach()), 1e-3)
                    loss = loss + self.lam * conf / conf0
                else:
                    loss = loss + self.lam * conf
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == self.steps - 1:
                self.log.append(dict(step=step, ret=float(loss_ret), conf=float(conf), conf_rel=(float(conf) / conf0 if adv and self.conf_norm else None)))
        self.f = f.eval()
        self.g = g if adv else None  # final in-loop adversary head (D3 engagement check)
        # collapse check: a map whose retrieval loss is still at the uniform-softmax value ln N never trained
        self.final_ret = float(np.mean([x["ret"] for x in self.log[-2:]]))
        self.collapsed = bool(self.final_ret > math.log(len(nc)) - 0.5)
        return self

    def _adv_step(self, f, g, og, Xa, Ya):
        ia = torch.randint(0, len(Xa), (256,), device=DEV)
        with torch.no_grad():
            z = f(Xa[ia])
        loss = F.binary_cross_entropy_with_logits(g(z), Ya[ia], reduction="none").sum(1).mean()
        og.zero_grad()
        loss.backward()
        og.step()

    @torch.no_grad()
    def _apply(self, X):
        out = []
        for s in range(0, len(X), 2048):
            out.append(self.f(torch.as_tensor(X[s:s + 2048], device=DEV)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def docs(self, X, rng):
        return self._apply(X)

    def queries(self, X):
        return self._apply(X)


def make_learned(spec):
    kw = {k: spec[k] for k in ("steps", "bs", "lr", "tau", "reinit", "warm", "k_adv", "conf_norm") if k in spec}
    return Learned(spec["family"], spec["d"], spec.get("lam", 0.0), **kw)
