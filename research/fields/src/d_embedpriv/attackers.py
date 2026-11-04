"""Token-bag attackers, always retrained from scratch on the RELEASED (defended) vectors of the attacker's own pool.

Each attacker maps a released vector z to a score per vocabulary word.  Hyper-parameters / early stopping are chosen on an inner
random 15% of the attacker pool (never on V or C).  The last-layer bias is initialised at the logit of the pool word frequency, so
an attacker given a constant vector (or shuffled pairs) falls back to the frequency prior: that is the "chance" reference.

  ridge : closed-form multi-output ridge regression on standardised z (linear attacker)
  mlp   : 2 x 1024 GELU MLP, BCE, AdamW, early stopping
  knn   : cosine k-nearest-neighbour of the pool, score = softmax-weighted mean of neighbours' bags (kNN-to-corpus)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _std(Z):
    mu = Z.mean(0, keepdim=True)
    sd = Z.std(0, keepdim=True).clamp(min=1e-6)
    return mu, sd


def _quick_rprec(S, Y, k_cap=None):
    """Mean R-precision over all vocabulary words (inner selection metric), torch."""
    k = Y.sum(1).long()
    order = torch.argsort(-S, dim=1)
    hit = torch.gather(Y, 1, order) > 0
    ar = torch.arange(S.shape[1], device=S.device)[None]
    ok = k > 0
    r = ((hit & (ar < k[:, None])).sum(1).float() / k.clamp(min=1))[ok]
    return r.mean().item()


class Ridge:
    name = "ridge"

    def fit(self, Z, Y, seed=0):
        g = torch.Generator().manual_seed(seed)
        Z = Z.to(DEV).float()
        Y = Y.to(DEV).float()
        n = len(Z)
        perm = torch.randperm(n, generator=g).to(DEV)
        nv = int(0.15 * n)
        va, tr = perm[:nv], perm[nv:]
        best = (-1, None)
        for lam in [1e-1, 1.0, 10.0, 100.0, 1000.0]:
            mu, sd = _std(Z[tr])
            W = self._solve((Z[tr] - mu) / sd, Y[tr], lam)
            s = ((Z[va] - mu) / sd) @ W[:-1] + W[-1]
            r = _quick_rprec(s, Y[va])
            if r > best[0]:
                best = (r, lam)
        self.lam = best[1]
        self.mu, self.sd = _std(Z)
        self.W = self._solve((Z - self.mu) / self.sd, Y, self.lam)
        return self

    @staticmethod
    def _solve(Zs, Y, lam):
        n, d = Zs.shape
        X = torch.cat([Zs, torch.ones(n, 1, device=Zs.device)], 1)
        A = X.T @ X + lam * torch.eye(d + 1, device=Zs.device)
        A[-1, -1] -= lam  # no penalty on the bias
        return torch.linalg.solve(A.double(), (X.T @ Y).double()).float()

    def score(self, Z):
        Z = Z.to(DEV).float()
        return (((Z - self.mu) / self.sd) @ self.W[:-1] + self.W[-1]).cpu().numpy()


class MLP:
    name = "mlp"

    def __init__(self, hidden=1024, epochs=80, patience=8, lr=1e-3, wd=1e-2, drop=0.2, bs=128):
        self.h, self.epochs, self.patience, self.lr, self.wd, self.drop, self.bs = hidden, epochs, patience, lr, wd, drop, bs

    def _net(self, d, V, prior):
        net = nn.Sequential(nn.Linear(d, self.h), nn.GELU(), nn.Dropout(self.drop), nn.Linear(self.h, self.h), nn.GELU(),
                            nn.Dropout(self.drop), nn.Linear(self.h, V))
        with torch.no_grad():
            net[-1].bias.copy_(torch.logit(prior.clamp(1e-5, 1 - 1e-5)))
            net[-1].weight.mul_(0.1)
        return net.to(DEV)

    def fit(self, Z, Y, seed=0):
        torch.manual_seed(seed)
        Z = Z.to(DEV).float()
        Y = Y.to(DEV).float()
        n = len(Z)
        perm = torch.randperm(n, device=DEV)
        nv = int(0.15 * n)
        va, tr = perm[:nv], perm[nv:]
        self.mu, self.sd = _std(Z[tr])
        Zs = (Z - self.mu) / self.sd
        prior = Y[tr].mean(0)
        self.net = self._net(Z.shape[1], Y.shape[1], prior)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, bad, state = 1e9, 0, None
        for ep in range(self.epochs):
            self.net.train()
            p = tr[torch.randperm(len(tr), device=DEV)]
            for s in range(0, len(p), self.bs):
                b = p[s:s + self.bs]
                loss = F.binary_cross_entropy_with_logits(self.net(Zs[b]), Y[b], reduction="none").sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(self.net(Zs[va]), Y[va], reduction="none").sum(1).mean().item()
            if vl < best - 1e-3:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        self.epochs_run = ep + 1
        return self

    @torch.no_grad()
    def score(self, Z):
        Z = Z.to(DEV).float()
        return self.net((Z - self.mu) / self.sd).cpu().numpy()


class KNN:
    name = "knn"

    def fit(self, Z, Y, seed=0):
        Z = Z.to(DEV).float()
        self.Y = Y.to(DEV).float()
        self.mu = Z.mean(0, keepdim=True)
        self.P = F.normalize(Z - self.mu, dim=1)
        self.prior = self.Y.mean(0)
        n = len(Z)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).to(DEV)
        nv = int(0.15 * n)
        va, tr = perm[:nv], perm[nv:]
        best = (-1, None)
        for k in (5, 20):
            for T in (0.02, 0.1):
                s = self._score(self.P[va], self.P[tr], self.Y[tr], k, T)
                r = _quick_rprec(s, self.Y[va])
                if r > best[0]:
                    best = (r, (k, T))
        self.k, self.T = best[1]
        return self

    @staticmethod
    def _score(Q, P, Y, k, T):
        S = Q @ P.T
        v, ix = S.topk(min(k, P.shape[0]), dim=1)
        w = torch.softmax(v / T, dim=1)
        return (w[:, :, None] * Y[ix]).sum(1)

    def score(self, Z):
        Q = F.normalize(Z.to(DEV).float() - self.mu, dim=1)
        return self._score(Q, self.P, self.Y, self.k, self.T).cpu().numpy()


SUITE = {"ridge": Ridge, "mlp": MLP, "knn": KNN}


def run_suite(Zpool, Ypool, evals, seed=0, names=("ridge", "mlp", "knn")):
    """Train every attacker on (Zpool, Ypool) and score each eval matrix.  Returns {attacker: {eval_name: scores}}."""
    out = {}
    for nm in names:
        a = SUITE[nm]().fit(Zpool, Ypool, seed=seed)
        out[nm] = {k: a.score(Z) for k, Z in evals.items()}
        del a
    return out
