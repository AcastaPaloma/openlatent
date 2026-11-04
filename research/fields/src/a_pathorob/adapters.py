"""Representation methods E(z).  Every method is fit on the adapter-train units A only and returns a dict
split-name -> representation.  Learned: vib (rate bottleneck, NO adversary), adv_retrain (adversary retrained from
scratch every round), dann, vq (discrete bottleneck, no adversary), supcon_xc (cross-centre contrastive).
Floors: raw, lda (task-label-only projection), leace, isp, centre_mean (transductive per-centre mean removal).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attackers import DEV, _t, Std, _cw, fit_linear

SPLITS = ("A", "B", "V", "C", "OOD1", "OOD2")


def mlp_enc(d, h, o, p_in=0.2):
    return nn.Sequential(nn.Dropout(p_in), nn.Linear(d, h), nn.GELU(), nn.Linear(h, o))


def batches(n, bs, gen=None):
    perm = torch.randperm(n, device=DEV)
    for i in range(0, n, bs):
        if len(perm[i:i + bs]) >= 8:
            yield perm[i:i + bs]


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


# ----------------------------------------------------------------------------- closed-form floors
def _eigh_sqrt(S, rho):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    lam = rho * w.mean()
    w = w + lam
    return V, w


_EIG = {}


def leace_fit(X, z, K, rho):
    """Belrose et al. 2023 closed-form linear concept erasure (least-squares sense); ridge rho * mean eigenvalue on the covariance."""
    key = (X.shape, float(X[:3].sum()), float(X[-3:].sum()))
    X = X.astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    n = len(X)
    if key not in _EIG:                      # one eigendecomposition per fit set, shared by the rho ladder
        _EIG.clear()
        w0, V0 = np.linalg.eigh(Xc.T @ Xc / n)
        _EIG[key] = (np.clip(w0, 0, None), V0)
    w0, V = _EIG[key]
    w = w0 + rho * w0.mean()
    Wh = (V / np.sqrt(w)) @ V.T          # Sigma^{-1/2}
    Wi = (V * np.sqrt(w)) @ V.T          # Sigma^{1/2}
    Z = np.eye(K)[z]
    Z = Z - Z.mean(0)
    Sxz = Xc.T @ Z / n
    U, s, _ = np.linalg.svd(Wh @ Sxz, full_matrices=False)
    U = U[:, s > 1e-8 * max(s.max(), 1e-30)]
    P = U @ U.T
    M = Wi @ P @ Wh                       # x' = x - M (x - mu)
    return mu, M


def isp_fit(X, z, K, n_iter, wd=1e-2):
    """Iterative nullspace projection (Ravfogel et al. 2020) with softmax-regression directions."""
    Xt = _t(X)
    d = X.shape[1]
    P = torch.eye(d, device=DEV)
    zt = _t(z, torch.long)
    for _ in range(n_iter):
        Xp = Xt @ P
        mu, sd = Xp.mean(0, keepdim=True), Xp.std(0, keepdim=True).clamp(min=1e-6)
        f = fit_linear((Xp - mu) / sd, zt, K, wd, iters=40)
        W = f.W / sd.T                         # (d,K) in the coordinates of Xp
        W = W - W.mean(1, keepdim=True)        # softmax is shift invariant
        U, s, _ = torch.linalg.svd(W, full_matrices=False)
        U = U[:, s > 1e-6 * s.max()]
        P = P @ (torch.eye(d, device=DEV) - U @ U.T)
    return P.cpu().numpy()


# ----------------------------------------------------------------------------- learned
def _prep(P, sp):
    """standardise with the adapter-train statistics (all methods see the same inputs)."""
    XA = _t(P.X[sp["A"]])
    st = Std().fit(XA)
    return st, st(XA), _t(P.y[sp["A"]], torch.long), _t(P.c[sp["A"]], torch.long)


def _release(f, P, sp, st, gen=None):
    out = {}
    with torch.no_grad():
        for s in SPLITS:
            idx = sp[s]
            Z = []
            for i in range(0, len(idx), 4096):
                Z.append(f(st(_t(P.X[idx[i:i + 4096]]))).cpu())
            out[s] = torch.cat(Z).numpy().astype(np.float32) if len(Z) else np.zeros((0, 1), np.float32)
    return out


def train_vib(P, sp, dl, beta, seed, epochs=40, hid=512):
    st, XA, yA, cA = _prep(P, sp)
    K = len(P.classes)
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(XA.shape[1], hid), nn.GELU(), nn.Linear(hid, 2 * dl)).to(DEV)
    head = nn.Linear(dl, K).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)
    cw = _cw(yA, K)
    for ep in range(epochs):
        enc.train()
        for idx in batches(len(XA), 256):
            mu, lv = enc(XA[idx]).chunk(2, -1)
            lv = lv.clamp(-10, 6)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
            loss = F.cross_entropy(head(z), yA[idx], weight=cw) + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    g = torch.Generator(device=DEV).manual_seed(5000 + seed)

    def f(x):   # the released latent is a SAMPLE from the channel
        mu, lv = enc(x).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        return mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, device=DEV, generator=g)
    rep = _release(f, P, sp, st)
    with torch.no_grad():                    # rate = KL(q(z|x) || N(0,I)) in bits per patch, measured on the sealed test units C
        mu, lv = enc(st(_t(P.X[sp["C"]]))).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        rate_bits = float((0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)).mean() / math.log(2))
    return rep, dict(rate_bits_per_patch=rate_bits)


def train_dann(P, sp, dl, lam, seed, epochs=40, hid=512):
    st, XA, yA, cA = _prep(P, sp)
    K, Kc = len(P.classes), len(P.centres)
    Kc = int(P.n_id_centres)
    torch.manual_seed(seed)
    enc = mlp_enc(XA.shape[1], hid, dl).to(DEV)
    head = nn.Linear(dl, K).to(DEV)
    adv = nn.Sequential(nn.Linear(dl, 256), nn.GELU(), nn.Linear(256, Kc)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()) + list(adv.parameters()), lr=1e-3, weight_decay=1e-2)
    cw = _cw(yA, K)
    nsteps = epochs * math.ceil(len(XA) / 256)
    t = 0
    for ep in range(epochs):
        enc.train()
        for idx in batches(len(XA), 256):
            p = t / nsteps
            l = lam * (2 / (1 + math.exp(-10 * p)) - 1)
            z = enc(XA[idx])
            loss = F.cross_entropy(head(z), yA[idx], weight=cw) + F.cross_entropy(adv(GRL.apply(z, l)), cA[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            t += 1
    enc.eval()
    return _release(enc, P, sp, st), {}


def train_adv_retrain(P, sp, dl, lam, seed, rounds=12, steps_e=60, steps_a=120, hid=512):
    """Adapter vs adversaries (linear + MLP) that are RE-INITIALISED and retrained to convergence-ish every round."""
    st, XA, yA, cA = _prep(P, sp)
    K = len(P.classes)
    Kc = int(P.n_id_centres)
    torch.manual_seed(seed)
    enc = mlp_enc(XA.shape[1], hid, dl).to(DEV)
    head = nn.Linear(dl, K).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)
    cw = _cw(yA, K)
    logs = []
    for r in range(rounds):
        enc.eval()
        with torch.no_grad():
            Z = enc(XA)
        zs = Std().fit(Z)
        Zn = zs(Z)
        advs = nn.ModuleList([nn.Linear(dl, Kc), nn.Sequential(nn.Linear(dl, 256), nn.GELU(), nn.Linear(256, Kc))]).to(DEV)
        oa = torch.optim.Adam(advs.parameters(), lr=3e-3)
        for s in range(steps_a):
            idx = torch.randint(0, len(Zn), (512,), device=DEV)
            loss = sum(F.cross_entropy(a(Zn[idx]), cA[idx]) for a in advs)
            oa.zero_grad(); loss.backward(); oa.step()
        with torch.no_grad():
            acc = float(np.mean([(a(Zn).argmax(1) == cA).float().mean().item() for a in advs]))
        enc.train()
        for s in range(steps_e):
            idx = torch.randint(0, len(XA), (256,), device=DEV)
            z = enc(XA[idx])
            zn = zs(z)                       # attackers were fitted on standardised latents
            conf = 0.0
            for a in advs:
                p = F.softmax(a(zn), -1)
                conf = conf + (p * torch.log(p.clamp(min=1e-8))).sum(-1).mean() + math.log(Kc)   # KL(p || uniform)
            loss = F.cross_entropy(head(z), yA[idx], weight=cw) + lam * conf / len(advs)
            opt.zero_grad(); loss.backward(); opt.step()
        logs.append(round(acc, 3))
    enc.eval()
    return _release(enc, P, sp, st), dict(adv_train_acc_per_round=logs)


def train_ae(P, sp, dl, lam, beta, seed, rounds=12, steps_e=60, steps_a=120, hid=512):
    """Task-agnostic adapter: reconstruct the (standardised) frozen embedding through a rate-limited Gaussian channel (beta * KL, beta=0
    -> almost no rate limit), NO class labels.  lam > 0 adds the retrained-from-scratch adversaries (as in adv_retrain) that try to read the
    centre from the released sample.  Class labels are never used, so held-out classes are on the same footing as seen ones."""
    st, XA, yA, cA = _prep(P, sp)
    d = XA.shape[1]
    Kc = int(P.n_id_centres)
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 2 * dl)).to(DEV)
    dec = nn.Sequential(nn.Linear(dl, hid), nn.GELU(), nn.Linear(hid, d)).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-2)

    def chan(x):
        mu, lv = enc(x).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
        return z, kl
    logs = []
    for r in range(rounds):
        advs, zs = None, None
        if lam > 0:
            enc.eval()
            with torch.no_grad():
                Z = chan(XA)[0]
            zs = Std().fit(Z)
            Zn = zs(Z)
            advs = nn.ModuleList([nn.Linear(dl, Kc), nn.Sequential(nn.Linear(dl, 256), nn.GELU(), nn.Linear(256, Kc))]).to(DEV)
            oa = torch.optim.Adam(advs.parameters(), lr=3e-3)
            for s in range(steps_a):
                idx = torch.randint(0, len(Zn), (512,), device=DEV)
                loss = sum(F.cross_entropy(a_(Zn[idx]), cA[idx]) for a_ in advs)
                oa.zero_grad(); loss.backward(); oa.step()
            with torch.no_grad():
                logs.append(round(float(np.mean([(a_(Zn).argmax(1) == cA).float().mean().item() for a_ in advs])), 3))
        enc.train()
        for s in range(steps_e):
            idx = torch.randint(0, len(XA), (256,), device=DEV)
            z, kl = chan(XA[idx])
            loss = F.mse_loss(dec(z), XA[idx]) + beta * kl
            if lam > 0:
                zn = zs(z)
                conf = 0.0
                for a_ in advs:
                    p = F.softmax(a_(zn), -1)
                    conf = conf + (p * torch.log(p.clamp(min=1e-8))).sum(-1).mean() + math.log(Kc)
                loss = loss + lam * conf / len(advs)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    g = torch.Generator(device=DEV).manual_seed(5000 + seed)

    def f(x):
        mu, lv = enc(x).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        return mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, device=DEV, generator=g)
    rep = _release(f, P, sp, st)
    with torch.no_grad():
        mu, lv = enc(st(_t(P.X[sp["C"]]))).chunk(2, -1)
        lv = lv.clamp(-10, 6)
        rate_bits = float((0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)).mean() / math.log(2))
    return rep, dict(rate_bits_per_patch=rate_bits, adv_train_acc_per_round=logs)


class VQ(nn.Module):
    def __init__(self, m, Kc, dg):
        super().__init__()
        self.m, self.Kc, self.dg = m, Kc, dg
        self.cb = nn.Parameter(torch.randn(m, Kc, dg) * 0.5)
        self.register_buffer("init", torch.zeros(()))

    def forward(self, z):
        n = z.shape[0]
        z = z.view(n, self.m, self.dg)
        if self.init.item() == 0 and self.training:         # data-dependent init from the first batch
            with torch.no_grad():
                for j in range(self.m):
                    self.cb.data[j] = z[torch.randint(0, n, (self.Kc,), device=z.device), j].detach()
                self.init.fill_(1)
        d = ((z.unsqueeze(2) - self.cb.unsqueeze(0)) ** 2).sum(-1)    # (n,m,Kc)
        idx = d.argmin(-1)
        zq = torch.stack([self.cb[j][idx[:, j]] for j in range(self.m)], 1)
        loss = F.mse_loss(zq, z.detach()) + 0.25 * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()
        return zq.reshape(n, -1), loss, idx


def train_vq(P, sp, m, Kc, seed, dg=16, epochs=40, hid=512):
    st, XA, yA, cA = _prep(P, sp)
    K = len(P.classes)
    torch.manual_seed(seed)
    enc = mlp_enc(XA.shape[1], hid, m * dg).to(DEV)
    vq = VQ(m, Kc, dg).to(DEV)
    head = nn.Linear(m * dg, K).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(vq.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)
    cw = _cw(yA, K)
    for ep in range(epochs):
        enc.train(); vq.train()
        used = torch.zeros(m, Kc, device=DEV)
        for idx in batches(len(XA), 256):
            z = enc(XA[idx])
            zq, lvq, ci = vq(z)
            for j in range(m):
                used[j] += torch.bincount(ci[:, j], minlength=Kc).float()
            loss = F.cross_entropy(head(zq), yA[idx], weight=cw) + lvq
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                 # dead-code reset
            dead = (used == 0)
            if dead.any() and ep < epochs - 5:
                zz = enc(XA[torch.randint(0, len(XA), (Kc,), device=DEV)]).view(Kc, m, dg)
                for j in range(m):
                    vq.cb.data[j][dead[j]] = zz[:dead[j].sum(), j].detach()
    enc.eval(); vq.eval()

    def f(x):
        return vq(enc(x))[0]
    return _release(f, P, sp, st), dict(rate_bits_per_patch=float(m * math.log2(Kc)))


def train_supcon(P, sp, dl, tau, seed, epochs=40, hid=512):
    """cross-centre supervised contrastive: positives = same class, DIFFERENT centre; negatives = everything else in the batch."""
    st, XA, yA, cA = _prep(P, sp)
    torch.manual_seed(seed)
    enc = mlp_enc(XA.shape[1], hid, dl).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        enc.train()
        for idx in batches(len(XA), 512):
            z = F.normalize(enc(XA[idx]), dim=1)
            y, c = yA[idx], cA[idx]
            sim = z @ z.T / tau
            eye = torch.eye(len(idx), device=DEV, dtype=torch.bool)
            pos = (y[:, None] == y[None]) & (c[:, None] != c[None])
            logp = sim - torch.logsumexp(sim.masked_fill(eye, -1e9), 1, keepdim=True)
            npos = pos.sum(1)
            ok = npos > 0
            if ok.sum() == 0:
                continue
            loss = -((logp * pos).sum(1)[ok] / npos[ok]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    return _release(lambda x: F.normalize(enc(x), dim=1), P, sp, st), {}


# ----------------------------------------------------------------------------- floors
def run_floor(method, hp, P, sp, seed):
    K, Kc = len(P.classes), int(P.n_id_centres)
    if method == "raw":
        return {s: P.X[sp[s]] for s in SPLITS}, {}
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        m = LDA(solver="eigen", shrinkage=hp.get("shrinkage", "auto")).fit(P.X[sp["A"]], P.y[sp["A"]])
        return {s: m.transform(P.X[sp[s]]).astype(np.float32) for s in SPLITS}, {}
    if method == "lda_pca":      # matched-dimensionality label-only arm: LDA (C-1 dims) + top PCs of the residual after removing the LDA subspace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        XA = P.X[sp["A"]].astype(np.float64)
        key = ("lda_pca", XA.shape, float(XA[:3].sum()), float(XA[-3:].sum()))
        if key not in _EIG:                       # LDA + residual SVD are shared by the whole d ladder
            _EIG.clear()
            m0 = LDA(solver="eigen", shrinkage="auto").fit(XA, P.y[sp["A"]])
            nl0 = m0.transform(XA[:2]).shape[1]
            mu0 = XA.mean(0)
            Q0, _ = np.linalg.qr(m0.scalings_[:, :nl0])
            resid = (XA - mu0) - ((XA - mu0) @ Q0) @ Q0.T
            _, _, Vt0 = np.linalg.svd(resid, full_matrices=False)
            _EIG[key] = (m0, nl0, mu0, Q0, Vt0[:128].T)
        m_, nl, mu, Q, Vall = _EIG[key]
        d = hp["d"]
        V = Vall[:, : max(d - nl, 0)]
        out = {}
        for s in SPLITS:
            X64 = P.X[sp[s]].astype(np.float64)
            Xc_ = X64 - mu
            parts = [m_.transform(X64)[:, :d]]
            if d > nl:
                parts.append((Xc_ - (Xc_ @ Q) @ Q.T) @ V)
            out[s] = np.concatenate(parts, 1).astype(np.float32)
        return out, {}
    if method == "leace":
        mu, M = leace_fit(P.X[sp["A"]], P.c[sp["A"]], Kc, hp["rho"])
        out = {}
        for s in SPLITS:
            X64 = P.X[sp[s]].astype(np.float64)
            out[s] = (X64 - (X64 - mu) @ M.T).astype(np.float32)
        return out, {}
    if method == "isp":
        XA = P.X[sp["A"]]
        mu, sd = XA.mean(0), XA.std(0) + 1e-6
        Pm = isp_fit((XA - mu) / sd, P.c[sp["A"]], Kc, hp["n_iter"])
        return {s: (((P.X[sp[s]] - mu) / sd) @ Pm).astype(np.float32) for s in SPLITS}, {}
    if method in ("centre_zscore", "centre_whiten"):
        # transductive per-centre standardisation inside each split (needs the centre id of every patch at release time)
        out = {}
        for s in SPLITS:
            X = P.X[sp[s]].astype(np.float32).copy()
            c = P.c[sp[s]]
            if method == "centre_whiten":
                # PCA-64 fitted on A (pooled), then per-centre whitening with ridge 1e-2 * mean eigenvalue
                if "pca" not in _EIG:
                    XA = P.X[sp["A"]].astype(np.float64)
                    mu = XA.mean(0)
                    _, _, Vt = np.linalg.svd(XA - mu, full_matrices=False)
                    _EIG["pca"] = (mu, Vt[:64].T)
                mu_p, Vp = _EIG["pca"]
                X = ((X - mu_p) @ Vp).astype(np.float32)
            for k in np.unique(c):
                m = c == k
                Xk = X[m] - X[m].mean(0)
                if method == "centre_zscore":
                    Xk = Xk / (Xk.std(0) + 1e-6)
                else:
                    cov = np.cov(Xk.T) + 1e-2 * np.eye(Xk.shape[1]) * np.trace(np.cov(Xk.T)) / Xk.shape[1]
                    w, V = np.linalg.eigh(cov)
                    Xk = Xk @ (V / np.sqrt(w)) @ V.T
                X[m] = Xk
            out[s] = X.astype(np.float32)
        return out, {}
    if method in ("pca", "randproj"):       # label-free rate controls: d numbers per patch
        d = hp["d"]
        XA = P.X[sp["A"]].astype(np.float64)
        mu = XA.mean(0)
        if method == "pca":
            _, _, Vt = np.linalg.svd(XA - mu, full_matrices=False)
            R = Vt[:d].T
        else:
            R = np.random.RandomState(31 + seed).randn(XA.shape[1], d) / np.sqrt(XA.shape[1])
        return {s: ((P.X[sp[s]] - mu) @ R).astype(np.float32) for s in SPLITS}, {}
    if method == "centre_mean":      # transductive reference: subtract the mean of each centre computed inside each split
        out = {}
        for s in SPLITS:
            X = P.X[sp[s]].copy()
            c = P.c[sp[s]]
            for k in np.unique(c):
                X[c == k] -= X[c == k].mean(0)
            out[s] = X
        return out, {}
    raise ValueError(method)


def run_control(method, P, sp, seed):
    rng = np.random.RandomState(99 + seed)
    d = P.X.shape[1]
    if method == "ctrl_const":
        return {s: np.ones((len(sp[s]), d), np.float32) for s in SPLITS}, {}
    if method == "ctrl_noise":
        return {s: rng.randn(len(sp[s]), d).astype(np.float32) for s in SPLITS}, {}
    raise ValueError(method)


def build(method, hp, P, sp, seed):
    if method in ("raw", "lda", "lda_pca", "leace", "isp", "centre_mean", "centre_zscore", "centre_whiten", "pca", "randproj"):
        return run_floor(method, hp, P, sp, seed)
    if method in ("ctrl_const", "ctrl_noise", "ctrl_shuffle", "ctrl_shuffle_orth"):
        if method in ("ctrl_shuffle", "ctrl_shuffle_orth"):
            return run_floor("raw", {}, P, sp, seed)
        return run_control(method, P, sp, seed)
    if method == "vib":
        return train_vib(P, sp, hp["dl"], hp["beta"], seed)
    if method == "dann":
        return train_dann(P, sp, hp["dl"], hp["lam"], seed)
    if method == "adv_retrain":
        return train_adv_retrain(P, sp, hp["dl"], hp["lam"], seed)
    if method == "vq":
        return train_vq(P, sp, hp["m"], hp["K"], seed)
    if method == "ae":
        return train_ae(P, sp, hp["dl"], hp["lam"], hp["beta"], seed)
    if method == "supcon_xc":
        return train_supcon(P, sp, hp["dl"], hp["tau"], seed)
    raise ValueError(method)


# grids.  GATE = floors + controls (run and read BEFORE any learned result); LEARNED = the adapters; RATE = label-free rate controls
GATE = (
    [("raw", {}), ("lda", {}), ("centre_mean", {}), ("centre_zscore", {}), ("centre_whiten", {})]
    + [("leace", dict(rho=r)) for r in (1e-3, 1e-2, 1e-1, 1.0)]
    + [("isp", dict(n_iter=n)) for n in (1, 5, 20)]
)
CONTROLS = [("ctrl_const", {}), ("ctrl_noise", {})] + [("ctrl_shuffle_orth", dict(rep=r)) for r in range(3)]
LEARNED = (
    [("vib", dict(dl=dl, beta=b)) for dl in (4, 32) for b in (1e-3, 1e-2, 1e-1, 1.0)]
    + [("vq", dict(m=m, K=K)) for m, K in ((1, 16), (2, 16), (4, 16), (4, 64), (8, 64))]
    + [("dann", dict(dl=64, lam=l)) for l in (0.3, 1.0, 3.0, 10.0)]
    + [("adv_retrain", dict(dl=64, lam=l)) for l in (0.3, 1.0, 3.0, 10.0)]
    + [("adv_retrain", dict(dl=8, lam=l)) for l in (1.0, 10.0)]
    + [("supcon_xc", dict(dl=dl, tau=0.1)) for dl in (16, 64)]
)
RATE = [(m, dict(d=d)) for m in ("pca", "randproj") for d in (2, 4, 8, 16, 32, 64, 256)]
GRID = GATE + LEARNED + RATE
# protocol 2: matched-dimensionality label-only arm, task-agnostic autoencoder adapters, per-centre whitening, held-out classes
LDA_ARMS = [("lda", {})] + [("lda_pca", dict(d=d)) for d in (4, 8, 16, 32, 64, 128)]
AE = [("ae", dict(dl=dl, lam=0.0, beta=b)) for dl in (8, 32) for b in (1e-3, 1e-2)] + \
     [("ae", dict(dl=dl, lam=l, beta=1e-3)) for dl in (8, 32) for l in (1.0, 3.0, 10.0)]
P2 = (
    [("raw", {}), ("centre_whiten", {}), ("leace", dict(rho=1.0))] + LDA_ARMS
    + [("pca", dict(d=d)) for d in (4, 16, 64)] + [("randproj", dict(d=d)) for d in (4, 16, 64)]
    + [("vib", dict(dl=dl, beta=b)) for dl in (4, 32) for b in (1e-3, 1e-2, 1e-1)]
    + [("vq", dict(m=m, K=K)) for m, K in ((1, 16), (2, 16), (4, 16), (4, 64), (8, 64))]
    + [("dann", dict(dl=64, lam=3.0))]
    + [("adv_retrain", dict(dl=64, lam=l)) for l in (1.0, 3.0, 10.0)] + [("adv_retrain", dict(dl=8, lam=3.0))]
    + [("supcon_xc", dict(dl=dl, tau=0.1)) for dl in (16, 64)]
    + AE
)
