"""Floors and references of JS-1 (results_notes/F.md 2.2).  Every inductive arm is a function of ONE well's encoder-0 vector,
fitted on the Fit set only (torch, float64 for the eigen problems), and never uses the source id at inference.
Transductive references are computed once per scenario (refs.py) and loaded here.
"""
import hashlib

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHRINK = 0.1


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def T(x, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=DEV)


class PCA:
    def __init__(self, Xfit):
        X = Xfit.double()
        self.mu = X.mean(0, keepdim=True)
        C = (X - self.mu).T @ (X - self.mu) / (X.shape[0] - 1)
        w, V = torch.linalg.eigh(C)
        o = torch.argsort(w, descending=True)
        self.w, self.V = w[o], V[:, o]

    def __call__(self, X, d):
        return (((X.double() - self.mu) @ self.V[:, :d])).float()


def lda_fit(Xfit, y, shrink=SHRINK):
    """Shrinkage LDA by the generalised eigenproblem S_b v = lam S_w' v with S_w' = (1-s) S_w + s tr(S_w)/p I.  y: int labels (>=0), classes with >= 2 wells only.
    Returns (mu, W) with W (p, C-1) ordered by eigenvalue and W^T S_w' W = I (unit regularised within-class variance)."""
    X = Xfit.double()
    y = torch.as_tensor(y, device=DEV)
    cnt = torch.bincount(y)
    keep = cnt[y] >= 2
    X, y = X[keep], y[keep]
    ids, y = torch.unique(y, return_inverse=True)
    C = len(ids)
    n, p = X.shape
    cnt = torch.bincount(y, minlength=C).double()
    M = torch.zeros(C, p, dtype=torch.float64, device=DEV).index_add_(0, y, X) / cnt[:, None]
    Xc = X - M[y]
    Sw = Xc.T @ Xc / max(n - C, 1)
    mu = X.mean(0, keepdim=True)
    Md = M - mu
    Sb = (Md * cnt[:, None]).T @ Md / n
    Swr = (1 - shrink) * Sw + shrink * (torch.trace(Sw) / p) * torch.eye(p, dtype=torch.float64, device=DEV)
    w, V = torch.linalg.eigh(Swr)
    Wm = (V / torch.sqrt(w)) @ V.T
    lam, U = torch.linalg.eigh(Wm @ Sb @ Wm)
    o = torch.argsort(lam, descending=True)[: min(C - 1, p)]
    return mu, (Wm @ U[:, o])


def lda_apply(X, mu, W, d):
    return ((X.double() - mu) @ W[:, :d]).float()


def sphering_fit(Xdmso, eps_list):
    """pycytominer Spherize(method='ZCA-cor') as in the benchmark repo: standardise, SVD, singular values + epsilon, ZCA rotation.  Returns a function eps-index -> (mean, std, W)."""
    X = Xdmso.double()
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True, unbiased=False)
    sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
    Z = (X - mu) / sd
    n = Z.shape[0]
    lam, V = torch.linalg.eigh(Z.T @ Z / (n - 1))
    sing = torch.sqrt(torch.clamp(lam, min=0) * (n - 1))
    out = []
    for e in eps_list:
        W = (V * (np.sqrt(n - 1) / (sing + e))) @ V.T
        out.append((mu, sd, W))
    return out


def sphering_apply(X, par):
    mu, sd, W = par
    return (((X.double() - mu) / sd) @ W).float()


def log_uniform_eps(size=25, lo=-5, hi=3, seed=(6, 12, 2022)):
    rng = np.random.default_rng(list(seed))
    return 10.0 ** rng.uniform(lo, hi, size=size)


DPCA_D = (16, 64)
DITHER_R = (0.25, 0.5, 1, 2, 4)


class ArmFactory:
    """Builds the pre-registered inductive floors lazily.  Xfit: (nf, D) float32 tensor of Fit wells;  fitlab: dict with 'lda_lab' (int labels, -1 = none)
    for the Target2/Fit-class LDA and 'ldaP_lab' (production compound labels, -1 = none) or None;  s5: bool."""

    def __init__(self, Xfit, lab_class, lab_prod, salt, s5):
        self.Xfit = Xfit
        self.D = Xfit.shape[1]
        self.salt = salt
        self.s5 = s5
        self.pca = PCA(Xfit)
        self.mu_z = Xfit.mean(0, keepdim=True)
        self.sd_z = Xfit.std(0, keepdim=True)
        self.sd_z = torch.where(self.sd_z < 1e-6, torch.ones_like(self.sd_z), self.sd_z)
        self.lab_class = lab_class
        self.lab_prod = lab_prod
        self._lda = {}
        self._ldaP = {}
        self.lda_label_name = "ldaT2" if s5 else "ldaT"

    # --- LDA caches
    def lda_class(self, shrink):
        if shrink not in self._lda:
            m = self.lab_class >= 0
            self._lda[shrink] = lda_fit(self.Xfit[torch.as_tensor(m, device=DEV)], self.lab_class[m], shrink)
        return self._lda[shrink]

    def ldaP(self, shrink):
        if shrink not in self._ldaP:
            m = torch.as_tensor(self.lab_prod >= 0, device=DEV)
            Z = self.pca(self.Xfit[m], 256)
            mu, W = lda_fit(Z, self.lab_prod[self.lab_prod >= 0], shrink)
            self._ldaP[shrink] = (mu, W)
        return self._ldaP[shrink]

    def names(self):
        """(name, family) in evaluation order; family in floor / labelonly / dither / ctrl."""
        L = [("raw", "floor")]
        L += [(f"pca{d}", "floor") for d in (16, 32, 64, 128, 301)]
        L += [(f"rp{d}", "floor") for d in (16, 64, 301)]
        lab = self.lda_label_name
        L += [(f"{lab}_{d}", "labelonly") for d in (16, 64, "max")]
        L += [(f"{lab}_max_sh{s:g}", "labelonly") for s in (0.01, 1)]
        if self.s5:
            L += [(f"ldaP_{d}", "labelonly") for d in (16, 64, 255)]
            L += [(f"ldaP_255_sh{s:g}", "labelonly") for s in (0.01, 1)]
        L += [(f"dpca{d}_r{r:g}", "dither") for d in DPCA_D for r in DITHER_R]
        L += [(f"noise_r{r:g}", "dither") for r in (0.5, 1, 2)]
        L += [(f"{lab}_max_r{r:g}", "dither") for r in (0.5, 1, 2)]
        if self.s5:
            L += [(f"ldaP_255_r{r:g}", "dither") for r in (0.5, 1, 2)]
        L += [("ctrl_noise16", "ctrl")]
        return L

    def _sd_fit(self, f):
        s = f(self.Xfit).std(0, keepdim=True)
        return torch.where(s < 1e-6, torch.ones_like(s), s)

    def build(self, name, X):
        """X: (m, D) float32 tensor on DEV -> (m, d) float32 tensor."""
        rng = torch.Generator(device=DEV).manual_seed(seed_of(self.salt, name))

        def noisy(f, r):
            sd = self._sd_fit(f)
            Y = f(X)
            return Y + r * sd * torch.randn(Y.shape, generator=rng, device=DEV)
        lab = self.lda_label_name
        if name == "raw":
            return X
        if name.startswith("pca"):
            return self.pca(X, int(name[3:]))
        if name.startswith("rp"):
            d = int(name[2:])
            R = torch.randn(self.D, d, generator=torch.Generator(device=DEV).manual_seed(seed_of(self.salt, "rpmat", d)), device=DEV) / np.sqrt(self.D)
            return ((X - self.mu_z) / self.sd_z) @ R
        if name.startswith("dpca"):
            d, r = name[4:].split("_r")
            d, r = int(d), float(r)
            return noisy(lambda Z: self.pca(Z, d), r)
        if name.startswith("noise_r"):
            return noisy(lambda Z: Z, float(name[7:]))
        if name.startswith(lab + "_"):
            rest = name[len(lab) + 1:]
            shrink = SHRINK
            r = None
            if "_sh" in rest:
                rest, s = rest.split("_sh")
                shrink = float(s)
            if "_r" in rest:
                rest, rr = rest.split("_r")
                r = float(rr)
            mu, W = self.lda_class(shrink)
            d = W.shape[1] if rest == "max" else min(int(rest), W.shape[1])
            f = lambda Z: lda_apply(Z, mu, W, d)  # noqa: E731
            return noisy(f, r) if r is not None else f(X)
        if name.startswith("ldaP_"):
            rest = name[5:]
            shrink = SHRINK
            r = None
            if "_sh" in rest:
                rest, s = rest.split("_sh")
                shrink = float(s)
            if "_r" in rest:
                rest, rr = rest.split("_r")
                r = float(rr)
            mu, W = self.ldaP(shrink)
            d = min(int(rest), W.shape[1])
            f = lambda Z: lda_apply(self.pca(Z, 256), mu, W, d)  # noqa: E731
            return noisy(f, r) if r is not None else f(X)
        raise ValueError(name)
