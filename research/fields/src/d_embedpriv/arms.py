"""Released-vector constructions ("arms") for Focus D.  Every arm maps RAW frozen-embedder vectors (unit norm) to RELEASED vectors
(unit norm after the map, except the additive-noise arm, which is renormalised after the noise).  The same deterministic map is applied to
documents and queries; additive noise is applied to stored DOCUMENT vectors only (favourable to the noise floor: queries stay clean).

Floors (no learning of a privacy objective):
  raw        identity
  randproj   Gaussian random projection to d dims
  pca        PCA-d fitted on the defender pool (NFCorpus D docs + SciFact ED docs), centred
  noise      doc <- normalise(doc + r/sqrt(768) * N(0, I)),  r = noise-to-signal norm ratio
  leace      closed-form linear erasure (LEACE, Belrose et al. 2023) of the span of the top-r SVD directions of the secret-token bag
             (fitted on the defender pool; concept = bag of rare+mid words)
  const      all-ones vector (control: attacker must fall back to the frequency prior)
Learned arms live in train.py (sup_lin, bott_mlp, adv).
"""
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd


def _norm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


class Arm:
    stochastic = False
    name = "arm"

    def fit(self, ctx, seed):
        return self

    def docs(self, X, rng):
        return X

    def queries(self, X):
        return X


class Raw(Arm):
    name = "raw"


class Const(Arm):
    name = "const"

    def docs(self, X, rng):
        return np.ones((len(X), 4), np.float32)

    def queries(self, X):
        return np.ones((len(X), 4), np.float32)


class RandProj(Arm):
    def __init__(self, d):
        self.d = d
        self.name = f"randproj_d{d}"

    def fit(self, ctx, seed):
        self.R = np.random.default_rng(1000 + seed).standard_normal((ctx.X_nf.shape[1], self.d)).astype(np.float32) / np.sqrt(self.d)
        return self

    def docs(self, X, rng):
        return _norm(X @ self.R)

    def queries(self, X):
        return _norm(X @ self.R)


class PCA(Arm):
    def __init__(self, d):
        self.d = d
        self.name = f"pca_d{d}"

    def fit(self, ctx, seed):
        Xd = ctx.defender_docs()
        self.mu = Xd.mean(0)
        _, _, Vt = np.linalg.svd(Xd - self.mu, full_matrices=False)
        self.P = Vt[: self.d].T.astype(np.float32)
        return self

    def docs(self, X, rng):
        return _norm((X - self.mu) @ self.P)

    def queries(self, X):
        return _norm((X - self.mu) @ self.P)


class Noise(Arm):
    stochastic = True

    def __init__(self, r):
        self.r = r
        self.name = f"noise_r{r:g}"

    def docs(self, X, rng):
        return _norm(X + (self.r / np.sqrt(X.shape[1])) * rng.standard_normal(X.shape).astype(np.float32))


class Leace(Arm):
    def __init__(self, r, rho=1e-2):
        self.r, self.rho = r, rho
        self.name = f"leace_r{r}"

    def fit(self, ctx, seed):
        X = ctx.defender_docs().astype(np.float64)
        Yd = ctx.defender_bags().toarray().astype(np.float32)  # 3.6k x 14.9k dense, fine
        n, d = X.shape
        mu = X.mean(0)
        Xc = X - mu
        Yd -= Yd.mean(0, keepdims=True)
        U, S, _ = randomized_svd(Yd, n_components=self.r, random_state=seed, n_iter=4)
        C = U * S
        C = C - C.mean(0)
        Sig = Xc.T @ Xc / n
        Sig += self.rho * np.trace(Sig) / d * np.eye(d)
        ev, evec = np.linalg.eigh(Sig)
        ev = np.maximum(ev, 1e-12)
        Sh = (evec * np.sqrt(ev)) @ evec.T
        Smh = (evec / np.sqrt(ev)) @ evec.T
        Sxc = Xc.T @ C / n
        M = Smh @ Sxc
        U2, s2, _ = np.linalg.svd(M, full_matrices=False)
        U2 = U2[:, s2 > 1e-8 * s2.max()]
        E = np.eye(d) - Sh @ (U2 @ U2.T) @ Smh
        self.mu, self.E = mu.astype(np.float32), E.T.astype(np.float32)
        self.rank = int(U2.shape[1])
        return self

    def docs(self, X, rng):
        return _norm((X - self.mu) @ self.E + self.mu)

    def queries(self, X):
        return self.docs(X, None)


def make_arm(spec):
    f = spec["family"]
    if f == "raw":
        return Raw()
    if f == "const":
        return Const()
    if f == "randproj":
        return RandProj(spec["d"])
    if f == "pca":
        return PCA(spec["d"])
    if f == "noise":
        return Noise(spec["r"])
    if f == "leace":
        return Leace(spec["r"], spec.get("rho", 1e-2))
    from train import make_learned
    return make_learned(spec)
