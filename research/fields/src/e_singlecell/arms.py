"""Encoder 0 and the floor / reference arms of SC-0.  Everything is fitted on the fold's TRAINING cells (mask T) and applied to all cells;
only the T_* references use batch ids of held-out cells (transductive).
"""
import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------------------------------------------ encoder 0
def encoder0(X, T, n_hvg=2000, n_pc=50):
    """X: csr log-normalised (cells x genes).  HVG (seurat, no batch key) on the training cells, standardise (train stats, clip +-10), PCA-50.
    Returns z0 (n_cells, n_pc) float32 and the HVG index."""
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import PCA
    a = ad.AnnData(X[T])
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_hvg)
    hv = np.where(a.var["highly_variable"].to_numpy())[0]
    Xh = X[:, hv].toarray().astype(np.float32)
    mu = Xh[T].mean(0)
    sd = Xh[T].std(0)
    sd[sd < 1e-6] = 1.0
    Xs = np.clip((Xh - mu) / sd, -10, 10)
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0).fit(Xs[T])
    return pca.transform(Xs).astype(np.float32), hv


# ------------------------------------------------------------------------------------------------ helpers
def _pca_fit(Z, T, k):
    mu = Z[T].mean(0)
    U, S, Vt = np.linalg.svd(Z[T] - mu, full_matrices=False)
    return mu, Vt[:k].T


def lda_fit(Z, T, y, max_dims=None):
    """LDA on rows T with labels y (-1 = no label); returns (project_fn, directions Q orthonormal in z-space)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    m = T & (y >= 0)
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(Z[m], y[m])
    k = min(len(lda.classes_) - 1, lda.scalings_.shape[1])     # svd solver keeps more columns than n_classes-1: only the first ones are discriminant
    if max_dims is not None:
        k = min(k, max_dims)
    Wd = lda.scalings_[:, :k]
    mu = lda.xbar_
    Q, _ = np.linalg.qr(Wd)
    return (lambda Zin: ((Zin - mu) @ Wd).astype(np.float32)), Q


def lda_pad(Z, T, y, d, max_dims=None):
    """LDA dims padded with PCA of the LDA-orthogonal residual of z0 to d dims (d=None: LDA dims only)."""
    proj, Q = lda_fit(Z, T, y, max_dims)
    L = proj(Z)
    if d is None or L.shape[1] >= d:
        return L[:, :d] if d is not None else L
    R = Z - (Z @ Q) @ Q.T
    mu, V = _pca_fit(R, T, d - L.shape[1])
    return np.concatenate([L, ((R - mu) @ V).astype(np.float32)], 1)


def leace_fit(Z, T, bt, ridge=1e-3):
    """closed-form LEACE of the (training) batch id from z0: erase the linear predictability of the one-hot batch label."""
    X = Z[T].astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    n, d = Xc.shape
    B = int(bt.max()) + 1
    Y = np.eye(B)[bt]
    Y -= Y.mean(0)
    S = Xc.T @ Xc / n
    S += ridge * np.trace(S) / d * np.eye(d)
    w, V = np.linalg.eigh(S)
    Wm = (V / np.sqrt(w)) @ V.T          # S^-1/2
    Wi = (V * np.sqrt(w)) @ V.T          # S^1/2
    C = Xc.T @ Y / n
    U, sv, _ = np.linalg.svd(Wm @ C, full_matrices=False)
    U = U[:, sv > 1e-6 * sv.max()]
    P = Wi @ U @ U.T @ Wm                # erased part of (x - mu)
    return lambda Zin: ((Zin - mu) - (Zin - mu) @ P.T + mu).astype(np.float32)


def per_batch(Z, b, fn):
    out = np.empty_like(Z)
    for k in np.unique(b):
        m = b == k
        out[m] = fn(Z[m])
    return out


def whiten(Zb):
    mu = Zb.mean(0)
    C = np.cov((Zb - mu).T) + 1e-3 * np.eye(Zb.shape[1])
    w, V = np.linalg.eigh(C)
    return ((Zb - mu) @ ((V / np.sqrt(w)) @ V.T)).astype(np.float32)


def harmony(Z, b, seed=0):
    import harmonypy
    import pandas as pd
    ho = harmonypy.run_harmony(Z, pd.DataFrame({"batch": b.astype(str)}), "batch", random_state=seed, verbose=False)
    Zc = np.asarray(ho.Z_corr)
    return (Zc.T if Zc.shape[0] == Z.shape[1] and Zc.shape[1] == Z.shape[0] else Zc).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the arm builder
ARMS_INDUCTIVE = ["raw50", "pca8", "pca16", "pca32", "rp16", "rp32", "noise_r0.5", "noise_r1", "noise_r2", "dith16_r0.5", "dith16_r1",
                  "ldaC", "ldaC_pca16", "ldaC_pca32", "ldaF16", "leace", "leace_pca16"]
ARMS_H = ["ldaC", "ldaC_pca16", "ldaC_pca32", "ldaF16"]            # also built with the held-out types hidden: name + "_H"
ARMS_REF = ["T_centre", "T_bwhiten16", "T_harmony", "T_scvi", "onehot_oracle"]
ARMS_CTRL = ["ctrl_const", "ctrl_noise16"]
# post-hoc dense floor frontier (sensitivity for SC-K3 only; NOT part of the pre-registered grid, run as separate jobs)
DENSE = ([f"pca{d}" for d in (4, 6, 12, 24)] + [f"dith{d}_r{r:g}" for d in (8, 16, 32) for r in (0.25, 0.5, 1, 2, 4)] +
         ["ldaC_pca8", "ldaC_pca24", "ldaC_pca8_H", "ldaC_pca24_H"])


def build_arms(Z, T, sc, seed, fold_batches, scvi=None, only=None):
    """Z: z0 (n,50) of all cells.  T: bool training mask.  sc: SC object.  Returns {arm: float32 matrix (n,d)}."""
    n = Z.shape[0]
    out = {}
    want = (lambda a: True) if only is None else (lambda a: a in only)
    sd = Z[T].std(0)
    zs = (Z - Z[T].mean(0)) / sd
    out["raw50"] = Z
    for d in (8, 16, 32):
        out[f"pca{d}"] = Z[:, :d]
    for d in (16, 32):
        R = np.random.RandomState(1000 + seed + d).randn(Z.shape[1], d).astype(np.float32) / np.sqrt(Z.shape[1])
        out[f"rp{d}"] = (zs @ R).astype(np.float32)
    for r in (0.5, 1, 2):
        out[f"noise_r{r:g}"] = (Z + r * sd * np.random.RandomState(2000 + seed + int(10 * r)).randn(*Z.shape)).astype(np.float32)
    for r in (0.5, 1):
        out[f"dith16_r{r:g}"] = (Z[:, :16] + r * sd[:16] * np.random.RandomState(3000 + seed + int(10 * r)).randn(n, 16)).astype(np.float32)
    # label-supervised floors (full fit, then the held-out-type versions)
    yc, yf = sc.coarse, sc.fine
    hide = np.isin(yf, sc.held_codes)
    for suffix, Tm_c, Tm_f in (("", T, T), ("_H", T & ~hide, T & ~hide)):
        ycl = np.where(Tm_c, yc, -1)
        yfl = np.where(Tm_f, yf, -1)
        out["ldaC" + suffix] = lda_pad(Z, Tm_c, ycl, None)
        out["ldaC_pca16" + suffix] = lda_pad(Z, Tm_c, ycl, 16)
        out["ldaC_pca32" + suffix] = lda_pad(Z, Tm_c, ycl, 32)
        out["ldaF16" + suffix] = lda_pad(Z, Tm_f, yfl, None, max_dims=16)
    f = leace_fit(Z, T, np.unique(sc.b[T], return_inverse=True)[1])
    E = f(Z)
    out["leace"] = E
    mu, V = _pca_fit(E, T, 16)
    out["leace_pca16"] = ((E - mu) @ V).astype(np.float32)
    # transductive references
    out["T_centre"] = per_batch(Z, sc.b, lambda x: x - x.mean(0))
    out["T_bwhiten16"] = per_batch(Z[:, :16], sc.b, whiten)
    if only is None or "T_harmony" in only:
        out["T_harmony"] = harmony(Z, sc.b, seed)
    if scvi is not None:
        out["T_scvi"] = scvi
    oh = np.zeros((n, len(sc.fine_names) + 1), np.float32)
    oh[np.arange(n), np.where(yf >= 0, yf, len(sc.fine_names))] = 1.0
    out["onehot_oracle"] = oh
    out["ctrl_const"] = np.ones((n, 1), np.float32)
    out["ctrl_noise16"] = np.random.RandomState(4000 + seed).randn(n, 16).astype(np.float32)
    if only is not None and any(x in DENSE for x in only):
        for d in (4, 6, 12, 24):
            out[f"pca{d}"] = Z[:, :d]
        for d in (8, 16, 32):
            for r in (0.25, 0.5, 1, 2, 4):
                out[f"dith{d}_r{r:g}"] = (Z[:, :d] + r * sd[:d] * np.random.RandomState(5000 + seed + 100 * d + int(100 * r)).randn(n, d)).astype(np.float32)
        for suffix, Tm in (("", T), ("_H", T & ~hide)):
            ycl = np.where(Tm, yc, -1)
            out["ldaC_pca8" + suffix] = lda_pad(Z, Tm, ycl, 8)
            out["ldaC_pca24" + suffix] = lda_pad(Z, Tm, ycl, 24)
    if only is not None:
        out = {k: v for k, v in out.items() if k in only}
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}
