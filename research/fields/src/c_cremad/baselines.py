"""Fixed (non-learned) representations: raw uniform-layer mix, pooling/smoothing, PCA, LEACE.

All operate on X0 = frame-level, per-layer-LayerNorm'ed, uniform average of the cached layers (a Ragged [Nfr,768]).
Fits (PCA, LEACE) use only the clips/actors passed in; nothing here touches test actors.
"""
import numpy as np
import torch
import torch.nn.functional as F
from lib import Ragged, Rep, DEV


def build_x0(frames_np, offsets, chunk=200_000):
    outs = []
    for s in range(0, frames_np.shape[0], chunk):
        x = torch.from_numpy(np.ascontiguousarray(frames_np[s:s + chunk])).to(DEV).float()  # [n,L,D]
        x = F.layer_norm(x, (x.shape[-1],))
        outs.append(x.mean(1).half())
    return Ragged(torch.cat(outs, 0), offsets)


def clip_frame_index(rag, clips):
    """Concatenated frame indices of the given clips (np array of clip ids)."""
    off = rag.off.cpu().numpy()
    return np.concatenate([np.arange(off[c], off[c + 1]) for c in clips])


def rep_from_ragged(rag, name):
    return Rep(rag.pooled_stats(), rag, name)


def mean_rep(rag):
    p = rag.pooled_stats()
    d = p.shape[1] // 2
    return Rep(p[:, :d], None, "mean")


def smooth(rag, P):
    """Average-pool non-overlapping windows of P frames inside each clip (rate reduction in time)."""
    off = rag.off
    lens = rag.len
    nseg = (lens + P - 1) // P
    new_off = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), nseg.cumsum(0)])
    clip_of_frame = torch.repeat_interleave(torch.arange(rag.n, device=DEV), lens)
    pos_in_clip = torch.arange(rag.flat.shape[0], device=DEV) - off[clip_of_frame]
    seg = new_off[clip_of_frame] + pos_in_clip // P
    out = torch.zeros(int(new_off[-1]), rag.flat.shape[1], device=DEV)
    out.index_add_(0, seg, rag.flat.float())
    cnt = torch.zeros(int(new_off[-1]), device=DEV).index_add_(0, seg, torch.ones_like(seg, dtype=torch.float))
    return Ragged((out / cnt[:, None]).half(), new_off.cpu().numpy())


def affine(rag, A, b):
    """frames -> frames @ A^T + b (chunked)."""
    outs = []
    for s in range(0, rag.flat.shape[0], 200_000):
        x = rag.flat[s:s + 200_000].float()
        outs.append((x @ A.T + b).half())
    return Ragged(torch.cat(outs, 0), rag.off.cpu().numpy())


def _moments(rag, frame_idx, labels_per_frame, ncls):
    d = rag.flat.shape[1]
    n = 0
    sx = torch.zeros(d, dtype=torch.float64, device=DEV)
    sxx = torch.zeros(d, d, dtype=torch.float64, device=DEV)
    sc = torch.zeros(ncls, d, dtype=torch.float64, device=DEV)
    cnt = torch.zeros(ncls, dtype=torch.float64, device=DEV)
    for s in range(0, len(frame_idx), 100_000):
        fi = torch.as_tensor(frame_idx[s:s + 100_000], device=DEV)
        x = rag.flat[fi].double()
        lab = torch.as_tensor(labels_per_frame[s:s + 100_000], device=DEV)
        n += len(fi); sx += x.sum(0); sxx += x.T @ x
        sc.index_add_(0, lab, x); cnt.index_add_(0, lab, torch.ones(len(fi), dtype=torch.float64, device=DEV))
    return n, sx, sxx, sc, cnt


def fit_leace(rag, clips, clip_labels, ncls, eps=1e-4):
    """Closed-form LEACE eraser for a categorical concept (Belrose et al. 2023) fit on the given clips.
    Returns (A, b) with eraser(x) = x A^T + b.  Concept labels are per clip (e.g. actor id)."""
    off = rag.off.cpu().numpy()
    lens = off[clips + 1] - off[clips]
    fidx = clip_frame_index(rag, clips)
    lab = np.repeat(clip_labels, lens)
    n, sx, sxx, sc, cnt = _moments(rag, fidx, lab, ncls)
    mu = sx / n
    cov = sxx / n - torch.outer(mu, mu)
    cov = cov + eps * torch.trace(cov) / cov.shape[0] * torch.eye(cov.shape[0], dtype=torch.float64, device=DEV)
    evals, U = torch.linalg.eigh(cov)
    W = U @ torch.diag(evals.rsqrt()) @ U.T
    Wp = U @ torch.diag(evals.sqrt()) @ U.T
    sxz = (sc / n).T - torch.outer(mu, cnt / n)  # [d, C]
    M = W @ sxz
    Um, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = S > 1e-6 * S.max()
    Um = Um[:, keep]
    B = Wp @ Um @ Um.T @ W  # x' = x - B (x - mu)
    d = B.shape[0]
    A = (torch.eye(d, dtype=torch.float64, device=DEV) - B).float()
    b = (B @ mu).float()
    return A, b, int(keep.sum())


def fit_pca(rag, clips, k):
    fidx = clip_frame_index(rag, clips)
    d = rag.flat.shape[1]
    n, sx, sxx, _, _ = _moments(rag, fidx, np.zeros(len(fidx), dtype=np.int64), 1)
    mu = sx / n
    cov = sxx / n - torch.outer(mu, mu)
    evals, U = torch.linalg.eigh(cov)
    V = U[:, -k:].flip(1)  # top-k
    A = V.T.float()  # [k,d]
    b = -(A.double() @ mu).float()
    return A, b


def trim_ragged(rag, spans, stride=320, min_frames=10):
    """Keep only frames inside the VAD speech span (sample units) of each clip; clips without a span keep everything.
    Works for [Nfr,D] and [Nfr,L,D] flats (SSL frames are 20 ms = 320 samples)."""
    off = rag.off.cpu().numpy()
    keep, lens = [], []
    n_trim = 0
    for c in range(rag.n):
        a, b_ = off[c], off[c + 1]
        s, t = spans[c]
        if s >= 0:
            fs, fe = a + int(s) // stride, min(b_, a + -(-int(t) // stride))
            if fe - fs >= min_frames:
                keep.append(np.arange(fs, fe)); lens.append(fe - fs); n_trim += 1
                continue
        keep.append(np.arange(a, b_)); lens.append(b_ - a)
    idx = torch.as_tensor(np.concatenate(keep), device=rag.flat.device)
    print(f"trim: {n_trim}/{rag.n} clips trimmed, frames {rag.flat.shape[0]} -> {len(idx)}", flush=True)
    return Ragged(rag.flat[idx], np.concatenate([[0], np.cumsum(lens)]), rag.dev)


def fit_lda_proj(P, y, tr_idx, val_idx, k, ncls_present, alphas=(0.05, 0.2, 0.5)):
    """Label-supervised linear projection (shrinkage LDA) of pooled features P [N,D] to k dims.
    LDA has at most C-1 informative directions (C = emotions present in y[tr_idx]); extra dims (k > C-1) are filled with the top
    principal directions of the residual after projecting the LDA subspace out.  Shrinkage chosen on val_idx by nearest-class-mean
    UAR in the projected space.  Returns (mu, sd, W [D,k])."""
    P = P.double()
    Xtr = P[tr_idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    X = (P - mu) / sd
    ytr = torch.as_tensor(y[tr_idx], device=P.device)
    classes = torch.unique(ytr)
    C = len(classes)
    Xt = X[tr_idx]
    m_c = torch.stack([Xt[ytr == c].mean(0) for c in classes])
    n_c = torch.stack([(ytr == c).sum() for c in classes]).double()
    m = Xt.mean(0)
    R = Xt - m_c[torch.searchsorted(classes, ytr)]
    Sw0 = R.T @ R / len(ytr)
    Sb = ((m_c - m).T * n_c) @ (m_c - m) / len(ytr)
    best = None
    for a in alphas:
        Sw = (1 - a) * Sw0 + a * (torch.trace(Sw0) / Sw0.shape[0]) * torch.eye(Sw0.shape[0], dtype=torch.float64, device=P.device)
        ev, U = torch.linalg.eigh(Sw)
        Wh = U @ torch.diag(ev.rsqrt()) @ U.T
        e2, V = torch.linalg.eigh(Wh @ Sb @ Wh)
        order = torch.argsort(e2, descending=True)[: C - 1]
        W = Wh @ V[:, order]  # [D, C-1]
        W = W / W.norm(dim=0, keepdim=True)
        if k > C - 1:  # residual PCA directions
            Q, _ = torch.linalg.qr(W)
            Xr = Xt - (Xt @ Q) @ Q.T
            cov = Xr.T @ Xr / len(Xr)
            e3, U3 = torch.linalg.eigh(cov)
            W = torch.cat([W, U3[:, -(k - (C - 1)):].flip(1)], 1)
        elif k < C - 1:
            W = W[:, :k]
        Z = X @ W
        Zc = torch.stack([Z[tr_idx][ytr == c].mean(0) for c in classes])
        yv = torch.as_tensor(y[val_idx], device=P.device)
        keep = torch.isin(yv, classes)
        d = torch.cdist(Z[val_idx][keep], Zc)
        pred = classes[d.argmin(1)]
        rec = [float((pred[yv[keep] == c] == c).double().mean()) for c in classes]
        sc = sum(rec) / len(rec)
        if best is None or sc > best[0]:
            best = (sc, a, W)
    return mu.float(), sd.float(), best[2].float(), best[1], best[0]
