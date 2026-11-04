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
