"""Distractors for the B' kill test.  Image-space sprites composited over the clean MuJoCo render.

Everything is a deterministic function of (kind, actions, seed): the clean frame, the distractor-only layer
(alpha + colour) and every nuisance label are exact.  Pixel units are 128x128 render pixels, one step = one
macro step (4 env steps).

kinds
  none        no distractor (oracle clean frames)
  indep       K sprites, AR(1) random-walk velocity independent of the action          (the case SAM masks solve)
  corr_det    sprite velocity = R_k a_xyz exactly (deterministic function of the action)  (action-correlated, no residual)
  corr_noisy  sprite velocity = R_k a_xyz + sigma * eps                                   (action-correlated + own noise)
Sprite 0 mimics the arm (R_0 = fitted EE-pixel response), sprite 1 mirrors it, sprite 2 uses a random fixed map.
Sprites start near the end effector, are opaque discs drawn over the arm, and half of the palette is arm-like
(red / grey / dark), so a segmenter cannot separate them by colour alone.
"""
import numpy as np
import torch

KINDS = ["none", "indep", "corr_det", "corr_noisy"]
K_SPR = 3
RADIUS = 7.0
SIGMA_CORR = 1.5       # px / macro step, independent part of corr_noisy
SPREAD0 = 22.0         # px, initial scatter around the end effector
PALETTE = torch.tensor([[0.75, 0.12, 0.12], [0.10, 0.70, 0.20], [0.15, 0.25, 0.85],
                        [0.90, 0.85, 0.10], [0.62, 0.62, 0.64], [0.12, 0.12, 0.14]])


def fit_ee_response(acts, eepix):
    """least squares  d eepix[t] = M a_xyz[t]  -> M (2,3) in px per unit action."""
    d = (eepix[:, 1:] - eepix[:, :-1]).reshape(-1, 2)
    a = acts[..., :3].reshape(-1, 3)
    M, *_ = np.linalg.lstsq(a, d, rcond=None)
    return M.T  # (2,3)


def make_maps(M_ee, seed=4242):
    rng = np.random.default_rng(seed)
    R2 = rng.normal(size=(2, 3))
    R2 *= np.linalg.norm(M_ee) / np.linalg.norm(R2)
    return np.stack([M_ee, -M_ee, R2])  # (K,2,3)


def sample(kind, acts, eepix0, maps, seed, indep_std=None, W=128):
    """acts (N,T,4), eepix0 (N,2).  Returns dict(pos (N,T+1,K,2), cls (N,K), vres (N,T,K,2), vtot (N,T,K,2))."""
    N, T = acts.shape[:2]
    K = K_SPR
    if kind == "none":
        return dict(pos=np.zeros((N, T + 1, K, 2), np.float32) - 100.0, cls=np.zeros((N, K), np.int64),
                    vres=np.zeros((N, T, K, 2), np.float32), vtot=np.zeros((N, T, K, 2), np.float32))
    rng = np.random.default_rng(seed)
    pos0 = eepix0[:, None, :] + rng.normal(0, SPREAD0, (N, K, 2))
    cls = np.stack([rng.permutation(6)[:K] for _ in range(N)])
    eps = rng.normal(0, 1, (N, T, K, 2))
    a3 = acts[..., :3]
    drive = np.einsum("kij,ntj->ntki", maps, a3)  # (N,T,K,2) deterministic action response
    if kind == "corr_det":
        vtot = drive; vres = np.zeros_like(drive)
    elif kind == "corr_noisy":
        vres = SIGMA_CORR * eps; vtot = drive + vres
    elif kind == "indep":
        s = (indep_std if indep_std is not None else 3.0) * np.sqrt(1 - 0.6 ** 2)
        v = np.zeros((N, T, K, 2)); prev = np.zeros((N, K, 2))
        for t in range(T):
            prev = 0.6 * prev + s * eps[:, t]; v[:, t] = prev
        vtot = v; vres = v
    else:
        raise ValueError(kind)
    pos = np.zeros((N, T + 1, K, 2)); pos[:, 0] = np.clip(pos0, RADIUS, W - RADIUS)
    for t in range(T):  # reflecting walls
        nxt = pos[:, t] + vtot[:, t]
        lo, hi = RADIUS, W - RADIUS
        nxt = np.where(nxt < lo, 2 * lo - nxt, nxt); nxt = np.where(nxt > hi, 2 * hi - nxt, nxt)
        pos[:, t + 1] = np.clip(nxt, lo, hi)
    return dict(pos=pos.astype(np.float32), cls=cls.astype(np.int64), vres=vres.astype(np.float32), vtot=vtot.astype(np.float32))


def composite(clean, pos, cls, kind_none=None):
    """clean (B,3,H,W) float [0,1]; pos (B,K,2) px; cls (B,K) long.  Returns (img, alpha (B,1,H,W)).
    The distractor-only layer is  layer = composite(zeros-ish) ; here alpha and the painted colour are exact."""
    B, _, H, W = clean.shape
    dev = clean.device
    ys = torch.arange(H, device=dev, dtype=torch.float32)[None, :, None]
    xs = torch.arange(W, device=dev, dtype=torch.float32)[None, None, :]
    pal = PALETTE.to(dev)
    out = clean
    alpha = torch.zeros(B, 1, H, W, device=dev)
    for k in range(pos.shape[1]):
        d = torch.sqrt((xs - pos[:, k, 0, None, None]) ** 2 + (ys - pos[:, k, 1, None, None]) ** 2)
        a = (RADIUS + 0.5 - d).clamp(0, 1)[:, None]
        col = pal[cls[:, k]][:, :, None, None]
        out = out * (1 - a) + col * a
        alpha = alpha + a * (1 - alpha)
    return out, alpha
