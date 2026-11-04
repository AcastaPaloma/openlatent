"""Releases = deterministic image-to-image maps applied to the 256x256 uint8 bank.  Floors first (T.md section 4), learned defences after.

A release is a pure function of (image id, release name): every random draw is seeded from SHA-256(id:name), so a defended image is
a fixed 'published' object (the attacker sees ONE defended copy per image, not fresh noise each epoch) and re-runs are byte-stable.
Output dtype is float16 in [0,1] (8-bit PNG fidelity is exceeded; the PriCheXy float path is preserved).

Name grammar (parameters after the underscore):
  raw | const | warp0                                   undefended / all-black control / PriCheXy pipeline with a ZERO flow (border artefact only)
  rw1_mu0.01   random flow, i.i.d. U(-1,1) per pixel, PriCheXy smoothing (the mechanism without learning; RMS tiny after smoothing)
  rw2_mu0.01   random SMOOTH flow (corr. length ~12 px) with sup-norm of the raw flow = 1, i.e. the same hard bound mu as the paper
  rw3_mu0.01   random flow with the radially averaged power spectrum and RMS distribution MEASURED on the learned generator at that mu
  blur_s2 | down_f4 | noise_s0.1 | quant_l8 | quantd_l8 (dithered) | jpeg_q10 | pca_k32 | dppix_b8
  rim_w24 | core_w24   keep ONLY the 24-px frame / ZERO the frame (acquisition-fingerprint controls; T.md stage 1)
  pcx_mu0.01   PriCheXy-Net released generator (learned)
"""
from __future__ import annotations

import io
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import generator as G
from .meta import unit


def _rng(img_id: str, name: str) -> np.random.Generator:
    return np.random.default_rng(int(unit(f"{img_id}:{name}", "t_cxr:release") * 2**32))


def _tg(img_id: str, name: str) -> torch.Generator:
    g = torch.Generator(); g.manual_seed(int(unit(f"{img_id}:{name}", "t_cxr:release") * 2**31)); return g


def _gblur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-ax ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def _random_smooth_flow(ids, name, n=256, corr_px=12.0) -> torch.Tensor:
    """(B,2,n,n) with sup-norm exactly 1 per image: Gaussian white noise smoothed with sigma=corr_px, divided by its max."""
    out = []
    for i in ids:
        w = torch.randn(1, 2, n, n, generator=_tg(i, name))
        s = _gblur(w.reshape(2, 1, n, n), corr_px).reshape(1, 2, n, n)
        out.append(s / s.abs().max())
    return torch.cat(out)


class Releases:
    """Factory.  `ctx` carries things fitted on TRAIN images only: pca basis, learned-generator flow statistics, generator weights."""

    def __init__(self, device="cpu", gen_dir: str | None = None, flow_stats: dict | None = None, pca: dict | None = None):
        self.dev, self.gen_dir, self.flow_stats, self.pca = device, gen_dir, flow_stats or {}, pca or {}
        self._gens: dict = {}

    # ------------------------------------------------------------------ one release on one batch
    def apply(self, name: str, x: torch.Tensor, ids: list[str]) -> torch.Tensor:
        """x: (B,n,n) uint8 or float in [0,1] on any device -> (B,n,n) float16 on self.dev."""
        x = (x.float() / 255.0 if x.dtype == torch.uint8 else x.float()).to(self.dev).unsqueeze(1)
        kind, _, par = name.partition("_")
        if name == "raw":
            y = x
        elif name == "const":
            y = torch.zeros_like(x)
        elif name == "warp0":
            y = G.warp(x, torch.zeros(x.shape[0], 2, *x.shape[-2:], device=self.dev), 0.0)
        elif kind in ("rw1", "rw2", "rw3"):
            mu = float(par.removeprefix("mu"))
            if kind == "rw1":
                raw = torch.stack([(torch.rand(2, *x.shape[-2:], generator=_tg(i, name)) * 2 - 1) for i in ids]).to(self.dev)
            elif kind == "rw2":
                raw = _random_smooth_flow(ids, name, x.shape[-1]).to(self.dev)
            else:
                st = self.flow_stats[f"mu{mu:g}"]
                disp = []
                for i in ids:
                    g = _tg(i, name)
                    r = st["rms_px"][int(torch.randint(len(st["rms_px"]), (1,), generator=g))]
                    disp.append(G.spectrum_matched_field(st["radial_power"], r, g, x.shape[-1]))
                disp = torch.stack(disp).to(self.dev)
                # the field IS the applied displacement (already smoothed spectrum): resample directly, no extra smoothing
                idg = G.identity_grid(x.shape[-1], self.dev)
                grid = (idg - disp / G.PX_PER_UNIT).permute(0, 2, 3, 1)
                return F.grid_sample(x, grid, padding_mode="border", align_corners=True).squeeze(1).half()
            y = G.warp(x, raw, mu)
        elif kind == "blur":
            y = _gblur(x, float(par.removeprefix("s")))
        elif kind == "down":
            f = int(par.removeprefix("f"))
            y = F.interpolate(F.interpolate(x, scale_factor=1 / f, mode="area"), size=x.shape[-2:], mode="bilinear", align_corners=False)
        elif kind == "noise":
            s = float(par.removeprefix("s"))
            nz = torch.stack([torch.randn(*x.shape[-2:], generator=_tg(i, name)) for i in ids]).unsqueeze(1).to(self.dev)
            y = (x + s * nz).clamp(0, 1)
            y = torch.round(y * 255) / 255
        elif kind in ("quant", "quantd"):
            L = int(par.removeprefix("l"))
            if kind == "quantd":
                nz = torch.stack([torch.rand(*x.shape[-2:], generator=_tg(i, name)) - 0.5 for i in ids]).unsqueeze(1).to(self.dev)
                y = torch.round((x * (L - 1)) + nz).clamp(0, L - 1) / (L - 1)
            else:
                y = torch.round(x * (L - 1)) / (L - 1)
        elif kind == "jpeg":
            from PIL import Image
            q = int(par.removeprefix("q"))
            arr = (x.squeeze(1).cpu().numpy() * 255).round().astype(np.uint8)
            out = []
            for a in arr:
                b = io.BytesIO(); Image.fromarray(a, "L").save(b, "JPEG", quality=q); b.seek(0)
                out.append(np.asarray(Image.open(b).convert("L")))
            y = torch.from_numpy(np.stack(out)).float().div(255).unsqueeze(1).to(self.dev)
        elif kind == "pca":
            k = int(par.removeprefix("k"))
            mean, comp = self.pca["mean"].to(self.dev), self.pca["components"][:k].to(self.dev)          # (4096,), (k,4096) orthonormal rows
            s = self.pca["size"]
            z = F.interpolate(x, size=(s, s), mode="area").flatten(1) - mean
            rec = (z @ comp.T @ comp + mean).view(-1, 1, s, s)
            y = F.interpolate(rec, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
        elif kind == "dppix":
            b = int(par.removeprefix("b")); m, eps = 1.0, 0.1                                              # the paper's setting (m=1, eps=0.1)
            p = F.avg_pool2d(x, b)
            nz = torch.stack([_laplace(_tg(i, name), p.shape[-2:], m / (b * b * eps)) for i in ids]).unsqueeze(1).to(self.dev)
            y = F.interpolate((p + nz).clamp(0, 1), scale_factor=b, mode="nearest")
        elif kind in ("rim", "core"):                                                              # acquisition-fingerprint controls
            w = int(par.removeprefix("w"))
            m = torch.zeros_like(x); m[..., w:-w, w:-w] = 1
            y = x * (1 - m) if kind == "rim" else x * m
        elif kind == "pcx":
            mu = float(par.removeprefix("mu"))
            if mu not in self._gens:
                self._gens[mu] = G.load_generator(f"{self.gen_dir}/generator_lowest_total_loss_mu_{par.removeprefix('mu')}.pth", self.dev)
            with torch.no_grad():
                y = G.warp(x, self._gens[mu](x), mu)
        else:
            raise ValueError(name)
        return y.squeeze(1).half()

    def materialize(self, name: str, bank: np.ndarray, rows: np.ndarray, ids: list[str], bs: int = 128) -> torch.Tensor:
        """bank: (N,n,n) uint8 (memmap ok); rows: indices; returns (len(rows), n, n) float16 on self.dev."""
        out = []
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            out.append(self.apply(name, torch.from_numpy(np.ascontiguousarray(bank[r])), ids[i:i + bs]))
        return torch.cat(out)


def _laplace(g: torch.Generator, shape, scale: float) -> torch.Tensor:
    u = torch.rand(*shape, generator=g) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2 * u.abs())


def fit_pca(bank: np.ndarray, rows: np.ndarray, size: int = 64, kmax: int = 256) -> dict:
    """PCA on `size`x`size` area-downsampled TRAIN images (rows must be train-split rows)."""
    x = torch.from_numpy(np.ascontiguousarray(bank[rows])).float().div(255).unsqueeze(1)
    z = F.interpolate(x, size=(size, size), mode="area").flatten(1)
    mean = z.mean(0)
    _, _, Vt = torch.linalg.svd(z - mean, full_matrices=False)
    return dict(mean=mean, components=Vt[:kmax].contiguous(), size=size)


FLOOR_GRID = (
    ["raw", "const", "warp0"]
    + [f"rw1_mu{m}" for m in (0.005, 0.01, 0.02)] + [f"rw2_mu{m}" for m in (0.005, 0.01, 0.02, 0.04)] + [f"rw3_mu{m}" for m in (0.005, 0.01)]
    + [f"blur_s{s}" for s in (0.5, 1, 2, 4)] + [f"down_f{f}" for f in (2, 4, 8)] + [f"noise_s{s}" for s in (0.02, 0.05, 0.1, 0.2)]
    + [f"quant_l{l}" for l in (32, 8, 4)] + [f"quantd_l{l}" for l in (8, 4)] + [f"jpeg_q{q}" for q in (30, 10, 3)] + [f"pca_k{k}" for k in (8, 32, 128)]
    + ["dppix_b8", "rim_w24", "core_w24"])
LEARNED = [f"pcx_mu{m}" for m in (0.001, 0.005, 0.01)]


import hashlib
FROZEN_FLOORS_SHA256 = hashlib.sha256("\n".join(sorted(FLOOR_GRID + ["labels14"])).encode()).hexdigest()   # the list frozen in T.md section 4 (labels14 = code floor, heads.py/utility.py)
