"""PriCheXy-Net flow-field mechanism (re-implemented from the paper, Sec. 2.2) and a loader for the released generator weights.

The U-Net block follows Buda et al. (MIT licence, https://github.com/mateuszbuda/brain-segmentation-pytorch); module names are kept so the
released `generator_lowest_total_loss_mu_*.pth` state dicts load.  Everything else (flow construction, smoothing, resampling) is written
from the paper text and checked against the released code's behaviour; no GPL-3 code from the PriCheXy-Net repository is copied.

Mechanism: F = F_id - mu * G(x);  F <- GaussianFilter(F; kernel 9, sigma 2, ZERO padding);  y = grid_sample(x, F, border, align_corners=True).
Units: grid coordinates are in [-1, 1] over 256 px, so ONE coordinate unit = 127.5 px and a flow bounded by mu displaces by <= 127.5*mu px
(1.275 px at mu = 0.01).  Zero padding in the smoothing pulls the outer ~4 grid rows towards 0, so even G = 0 is NOT the identity map near the
border (see `border_artifact`); the "mu = 0 pipeline" arm in T.md measures what that does to a verifier.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

PX_PER_UNIT = 127.5      # (256 - 1) / 2


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, init_features: int = 32):
        super().__init__()
        f = init_features
        self.encoder1 = UNet._block(in_channels, f, "enc1"); self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet._block(f, f * 2, "enc2"); self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = UNet._block(f * 2, f * 4, "enc3"); self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = UNet._block(f * 4, f * 8, "enc4"); self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = UNet._block(f * 8, f * 16, "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2); self.decoder4 = UNet._block(f * 16, f * 8, "dec4")
        self.upconv3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2); self.decoder3 = UNet._block(f * 8, f * 4, "dec3")
        self.upconv2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2); self.decoder2 = UNet._block(f * 4, f * 2, "dec2")
        self.upconv1 = nn.ConvTranspose2d(f * 2, f, 2, 2); self.decoder1 = UNet._block(f * 2, f, "dec1")
        self.conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x); e2 = self.encoder2(self.pool1(e1)); e3 = self.encoder3(self.pool2(e2)); e4 = self.encoder4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.decoder4(torch.cat((self.upconv4(b), e4), 1)); d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), 1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), 1)); d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), 1))
        return torch.tanh(self.conv(d1))

    @staticmethod
    def _block(cin, f, name):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(cin, f, 3, padding=1, bias=False)), (name + "norm1", nn.BatchNorm2d(f)), (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(f, f, 3, padding=1, bias=False)), (name + "norm2", nn.BatchNorm2d(f)), (name + "relu2", nn.ReLU(inplace=True))]))


def gaussian_kernel(size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def smooth(field: torch.Tensor, size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    """depthwise Gaussian filter with ZERO padding (as in the released code)."""
    c = field.shape[1]
    k = gaussian_kernel(size, sigma).to(field)[None, None].repeat(c, 1, 1, 1)
    return F.conv2d(field, k, groups=c, padding=(size - 1) // 2)


def identity_grid(n: int = 256, device="cpu") -> torch.Tensor:
    d = torch.linspace(-1, 1, n, device=device)
    mx, my = torch.meshgrid(d, d, indexing="ij")
    return torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2)      # (1, 2, n, n); channel 0 = x (width), 1 = y (height)


def warp(x: torch.Tensor, raw_flow: torch.Tensor, mu: float, pad_smooth: bool = True) -> torch.Tensor:
    """x (B,1,n,n) in [0,1]; raw_flow (B,2,n,n) in [-1,1] (U-Net output or any random field)."""
    g = identity_grid(x.shape[-1], x.device).to(x.dtype) - mu * raw_flow
    g = smooth(g) if pad_smooth else g
    return F.grid_sample(x, g.permute(0, 2, 3, 1), padding_mode="border", align_corners=True)


def displacement_px(raw_flow: torch.Tensor, mu: float) -> torch.Tensor:
    """pixel displacement field actually applied (relative to the identity), INCLUDING the smoothing and its zero-padding border effect."""
    n = raw_flow.shape[-1]
    idg = identity_grid(n, raw_flow.device).to(raw_flow.dtype)
    return (smooth(idg - mu * raw_flow) - idg) * PX_PER_UNIT


def border_artifact(n: int = 256) -> dict:
    """what the pipeline does when the generator outputs exactly zero (mu irrelevant): displacement in px at the border rows."""
    d = displacement_px(torch.zeros(1, 2, n, n), 0.0)[0, 0]           # x-displacement, constant across rows
    row = d[n // 2]
    return dict(max_abs_px=float(row.abs().max()), at_px0=float(row[0]), at_px3=float(row[3]), at_px4=float(row[4]), interior_max=float(row[8:-8].abs().max()))


def load_generator(path: str, device="cuda") -> UNet:
    g = UNet(1, 2, 32).to(device)
    g.load_state_dict(torch.load(path, map_location=device))
    return g.eval()


@torch.no_grad()
def release_pricheXy(gen: UNet, x: torch.Tensor, mu: float, bs: int = 64) -> torch.Tensor:
    """x (N,1,256,256) float in [0,1] -> deformed (N,1,256,256) (float32; callers may store fp16)."""
    out = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].float()
        out.append(warp(xb, gen(xb), mu))
    return torch.cat(out)


@torch.no_grad()
def flow_statistics(gen: UNet, x: torch.Tensor, mu: float, bs: int = 64) -> dict:
    """Per-image RMS / sup displacement (px) and the mean radially averaged power spectrum of the applied displacement field."""
    rms, sup, spec = [], [], None
    n = x.shape[-1]
    fy = torch.fft.fftfreq(n)[:, None].expand(n, n); fx = torch.fft.fftfreq(n)[None, :].expand(n, n)
    rad = ((fx ** 2 + fy ** 2).sqrt() * n).round().long().clamp(max=n // 2)          # integer radial bin 0..n/2
    cnt = torch.bincount(rad.flatten(), minlength=n // 2 + 1).float()
    acc = torch.zeros(n // 2 + 1)
    seen = 0
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].float().to(next(gen.parameters()).device)
        disp = displacement_px(gen(xb), mu)                                          # (B,2,n,n)
        rms.append(disp.flatten(1).pow(2).mean(1).sqrt().cpu()); sup.append(disp.flatten(1).abs().max(1).values.cpu())
        P = (torch.fft.fft2(disp.cpu()).abs() ** 2).sum(1)                           # (B,n,n)  sum over x/y channels
        for p in P:
            acc += torch.bincount(rad.flatten(), weights=p.flatten(), minlength=n // 2 + 1)
        seen += xb.shape[0]
    return dict(rms_px=torch.cat(rms), sup_px=torch.cat(sup), radial_power=(acc / cnt.clamp(min=1) / seen))


def spectrum_matched_field(radial_power: torch.Tensor, rms_px: torch.Tensor, generator: torch.Generator, n: int = 256) -> torch.Tensor:
    """Random displacement field (2,n,n) in PIXELS with the given radially averaged power spectrum and per-channel-pair RMS."""
    fy = torch.fft.fftfreq(n)[:, None].expand(n, n); fx = torch.fft.fftfreq(n)[None, :].expand(n, n)
    rad = ((fx ** 2 + fy ** 2).sqrt() * n).round().long().clamp(max=n // 2)
    amp = radial_power.clamp(min=0).sqrt()[rad]                                    # (n,n)
    amp[0, 0] = 0.0                                                                 # zero-mean field (no global shift)
    w = torch.randn(2, n, n, generator=generator)
    f = torch.fft.ifft2(torch.fft.fft2(w) * amp).real
    cur = f.flatten().pow(2).mean().sqrt().clamp(min=1e-12)
    return f * (rms_px / cur)
