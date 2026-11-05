"""Attackers for R-CXR.  Every learned attacker is trained FROM SCRATCH (ImageNet init only) on the DEFENDED training images and scored on
patient-disjoint test pairs; nothing is shared with the defender.

Pair-score interface (all families): score[n] in R, higher = 'same patient'.  Families (T.md section 3):
  F1 SiameseR50  (PriCheXy's own verifier: 2 shared ResNet-50 branches, sigmoid 128-d, |diff|, linear head)   conditions: mismatch | match | anon2
  F2 SiameseConvNeXt-T (same head)                                                                              condition: match
  F3 frozen-feature heads (see heads.py)       F4 structural NCC with affine registration (this file, non-learned)
  F5 demographic attacker (heads.py)           F6 patient-ID InfoNCE metric learner (heads.py)  <- HELD OUT
Conditions: mismatch = train on (defended, defended) pairs, test on (defended, REAL)  [PriCheXy's code path]; match = train and test on
(defended, REAL); anon2 = train and test on (defended, defended).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep(x: torch.Tensor) -> torch.Tensor:
    """(B,n,n) float in [0,1] -> (B,3,n,n) ImageNet-normalised."""
    x = x.float().unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - IM_MEAN.to(x.device)) / IM_STD.to(x.device)


class Siamese(nn.Module):
    """Packhauser et al. 2022 verifier: shared backbone, 128-d sigmoid embedding, |difference|, linear head -> logit."""

    def __init__(self, arch: str = "resnet50", pretrained: bool = True, head: str = "linear"):
        super().__init__()
        import torchvision.models as tvm
        d = 128 if head == "linear" else 512                    # head "xl" (T.md section 12, R5): 512-d embedding + MLP on [|ea-eb|, ea*eb, ea+eb]
        if arch == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            m.fc = nn.Linear(2048, d)
        elif arch == "convnext_tiny":
            m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            m.classifier[2] = nn.Linear(768, d)
        elif arch == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            m.fc = nn.Linear(512, d)
        else:
            raise ValueError(arch)
        self.head = head
        self.model = m
        if head == "linear":
            self.fc_end = nn.Linear(128, 1)
        else:
            self.pair = nn.Sequential(nn.Linear(3 * d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def embed(self, x):
        return torch.sigmoid(self.model(x))

    def forward(self, a, b):
        ea, eb = self.embed(a), self.embed(b)
        if self.head == "linear":
            return self.fc_end(torch.abs(ea - eb)).squeeze(-1)
        return self.pair(torch.cat([torch.abs(ea - eb), ea * eb, ea + eb], dim=1)).squeeze(-1)


def _batches(n: int, bs: int, shuffle: bool, gen: torch.Generator | None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(net: Siamese, xa: torch.Tensor, xb: torch.Tensor, ia, ib, bs: int = 128, amp: bool = True) -> np.ndarray:
    net.eval()
    dev = next(net.parameters()).device
    out = []
    for sl in _batches(len(ia), bs, False, None):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            out.append(net(_prep(xa[ia[sl]].to(dev)), _prep(xb[ib[sl]].to(dev))).float().cpu())
    return torch.cat(out).numpy()


def augment(x: torch.Tensor, shift_px: int = 8, jitter: float = 0.10) -> torch.Tensor:
    """R5 training-time augmentation, independent per image: random translation up to +-shift_px and brightness / contrast jitter; x (B,n,n) float in [0,1]."""
    B, n = x.shape[0], x.shape[-1]
    th = torch.zeros(B, 2, 3, device=x.device); th[:, 0, 0] = 1; th[:, 1, 1] = 1
    th[:, :, 2] = (torch.rand(B, 2, device=x.device) * 2 - 1) * shift_px * 2.0 / n
    g = F.affine_grid(th, (B, 1, n, n), align_corners=False)
    y = F.grid_sample(x.float().unsqueeze(1), g, padding_mode="border", align_corners=False).squeeze(1)
    c = 1 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * jitter
    b = (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * jitter * 0.5
    return ((y - 0.5) * c + 0.5 + b).clamp(0, 1)


def train_siamese(tr: dict, va: dict, te: dict, *, arch="resnet50", pretrained=True, seed=0, lr=1e-4, bs=32, max_epochs=30, patience=5, amp=True, device="cuda",
                  log=print, return_model=False, head="linear", select="loss", aug=False, cosine=False, time_budget_s=None) -> dict:
    """tr/va/te = dict(xa=(Na,n,n) float16 tensor of the images used as element A, xb= same for element B, ia, ib = index arrays, y = labels).
    The caller decides what xa/xb ARE (defended / raw) for each phase -> the three conditions.  Early stopping on validation BCE (paper: patience 5).
    Returns val/test scores at the best validation epoch, the per-epoch curve, and the epoch used."""
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    dev = torch.device(device)
    net = Siamese(arch, pretrained, head).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs) if cosine else None
    import time as _time
    from .stats import auc as _auc
    t_start = _time.time(); truncated = False; best_auc = -1.0
    ytr = torch.as_tensor(tr["y"], dtype=torch.float32)
    best, best_state, bad, curve = math.inf, None, 0, []
    for ep in range(max_epochs):
        net.train()
        tl = 0.0
        for sl in _batches(len(ytr), bs, True, gen):
            xa_, xb_ = tr["xa"][tr["ia"][sl]].to(dev), tr["xb"][tr["ib"][sl]].to(dev)
            if aug:
                xa_, xb_ = augment(xa_), augment(xb_)
            a, b = _prep(xa_), _prep(xb_)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                logit = net(a, b)
            loss = F.binary_cross_entropy_with_logits(logit.float(), ytr[sl].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += float(loss) * len(sl)
        sv = predict(net, va["xa"], va["xb"], va["ia"], va["ib"], amp=amp)
        vl = float(F.binary_cross_entropy_with_logits(torch.as_tensor(sv), torch.as_tensor(va["y"], dtype=torch.float32)))
        va_auc = _auc(va["y"], sv) if select == "auc" else None
        curve.append((ep, tl / len(ytr), vl) if va_auc is None else (ep, tl / len(ytr), vl, va_auc))
        log(f"  ep {ep} train {tl / len(ytr):.4f} val {vl:.4f}" + ("" if va_auc is None else f" val_auc {va_auc:.4f}"))
        improved = best_state is None or ((va_auc > best_auc + 1e-6) if select == "auc" else (vl < best))     # first epoch always accepted (also guards a one-class validation fold)
        if improved:
            best, bad, best_state = vl, 0, copy.deepcopy(net.state_dict())          # best_val_loss = validation BCE at the selected epoch (whichever criterion selected it)
            if va_auc is not None:
                best_auc = va_auc
        else:
            bad += 1
            if bad >= patience:
                break
        if sched is not None:
            sched.step()
        if time_budget_s and _time.time() - t_start > time_budget_s:
            truncated = True; log(f"  STOP: time budget {time_budget_s}s reached after epoch {ep}")
            break
    net.load_state_dict(best_state)
    extra = dict(model=net) if return_model else {}
    return dict(**extra, val_score=predict(net, va["xa"], va["xb"], va["ia"], va["ib"], amp=amp), test_score=predict(net, te["xa"], te["xb"], te["ia"], te["ib"], amp=amp),
                curve=curve, best_val_loss=best, epochs=len(curve), truncated=truncated, best_val_auc=best_auc)


# ---------------------------------------------------------------------------------------------- F4 structural (non-learned)
AFFINE_GRID = [(s, tx, ty) for s in (0.94, 1.0, 1.06) for tx in (-0.08, -0.04, 0.0, 0.04, 0.08) for ty in (-0.08, -0.04, 0.0, 0.04, 0.08)]


def _hp(x: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """(B,1,n,n) high-pass: subtract a Gaussian blur (removes exposure / low-frequency shading)."""
    r = int(3 * sigma)
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-ax ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x - y


def _ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """normalised cross-correlation over the masked region, per batch element."""
    m = mask.float()
    n = m.sum((1, 2, 3)).clamp(min=1)
    am = (a * m).sum((1, 2, 3)) / n; bm = (b * m).sum((1, 2, 3)) / n
    a0, b0 = (a - am.view(-1, 1, 1, 1)) * m, (b - bm.view(-1, 1, 1, 1)) * m
    return (a0 * b0).sum((1, 2, 3)) / ((a0.pow(2).sum((1, 2, 3)).sqrt() * b0.pow(2).sum((1, 2, 3)).sqrt()).clamp(min=1e-8))


@torch.no_grad()
def structural_scores(xa: torch.Tensor, xb: torch.Tensor, ia, ib, size: int = 128, highpass: bool = True, grid=AFFINE_GRID, bs: int = 256, device="cpu") -> np.ndarray:
    """max over an affine grid (scale, tx, ty) of the NCC between image A and the warped image B.  No training, no labels, deterministic."""
    out = []
    for sl in _batches(len(ia), bs, False, None):
        a = F.interpolate(xa[ia[sl]].float().unsqueeze(1).to(device), size=(size, size), mode="area")
        b = F.interpolate(xb[ib[sl]].float().unsqueeze(1).to(device), size=(size, size), mode="area")
        if highpass:
            a, b = _hp(a, 8.0 * size / 256), _hp(b, 8.0 * size / 256)
        best = torch.full((a.shape[0],), -2.0, device=a.device)
        ones = torch.ones_like(b)
        for s, tx, ty in grid:
            th = torch.tensor([[s, 0, tx], [0, s, ty]], dtype=torch.float32, device=a.device).expand(a.shape[0], 2, 3)
            g = F.affine_grid(th, a.shape, align_corners=False)
            bw = F.grid_sample(b, g, padding_mode="zeros", align_corners=False)
            mw = F.grid_sample(ones, g, padding_mode="zeros", align_corners=False) > 0.99
            best = torch.maximum(best, _ncc(a, bw, mw))
        out.append(best.cpu())
    return torch.cat(out).numpy()
