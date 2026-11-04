"""Utility U: mean AUC over the 14 findings of the paper's DenseNet-121 (CheXNet) FINE-TUNED on defended images (retrained, as the attackers are) and
scored on defended held-out-patient images; U_fixed = the paper's fixed classifier on defended images (replication of 80.5 / 76.2).
The pickled classifier is a torchvision DenseNet-121 (14 sigmoid outputs); its pickle was audited with pickletools before loading (T.md section 0)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    x = F.interpolate(x.float().unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_classifier(path: str, device="cuda") -> nn.Module:
    m = torch.load(path, map_location=device, weights_only=False)["model"]
    return m.to(device)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device="cuda", bs: int = 128, size: int = 224) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, X.shape[0], bs):
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            out.append(model(_prep(X[i:i + bs].to(device), size)).float().cpu())
    return torch.cat(out).numpy()


def mean_auc(Y: np.ndarray, P: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """macro AUC over labels with >= 1 positive and >= 1 negative.  weights = patient-bootstrap multiplicities per image (None = 1)."""
    from .stats import auc as _auc
    per = np.full(Y.shape[1], np.nan)
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < len(Y):
            if weights is None:
                per[k] = _auc(Y[:, k], P[:, k])
            else:
                from sklearn.metrics import roc_auc_score
                per[k] = roc_auc_score(Y[:, k], P[:, k], sample_weight=weights)
    return float(np.nanmean(per)), per


def finetune(model: nn.Module, Xtr: torch.Tensor, Ytr: np.ndarray, epochs: int = 3, lr: float = 1e-4, bs: int = 32, seed: int = 0, device="cuda", size: int = 224, log=print) -> nn.Module:
    """fine-tune ALL layers with BCE on the defended training images (3 epochs; no selection on any held-out role)."""
    torch.manual_seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Y = torch.as_tensor(Ytr, dtype=torch.float32)
    for ep in range(epochs):
        tot = 0.0
        for idx in torch.randperm(len(Y), generator=gen).split(bs):
            model.train()
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                p = model(_prep(Xtr[idx].to(device), size)).float().clamp(1e-6, 1 - 1e-6)           # the paper's classifier ends in a Sigmoid
            loss = F.binary_cross_entropy(p, Y[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        log(f"  utility ep {ep} train bce {tot / len(Y):.4f}")
    return model.eval()
