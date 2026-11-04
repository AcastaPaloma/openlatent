"""Frozen DINOv2-B/14 features (Apache-2.0, `facebook/dinov2-base`): CLS token and mean patch token concatenated (1,536-d), 224 px, ImageNet normalisation.
Grey 256-px images are replicated to 3 channels.  Used by F3 (pair heads / retrieval scores) and F5 (demographic regressors)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_MODEL = {}


@torch.no_grad()
def dino_features(X: torch.Tensor, device: str = "cuda", model_id: str = "facebook/dinov2-base", bs: int = 128) -> np.ndarray:
    from transformers import AutoModel
    if model_id not in _MODEL:
        _MODEL[model_id] = AutoModel.from_pretrained(model_id).to(device).eval()
    m = _MODEL[model_id]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, X.shape[0], bs):
        x = F.interpolate(X[i:i + bs].to(device).float().unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            h = m(pixel_values=(x - mean) / std).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
    return torch.cat(out).numpy()
