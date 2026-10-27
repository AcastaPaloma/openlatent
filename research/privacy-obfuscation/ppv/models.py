"""Utility (action) and privacy (identity/attribute) models, all from public pretrained weights.

Every model takes a float clip in [0, 1], shape (B, C, T, H, W) with C in {1, 3}; grayscale is
repeated to 3 channels and normalised with the backbone's own statistics.

    r2p1d   : torchvision R(2+1)D-18, Kinetics-400   (video; action model / video attacker)
    r3d     : torchvision R3D-18, Kinetics-400       (video; gait-capable identity attacker)
    frame_r50 / frame_r18 : timm ResNet, ImageNet    (per frame; logits averaged over frames)
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import video as tvv

K400_MEAN = (0.43216, 0.394666, 0.37645)
K400_STD = (0.22803, 0.22145, 0.216989)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def to_rgb(x):
    return x.expand(-1, 3, *x.shape[2:]) if x.shape[1] == 1 else x


def normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    return (to_rgb(x) - m) / s


class VideoNet(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        ctor, weights = {"r2p1d": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights.KINETICS400_V1),
                         "r3d": (tvv.r3d_18, tvv.R3D_18_Weights.KINETICS400_V1)}[arch]
        self.net = ctor(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(normalize(x, K400_MEAN, K400_STD))


class FrameNet(nn.Module):
    """2D ImageNet classifier applied to every frame; clip logits = mean over frames.
    `scale` upsamples frames first (KTH frames are 120x160, small for ImageNet features)."""

    def __init__(self, arch, num_classes, scale=2.0):
        super().__init__()
        self.net = timm.create_model(arch, pretrained=True, num_classes=num_classes)
        self.scale = scale

    def forward(self, x):
        b, c, t, h, w = x.shape
        f = normalize(x, IMNET_MEAN, IMNET_STD).transpose(1, 2).reshape(b * t, 3, h, w)
        if self.scale != 1:
            f = F.interpolate(f, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.net(f).view(b, t, -1).mean(1)


class DinoFrame(nn.Module):
    """DINOv2 ViT-B/14 (timm, LVD-142M) per frame; clip logits = mean over frames. DINOv2 features are
    strong for re-identification, so this is the strongest appearance attacker we use.
    frozen=True: linear probe on frozen features (head lr x5); frozen=False: full fine-tune (lr x0.1)."""

    def __init__(self, num_classes, frozen=True, size=(224, 294)):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                          dynamic_img_size=True)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        self.frozen, self.size = frozen, size
        self.lr_scale = 5.0 if frozen else 0.1
        if frozen:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        b, c, t, h, w = x.shape
        f = normalize(x, IMNET_MEAN, IMNET_STD).transpose(1, 2).reshape(b * t, 3, h, w)
        f = F.interpolate(f, size=self.size, mode="bilinear", align_corners=False)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            feats = self.backbone(f)
        return self.head(feats.float()).view(b, t, -1).mean(1)


def build(name, num_classes):
    if name in ("r2p1d", "r3d"):
        return VideoNet(name, num_classes)
    if name == "frame_r50":
        return FrameNet("resnet50.a1_in1k", num_classes)
    if name == "frame_r18":
        return FrameNet("resnet18.a1_in1k", num_classes)
    if name == "frame_dino_probe":
        return DinoFrame(num_classes, frozen=True)
    if name == "frame_dino_ft":
        return DinoFrame(num_classes, frozen=False)
    raise ValueError(name)
