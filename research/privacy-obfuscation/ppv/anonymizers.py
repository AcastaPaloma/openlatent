"""Video anonymizers f: clip -> clip. Input and output are float (B, C, T, H, W) in [0, 1].

Fixed (no parameters):
    raw                       identity (upper bound on utility and on leakage)
    blur      sigma           per-frame Gaussian blur
    down      factor          area downsample, bilinear upsample back
    pixelate  factor          area downsample, nearest upsample back
    diff      gain            |x_t - x_{t-1}| (motion only)
    bdq_fixed sigma, levels   blur -> temporal difference -> uniform quantisation

Learned (trained adversarially by train.train_anonymizer; all start as or near identity):
    bdq        BDQ-style: learnable blur sigma, temporal difference, learnable quantisation
               thresholds (hard steps forward, soft sigmoid gradients backward)
    unet3d     residual spatio-temporal conv net, last layer zero-initialised
    vae_latent frozen SD VAE (stabilityai/sd-vae-ft-mse) encoder -> learnable residual 3D-conv
               latent edit (zero-initialised) -> frozen decoder; output stays a pixel video
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(sigma, size=None):
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    size = size or int(2 * math.ceil(3 * float(sigma)) + 1)
    r = torch.arange(size, dtype=torch.float32, device=sigma.device) - size // 2
    k = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur2d(x, sigma, size=None):
    """Separable Gaussian blur of every frame; sigma may be a tensor with gradients."""
    b, c, t, h, w = x.shape
    k = gaussian_kernel(sigma, size).to(x)
    f = x.transpose(1, 2).reshape(b * t * c, 1, h, w)
    p = k.numel() // 2
    f = F.conv2d(F.pad(f, (p, p, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    f = F.conv2d(F.pad(f, (0, 0, p, p), mode="replicate"), k.view(1, 1, -1, 1))
    return f.view(b, t, c, h, w).transpose(1, 2)


def per_frame(fn, x):
    b, c, t, h, w = x.shape
    y = fn(x.transpose(1, 2).reshape(b * t, c, h, w))
    return y.view(b, t, *y.shape[1:]).transpose(1, 2)


def tdiff(x):
    d = x[:, :, 1:] - x[:, :, :-1]
    return torch.cat([torch.zeros_like(x[:, :, :1]), d], 2)


class Fixed(nn.Module):
    learned = False

    def __init__(self, kind, **p):
        super().__init__()
        self.kind, self.p = kind, p

    def forward(self, x):
        k, p = self.kind, self.p
        h, w = x.shape[-2:]
        if k in ("raw", "bg_median", "fg_only"):  # whole-video controls, applied in run.py
            return x
        if k == "const":  # sanity control: every frame black, so no information survives
            return torch.zeros_like(x)
        if k == "tshuffle":  # scramble frame order inside each clip/window: removes motion rhythm (gait), keeps frames
            g = torch.Generator(device="cpu").manual_seed(p.get("seed", 0))
            return x[:, :, torch.randperm(x.shape[2], generator=g).to(x.device)]
        if k == "mean":  # brightness-only control: each frame becomes its own mean intensity (1 number per frame)
            return x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
        if k == "blur":
            return blur2d(x, p["sigma"])
        if k in ("down", "pixelate"):
            s = p["factor"]
            small = per_frame(lambda f: F.interpolate(f, size=(h // s, w // s), mode="area"), x)
            mode = "bilinear" if k == "down" else "nearest"
            return per_frame(lambda f: F.interpolate(f, size=(h, w), mode=mode), small).clamp(0, 1)
        if k == "diff":
            return (tdiff(x).abs() * p.get("gain", 4.0)).clamp(0, 1)
        if k == "hold":  # temporal hold: every k-th frame repeated k times (removes fine motion timing)
            kk = p["k"]
            idx = (torch.arange(x.shape[2], device=x.device) // kk) * kk
            return x[:, :, idx]
        if k == "bdq_fixed":
            d = tdiff(blur2d(x, p["sigma"]))  # [-1, 1]
            n = p["levels"]
            return (torch.round((d + 1) / 2 * (n - 1)) / (n - 1)).clamp(0, 1)
        raise ValueError(k)


class BDQ(nn.Module):
    """Blur -> Difference -> Quantisation, after Kumawat & Nagahara (ECCV 2022), reimplemented.

    The quantiser is q(d) = (1/N) sum_i step(d - b_i) with N learnable thresholds b_i. Forward uses
    hard steps; backward uses the gradient of sigmoid(hardness * (d - b_i)) (straight-through).
    """
    learned = True

    def __init__(self, sigma=2.0, levels=15, hardness=20.0, gain=4.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.thresholds = nn.Parameter(torch.linspace(-1, 1, levels + 2)[1:-1])
        self.hardness, self.gain = hardness, gain

    def forward(self, x):
        sigma = self.log_sigma.exp().clamp(0.3, 8.0)
        d = (tdiff(blur2d(x, sigma, size=13)) * self.gain).clamp(-1, 1)
        z = d.unsqueeze(-1) - self.thresholds
        soft = torch.sigmoid(self.hardness * z).mean(-1)
        hard = (z > 0).float().mean(-1)
        return soft + (hard - soft).detach()


def conv3(cin, cout, k=(3, 3, 3)):
    return nn.Conv3d(cin, cout, k, padding=tuple(v // 2 for v in k))


class ResBlock3d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), conv3(ch, ch), nn.GroupNorm(8, ch), nn.SiLU(), conv3(ch, ch))

    def forward(self, x):
        return x + self.body(x)


class UNet3d(nn.Module):
    """Residual anonymizer: y = clamp(x + g(x)), g a small encoder-decoder with skips whose
    last layer starts at zero, so training starts from the identity map."""
    learned = True

    def __init__(self, cin=1, ch=32):
        super().__init__()
        self.inp = conv3(cin, ch)
        self.e1 = ResBlock3d(ch)
        self.down = nn.Conv3d(ch, 2 * ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.mid = nn.Sequential(ResBlock3d(2 * ch), ResBlock3d(2 * ch))
        self.up = nn.ConvTranspose3d(2 * ch, ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.d1 = ResBlock3d(ch)
        self.out = conv3(ch, cin)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h1 = self.e1(self.inp(x))
        h = self.up(self.mid(self.down(h1)))
        h = h[..., : h1.shape[-2], : h1.shape[-1]]
        return (x + self.out(self.d1(h + h1))).clamp(0, 1)


class VAELatent(nn.Module):
    """Frozen SD VAE around a learnable latent edit. Frames are encoded independently; the
    edit is a residual 3D-conv net over (latent channels, time, h/8, w/8), zero-initialised,
    so the starting point is plain VAE reconstruction."""
    learned = True

    def __init__(self, vae_id="stabilityai/sd-vae-ft-mse", ch=128, blocks=4, out_gray=True, checkpoint=True):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_id)
        self.vae.requires_grad_(False).eval()
        if checkpoint:
            self.vae.enable_gradient_checkpointing()
        zc = self.vae.config.latent_channels
        self.edit = nn.Sequential(conv3(zc, ch), *[ResBlock3d(ch) for _ in range(blocks)], conv3(ch, zc))
        nn.init.zeros_(self.edit[-1].weight)
        nn.init.zeros_(self.edit[-1].bias)
        self.out_gray = out_gray

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def forward(self, x):
        b, c, t, h, w = x.shape
        f = x.expand(-1, 3, -1, -1, -1) if c == 1 else x
        f = f.transpose(1, 2).reshape(b * t, 3, h, w) * 2 - 1
        # the VAE works at 1/8 resolution: pad to a multiple of 8 (UCF101 frames are 112x150)
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            f = F.pad(f, (0, pw, 0, ph), mode="replicate")
        with torch.no_grad():
            z = self.vae.encode(f).latent_dist.mean * self.vae.config.scaling_factor
        z = z.view(b, t, *z.shape[1:]).transpose(1, 2)
        z = z + self.edit(z)
        z = z.transpose(1, 2).reshape(b * t, *z.shape[1:2], *z.shape[3:])
        y = self.vae.decode(z / self.vae.config.scaling_factor).sample[:, :, :h, :w]
        y = ((y + 1) / 2).clamp(0, 1).reshape(b, t, 3, h, w).transpose(1, 2)
        return y.mean(1, keepdim=True) if self.out_gray else y


class PersonBox(nn.Module):
    """Detect people per frame (torchvision Faster R-CNN MobileNetV3-320, COCO) and black out
    (fill=0) or blur their boxes: the person-removal baseline used by VITA/SPAct ("blackening").
    Fixed; no learned parameters of its own."""
    learned = False

    def __init__(self, score=0.5, mode="black", sigma=8.0, every=1, dilate=4):
        super().__init__()
        from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                  fasterrcnn_mobilenet_v3_large_320_fpn)
        self.det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1).eval()
        self.det.requires_grad_(False)
        self.score, self.mode, self.sigma = score, mode, sigma
        self.every, self.dilate = every, dilate  # detect on every `every`-th frame; grow boxes by `dilate` px

    def train(self, mode=True):
        super().train(mode)
        self.det.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        rgb = frames.expand(-1, 3, -1, -1) if c == 1 else frames
        mask = torch.zeros(b * t, 1, h, w, device=x.device, dtype=x.dtype)
        idx = [k for k in range(b * t) if (k % t) % self.every == 0]  # keyframes within each clip
        d = self.dilate
        for i in range(0, len(idx), 256):
            chunk = idx[i:i + 256]
            for k, det in zip(chunk, self.det(list(rgb[chunk].float()))):
                keep = (det["labels"] == 1) & (det["scores"] >= self.score)
                last = min(k + self.every, (k // t + 1) * t)  # boxes cover the frames up to the next keyframe
                for x0, y0, x1, y1 in det["boxes"][keep].round().long().tolist():
                    mask[k:last, :, max(0, y0 - d):y1 + d, max(0, x0 - d):x1 + d] = 1
        fill = torch.zeros_like(frames) if self.mode == "black" else blur2d(frames.view(b, t, c, h, w).transpose(1, 2), self.sigma).transpose(1, 2).reshape(b * t, c, h, w)
        out = frames * (1 - mask) + fill * mask
        return out.view(b, t, c, h, w).transpose(1, 2)


COCO_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]


class Pose(nn.Module):
    """Replace every frame by stick figures of the people in it (torchvision Keypoint R-CNN, COCO 17 keypoints) drawn on a
    plain background: no face, skin, clothing or room survives, only body configuration and motion. People are detected on
    every `every`-th frame (input upscaled to `det_size` short side) and the drawing is held until the next keyframe.
    Fixed; no learned parameters of its own."""
    learned = False

    def __init__(self, every=4, score=0.7, kp_score=2.0, det_size=320, width=1):
        super().__init__()
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        self.det = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=det_size, max_size=int(det_size * 1.6)).eval()
        self.det.requires_grad_(False)
        self.every, self.score, self.kp_score, self.width = every, score, kp_score, width
        self.register_buffer("limbs", torch.tensor(COCO_LIMBS), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.det.eval()
        return self

    def draw(self, people, h, w, device):
        """people: list of (17, 3) keypoints (x, y, visibility score) -> (h, w) canvas with limbs set to 1."""
        canvas = torch.zeros(h * w, device=device)
        t = torch.linspace(0, 1, 32, device=device)
        for kp in people:
            a, b = kp[self.limbs[:, 0]], kp[self.limbs[:, 1]]  # (L, 3)
            ok = (a[:, 2] > self.kp_score) & (b[:, 2] > self.kp_score)
            if not ok.any():
                continue
            a, b = a[ok, :2], b[ok, :2]
            pts = a[:, None] + (b - a)[:, None] * t[None, :, None]  # (L, 32, 2) points along each limb
            xs = pts[..., 0].round().long().clamp(0, w - 1)
            ys = pts[..., 1].round().long().clamp(0, h - 1)
            canvas[(ys * w + xs).flatten()] = 1.0
        canvas = canvas.view(1, 1, h, w)
        if self.width > 1:
            canvas = F.max_pool2d(canvas, self.width * 2 - 1, stride=1, padding=self.width - 1)
        return canvas[0, 0]

    @torch.no_grad()
    def forward(self, x):
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        rgb = frames.expand(-1, 3, -1, -1) if c == 1 else frames
        out = torch.zeros(b * t, 1, h, w, device=x.device, dtype=x.dtype)
        idx = [k for k in range(b * t) if (k % t) % self.every == 0]
        for i in range(0, len(idx), 64):
            chunk = idx[i:i + 64]
            for k, det in zip(chunk, self.det(list(rgb[chunk].float()))):
                keep = det["scores"] >= self.score
                # keypoints[..., 2] is a constant visibility flag; the per-keypoint confidence is keypoints_scores
                kps = torch.cat([det["keypoints"][keep][..., :2], det["keypoints_scores"][keep][..., None]], -1)
                canvas = self.draw(list(kps), h, w, x.device).to(x.dtype)
                last = min(k + self.every, (k // t + 1) * t)
                out[k:last, 0] = canvas
        return out.view(b, t, 1, h, w).transpose(1, 2).expand(-1, c, -1, -1, -1)


class TaskMask(nn.Module):
    """Keep only the task-relevant part of each frame, chosen on frozen DINOv2 patch features.

    Each frame is encoded by a frozen DINOv2 ViT-S/14; a head scores every patch and the top `keep` fraction of patches
    is kept (hard top-k forward, sigmoid straight-through backward, so the action loss and an optional adversary decide
    what is "important": the person, but also the pool table in "playing pool"). Inside the mask an optional `inner`
    anonymizer runs (e.g. BDQ); outside it the frame becomes `fill`:
        gray  constant 0.5 (foreground only: nothing of the room survives)
        blur  the frame blurred with sigma 8 (a soft room)
        raw   the original frame (the obfuscated region pasted back into the real background)
    head="saliency" needs no training: it keeps the patches whose DINOv2 features are furthest from the frame's mean
    feature (an unsupervised "salient object" baseline).
    """

    def __init__(self, keep=0.25, fill="gray", head="learned", inner=None, size=(126, 168), in_ch=1):
        super().__init__()
        import timm
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
        self.dino.requires_grad_(False).eval()
        self.keep, self.fill, self.size, self.head_kind = keep, fill, size, head
        self.head = nn.Sequential(nn.Linear(self.dino.num_features, 128), nn.GELU(), nn.Linear(128, 1)) if head == "learned" else None
        self.inner = build(inner, in_ch) if inner else None
        self.learned = head == "learned" or bool(getattr(self.inner, "learned", False))
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def patch_scores(self, x):
        """x (B, C, T, H, W) -> per-patch scores (B*T, gh, gw)."""
        b, c, t, h, w = x.shape
        f = x.transpose(1, 2).reshape(b * t, c, h, w)
        f = f.expand(-1, 3, -1, -1) if c == 1 else f
        f = F.interpolate(f.float(), size=self.size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            tok = self.dino.forward_features((f - self.mean) / self.std)[:, self.dino.num_prefix_tokens:]
        gh, gw = self.size[0] // 14, self.size[1] // 14
        if self.head is None:
            return (tok - tok.mean(1, keepdim=True)).norm(dim=-1).view(b * t, gh, gw)
        return self.head(tok.float()).squeeze(-1).view(b * t, gh, gw)

    def mask(self, x):
        b, c, t, h, w = x.shape
        s = self.patch_scores(x).flatten(1)  # (BT, P)
        k = max(1, int(round(self.keep * s.shape[1])))
        thr = s.topk(k, dim=1).values[:, -1:].detach()
        hard = (s >= thr).float()
        m = hard if self.head is None else torch.sigmoid(s - thr) + (hard - torch.sigmoid(s - thr)).detach()
        gh, gw = self.size[0] // 14, self.size[1] // 14
        m = F.interpolate(m.view(b * t, 1, gh, gw), size=(h, w), mode="nearest")
        return m.view(b, t, 1, h, w).transpose(1, 2)  # (B, 1, T, H, W)

    def forward(self, x):
        m = self.mask(x)
        inside = self.inner(x) if self.inner is not None else x
        if self.fill == "raw":
            outside = x
        elif self.fill == "blur":
            outside = blur2d(x, torch.tensor(8.0, device=x.device), size=33)
        else:
            outside = torch.full_like(x, 0.5)
        return m * inside + (1 - m) * outside


class Chain(nn.Module):
    """Apply anonymizers in order, e.g. [{"type": "down", "factor": 4}, {"type": "bdq"}]."""

    def __init__(self, stages, in_ch=1):
        super().__init__()
        self.stages = nn.ModuleList([build(c, in_ch) for c in stages])
        self.learned = any(getattr(m, "learned", False) for m in self.stages)

    def forward(self, x):
        for m in self.stages:
            x = m(x)
        return x


def frozen_key(k):
    """State-dict keys of frozen pretrained parts (SD-VAE, DINOv2): rebuilt from their own weights, never checkpointed."""
    return any(k.startswith(p) or f".{p}" in k for p in ("vae.", "dino."))


def build(cfg, in_ch=1):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "chain":
        return Chain(cfg["stages"], in_ch)
    if kind in ("raw", "bg_median", "fg_only", "hold", "const", "mean", "tshuffle", "blur", "down", "pixelate", "diff", "bdq_fixed"):
        return Fixed(kind, **cfg)
    if kind == "person_box":
        return PersonBox(**cfg)
    if kind == "bdq":
        return BDQ(**cfg)
    if kind == "unet3d":
        return UNet3d(cin=in_ch, **cfg)
    if kind == "vae_latent":
        cfg.setdefault("out_gray", in_ch == 1)  # keep the dataset's channel count
        return VAELatent(**cfg)
    if kind == "taskmask":
        return TaskMask(in_ch=in_ch, **cfg)
    if kind == "pose":
        return Pose(**cfg)
    raise ValueError(kind)
