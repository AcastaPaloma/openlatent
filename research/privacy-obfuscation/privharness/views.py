"""Cue-isolating views of an item and frozen features for linear probes (GPU).

A view keeps one kind of information and removes the rest, so a probe trained on it measures how much a target
(the task, or a secret) depends on that cue:

    raw        the frames                                           (everything)
    colour     per-frame mean / std colour and a 4x4 grid of means  (lighting, session, dominant colours)
    lowres     frames downsampled x16 then upsampled                (coarse layout and colour)
    scene      frames with every detected person blacked out        (background, room, objects not on people)
    person     crop around the detected people, background removed (appearance of the people)
    body       2D keypoints of the most confident person, centred on the hips, scaled by the torso, plus bone
               length ratios                                         (body proportions, posture, motion)
    posture    limb directions only (unit vectors, no lengths)      (posture and motion, no body proportions)
    motion     |frame difference| pooled to an 8x10 grid             (where and how much things move)
    label      one-hot primary utility label (secrets only)          (what the task label alone implies)

Appearance views are embedded with DINOv2 ViT-S/14 (per frame CLS token; mean and std over K frames). Video
features for screening obfuscated data come from Kinetics-400 R(2+1)D-18 (penultimate layer) on a 16-frame clip,
which also sees motion.
"""

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def sample_frames(arr, start, end, k):
    """k frames evenly spaced in [start, end] (1-based inclusive) as uint8 (k, H, W, 3)."""
    idx = np.linspace(start - 1, end - 1, k).round().astype(int)
    fr = np.asarray(arr[np.clip(idx, 0, len(arr) - 1)])
    return np.repeat(fr[..., None], 3, -1) if fr.ndim == 3 else fr


def sample_clip(arr, start, end, t=16, stride=2):
    """A centred t-frame clip (stride reduced to fit) as uint8 (t, H, W, 3)."""
    n = end - start + 1
    s = stride if n >= (t - 1) * stride + 1 else 1
    span = (t - 1) * s + 1
    a = start - 1 + max(0, (n - span) // 2)
    idx = np.clip(np.arange(a, a + span, s), 0, len(arr) - 1)
    fr = np.asarray(arr[idx])
    return np.repeat(fr[..., None], 3, -1) if fr.ndim == 3 else fr


class Detectors:
    """Person boxes (Faster R-CNN MobileNetV3, COCO) and keypoints (Keypoint R-CNN R50, COCO) on uint8 frames."""

    def __init__(self, det_size=320):
        from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                  KeypointRCNN_ResNet50_FPN_Weights,
                                                  fasterrcnn_mobilenet_v3_large_320_fpn, keypointrcnn_resnet50_fpn)
        self.box = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1).eval().to(DEV)
        self.kp = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=det_size,
                                            max_size=int(det_size * 1.6)).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, frames, score=0.5, kp_det_score=0.7, chunk=32):
        """frames uint8 (N, H, W, 3) -> per frame dict(boxes (m, 4), kps (17, 3) of the top person or None)."""
        out = []
        x = torch.from_numpy(frames).to(DEV).permute(0, 3, 1, 2).float() / 255
        for i in range(0, len(x), chunk):
            xs = list(x[i:i + chunk])
            boxes = self.box(xs)
            kps = self.kp(xs)
            for b, k in zip(boxes, kps):
                keep = (b["labels"] == 1) & (b["scores"] >= score)
                good = k["scores"] >= kp_det_score
                kp = None
                if good.any():
                    j = int(good.nonzero()[0, 0])  # detections are sorted by score
                    kp = torch.cat([k["keypoints"][j, :, :2], k["keypoints_scores"][j, :, None]], -1).cpu().numpy()
                out.append(dict(boxes=b["boxes"][keep].cpu().numpy(), kps=kp))
        return out


class Dino:
    """DINOv2 ViT-S/14 CLS features, mean and std over each item's frames (768-d)."""

    def __init__(self, size=(224, 280)):
        import timm
        self.net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                     dynamic_img_size=True).eval().to(DEV)
        self.size = size

    @torch.no_grad()
    def __call__(self, frames_per_item, chunk=256):
        """list of uint8 (k, H, W, 3) arrays -> (n_items, 768) float32."""
        k = len(frames_per_item[0])
        flat = np.concatenate(frames_per_item)
        feats = []
        for i in range(0, len(flat), chunk):
            x = torch.from_numpy(flat[i:i + chunk]).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
            x = (x - IMNET_MEAN.to(DEV)) / IMNET_STD.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(self.net(x).float().cpu())
        f = torch.cat(feats).view(len(frames_per_item), k, -1)
        return torch.cat([f.mean(1), f.std(1)], 1).numpy()


class VideoFeat:
    """Kinetics-400 R(2+1)D-18 penultimate features of a 16-frame clip (512-d): appearance and motion."""

    def __init__(self):
        from torchvision.models import video as tvv
        self.net = tvv.r2plus1d_18(weights=tvv.R2Plus1D_18_Weights.KINETICS400_V1)
        self.net.fc = torch.nn.Identity()
        self.net.eval().to(DEV)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=DEV).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=DEV).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, clips, chunk=32):
        """list of uint8 (16, H, W, 3) -> (n, 512)."""
        out = []
        for i in range(0, len(clips), chunk):
            x = torch.from_numpy(np.stack(clips[i:i + chunk])).to(DEV).permute(0, 4, 1, 2, 3).float() / 255
            x = (x - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self.net(x).float().cpu())
        return torch.cat(out).numpy()


def colour_feats(frames):
    """(k, H, W, 3) uint8 -> mean/std colour, 4x4 grid means; mean and std over frames."""
    f = frames.astype(np.float32) / 255
    k, h, w, _ = f.shape
    grid = f[:, : h // 4 * 4, : w // 4 * 4].reshape(k, 4, h // 4, 4, w // 4, 3).mean((2, 4)).reshape(k, -1)
    per = np.concatenate([f.mean((1, 2)), f.std((1, 2)), grid], 1)
    return np.concatenate([per.mean(0), per.std(0)])


def lowres(frames, factor=16):
    k, h, w, _ = frames.shape
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    small = F.interpolate(x, size=(max(1, h // factor), max(1, w // factor)), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def mask_people(frames, dets, dilate=4):
    out = frames.copy()
    h, w = frames.shape[1:3]
    for i, d in enumerate(dets):
        for x0, y0, x1, y1 in d["boxes"].round().astype(int):
            out[i, max(0, y0 - dilate):min(h, y1 + dilate), max(0, x0 - dilate):min(w, x1 + dilate)] = 0
    return out


def crop_people(frames, dets, pad=0.1):
    """Union box of the people in each frame, padded, cropped and resized back to the frame size (black if none)."""
    import cv2
    k, h, w, _ = frames.shape
    out = np.zeros_like(frames)
    for i, d in enumerate(dets):
        if len(d["boxes"]) == 0:
            continue
        x0, y0 = d["boxes"][:, 0].min(), d["boxes"][:, 1].min()
        x1, y1 = d["boxes"][:, 2].max(), d["boxes"][:, 3].max()
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        a, b = int(max(0, y0 - py)), int(min(h, y1 + py))
        c, e = int(max(0, x0 - px)), int(min(w, x1 + px))
        if b - a >= 4 and e - c >= 4:
            out[i] = cv2.resize(frames[i, a:b, c:e], (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def body_feats(dets, kp_min=2.0):
    """Hip-centred, torso-scaled keypoints (17 x 2) and 12 bone lengths / torso, mean and std over frames, plus the
    fraction of frames with a reliable skeleton. Missing joints are zeros."""
    per, ok = [], 0
    for d in dets:
        kp = d["kps"]
        v = np.zeros(17 * 2 + len(LIMBS))
        if kp is not None and kp[11, 2] > kp_min and kp[12, 2] > kp_min and kp[5, 2] > kp_min and kp[6, 2] > kp_min:
            hip, sho = kp[[11, 12], :2].mean(0), kp[[5, 6], :2].mean(0)
            torso = max(np.linalg.norm(sho - hip), 1e-3)
            rel = (kp[:, :2] - hip) / torso * (kp[:, 2:3] > kp_min)
            bones = [np.linalg.norm(kp[a, :2] - kp[b, :2]) / torso if kp[a, 2] > kp_min and kp[b, 2] > kp_min else 0
                     for a, b in LIMBS]
            v = np.concatenate([rel.ravel(), bones])
            ok += 1
        per.append(v)
    per = np.stack(per)
    return np.concatenate([per.mean(0), per.std(0), [ok / len(dets)]])


def posture_feats(dets, kp_min=2.0):
    """Limb unit directions (12 x 2) only: posture and motion without body proportions or size."""
    per = []
    for d in dets:
        kp = d["kps"]
        v = np.zeros(len(LIMBS) * 2)
        if kp is not None:
            for j, (a, b) in enumerate(LIMBS):
                if kp[a, 2] > kp_min and kp[b, 2] > kp_min:
                    dv = kp[b, :2] - kp[a, :2]
                    n = np.linalg.norm(dv)
                    if n > 0:
                        v[2 * j:2 * j + 2] = dv / n
        per.append(v)
    per = np.stack(per)
    return np.concatenate([per.mean(0), per.std(0)])


def motion_feats(clip, grid=(8, 10)):
    """|frame difference| of a clip pooled to a grid: mean and std over time, plus global motion energy."""
    x = torch.from_numpy(clip).float().mean(-1)  # (t, H, W) luminance
    d = (x[1:] - x[:-1]).abs()[:, None]
    g = F.adaptive_avg_pool2d(d, grid)[:, 0].reshape(len(d), -1).numpy() / 255
    return np.concatenate([g.mean(0), g.std(0), [g.mean(), g.std()]])


def scene_stats(frames, dets):
    """Per-item person statistics for the dataset card."""
    h, w = frames.shape[1:3]
    n_people = [len(d["boxes"]) for d in dets]
    areas = [((d["boxes"][:, 2] - d["boxes"][:, 0]) * (d["boxes"][:, 3] - d["boxes"][:, 1])).sum() / (h * w)
             for d in dets if len(d["boxes"])]
    return dict(person_rate=float(np.mean([n > 0 for n in n_people])), mean_people=float(np.mean(n_people)),
                person_area=float(np.median(areas)) if areas else 0.0,
                skeleton_rate=float(np.mean([d["kps"] is not None for d in dets])))
