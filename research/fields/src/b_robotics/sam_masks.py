"""Zero-shot SAM mask baseline (ungated facebook/sam-vit-base), *oracle-prompted* = most generous version.

Per frame: 3 positive point prompts sampled from the exact arm silhouette of the CLEAN render, restricted to
pixels NOT covered by a distractor sprite (so the prompt never lands on a decoy), then SAM ViT-B with
multimask_output=False.  The resulting mask is "SAM's answer to: segment the arm".  We store masks
(packed bits) per (split, kind) and mask-quality stats (IoU vs exact silhouette, decoy pixels inside the mask).

python sam_masks.py --kind corr_noisy --splits train,val,test --out $OUT [--limit N]  (needs a GPU unless --cpu)
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from scipy import ndimage
import data as Dt, distractors as D


def pick_points(valid, n=3):
    """deterministic: nearest-to-centroid, then farthest-point sampling inside the (eroded) valid region."""
    v = ndimage.binary_erosion(valid, iterations=2)
    if v.sum() < 12:
        v = valid
    ys, xs = np.nonzero(v)
    if len(ys) < 12:
        return None
    P = np.stack([xs, ys], 1).astype(np.float32)
    c = P.mean(0)
    pts = [P[np.argmin(((P - c) ** 2).sum(1))]]
    for _ in range(n - 1):
        d = np.min([((P - q) ** 2).sum(1) for q in pts], axis=0)
        pts.append(P[np.argmax(d)])
    return np.stack(pts)


@torch.no_grad()
def run(kind, splits, out, limit=None, device="cuda", bs=32, model_id="facebook/sam-vit-base"):
    from transformers import SamModel
    torch.set_num_threads(4)   # CPU-side work is tiny; many OpenMP threads just thrash on a shared box
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SamModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    summary = {}
    os.makedirs(out, exist_ok=True)
    for split in splits:
        ep = Dt.load_split(split)
        st = Dt.dist_states(split, kind)
        N = len(ep["frames"]) if limit is None else limit
        T1 = ep["frames"].shape[1]
        arm = Dt.unpack(ep["arm"][:N])
        masks = np.zeros((N, T1, 128, 128), bool)
        iou_l, inc_l, cov_l = [], [], []
        t0 = time.time()
        for e in range(N):
            clean = torch.from_numpy(ep["frames"][e]).to(device).permute(0, 3, 1, 2).float() / 255  # (T1,3,128,128)
            img, alpha = D.composite(clean, torch.from_numpy(st["pos"][e]).to(device), torch.from_numpy(st["cls"][e]).to(device)[None].repeat(T1, 1))
            alpha = (alpha[:, 0] > 0.5).cpu().numpy()
            pts, ok = [], []
            for t in range(T1):
                p = pick_points(arm[e, t] & ~alpha[t])
                ok.append(p is not None)
                pts.append(p if p is not None else np.zeros((3, 2), np.float32) + 64)
            pts = torch.from_numpy(np.stack(pts)).float()  # (T1,3,2) in 128-px coords
            ok = np.array(ok)
            X = F.interpolate((img * 255), size=(1024, 1024), mode="bicubic", align_corners=False).clamp(0, 255)
            X = ((X - mean) / std).to(dtype)
            outm = []
            for i in range(0, T1, bs):
                emb = model.get_image_embeddings(X[i:i + bs])
                p = (pts[i:i + bs] * 8.0).to(device).to(dtype)[:, None]  # (b,1,3,2) in 1024-px coords
                lab = torch.ones(p.shape[:3], device=device, dtype=torch.long)
                o = model(image_embeddings=emb, input_points=p, input_labels=lab, multimask_output=False)
                lo = o.pred_masks[:, 0].float()  # (b,1,256,256) logits
                outm.append((F.interpolate(lo, size=(128, 128), mode="bilinear", align_corners=False)[:, 0] > 0).cpu().numpy())
            m = np.concatenate(outm) & ok[:, None, None]
            masks[e] = m
            gt = arm[e]
            inter = (m & gt).sum((1, 2)); uni = (m | gt).sum((1, 2))
            iou_l.append(inter / np.maximum(uni, 1))
            inc_l.append(((m & alpha).sum((1, 2))) / np.maximum(alpha.sum((1, 2)), 1))   # fraction of decoy pixels SAM put in the arm mask
            cov_l.append(((m & gt).sum((1, 2))) / np.maximum(gt.sum((1, 2)), 1))
            if e % 50 == 0:
                print(f"{kind} {split} ep {e}/{N} iou {np.mean(iou_l):.3f} decoy_in_mask {np.mean(inc_l):.3f} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(out, f"sam_{D.MASK_PREFIX}{kind}_{split}.npz"), mask=np.packbits(masks, axis=-1))
        summary[split] = dict(iou_vs_exact_arm=float(np.mean(iou_l)), decoy_pixels_in_mask=float(np.mean(inc_l)),
                              arm_recall=float(np.mean(cov_l)), iou_per_episode=[float(x.mean()) for x in iou_l][:2000], n_eps=N)
        print(kind, split, {k: v for k, v in summary[split].items() if k != "iou_per_episode"}, flush=True)
    json.dump(summary, open(os.path.join(out, f"sam_summary_{D.MASK_PREFIX}{kind}.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=os.path.expanduser("~/work/data/b_mw/masks"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    run(a.kind, a.splits.split(","), a.out, a.limit, "cpu" if a.cpu else "cuda")
