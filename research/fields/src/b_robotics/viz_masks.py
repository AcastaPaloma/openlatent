"""Overlay SAM masks (green) and exact silhouette outline on distractor frames.  python viz_masks.py kind split ep out.png [masks_dir]"""
import sys, numpy as np, torch, imageio
import data as Dt, distractors as D
from scipy import ndimage

def main(kind, split, e, out, mdir):
    ep = Dt.load_split(split); st = Dt.dist_states(split, kind)
    sam = Dt.unpack(np.load(f"{mdir}/sam_{kind}_{split}.npz")["mask"]); arm = Dt.unpack(ep["arm"])
    row = []
    for t in [0, 3, 6, 9, 12, 15]:
        clean = torch.from_numpy(ep["frames"][e, t]).permute(2, 0, 1)[None].float() / 255
        img, _ = D.composite(clean, torch.from_numpy(st["pos"][e, t])[None], torch.from_numpy(st["cls"][e])[None])
        im = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ov = im.copy(); m = sam[e, t]; ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        edge = arm[e, t] ^ ndimage.binary_erosion(arm[e, t]); ov[edge] = [255, 255, 0]
        row.append(np.concatenate([im, ov], 0))
    imageio.imwrite(out, np.concatenate(row, 1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
