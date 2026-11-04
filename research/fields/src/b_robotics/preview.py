"""Montage + overlap statistics of the four distractor kinds (CPU).  python preview.py out.png"""
import sys, json, numpy as np, torch, imageio
import data as Dt, distractors as D

def main(out="preview.png"):
    ep = Dt.load_split("test"); gp = Dt.global_params()
    print("M_ee (px per unit action)\n", gp["M_ee"].round(2), "indep_std", round(gp["indep_std"], 2))
    arm = Dt.unpack(ep["arm"])
    stats = {}
    rows = []
    for kind in D.KINDS:
        st = Dt.dist_states("test", kind)
        e = 7
        row = []
        occ = []
        for t in [0, 4, 8, 12, 16]:
            clean = torch.from_numpy(ep["frames"][e, t]).permute(2, 0, 1)[None].float() / 255
            img, al = D.composite(clean, torch.from_numpy(st["pos"][e, t])[None], torch.from_numpy(st["cls"][e])[None])
            row.append((img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        rows.append(np.concatenate(row, 1))
        # overlap statistic over all test frames: fraction of arm pixels covered by sprites, fraction of sprite pixels on the arm
        cov, onarm = [], []
        for e2 in range(0, 300, 3):
            clean = torch.from_numpy(ep["frames"][e2]).permute(0, 3, 1, 2).float() / 255
            _, al = D.composite(clean, torch.from_numpy(st["pos"][e2]), torch.from_numpy(st["cls"][e2])[None].repeat(17, 1))
            a = (al[:, 0] > 0.5).numpy(); m = arm[e2]
            cov.append((a & m).sum() / max(m.sum(), 1)); onarm.append((a & m).sum() / max(a.sum(), 1))
        stats[kind] = dict(arm_covered_by_sprites=float(np.mean(cov)), sprite_pixels_on_arm=float(np.mean(onarm)) if kind != "none" else 0.0)
    print(json.dumps(stats, indent=1))
    imageio.imwrite(out, np.concatenate(rows, 0))

if __name__ == "__main__":
    main(*sys.argv[1:])
