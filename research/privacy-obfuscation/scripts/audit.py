"""Leakage audit: is high identity accuracy on obfuscated KTH real leakage, or an evaluation bug?

1. Split check: train and test share no video; reports how many test videos share a recording session
   (same person + scenario) with some train video (a legitimate scene cue, not a bug).
2. Pixel nearest neighbour: one obfuscated frame per subsequence (the centre of a 16-frame clip), compared by
   cosine similarity of contrast-normalised 30x40 thumbnails. Identity accuracy of "copy the nearest
   train frame's label" shows how much identity is readable without any learned model.
3. Figures: for each technique, test frames as released, the same frames contrast-stretched (min-max per
   frame, which exposes faint structure), and the nearest train frame with its identity and whether it
   comes from the same recording session.

Run from research/privacy-obfuscation on a GPU VM:
    python3 scripts/audit.py --out ~/work/audit
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("USE_TF", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppv import anonymizers as A  # noqa: E402
from ppv import kth  # noqa: E402
from ppv.run import HF_REPO  # noqa: E402

DEV = "cuda"
TECHNIQUES = [
    ("raw", {"type": "raw"}, None),
    ("blur8", {"type": "blur", "sigma": 8}, None),
    ("down16", {"type": "down", "factor": 16}, None),
    ("down64", {"type": "down", "factor": 64}, None),
    ("diff", {"type": "diff", "gain": 4.0}, None),
    ("person_box", {"type": "person_box", "every": 1, "dilate": 6}, None),
    ("bdq_a5_ens", {"type": "bdq"}, "p2_bdq_a5_ens"),
    ("vae_a1_ens", {"type": "vae_latent", "out_gray": True}, "p2_vae_a1_ens"),
]


def build(cfg, src):
    f = A.build(cfg, in_ch=1)
    if src:
        from huggingface_hub import hf_hub_download
        f.load_state_dict(torch.load(hf_hub_download(HF_REPO, f"runs/{src}/anonymizer.pt"), map_location="cpu"), strict=False)
    return f.to(DEV).eval()


@torch.no_grad()
def centre_frames(f, rows, frames, batch=32):
    """Apply f to a 16-frame clip (stride 2) centred in each row; return the obfuscated centre frame per row."""
    out = []
    for i in range(0, len(rows), batch):
        clips = []
        for r in rows[i:i + batch]:
            arr = frames[r["video"]]
            a, b = r["start"] - 1, r["end"] - 1
            s = max(a, min((a + b) // 2 - 15, b - 30))
            idx = np.clip(np.arange(s, s + 31, 2), 0, len(arr) - 1)
            clips.append(arr[idx])
        x = torch.from_numpy(np.stack(clips)).to(DEV).float().div(255).unsqueeze(1)  # (B,1,16,H,W)
        y = f(x).float().clamp(0, 1)
        y = (y * 255).round() / 255  # as released (8-bit)
        out.append(y[:, 0, 8].cpu())
    return torch.cat(out)  # (N, H, W)


def thumbs(imgs):
    t = F.interpolate(imgs.unsqueeze(1), size=(30, 40), mode="area").flatten(1)
    t = t - t.mean(1, keepdim=True)
    return t / t.norm(dim=1, keepdim=True).clamp(min=1e-6)


def stretch(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/work/audit"))
    ap.add_argument("--split", default="ilic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = os.path.expanduser("~/work/data/kth")
    rows = kth.load_meta(root)
    tr, te = kth.split(rows, args.split)
    frames = kth.load_frames(root, [r["video"] for r in rows])

    shared_videos = {r["video"] for r in tr} & {r["video"] for r in te}
    sessions_tr = {(r["person"], r["scenario"]) for r in tr}
    te_videos = sorted({r["video"] for r in te})
    te_sess = {v: (int(v.split("_")[0][6:]), int(v.split("_")[2][1:])) for v in te_videos}
    report = dict(split=args.split, n_train_rows=len(tr), n_test_rows=len(te), shared_videos=len(shared_videos),
                  test_videos=len(te_videos),
                  test_videos_sharing_session_with_train=sum(te_sess[v] in sessions_tr for v in te_videos))
    print(json.dumps(report))

    y_tr = torch.tensor([r["person_id"] for r in tr])
    y_te = torch.tensor([r["person_id"] for r in te])
    a_tr = torch.tensor([r["action_id"] for r in tr])
    a_te = torch.tensor([r["action_id"] for r in te])
    rng = np.random.default_rng(0)
    show = sorted(rng.choice(len(te), 4, replace=False))
    results = {}
    for name, cfg, src in TECHNIQUES:
        f = build(cfg, src)
        im_tr, im_te = centre_frames(f, tr, frames), centre_frames(f, te, frames)
        sim = thumbs(im_te) @ thumbs(im_tr).T
        nn = sim.argmax(1)
        same_session = [(tr[j]["person"], tr[j]["scenario"]) == (te[i]["person"], te[i]["scenario"]) for i, j in enumerate(nn.tolist())]
        results[name] = dict(nn_identity=100 * float((y_tr[nn] == y_te).float().mean()),
                             nn_action=100 * float((a_tr[nn] == a_te).float().mean()),
                             nn_same_session=100 * float(np.mean(same_session)),
                             pixel_std=float(im_te.std()))
        print(name, json.dumps(results[name]))

        fig, axes = plt.subplots(len(show), 4, figsize=(12, 2.6 * len(show)))
        for k, i in enumerate(show):
            j = int(nn[i])
            raw_te = frames[te[i]["video"]][(te[i]["start"] + te[i]["end"]) // 2 - 1] / 255
            panels = [(raw_te, f"test raw: person {te[i]['person']} {te[i]['action']} d{te[i]['scenario']}"),
                      (im_te[i].numpy(), f"{name} (as released)"),
                      (stretch(im_te[i].numpy()), f"{name} contrast-stretched"),
                      (stretch(im_tr[j].numpy()), f"nearest TRAIN: person {tr[j]['person']} {tr[j]['action']} d{tr[j]['scenario']}"
                       + (" (same person)" if tr[j]["person"] == te[i]["person"] else ""))]
            for ax, (img, title) in zip(axes[k], panels):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
        fig.suptitle(f"{name}: pixel nearest-neighbour identity {results[name]['nn_identity']:.1f}% "
                     f"(same recording session as the neighbour in {results[name]['nn_same_session']:.0f}% of test clips)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"audit_{args.split}_{name}.png"), dpi=90)
        plt.close(fig)
        del f
        torch.cuda.empty_cache()
    report["techniques"] = results
    json.dump(report, open(os.path.join(args.out, f"audit_{args.split}.json"), "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
