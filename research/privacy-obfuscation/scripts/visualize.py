"""Side-by-side panels: original frames vs every anonymizer, with intermediates for the learned ones.

For each chosen test clip, one PNG with a grid of panels (all showing the same frame of the clip):
fixed filters, scene-only / person-only controls, person-box blackout, learned BDQ broken into its
stages (learned blur -> temporal difference -> learned quantisation), and the frozen-VAE anonymizer
(plain reconstruction, learned latent edit, and |edit - reconstruction|).

Learned anonymizers are loaded from the HF checkpoint repo (runs/<name>/anonymizer.pt).
Run from research/privacy-obfuscation on a GPU VM:
    python3 scripts/visualize.py --dataset kth --out ~/work/viz
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("USE_TF", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppv import anonymizers as A  # noqa: E402
from ppv import kth, ucf101  # noqa: E402
from ppv.run import HF_REPO  # noqa: E402

DEV = "cuda"


def load_learned(cfg, run, in_ch):
    from huggingface_hub import hf_hub_download
    f = A.build(cfg, in_ch=in_ch)
    sd = torch.load(hf_hub_download(HF_REPO, f"runs/{run}/anonymizer.pt"), map_location="cpu")
    f.load_state_dict(sd, strict=False)
    return f.to(DEV).eval()


def to_img(t):
    """(C, H, W) in [0,1] or signed -> HxW or HxWx3 numpy for imshow."""
    t = t.float().cpu()
    return t[0].numpy() if t.shape[0] == 1 else t.permute(1, 2, 0).clamp(0, 1).numpy()


@torch.no_grad()
def panels_for(clip_u8, full_video_u8, dataset, learned, mid):
    """clip_u8: (T, H, W[, 3]) uint8. Returns [(title, image, cmap, vmin, vmax)]."""
    x = torch.from_numpy(clip_u8).to(DEV).float() / 255
    x = x.unsqueeze(0).unsqueeze(0) if x.dim() == 3 else x.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
    c = x.shape[1]
    gray = c == 1
    cm = "gray" if gray else None
    out = [("original", to_img(x[0, :, mid]), cm, 0, 1)]
    for title, cfg in (("blur sigma 8", {"type": "blur", "sigma": 8}), ("pixelate x8", {"type": "pixelate", "factor": 8}),
                       ("downsample x16", {"type": "down", "factor": 16}), ("frame difference", {"type": "diff", "gain": 4.0})):
        out.append((title, to_img(A.build(cfg, c)(x)[0, :, mid]), cm, 0, 1))
    if dataset == "kth":
        v = full_video_u8.astype(np.float32)
        med = np.median(v, 0)
        out.append(("scene only (median)", med / 255, cm, 0, 1))
        fg = np.where(np.abs(v - med) > 20, v, 0)
        out.append(("person only (fg mask)", fg[len(fg) // 2] / 255, cm, 0, 1))
    pb = A.PersonBox(every=1, dilate=6).to(DEV)
    out.append(("person box blackout", to_img(pb(x)[0, :, mid]), cm, 0, 1))
    if "bdq" in learned:
        m = learned["bdq"][1]
        sigma = m.log_sigma.exp().clamp(0.3, 8.0)
        blurred = A.blur2d(x, sigma, size=13)
        d = (A.tdiff(blurred) * m.gain).clamp(-1, 1)
        q = m(x)
        out += [(f"BDQ 1: learned blur (sigma {float(sigma):.2f})", to_img(blurred[0, :, mid]), cm, 0, 1),
                ("BDQ 2: temporal difference", to_img(d[0, :, mid])[..., 0] if not gray else to_img(d[0, :, mid]), "RdBu", -1, 1),
                (f"BDQ 3: quantised ({learned['bdq'][0]})", to_img(q[0, :, mid]), cm, 0, 1)]
    if "vae" in learned:
        recon = A.VAELatent(out_gray=gray).to(DEV).eval()  # zero-initialised edit = plain reconstruction
        r, e = recon(x), learned["vae"][1](x)
        out += [("VAE reconstruction (no edit)", to_img(r[0, :, mid]), cm, 0, 1),
                (f"VAE latent edit ({learned['vae'][0]})", to_img(e[0, :, mid]), cm, 0, 1),
                ("|edit - reconstruction| x4", (4 * (e - r).abs()[0, :, mid].mean(0)).clamp(0, 1).cpu().numpy(), "magma", 0, 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kth", choices=["kth", "ucf101_vp"])
    ap.add_argument("--out", default=os.path.expanduser("~/work/viz"))
    ap.add_argument("--bdq", default=None, help="run name of a learned BDQ anonymizer")
    ap.add_argument("--vae", default=None, help="run name of a learned VAE latent-edit anonymizer")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "kth":
        root = os.path.expanduser("~/work/data/kth")
        rows = kth.load_meta(root)
        _, te = kth.split(rows, "ilic")
        picks = []
        for a in ("walking", "boxing", "handwaving", "running", "jogging", "handclapping")[: args.n]:
            picks.append(next(r for r in te if r["action"] == a and r["person"] in (2, 5, 9)))
        frames = kth.load_frames(root, [r["video"] for r in picks])
        in_ch, stride = 1, 2
    else:
        tr, te, frames = ucf101.load(os.path.expanduser("~/work/data/ucf101"))
        idx = np.linspace(0, len(te) - 1, args.n).astype(int)
        picks = [te[i] for i in idx]
        in_ch, stride = 3, 1

    learned = {}
    if args.bdq:
        learned["bdq"] = (args.bdq, load_learned({"type": "bdq"}, args.bdq, in_ch))
    if args.vae:
        learned["vae"] = (args.vae, load_learned({"type": "vae_latent", "out_gray": in_ch == 1}, args.vae, in_ch))

    for r in picks:
        arr = np.asarray(frames[r["video"]])
        a, b = r["start"] - 1, r["end"] - 1
        s = max(a, (a + b) // 2 - 16)
        clip = arr[s: s + 32: stride]
        full = arr[a: b + 1]
        ps = panels_for(clip, full, args.dataset, learned, mid=len(clip) // 2)
        cols = 5
        rows_n = (len(ps) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 2.8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (title, img, cmap, vmin, vmax) in zip(axes.ravel(), ps):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
        label = r["video"].replace("/", "_")
        fig.suptitle(f"{args.dataset}: {label}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(args.out, f"{args.dataset}_{label}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
