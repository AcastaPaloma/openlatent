"""Contact sheet: candidate obfuscators x six KTH actions (person 25), d1 and d2."""
import json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
from make_stimuli import fetch, window, activity, ranked, run, save_gif, parse_sequences, CTX, SPAN, N_FRAMES, HF_REPO
from ppv.anonymizers import build, frozen_key
from huggingface_hub import hf_hub_download

S = sys.argv[1]
CFG = S + "/pr/research/privacy-obfuscation/configs"
def cfg(run_name):
    import subprocess
    out = subprocess.run(["git", "-C", "/Users/kuanw/Me/code/openlatent", "show",
                          f"privacy-research:research/privacy-obfuscation/results/runs/{run_name}/result.json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["config"]["anonymizer"]
CANDS = [("raw", None, None), ("blur8", "p1_blur8", {"type": "blur", "sigma": 8}),
         ("bdq_a3_ens", "p2_bdq_a3_ens", None), ("bdq_a3_lr_s1", "p2_bdq_a3_lr3e-3_s1", None),
         ("mask_bdq", "p11_kth_mask_k25_bdq_a1", None), ("mask_cut", "p11_kth_mask_k25_a1", None),
         ("unet", "p2_unet_a1_ens", None), ("pose", "p12_kth_pose_canon_e16", None)]
fs = {}
for name, r, c in CANDS:
    if r is None:
        fs[name] = None; continue
    c = c or cfg(r)
    f = build(c, in_ch=1).eval()
    if name not in ("blur8", "pose"):
        sd = torch.load(hf_hub_download(HF_REPO, f"runs/{r}/anonymizer.pt"), map_location="cpu")
        miss, unexp = f.load_state_dict(sd, strict=False)
        assert not [k for k in miss if not frozen_key(k)] and not unexp, (r, miss, unexp)
    fs[name] = f
    print("loaded", name, c, flush=True)
seqs = parse_sequences(S + "/kth/00sequences.txt")
ACTIONS = ["boxing", "handclapping", "handwaving", "walking", "jogging", "running"]
stats = {}
for d in (1, 2):
    rows = []
    for a in ACTIONS:
        v = f"person25_{a}_d{d}"
        raw = fetch(S + "/kth", v)
        _, s = ranked(seqs, v, run(fs["bdq_a3_ens"], raw))[0]
        lo, hi = max(0, s - CTX), min(len(raw), s + SPAN + CTX)
        row = []
        for name, _, _ in CANDS:
            q = window(raw, s) if fs[name] is None else window(run(fs[name], raw[lo:hi]), s - lo)
            stats[f"d{d} {a} {name}"] = round(100 * activity(q).mean(), 2)
            row.append(q)
        rows.append(row)
        print("rendered", v, flush=True)
    H, W, g = 120, 160, 4
    out = np.full((N_FRAMES, 18 + len(ACTIONS) * (H + g), 70 + len(CANDS) * (W + g)), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, q in enumerate(row):
            out[:, 18 + i * (H + g):18 + i * (H + g) + H, 70 + j * (W + g):70 + j * (W + g) + W] = q
    frames = []
    for f in out:
        im = Image.fromarray(f); dr = ImageDraw.Draw(im)
        for j, (name, _, _) in enumerate(CANDS):
            dr.text((72 + j * (W + g), 3), name, fill=0)
        for i, a in enumerate(ACTIONS):
            dr.text((2, 18 + i * (H + g) + 55), a, fill=0)
        frames.append(np.asarray(im))
    save_gif(f"{S}/filters_d{d}.gif", np.stack(frames))
    act = np.stack(frames).astype(int)
    k = int(np.argmax(np.abs(act - np.median(act, 0)).sum((1, 2))))
    Image.fromarray(frames[N_FRAMES // 2]).save(f"{S}/filters_d{d}.png")
json.dump(stats, open(f"{S}/filters_stats.json", "w"), indent=0)
