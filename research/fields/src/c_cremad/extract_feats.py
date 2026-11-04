"""Cache frozen SSL features for CREMA-D (audio track).  GPU job (queue prefix c_).

  python extract_feats.py --model microsoft/wavlm-base-plus --name wavlm-base-plus --layers 3 6 9 12

Writes (atomically, tmp -> rename) to $CREMAD_FEATS/<name>/ :
  frames.npy   [Nframes_total, len(layers), D] float16   frame-level hidden states of the chosen layers
  offsets.npy  [N+1] int64                                  clip c occupies frames[offsets[c]:offsets[c+1]]
  stats.npy    [N, n_hidden, 2, D] float32                  per-clip mean / std over time for EVERY hidden state
  clip_ids.txt, wav_sha1.txt, info.json
Derived features of an ODbL database: kept on the VM disk, NOT written to $OUT (so never uploaded by the worker).
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np, torch
import scipy.io.wavfile as wavfile
from transformers import AutoFeatureExtractor, AutoModel
import data


def atomic_save(path, arr):
    tmp = str(path) + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--name", default="wavlm-base-plus")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 12])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    outdir = data.FEAT_ROOT / a.name
    outdir.mkdir(parents=True, exist_ok=True)
    meta = data.load_meta()
    ids = meta["ids"][: a.limit] if a.limit else meta["ids"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    print("model", a.model, "do_normalize", getattr(fe, "do_normalize", None), "device", dev, flush=True)
    frames, offs, stats, sha = [], [0], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, cid in enumerate(ids):
            p = data.DATA_ROOT / "AudioWAV" / f"{cid}.wav"
            raw = p.read_bytes()
            sha.append(hashlib.sha1(raw).hexdigest())
            sr, x = wavfile.read(p)
            assert sr == 16000 and x.ndim == 1, (cid, sr, x.shape)
            x = x.astype(np.float32) / 32768.0
            if getattr(fe, "do_normalize", False):
                x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            out = model(torch.from_numpy(x)[None].to(dev), output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 0)[:, 0].float()  # [L+1, T, D]
            stats.append(torch.stack([hs.mean(1), hs.std(1)], 1).cpu().numpy())  # [L+1, 2, D]
            sel = hs[a.layers].permute(1, 0, 2).half().cpu().numpy()  # [T, nl, D]
            frames.append(sel); offs.append(offs[-1] + sel.shape[0])
            if i % 500 == 0:
                print(f"{i}/{len(ids)} {time.time()-t0:.0f}s T={sel.shape[0]}", flush=True)
    atomic_save(outdir / "frames.npy", np.concatenate(frames, 0))
    atomic_save(outdir / "offsets.npy", np.array(offs, dtype=np.int64))
    atomic_save(outdir / "stats.npy", np.stack(stats, 0))
    (outdir / "clip_ids.txt").write_text("\n".join(ids) + "\n")
    (outdir / "wav_sha1.txt").write_text("\n".join(sha) + "\n")
    info = dict(model=a.model, layers=a.layers, n_clips=len(ids), n_frames=int(offs[-1]), dim=int(sel.shape[-1]),
                n_hidden=int(hs.shape[0]), seconds=round(time.time() - t0, 1))
    (outdir / "info.json").write_text(json.dumps(info, indent=1))
    # small manifest goes to $OUT (uploaded); the features themselves stay on the VM
    out = os.environ.get("OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "feats_info.json").write_text(json.dumps(info, indent=1))
    print("done", info, flush=True)


if __name__ == "__main__":
    main()
