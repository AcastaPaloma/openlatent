#!/usr/bin/env python
"""Extract frozen-encoder embeddings for the three PathoROB datasets (GPU; run through the queue).

    python -m research.fields.src.a_pathorob.extract --model phikonv2_clsmean
Reads parquet shards from $PATHOROB_DATA row group by row group (order == parquet order), applies the official PathoROB
preprocessing (Resize(224), CenterCrop(224), ToTensor, Normalize) and writes
$PATHOROB_CACHE/<model>/<dataset>__<key>.npy atomically.  A small JSON summary goes to $OUT (no embeddings).
"""
import argparse, io, json, os, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from a_pathorob import common as C


class RowGroupDS(torch.utils.data.Dataset):
    def __init__(self, files, tf):
        self.tf = tf
        self.items = [(str(f), g) for f in files for g in range(pq.ParquetFile(str(f)).num_row_groups)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, g = self.items[i]
        col = pq.ParquetFile(f).read_row_group(g, columns=["image"]).column("image").to_pylist()
        return torch.stack([self.tf(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in col])


def load_model(spec):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(spec["hf"], revision=spec["revision"])
    return m.eval()


@torch.no_grad()
def run(model_name, datasets, num_workers, force):
    spec = C.ENCODERS[model_name]
    mean, std = C.NORMS[spec["norm"]]
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    todo = [d for d in datasets if force or not C.emb_path(model_name, d).exists()]
    summary = dict(model=model_name, spec=spec, datasets={}, commit=C.git_commit())
    if todo:
        model = load_model(spec).cuda()
        amp = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[spec["dtype"]]
    for d in datasets:
        p = C.emb_path(model_name, d)
        if d not in todo:
            print(f"[{model_name}/{d}] cached at {p}", flush=True)
            summary["datasets"][d] = dict(cached=True, path=str(p))
            continue
        t0 = time.time()
        C.load_index(d)  # also writes the row-order index
        ds = RowGroupDS(C.parquet_files(d), tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers, prefetch_factor=4)
        feats = []
        for i, x in enumerate(dl):
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = model(pixel_values=x).last_hidden_state
            f = torch.cat([h[:, 0].float(), h[:, 1:].float().mean(1)], dim=1)
            feats.append(f.cpu())
            if i % 50 == 0:
                print(f"[{model_name}/{d}] {i}/{len(dl)} {time.time()-t0:.0f}s", flush=True)
        X = torch.cat(feats).numpy().astype(np.float32)
        idx = C.load_index(d)
        assert len(X) == len(idx), (len(X), len(idx))
        assert np.isfinite(X).all(), "non-finite embeddings"
        p.parent.mkdir(parents=True, exist_ok=True)
        C.atomic_save_npy(p, X)
        summary["datasets"][d] = dict(n=int(X.shape[0]), dim=int(X.shape[1]), seconds=round(time.time() - t0, 1),
                                      mean_norm=float(np.linalg.norm(X, axis=1).mean()), path=str(p))
        print(f"[{model_name}/{d}] saved {X.shape} in {time.time()-t0:.0f}s -> {p}", flush=True)
    out = Path(os.environ.get("OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"extract_{model_name}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.ENCODERS))
    ap.add_argument("--datasets", nargs="+", default=C.DATASETS)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.model, a.datasets, a.num_workers, a.force)
