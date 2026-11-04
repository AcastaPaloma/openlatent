"""Build the 256x256 uint8 image bank from the HF mirror zips (post-gate only).  Never runs the repo's loading script; downloads the 12 zips
(`data/images/images_00N.zip`) with huggingface_hub, resizes with PIL bilinear (the same op as torchvision Resize((256,256)) on a PIL image, as
PriCheXy-Net does), writes `cxr256.npy` (N,256,256 uint8, row order = metadata row order of t_cxr.meta.load) and `prep.json` (integrity report).
Integrity (FOCUS A4 / whitelist condition d; no official copy is reachable, so this is weaker than 'against an official copy'): every CSV image id present,
count 112,120, every PNG 1024x1024, per-zip sha256, exact-duplicate count by sha256 of the decoded 256-px array."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import meta

REPO = "alkzar90/NIH-Chest-X-ray-dataset"


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _process_zip(args):
    """worker: decode every PNG of one zip, resize, write into the shared memmap rows; returns (zip name, written row ids, bad_dims, missing_in_csv)."""
    zpath, bank_path, row, size = args
    bank = np.load(bank_path, mmap_mode="r+")
    written, bad, miss = [], 0, 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            base = Path(info.filename).name
            if not base.endswith(".png"):
                continue
            if base not in row:
                miss += 1
                continue
            im = Image.open(io.BytesIO(z.read(info)))
            if im.size != (1024, 1024):
                bad += 1
            bank[row[base]] = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR))
            written.append(row[base])
    bank.flush()
    return Path(zpath).name, written, bad, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with Data_Entry_2017_v2020.csv, test_list.txt, train_val_list.txt")
    ap.add_argument("--out", required=True, help="output dir for cxr256.npy and prep.json (NOT the HF-uploaded $OUT)")
    ap.add_argument("--zips", default=None, help="dir holding images_00N.zip (default: download from the HF mirror into <out>/zips)")
    ap.add_argument("--report", default=None, help="small json report copied to $OUT (uploaded)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", type=int, default=12, help="number of zips to process (tests only; the integrity check then fails by design)")
    a = ap.parse_args()
    from multiprocessing import Pool
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt")
    row = dict(zip(d["img"], d["row"]))
    N = len(d)
    bank_path = out / "cxr256.npy"
    bank = np.lib.format.open_memmap(bank_path, mode="w+", dtype=np.uint8, shape=(N, a.size, a.size)); del bank
    zdir = Path(a.zips) if a.zips else out / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    rep = dict(repo=REPO, zips={}, bad_dims=0, missing_in_csv=0, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    from huggingface_hub import hf_hub_download
    paths = []
    for k in range(1, a.only + 1):
        name = f"images_{k:03d}.zip"
        p = zdir / name
        if not p.exists():
            got = hf_hub_download(REPO, f"data/images/{name}", repo_type="dataset", local_dir=str(zdir / "_hf"))
            Path(got).rename(p)
        rep["zips"][name] = dict(bytes=p.stat().st_size, sha256=sha256_file(p))
        print("downloaded", name, rep["zips"][name]["bytes"], flush=True)
        paths.append(p)
    filled = np.zeros(N, bool)
    with Pool(min(a.procs, len(paths))) as pool:
        for name, written, bad, miss in pool.imap_unordered(_process_zip, [(str(p), str(bank_path), row, a.size) for p in paths]):
            filled[written] = True; rep["bad_dims"] += bad; rep["missing_in_csv"] += miss
            rep["zips"][name]["images"] = len(written)
            print(name, "done; filled so far", int(filled.sum()), flush=True)
    bank = np.load(bank_path, mmap_mode="r")
    rep["filled"] = int(filled.sum()); rep["expected"] = N
    h = [hashlib.sha256(bank[i].tobytes()).hexdigest() for i in range(N)]
    rep["exact_duplicate_images_256px"] = int(N - len(set(h)))
    rep["ok"] = bool(filled.all() and rep["missing_in_csv"] == 0 and N == 112120 and rep["bad_dims"] == 0)
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "prep.json").write_text(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).mkdir(parents=True, exist_ok=True)
        (Path(a.report) / "prep.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "zips"}))
    if not rep["ok"]:
        raise SystemExit("integrity check failed")


if __name__ == "__main__":
    main()
