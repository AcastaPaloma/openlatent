"""Shared paths, hashing, atomic IO and the encoder registry for Focus A (PathoROB).

Nothing in here touches the GPU.  Embeddings are cached under $PATHOROB_CACHE (default ~/work/cache/pathorob) keyed by
encoder spec + data hash, written atomically (tmp file then os.replace).  Embeddings are never written to $OUT, so the
queue worker never uploads them to HF (some encoders have ND licences).
"""
import hashlib, json, os, subprocess, time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("PATHOROB_DATA", "~/work/data/pathorob")).expanduser()
CACHE = Path(os.environ.get("PATHOROB_CACHE", "~/work/cache/pathorob")).expanduser()
PATHOROB_REPO = Path(os.environ.get("PATHOROB_REPO", "~/work/ext/PathoROB")).expanduser()
DATASETS = ["camelyon", "tcga", "tolkach_esca"]

# Registry.  `revision` pins the HF commit; `pre` is the preprocessing (all: Resize(224), CenterCrop(224), ToTensor,
# Normalize).  Feature = concat(CLS, mean of patch tokens) for every encoder (PathoROB "clsmean").
ENCODERS = {
    "phikonv2_clsmean": dict(hf="owkin/phikon-v2", revision="2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd",
                             norm="imagenet", dtype="fp32", licence="Owkin non-commercial (other)"),
    "phikon_clsmean": dict(hf="owkin/phikon", revision="057cc0295895c2df3dd7681a89680da6015cbefe",
                           norm="imagenet", dtype="fp32", licence="Owkin non-commercial"),
    "midnight_clsmean": dict(hf="kaiko-ai/midnight", revision="adc6b15679c981cce6f9b018bbad09d16eeeda9f",
                             norm="half", dtype="bf16", licence="MIT"),
    "dinov2L_clsmean": dict(hf="facebook/dinov2-large", revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                            norm="imagenet", dtype="fp32", licence="Apache-2.0"),
}
NORMS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), "half": ([0.5] * 3, [0.5] * 3)}


def parquet_files(dataset):
    return sorted((DATA / dataset / "data").glob("*.parquet"))


def data_hash(dataset):
    """sha256 over the parquet shards; cached in a sidecar keyed by (name, size, mtime)."""
    side = CACHE / "data_hash.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = json.loads(side.read_text()) if side.exists() else {}
    h = hashlib.sha256()
    for f in parquet_files(dataset):
        st = f.stat()
        key = f"{f}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            fh = hashlib.sha256()
            with open(f, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 24), b""):
                    fh.update(chunk)
            cache[key] = fh.hexdigest()
            atomic_write_text(side, json.dumps(cache, indent=1))
        h.update(cache[key].encode())
    return h.hexdigest()


def enc_key(model, dataset):
    spec = ENCODERS[model]
    s = json.dumps(dict(spec, model=model, pre="Resize224-CenterCrop224"), sort_keys=True) + "|" + data_hash(dataset)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_save_npy(path, arr):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def emb_path(model, dataset):
    return CACHE / model / f"{dataset}__{enc_key(model, dataset)}.npy"


def index_path(dataset):
    return CACHE / f"index_{dataset}__{data_hash(dataset)[:12]}.parquet"


def load_index(dataset):
    import pandas as pd
    p = index_path(dataset)
    if not p.exists():
        import pyarrow.parquet as pq
        t = pq.read_table([str(f) for f in parquet_files(dataset)][0], columns=["slide_id"]).schema  # touch schema
        frames = [pq.read_table(str(f), columns=["slide_id", "patch_id", "biological_class", "medical_center"]).to_pandas()
                  for f in parquet_files(dataset)]
        df = pd.concat(frames, ignore_index=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        df.to_parquet(tmp)
        os.replace(tmp, p)
    return pd.read_parquet(p)


def load_embeddings(model, dataset):
    p = emb_path(model, dataset)
    if not p.exists():
        raise FileNotFoundError(f"no cached embeddings for {model}/{dataset} at {p}; run extract.py through the queue")
    return np.load(p)


def git_commit(path=None):
    try:
        return subprocess.run(["git", "-C", str(path or Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"
