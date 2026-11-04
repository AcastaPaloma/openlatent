"""The sealed planted-leak manifest: build, commit-hash, generate releases.

    build     python -m n_control.manifest build --calib ~/n_control_data/calib.json --base-record ~/n_control_data/base_record.json
                                               --out ~/sealed_manifest/manifest.json
              writes the manifest as CANONICAL JSON (so sha256(file) == sha256(canon(obj))), mode 0400, prints the sha256 to commit.
    generate  python -m n_control.manifest generate --manifest ~/sealed_manifest/manifest.json --base-dir ~/n_control_data
                                               --out-dir ~/n_releases --sealed-dir ~/sealed_manifest --procs 8
              builds every release deterministically from the manifest, names it by its digest, writes the SEALED digest -> entry map and the sealed
              oracle report (mode 0400), prints the sha256 of the sorted digest list to commit before any certificate is run.

Entry fields: rid, kind (plant | null | shuf), family, level (1..5; 0 for null), base (msda | rep), amp, rep, assign_seed, key (hex), shuffle_seed.
Design counts (also in N.md): plant = 2 bases x 6 families x 5 levels x REPS; null = 2 bases x N_NULL; shuf = 2 bases x 6 families x len(SHUF_LEVELS) x N_SHUF.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import stat
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from .base import BASES, load_base  # noqa: E402
from .common import FAMILIES, LEVELS, canon, sha256_hex  # noqa: E402
from .release import build_release, digest, oracle_report, save  # noqa: E402

REPS = 12            # plant replicates per (base, family, level)
N_NULL = 60          # null releases per base (unplanted, random pseudo-identities)
SHUF_LEVELS = (2, 4, 5)
N_SHUF = 6           # label-shuffled planted twins per (base, family, shuffle level)
VERSION = "N-manifest-v1"


def _seed():
    return int.from_bytes(os.urandom(8), "big") >> 2


def build(calib_path, base_record_path, out_path, reps=REPS, n_null=N_NULL, n_shuf=N_SHUF, shuf_levels=SHUF_LEVELS):
    calib = json.load(open(os.path.expanduser(calib_path)))
    brec = json.load(open(os.path.expanduser(base_record_path)))
    entries = []
    for base in BASES:
        for fam in FAMILIES:
            for lvl in LEVELS:
                amp = calib["bases"][base][fam]["levels"][lvl - 1]["amp"]
                for r in range(reps):
                    entries.append(dict(kind="plant", family=fam, level=lvl, base=base, amp=float(amp), rep=r, assign_seed=_seed(),
                                        key=os.urandom(16).hex(), shuffle_seed=0))
            for lvl in shuf_levels:
                amp = calib["bases"][base][fam]["levels"][lvl - 1]["amp"]
                for r in range(n_shuf):
                    entries.append(dict(kind="shuf", family=fam, level=lvl, base=base, amp=float(amp), rep=r, assign_seed=_seed(),
                                        key=os.urandom(16).hex(), shuffle_seed=_seed()))
        for r in range(n_null):
            entries.append(dict(kind="null", family=None, level=0, base=base, amp=0.0, rep=r, assign_seed=_seed(), key=os.urandom(16).hex(), shuffle_seed=0))
    order = np.random.RandomState(int.from_bytes(os.urandom(4), "big")).permutation(len(entries))
    entries = [entries[i] for i in order]
    for i, e in enumerate(entries):
        e["rid"] = i
    man = dict(version=VERSION, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), base_record=brec,
               design=dict(reps=reps, n_null=n_null, n_shuf=n_shuf, shuf_levels=list(shuf_levels), families=list(FAMILIES), bases=list(BASES)),
               calibration={b: {f: calib["bases"][b][f]["levels"] for f in FAMILIES} for b in BASES}, entries=entries)
    raw = canon(man)
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite an existing manifest")
    with open(out_path, "wb") as f:
        f.write(raw)
    os.chmod(out_path, stat.S_IRUSR)
    kinds = {k: sum(e["kind"] == k for e in entries) for k in ("plant", "shuf", "null")}
    print(json.dumps(dict(sha256=sha256_hex(raw), bytes=len(raw), n_entries=len(entries), kinds=kinds), indent=1))


def _gen(args):
    e, base_dir, out_dir = args
    b = load_base(os.path.join(os.path.expanduser(base_dir), f"base_{e['base']}.npz"))
    rel = build_release(b, e)
    d = digest(rel)
    save(rel, os.path.join(out_dir, f"{d}.npz"))
    rep = oracle_report(b, e, rel)
    return e["rid"], d, rep


def generate(manifest_path, base_dir, out_dir, sealed_dir, procs=8):
    raw = open(os.path.expanduser(manifest_path), "rb").read()
    man = json.loads(raw)
    assert canon(man) == raw, "manifest is not canonical"
    out_dir = os.path.expanduser(out_dir)
    sealed_dir = os.path.expanduser(sealed_dir)
    os.makedirs(out_dir, exist_ok=True)
    with mp.Pool(procs) as pool:
        res = pool.map(_gen, [(e, base_dir, out_dir) for e in man["entries"]], chunksize=2)
    digests = sorted(d for _, d, _ in res)
    assert len(set(digests)) == len(digests), "duplicate release digests"
    mp_ = {d: rid for rid, d, _ in res}
    orc = {str(rid): rep for rid, _, rep in res}
    for name, obj in (("release_map.json", mp_), ("oracle.json", orc)):
        p = os.path.join(sealed_dir, name)
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
        with open(p, "wb") as f:
            f.write(canon(obj))
        os.chmod(p, stat.S_IRUSR)
    idx = "\n".join(digests) + "\n"
    open(os.path.join(out_dir, "INDEX.txt"), "w").write(idx)
    print(json.dumps(dict(n_releases=len(digests), digest_list_sha256=sha256_hex(idx.encode()), manifest_sha256=sha256_hex(raw)), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "generate"])
    ap.add_argument("--calib", default="~/n_control_data/calib.json")
    ap.add_argument("--base-record", default="~/n_control_data/base_record.json")
    ap.add_argument("--out", default="~/sealed_manifest/manifest.json")
    ap.add_argument("--manifest", default="~/sealed_manifest/manifest.json")
    ap.add_argument("--base-dir", default="~/n_control_data")
    ap.add_argument("--out-dir", default="~/n_releases")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.calib, a.base_record, a.out)
    else:
        generate(a.manifest, a.base_dir, a.out_dir, a.sealed_dir, a.procs)
