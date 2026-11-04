"""AFTER unsealing (exploratory): run the key-free reference attackers (reference.py) on the manifest's releases at the primary levels and on the null releases,
and tabulate, per family, which view reads the planted leak. Uses the same release files S's instrument was shown.

    OMP_NUM_THREADS=1 ... python -m n_control.ref_run --manifest ~/sealed_manifest/manifest.json --sealed-dir ~/sealed_manifest --releases ~/n_releases --out ~/n_results --procs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from .base import load_base  # noqa: E402
from .reference import run  # noqa: E402
from .release import load  # noqa: E402

_Q = {}


def _job(a):
    d, base, rel_dir, base_dir = a
    if base not in _Q:
        _Q[base] = load_base(os.path.join(os.path.expanduser(base_dir), f"base_{base}.npz")).stats()["qedges"]
    r = load(os.path.join(os.path.expanduser(rel_dir), f"{d}.npz"))
    return d, run(r, _Q[base], None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/sealed_manifest/manifest.json")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest")
    ap.add_argument("--releases", default="~/n_releases")
    ap.add_argument("--base-dir", default="~/n_control_data")
    ap.add_argument("--out", default="~/n_results")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    assert a.procs <= 8
    man = json.loads(open(os.path.expanduser(a.manifest), "rb").read())
    rmap = json.load(open(os.path.join(os.path.expanduser(a.sealed_dir), "release_map.json")))
    ent = {e["rid"]: e for e in man["entries"]}
    sel = [(d, ent[rid]["base"]) for d, rid in rmap.items() if (ent[rid]["kind"] == "plant" and ent[rid]["level"] >= 4) or ent[rid]["kind"] == "null"]
    with mp.Pool(a.procs) as pool:
        res = dict(pool.imap_unordered(_job, [(d, b, a.releases, a.base_dir) for d, b in sel], chunksize=1))
    out = {}
    for d, r in res.items():
        e = ent[rmap[d]]
        key = f"{e['kind']}|{e['family']}|{e['level']}"
        out.setdefault(key, []).append(r)
    summ = {k: {v: dict(win=float(np.mean([x[v]["win"] for x in L])), sess=float(np.mean([x[v]["sess"] for x in L])), n=len(L)) for v in L[0]} for k, L in out.items()}
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    json.dump(dict(per_release=res, summary=summ), open(os.path.join(os.path.expanduser(a.out), "reference.json"), "w"), indent=1, sort_keys=True)
    for k in sorted(summ):
        print(k, {v: (round(s["win"], 3), round(s["sess"], 3)) for v, s in summ[k].items()})


if __name__ == "__main__":
    main()
