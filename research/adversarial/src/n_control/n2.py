"""N2: release assembly, dose calibration, sealed manifest and release generation for the evaluation of evaluator v3 (tag s-suite-eval-v3).

    python -m n_control.n2 calibrate --base-dir ~/n_control_data --out ~/n_control_data/calib2.json --procs 4
    python -m n_control.n2 build     --calib ~/n_control_data/calib2.json --base-record ~/n_control_data/base_record.json --out ~/sealed_manifest_n2/manifest.json
    python -m n_control.n2 generate  --manifest ~/sealed_manifest_n2/manifest.json --base-dir ~/n_control_data --out-dir ~/n2_releases --sealed-dir ~/sealed_manifest_n2 --procs 6

Same bases, same pseudo-identity / subject-disjoint split machinery (assign.py), same release file format (X, y, unit, split) and digest as N1 (release.py).
Design counts: plant = 2 bases x 7 families x 4 levels x REPS; null = 2 x N_NULL; shuf = 2 x 7 x len(SHUF_LEVELS) x N_SHUF.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import stat
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from .assign import assign_sessions, unit_shuffle  # noqa: E402
from .base import BASES, Base, load_base  # noqa: E402
from .common import canon, sha256_hex  # noqa: E402
from .families2 import (FAMILIES2, LEVELS2, TARGET_SESSION2, TARGET_WINDOW2, inject2, omega2, session_scale, stats2)  # noqa: E402
from .release import digest, save  # noqa: E402

REPS, N_NULL, SHUF_LEVELS, N_SHUF = 5, 25, (2, 4), 2
VERSION = "N2-manifest-v1"
GRID = {"bias": (0.04, 8.0), "specin": (0.01, 3.0), "hb2": (0.003, 2.0), "phase2": (0.01, 3.0), "af2": (0.01, 3.0), "pac": (0.02, 40.0), "order": (0.01, 4.0)}
N_GRID, N_DRAW = 16, 4


def build_release2(base: Base, entry: dict) -> dict:
    split_s, pid_s = assign_sessions(base.sess_sub, base.sess_act, entry["assign_seed"])
    split_w, pid_w = split_s[base.sess], pid_s[base.sess]
    kind = entry["kind"]
    X = base.X if kind == "null" else inject2(base.X, pid_w, base.sess, entry["family"], entry["amp"], entry["key"], stats2(base))
    y_s = unit_shuffle(pid_s, split_s, entry["shuffle_seed"]) if kind == "shuf" else pid_s
    return dict(X=np.ascontiguousarray(X, dtype=np.float32), y=y_s[base.sess].astype(np.int64), unit=base.sess_name[base.sess], split=split_w.astype(np.int8),
                _pid_planted_session=pid_s, _y_session=y_s)


def oracle_report2(base: Base, entry: dict, rel: dict) -> dict:
    if entry["kind"] == "null":
        return dict(kind="null")
    st = stats2(base)
    out = {"kind": entry["kind"]}
    for nm, lab in (("planted", rel["_pid_planted_session"]), ("released", rel["_y_session"])):
        if nm == "released" and entry["kind"] != "shuf":
            out["released"] = out["planted"]
            continue
        out[nm] = omega2(entry["family"], rel["X"], lab[base.sess], base.sess, base.act, rel["split"], lab, base.sess_act, entry["key"], entry["amp"], st)
    return out


# ------------------------------------------------------------------------------------------------ calibration (oracle only)
_BASE = {}


def _cal_job(a):
    path, family, amp, d = a
    if path not in _BASE:
        _BASE[path] = load_base(path)
    b = _BASE[path]
    e = dict(kind="plant", family=family, amp=float(amp), assign_seed=20_000_000 + d, key=sha256_hex(f"calib2|{family}|{os.path.basename(path)}|{d}".encode())[:32], shuffle_seed=0)
    r = oracle_report2(b, e, build_release2(b, e))["planted"]
    return family, float(amp), d, r["win"], r["sess"]


def invert(amps, omega, target):
    om = np.maximum.accumulate(np.asarray(omega))
    la = np.log(np.asarray(amps))
    if target > om[-1]:
        return float(amps[-1]), False
    if target <= om[0]:
        return float(amps[0]), True
    i = int(np.searchsorted(om, target))
    j = i - 1
    w = (target - om[j]) / max(om[i] - om[j], 1e-12)
    return float(np.exp(la[j] + w * (la[i] - la[j]))), True


def calibrate(base_dir, out, procs):
    bd = os.path.expanduser(base_dir)
    jobs = []
    for nm in BASES:
        p = os.path.join(bd, f"base_{nm}.npz")
        for fam in FAMILIES2:
            lo, hi = GRID[fam]
            for amp in np.geomspace(lo, hi, N_GRID):
                for d in range(N_DRAW):
                    jobs.append((p, fam, amp, d))
    with mp.Pool(procs) as pool:
        res = pool.map(_cal_job, jobs, chunksize=4)
    out_d = {"n_grid": N_GRID, "n_draw": N_DRAW, "bases": {}}
    for nm in BASES:
        out_d["bases"][nm] = {}
        for fam in FAMILIES2:
            rows = [r for r, j in zip(res, jobs) if j[1] == fam and j[0].endswith(f"base_{nm}.npz")]
            amps = sorted({r[1] for r in rows})
            mw = [float(np.mean([r[3] for r in rows if r[1] == x])) for x in amps]
            ms = [float(np.mean([r[4] for r in rows if r[1] == x])) for x in amps]
            tgt, om = (TARGET_SESSION2, ms) if session_scale(fam) else (TARGET_WINDOW2, mw)
            levels = []
            for L, t in enumerate(tgt, start=1):
                amp, ok = invert(amps, om, t)
                levels.append(dict(level=L, target=t, scale="sess" if session_scale(fam) else "win", amp=amp, reached=ok))
            out_d["bases"][nm][fam] = dict(amps=amps, omega_win=mw, omega_sess=ms, levels=levels)
    json.dump(out_d, open(os.path.expanduser(out), "w"), indent=1, sort_keys=True)
    for nm in BASES:
        for fam in FAMILIES2:
            c = out_d["bases"][nm][fam]
            print(nm, fam, "max win %.2f sess %.2f" % (max(c["omega_win"]), max(c["omega_sess"])), [(l["target"], round(l["amp"], 4), l["reached"]) for l in c["levels"]])


# ------------------------------------------------------------------------------------------------ manifest
def _seed():
    return int.from_bytes(os.urandom(8), "big") >> 2


def build(calib_path, base_record_path, out_path):
    calib = json.load(open(os.path.expanduser(calib_path)))
    brec = json.load(open(os.path.expanduser(base_record_path)))
    entries = []
    for base in BASES:
        for fam in FAMILIES2:
            for lvl in LEVELS2:
                amp = calib["bases"][base][fam]["levels"][lvl - 1]["amp"]
                for r in range(REPS):
                    entries.append(dict(kind="plant", family=fam, level=lvl, base=base, amp=float(amp), rep=r, assign_seed=_seed(), key=os.urandom(16).hex(), shuffle_seed=0))
            for lvl in SHUF_LEVELS:
                amp = calib["bases"][base][fam]["levels"][lvl - 1]["amp"]
                for r in range(N_SHUF):
                    entries.append(dict(kind="shuf", family=fam, level=lvl, base=base, amp=float(amp), rep=r, assign_seed=_seed(), key=os.urandom(16).hex(), shuffle_seed=_seed()))
        for r in range(N_NULL):
            entries.append(dict(kind="null", family=None, level=0, base=base, amp=0.0, rep=r, assign_seed=_seed(), key=os.urandom(16).hex(), shuffle_seed=0))
    order = np.random.RandomState(int.from_bytes(os.urandom(4), "big")).permutation(len(entries))
    entries = [entries[i] for i in order]
    for i, e in enumerate(entries):
        e["rid"] = i
    man = dict(version=VERSION, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), base_record=brec,
               design=dict(reps=REPS, n_null=N_NULL, n_shuf=N_SHUF, shuf_levels=list(SHUF_LEVELS), families=list(FAMILIES2), bases=list(BASES)),
               calibration={b: {f: calib["bases"][b][f]["levels"] for f in FAMILIES2} for b in BASES}, entries=entries)
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
    rel = build_release2(b, e)
    d = digest(rel)
    save(rel, os.path.join(out_dir, f"{d}.npz"))
    return e["rid"], d, oracle_report2(b, e, rel)


def generate(manifest_path, base_dir, out_dir, sealed_dir, procs):
    raw = open(os.path.expanduser(manifest_path), "rb").read()
    man = json.loads(raw)
    assert canon(man) == raw, "manifest is not canonical"
    out_dir, sealed_dir = os.path.expanduser(out_dir), os.path.expanduser(sealed_dir)
    os.makedirs(out_dir, exist_ok=True)
    with mp.Pool(procs) as pool:
        res = pool.map(_gen, [(e, base_dir, out_dir) for e in man["entries"]], chunksize=2)
    digests = sorted(d for _, d, _ in res)
    assert len(set(digests)) == len(digests)
    for name, obj in (("release_map.json", {d: rid for rid, d, _ in res}), ("oracle.json", {str(rid): rep for rid, _, rep in res})):
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
    ap.add_argument("cmd", choices=["calibrate", "build", "generate"])
    ap.add_argument("--base-dir", default="~/n_control_data")
    ap.add_argument("--calib", default="~/n_control_data/calib2.json")
    ap.add_argument("--base-record", default="~/n_control_data/base_record.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default="~/sealed_manifest_n2/manifest.json")
    ap.add_argument("--out-dir", default="~/n2_releases")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest_n2")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    assert a.procs <= 8
    if a.cmd == "calibrate":
        calibrate(a.base_dir, a.out or "~/n_control_data/calib2.json", a.procs)
    elif a.cmd == "build":
        build(a.calib, a.base_record, a.out or "~/sealed_manifest_n2/manifest.json")
    else:
        generate(a.manifest, a.base_dir, a.out_dir, a.sealed_dir, a.procs)
