"""Dose calibration: find, per (family, base release, level), the tone/offset amplitude at which the key-aware matched filter reaches a target
identification rate. Uses ONLY the oracle (families.py); no attacker of Builder S's is ever run here. Calibration draws use their own keys and
assignment seeds (namespace `calib`), disjoint from the manifest's random 64-bit seeds.

    python -m n_control.calibrate --base-dir ~/n_control_data --out ~/n_control_data/calib.json --procs 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from .base import BASES, load_base  # noqa: E402
from .common import FAMILIES, TARGET_SESSION, TARGET_WINDOW, sha256_hex  # noqa: E402
from .release import build_release, oracle_report  # noqa: E402

GRID = {   # amplitude grids (log-spaced); widened automatically if the top of the grid does not reach the highest target
    "bias": (0.04, 8.0), "pooled": (0.004, 1.5), "specin": (0.01, 3.0), "xphase": (0.01, 3.0), "ampfreq": (0.01, 3.0), "stealth": (0.001, 0.5),
}
N_GRID, N_DRAW = 16, 4
_BASE = {}


def _get_base(path):
    if path not in _BASE:
        _BASE[path] = load_base(path)
    return _BASE[path]


def _job(args):
    base_path, family, amp, d = args
    b = _get_base(base_path)
    entry = dict(kind="plant", family=family, amp=float(amp), assign_seed=10_000_000 + d,
                 key=sha256_hex(f"calib|{family}|{os.path.basename(base_path)}|{d}".encode())[:32], shuffle_seed=0)
    rel = build_release(b, entry)
    r = oracle_report(b, entry, rel)["planted"]
    return family, float(amp), d, r["win"], r["sess"]


def invert(amps, omega, target):
    """Monotone interpolation in log-amplitude; returns (amp, reached)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="~/n_control_data")
    ap.add_argument("--out", default="~/n_control_data/calib.json")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    bd = os.path.expanduser(a.base_dir)
    jobs = []
    for nm in BASES:
        p = os.path.join(bd, f"base_{nm}.npz")
        for fam in FAMILIES:
            lo, hi = GRID[fam]
            for amp in np.geomspace(lo, hi, N_GRID):
                for d in range(N_DRAW):
                    jobs.append((p, fam, amp, d))
    with mp.Pool(a.procs) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    out = {"n_grid": N_GRID, "n_draw": N_DRAW, "bases": {}}
    k = 0
    for nm in BASES:
        out["bases"][nm] = {}
        for fam in FAMILIES:
            rows = [r for r, j in zip(res, jobs) if j[1] == fam and j[0].endswith(f"base_{nm}.npz")]
            amps = sorted({r[1] for r in rows})
            mw = [float(np.mean([r[3] for r in rows if r[1] == x])) for x in amps]
            ms = [float(np.mean([r[4] for r in rows if r[1] == x])) for x in amps]
            sdw = [float(np.std([r[3] for r in rows if r[1] == x])) for x in amps]
            sds = [float(np.std([r[4] for r in rows if r[1] == x])) for x in amps]
            tgt, key = (TARGET_SESSION, "sess") if fam == "pooled" else (TARGET_WINDOW, "win")
            om = ms if key == "sess" else mw
            levels = []
            for L, t in enumerate(tgt, start=1):
                amp, ok = invert(amps, om, t)
                levels.append(dict(level=L, target=t, scale=key, amp=amp, reached=ok))
            out["bases"][nm][fam] = dict(amps=amps, omega_win=mw, omega_sess=ms, sd_win=sdw, sd_sess=sds, levels=levels)
    json.dump(out, open(os.path.expanduser(a.out), "w"), indent=1, sort_keys=True)
    for nm in BASES:
        for fam in FAMILIES:
            c = out["bases"][nm][fam]
            print(nm, fam, "max win %.3f max sess %.3f" % (max(c["omega_win"]), max(c["omega_sess"])), [(l["target"], round(l["amp"], 4), l["reached"]) for l in c["levels"]])


if __name__ == "__main__":
    main()
