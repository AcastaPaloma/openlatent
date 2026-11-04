"""Acceptance runs of evaluator v3 on SYNTHETIC releases only (never on Builder N's data). CPU, multiprocessing.

python -m s_suite.run_v3_acceptance pilot  --out F.jsonl --procs 8     canary-dose pilot (sets canary.CANARY_SNR; seeds 5000+)
python -m s_suite.run_v3_acceptance accept --out F.jsonl --procs 8     N1 nulls, N2 canary detection, N3 non-inferiority vs v2 on planted data (seeds 20000+)

Synthetic release: 24 pseudo-identities dealt to sessions (train 6, val 2, test 4 sessions per identity = 144 / 48 / 96 sessions), 20 windows per session,
windows (T=128, C=2). 'white': iid Gaussian channels; 'smooth': AR(1) phi 0.95 plus a low-frequency periodic component; both with a 0.5 cross-channel
correlation, a per-session offset and gain. Labels are independent of the windows (null) until a canary is injected. Planted data (N3) is planted.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

from . import canary as cy  # noqa: E402
from . import certificate as cf  # noqa: E402
from . import suite_v3 as sv  # noqa: E402
from .attackers import SUITE12  # noqa: E402
from .planted import PlantedSpec, generate_replicate  # noqa: E402

N_ID, T, C, W = 24, 128, 2, 20
PER = dict(train=6, val=2, test=4)


def make_release(noise: str, seed: int) -> dict:
    rs = np.random.RandomState(seed)
    out = {}
    for split, per in PER.items():
        n_s = N_ID * per
        ident_s = rs.permutation(np.repeat(np.arange(N_ID), per))
        n = n_s * W
        unit = np.repeat(np.arange(n_s), W)
        a = rs.randn(n, T + 32)
        if noise == "smooth":
            a = lfilter([np.sqrt(1 - 0.95 ** 2)], [1.0, -0.95], a, axis=1)
            a = a + 1.0 * np.sin(2 * np.pi * 4 * np.arange(T + 32)[None] / T + rs.uniform(0, 2 * np.pi, (n, 1)))
        a = a[:, 32:]
        b = 0.5 * a + np.sqrt(0.75) * (lfilter([np.sqrt(1 - 0.95 ** 2)], [1.0, -0.95], rs.randn(n, T + 32), axis=1)[:, 32:] if noise == "smooth" else rs.randn(n, T))
        off = rs.randn(n_s, 1, 2) * 0.5
        gain = np.exp(0.2 * rs.randn(n_s, 1, 2))
        X = (np.stack([a, b], axis=2) * gain[unit] + off[unit]).astype(np.float32)
        out[split] = (X, ident_s[unit].astype(int), np.array([f"{split}{u}" for u in unit]))
    return out


def _unit_shuffle(y, cluster, rs):
    uu, inv = np.unique(cluster, return_inverse=True)
    first = np.zeros(len(uu), dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return y[first][rs.permutation(len(uu))][inv]


def _pipeline(splits, decl, n_classes, X_over=None, shuffle_rs=None):
    X = {s: (X_over[s] if X_over else v[0]) for s, v in splits.items()}
    P = sv.fit_predict_v3(X["train"], splits["train"][1], {"val": X["val"], "test": X["test"]}, n_classes)
    yv, ct_v = splits["val"][1], splits["val"][2]
    yt, ct_t = splits["test"][1], splits["test"][2]
    val = cf.SplitPreds(yv, ct_v, {a: P["val"][a] for a in decl.attackers})
    test = cf.SplitPreds(yt, ct_t, {a: P["test"][a] for a in decl.attackers})
    cert = cf.certify(decl, val, test, split_status="synthetic")
    shuf = None
    if shuffle_rs is not None:
        val_s = cf.SplitPreds(_unit_shuffle(yv, ct_v, shuffle_rs), ct_v, val.scores)
        test_s = cf.SplitPreds(_unit_shuffle(yt, ct_t, shuffle_rs), ct_t, test.scores)
        shuf = cf.certify(decl, val_s, test_s, split_status="synthetic")["verdict"]["label"]
    return cert, shuf, P


def _brief(c):
    s = c["selected"]
    return dict(verdict=c["verdict"]["label"], sel=s["name"], sel_test=s["test"], ci_cons=s["ci95_conservative"], baseline=c["baseline_used"],
                ub_bonf=c["suite_bound"]["bonferroni"], ub_maxt=c["suite_bound"]["maxt"], withheld=c["verdict"]["assurance_withheld"], digest=c["digest"][:12])


def job(j):
    t0 = time.time()
    kind = j["kind"]
    decl = sv.declaration(N_ID, C, name="v3-synthetic", n_boot=1000)
    if kind == "null":
        rel = make_release(j["noise"], j["seed"])
        cert, _, _ = _pipeline(rel, decl, N_ID)
        lo, hi = cert["selected"]["ci95_conservative"]
        out = dict(brief=_brief(cert), cover_cons=bool(lo <= 1.0 / N_ID <= hi), bonf_covers_chance=bool(cert["suite_bound"]["bonferroni"] >= 1.0 / N_ID))
    elif kind in ("pilot", "canary"):
        rel = make_release(j["noise"], j["seed"])
        g = j["group"]
        snr = {**cy.CANARY_SNR, g: j["snr"]} if kind == "pilot" else cy.CANARY_SNR
        ref = cy.Reference(rel["train"][0])
        Xc = {s: cy.inject(v[0], v[1], cy.KIND_OF_GROUP[g], 100 * j["seed"] + 7, snr[g], ref) for s, v in rel.items()}
        cert, shuf, P = _pipeline(rel, decl, N_ID, X_over=Xc, shuffle_rs=np.random.RandomState(j["seed"]))
        accs = {a: float((P["test"][a].argmax(1) == rel["test"][1]).mean()) for a in decl.attackers}
        grp = {g2: max(accs[a] for a in names) for g2, names in sv.GROUPS.items() for a in names[:1]}
        out = dict(brief=_brief(cert), shuffled_verdict=shuf, best_by_group={g2: max(accs[a] for a in names) for g2, names in sv.GROUPS.items()},
                   best_base=max(accs[a] for a in SUITE12), snr=snr[g])
    elif kind == "planted":
        spec = PlantedSpec(channel=j["channel"], dose=j["dose"], ident_seed=30000 + j["seed"])
        data = generate_replicate(spec, seed=900000 + j["seed"], units_train=12, units_val=6, units_test=6, units_pop=1)
        splits = {s: (data[s]["X"], data[s]["y"], np.array([f"{s}{u}" for u in data[s]["unit"]])) for s in ("train", "val", "test")}
        d3 = sv.declaration(25, 4, name="v3-planted", n_boot=1000)
        cert3, _, P = _pipeline(splits, d3, 25)
        d2 = cf.SuiteDecl("v2-equivalent", SUITE12, "accuracy", 0.05, 1000, 0, 0.02, 25)
        val = cf.SplitPreds(splits["val"][1], splits["val"][2], {a: P["val"][a] for a in SUITE12})
        test = cf.SplitPreds(splits["test"][1], splits["test"][2], {a: P["test"][a] for a in SUITE12})
        cert2 = cf.certify(d2, val, test, split_status="synthetic")
        out = dict(v3=_brief(cert3), v2=_brief(cert2))
    else:
        raise ValueError(kind)
    return dict(job=j, **out, secs=time.time() - t0)


def jobs(mode):
    js = []
    if mode == "pilot":
        for snr in (0.3, 0.5, 0.8):
            for g in ("hb", "phase", "inter"):
                for noise in ("white", "smooth"):
                    for r in range(5):
                        js.append(dict(kind="pilot", group=g, noise=noise, snr=snr, seed=5000 + r))
    else:
        for r in range(40):
            for noise in ("white", "smooth"):
                js.append(dict(kind="null", noise=noise, seed=20000 + r))
        for r in range(15):
            for g in ("hb", "phase", "inter"):
                for noise in ("white", "smooth"):
                    js.append(dict(kind="canary", group=g, noise=noise, seed=21000 + r))
        for r in range(20):
            for ch, dose in (("bias", 0.31), ("gain", 0.092), ("spectral", 0.24)):
                js.append(dict(kind="planted", channel=ch, dose=dose, seed=22000 + r))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "accept"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    todo = jobs(a.mode)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.dumps(json.loads(line)["job"], sort_keys=True))
            except Exception:  # noqa: BLE001
                pass
    todo = [j for j in todo if json.dumps(j, sort_keys=True) not in done]
    print(f"{len(todo)} jobs, {len(done)} done", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(a.procs) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
