"""Run Builder S's FROZEN certificate (tag s-suite-eval-v1) on released files, one sealed query per release digest, through S's ledger.

Instrument under test, unchanged: s_suite.attackers.SUITE12 (retrained on the release's train split; views `raw` = flattened window, `feat` = S's hand features
`planted.features` on the (n, T, C) window), s_suite.certificate.certify with SuiteDecl("n-motionsense-suite12", SUITE12, accuracy, alpha 0.05, n_boot 2000,
seed 0, margin 0.02, n_classes 24) -- the planted-engine declaration of S.md section 2 with n_classes = 24. val = the release's val split (selection only),
test = its test split (scored once). Units = sessions. The ledger directory is outside the checkout; a second query of a digest is refused.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/adversarial/src \
      python -m n_control.run_cert --releases ~/n_releases --out ~/n_certs --ledger ~/sealed_ledger_N --procs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from s_suite import certificate as cf  # noqa: E402
from s_suite import freeze  # noqa: E402
from s_suite import ledger  # noqa: E402
from s_suite.attackers import SUITE12, fit_predict_pool  # noqa: E402
from s_suite.planted import features, raw_view  # noqa: E402

from .common import N_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from .release import digest, load  # noqa: E402

DECL = cf.SuiteDecl("n-motionsense-suite12", SUITE12, "accuracy", 0.05, 2000, 0, 0.02, N_ID)
SPLIT_ID = "test"


def views(X):
    Xt = np.ascontiguousarray(X.transpose(0, 2, 1))     # (n, T, C) as S's feature code expects
    return {"raw": raw_view(Xt), "feat": features(Xt)}


def run_one(args):
    d, rel_dir, out_dir, ledger_dir = args
    outp = os.path.join(out_dir, f"{d}.json")
    if os.path.exists(outp):
        return d, "skip", 0.0
    t0 = time.time()
    try:
        r = load(os.path.join(rel_dir, f"{d}.npz"))
        assert digest(r) == d, "release content does not match its digest"
        tr, va, te = (r["split"] == s for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST))
        P = fit_predict_pool(views(r["X"][tr]), r["y"][tr], {"val": views(r["X"][va]), "test": views(r["X"][te])}, names=SUITE12, n_classes=N_ID)
        val = cf.SplitPreds(r["y"][va], r["unit"][va], {a: P["val"][a] for a in SUITE12})
        test = cf.SplitPreds(r["y"][te], r["unit"][te], {a: P["test"][a] for a in SUITE12})
        cert = cf.certify(DECL, val, test, split_status="sealed", sealed_ledger_dir=ledger_dir, release_digest=d, split_id=SPLIT_ID)
        cert["_n_control"] = dict(release=d, seconds=round(time.time() - t0, 1))
        tmp = outp + ".tmp"
        json.dump(cert, open(tmp, "w"), sort_keys=True)
        os.replace(tmp, outp)
        return d, "ok", time.time() - t0
    except Exception:  # noqa: BLE001
        return d, "ERR " + traceback.format_exc()[-600:], time.time() - t0


def preflight(repo_hint):
    bad = freeze.check()
    if bad:
        sys.exit("freeze check failed: " + "; ".join(bad))
    root = ledger.repo_root()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("work tree is dirty (tracked files); the ledger would record a dirty entry:\n" + dirty)
    tag = freeze.load_freeze().get("tag")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", f"{tag}^{{commit}}", "HEAD"], cwd=root)
    if anc.returncode != 0:
        sys.exit(f"tag {tag} is not an ancestor of HEAD")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases", default="~/n_releases")
    ap.add_argument("--out", default="~/n_certs")
    ap.add_argument("--ledger", default="~/sealed_ledger_N")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N digests of the (random-order) index; 0 = all")
    a = ap.parse_args()
    assert a.procs <= 8, "CPU etiquette: at most 8 worker processes"
    head = preflight(None)
    rel_dir, out_dir, ledger_dir = (os.path.expanduser(x) for x in (a.releases, a.out, a.ledger))
    os.makedirs(out_dir, exist_ok=True)
    digests = [x.strip() for x in open(os.path.join(rel_dir, "INDEX.txt")) if x.strip()]
    rs = np.random.RandomState(int.from_bytes(os.urandom(4), "big"))
    digests = [digests[i] for i in rs.permutation(len(digests))]       # run order unrelated to the manifest
    if a.limit:
        digests = digests[: a.limit]
    print(f"HEAD {head}; {len(digests)} releases; decl digest {DECL.digest()}", flush=True)
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        for i, (d, st, sec) in enumerate(pool.imap_unordered(run_one, [(d, rel_dir, out_dir, ledger_dir) for d in digests], chunksize=1)):
            if st.startswith("ERR"):
                print("ERROR", d[:12], st, flush=True)
            elif i % 10 == 0:
                print(f"{i + 1}/{len(digests)} {st} {sec:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
    bad = ledger.check(ledger_dir)
    print("ledger:", "clean" if not bad else bad, flush=True)
