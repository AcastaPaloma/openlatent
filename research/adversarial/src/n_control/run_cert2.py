"""Run Builder S's FROZEN evaluator v3 (tag s-suite-eval-v3, evaluator sha256 b5c75e11...57303) on the N2 releases, one sealed query per release digest.

Per release: retrain the declared suite-v3 (K = 24: SUITE12 + hb / phase / inter members) on the release's train split (`suite_v3.fit_predict_v3`, windows (n, T, C) = (n, 128, 2)),
certify with `suite_v3.declaration(24, 2)` (alpha 0.05, 2000 resamples, seed 0, margin 0.02, canary groups hb / phase / inter).
Canaries (S's `canary.run_canaries`, three extra pipelines) are needed ONLY when they can change the printed verdict: the v3 verdict is LEAK if the one-sided lower bound
of (selected - baseline) is > 0; else BELOW_MARGIN only if both suite bounds are under the margin AND every canary group was read; else NOT_RESOLVED. So a DRY certify
(split_status='development', no ledger, used only to decide whether to spend the canary compute) tells whether canaries matter; if the verdict is LEAK, or the bounds are not both
under the margin, the sealed certificate is identical with or without canaries and they are not run (recorded as `canaries_run: false`). The sealed certify call is made once.
All 24 attackers' val/test probabilities are saved (`--preds`) so the v1 suite (SUITE12 alone) can be re-scored on the SAME predictions afterwards (not a second query).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=research/adversarial/src python -m n_control.run_cert2 \
      --releases ~/n2_releases --out ~/n2_certs --preds ~/n2_preds --ledger ~/sealed_ledger_N2 --procs 8
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

from s_suite import canary as cy  # noqa: E402
from s_suite import certificate as cf  # noqa: E402
from s_suite import freeze, ledger  # noqa: E402
from s_suite import suite_v3 as sv  # noqa: E402

from .common import N_ID, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from .release import digest, load  # noqa: E402

EXPECTED_EVALUATOR = "b5c75e113792b75202673ebbdf6c437051e765db258a7344cc30942044957303"
EXPECTED_TAG = "s-suite-eval-v3"
DECL = sv.declaration(N_ID, 2)
SPLIT_ID = "test"


def run_one(args):
    d, rel_dir, out_dir, pred_dir, ledger_dir = args
    outp = os.path.join(out_dir, f"{d}.json")
    if os.path.exists(outp):
        return d, "skip", 0.0
    t0 = time.time()
    try:
        r = load(os.path.join(rel_dir, f"{d}.npz"))
        assert digest(r) == d, "release content does not match its digest"
        sp = {s: r["split"] == c for s, c in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))}
        W = {s: np.ascontiguousarray(r["X"][m].transpose(0, 2, 1)) for s, m in sp.items()}      # (n, T, C)
        Y = {s: r["y"][m] for s, m in sp.items()}
        U = {s: r["unit"][m] for s, m in sp.items()}
        P = sv.fit_predict_v3(W["train"], Y["train"], {"val": W["val"], "test": W["test"]}, N_ID)
        val = cf.SplitPreds(Y["val"], U["val"], {a: P["val"][a] for a in DECL.attackers})
        test = cf.SplitPreds(Y["test"], U["test"], {a: P["test"][a] for a in DECL.attackers})
        np.savez_compressed(os.path.join(pred_dir, f"{d}.npz"), **{f"{s}|{a}": P[s][a] for s in ("val", "test") for a in DECL.attackers})
        dry = cf.certify(DECL, val, test, split_status="development")
        v = dry["verdict"]
        need = bool(v["label"] != "LEAK" and v["below_margin_bonferroni"] and v["below_margin_maxt"])
        canaries = cy.run_canaries({s: (W[s], Y[s], U[s]) for s in ("train", "val", "test")}, N_ID, DECL) if need else None
        cert = cf.certify(DECL, val, test, split_status="sealed", sealed_ledger_dir=ledger_dir, release_digest=d, split_id=SPLIT_ID, canaries=canaries)
        cert["_n_control"] = dict(release=d, seconds=round(time.time() - t0, 1), canaries_run=need, dry_label=v["label"])
        tmp = outp + ".tmp"
        json.dump(cert, open(tmp, "w"), sort_keys=True)
        os.replace(tmp, outp)
        return d, "ok", time.time() - t0
    except Exception:  # noqa: BLE001
        return d, "ERR " + traceback.format_exc()[-800:], time.time() - t0


def preflight():
    bad = freeze.check()
    if bad:
        sys.exit("freeze check failed: " + "; ".join(bad))
    rec = freeze.load_freeze()
    if rec.get("tag") != EXPECTED_TAG or freeze.evaluator_sha256() != EXPECTED_EVALUATOR:
        sys.exit(f"evaluator is not {EXPECTED_TAG} / {EXPECTED_EVALUATOR[:8]}")
    root = ledger.repo_root()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("work tree is dirty (tracked files):\n" + dirty)
    if subprocess.run(["git", "merge-base", "--is-ancestor", f"{EXPECTED_TAG}^{{commit}}", "HEAD"], cwd=root).returncode != 0:
        sys.exit(f"tag {EXPECTED_TAG} is not an ancestor of HEAD")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases", default="~/n2_releases")
    ap.add_argument("--out", default="~/n2_certs")
    ap.add_argument("--preds", default="~/n2_preds")
    ap.add_argument("--ledger", default="~/sealed_ledger_N2")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    assert a.procs <= 8, "CPU etiquette: at most 8 worker processes"
    head = preflight()
    rel_dir, out_dir, pred_dir, ledger_dir = (os.path.expanduser(x) for x in (a.releases, a.out, a.preds, a.ledger))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    digests = [x.strip() for x in open(os.path.join(rel_dir, "INDEX.txt")) if x.strip()]
    rs = np.random.RandomState(int.from_bytes(os.urandom(4), "big"))
    digests = [digests[i] for i in rs.permutation(len(digests))]
    if a.limit:
        digests = digests[: a.limit]
    print(f"HEAD {head}; {len(digests)} releases; decl digest {DECL.digest()}; K = {len(DECL.attackers)}", flush=True)
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        for i, (d, st, sec) in enumerate(pool.imap_unordered(run_one, [(d, rel_dir, out_dir, pred_dir, ledger_dir) for d in digests], chunksize=1)):
            if st.startswith("ERR"):
                print("ERROR", d[:12], st, flush=True)
            elif i % 5 == 0:
                print(f"{i + 1}/{len(digests)} {st} {sec:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
    bad = ledger.check(ledger_dir)
    print("ledger:", "clean" if not bad else bad, flush=True)
