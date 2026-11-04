"""Planted-leak ENGINE checks (CPU, multiprocessing). Pre-registered in results_notes/S.md.

SCOPE (audit F10, FOCUS A3): the planted channels map onto the attackers' hand features and the oracle is ours, so detection is guaranteed by
construction. This runner is a unit test of the statistical engine (coverage of the bounds against known true accuracies, false-LEAK rate under
the null and under label shuffles, truth-based selection bias). It does NOT estimate detection power for real leaks; the detection-vs-dose curve
is recorded but is not evidence of power. The real calibration belongs to the negative-control builder (sealed manifest, unseen leak families).

python -m s_suite.run_planted --out OUT.jsonl --reps 120 --null-reps 300 --procs 24 [--smoke]

One job = (channel, dose, replicate). A replicate draws new identities (parameters), generates independent train / val / test / population
splits, retrains the 30-attacker pool on the train split, and records, for each n_test in N_TEST units:
  * the certificate of the pre-declared SUITE12 (val-selected attacker scored on the sealed test) and whether its bounds cover the TRUE
    accuracies (population sample of 2,500 sessions);
  * one-line statistics (kNN-5 and LDA on raw windows) against the suite's selected attacker (paired cluster bootstrap);
  * a unit-level label-shuffled control for the same predictions;
  * truth-based selection-bias quantities for K in K_LIST over the 30-attacker pool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from . import certificate as cf  # noqa: E402
from . import selection_bias as sb  # noqa: E402
from .attackers import POOL_NAMES, SUITE12, fit_predict_pool  # noqa: E402
from .planted import PlantedSpec, features, generate_replicate, raw_view  # noqa: E402

DOSES = {  # frozen before any certificate number; chosen from a synthetic oracle pilot to span oracle accuracy ~6% .. 60% (chance 4%)
    "bias": [0.125, 0.18, 0.23, 0.31, 0.40, 0.60, 1.0],
    "gain": [0.037, 0.053, 0.068, 0.092, 0.122, 0.19, 0.31],
    "spectral": [0.12, 0.17, 0.195, 0.24, 0.29, 0.41, 0.66],
}
N_TEST = (150, 600, 1425, 3000)
UNITS_TEST = 120           # per identity => 3,000 sealed units available, sub-sampled for smaller n
K_LIST = [1, 2, 3, 5, 10, 20, 30]
ONE_LINERS = ("knn5_raw", "lda_raw_auto")
DECL = cf.SuiteDecl("planted-suite12", SUITE12, "accuracy", 0.05, 2000, 0, 0.02, 25)


ENGINE_DOSE_IDX = (1, 3, 5)   # engine check uses 3 doses per channel (oracle accuracy about 8%, 14%, 35%) plus the null


def cells():
    out = [("null", 0.0)]
    for ch, ds in DOSES.items():
        out += [(ch, ds[i]) for i in ENGINE_DOSE_IDX]
    return out


def _subset(unit_ids_by_item, n_units, n_id=25, per_id=UNITS_TEST):
    k = n_units // n_id
    keep = (unit_ids_by_item % per_id) < k
    return keep


def _unit_shuffle(y, cluster, rs):
    """Permute identities among units (all items of a unit keep one label): the unit-level label-shuffled control."""
    uu, inv = np.unique(cluster, return_inverse=True)
    first = np.zeros(len(uu), dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    lab = y[first]
    return lab[rs.permutation(len(uu))][inv]


def run_job(job):
    channel, dose, rep, cell_idx = job
    t0 = time.time()
    spec = PlantedSpec(channel=channel, dose=dose, ident_seed=rep + 1000 * cell_idx)
    data = generate_replicate(spec, seed=rep + 100000 * cell_idx, units_train=12, units_val=6, units_test=UNITS_TEST, units_pop=100)
    vt = {"raw": raw_view(data["train"]["X"]), "feat": features(data["train"]["X"])}
    ve = {s: {"raw": raw_view(data[s]["X"]), "feat": features(data[s]["X"])} for s in ("val", "test", "pop")}
    P = fit_predict_pool(vt, data["train"]["y"], ve, n_classes=spec.n_id)
    corr = {s: np.stack([(P[s][a].argmax(1) == data[s]["y"]) for a in POOL_NAMES], 1).astype(float) for s in P}
    true_acc = corr["pop"].mean(0)
    true12 = true_acc[:12]
    rs = np.random.RandomState(rep + 7 * cell_idx)
    out = dict(channel=channel, dose=dose, rep=rep, true_acc=true_acc.tolist(), cert={}, oneliner={}, shuffled={}, truth_infl={})
    val_u = data["val"]["unit"]
    test_u_all = data["test"]["unit"]
    # unit index inside identity for the sub-sampling rule: units are numbered identity-major with UNITS_TEST per identity
    for n in N_TEST:
        keep = _subset(test_u_all, n)
        yt = data["test"]["y"][keep]
        ct = np.array(["t%d" % u for u in test_u_all[keep]])
        test = cf.SplitPreds(yt, ct, {a: P["test"][a][keep] for a in SUITE12})
        val = cf.SplitPreds(data["val"]["y"], np.array(["v%d" % u for u in val_u]), {a: P["val"][a] for a in SUITE12})
        c = cf.certify(DECL, val, test, reference=None, split_status="synthetic")
        sel = c["selected"]
        k_sel = SUITE12.index(sel["name"])
        sb_ = c["suite_bound"]
        out["cert"][str(n)] = dict(
            sel=sel["name"], sel_val=sel["val"], sel_test=sel["test"], sel_se=sel["se"], sel_mde=sel["mde80"], excess_lcb=sel["excess_lcb_onesided"],
            verdict=c["verdict"]["label"], ub_bonf=sb_["bonferroni"], ub_maxt=sb_["maxt"], q_maxt=sb_["q_maxt"], z_bonf=sb_["z_bonferroni"],
            naive_max=c["naive_max"]["test"], baseline=c["baseline_used"], floor_k1=c["floor"]["ucb_at_baseline_k1"],
            floor_suite=c["floor"]["ucb_at_baseline_suite"], true_sel=float(true12[k_sel]), true_best12=float(true12.max()),
            cover_bonf=bool(sb_["bonferroni"] >= true12.max()), cover_maxt=bool(sb_["maxt"] >= true12.max()),
            cover_ci=bool(sel["ci95"][0] <= true12[k_sel] <= sel["ci95"][1]),
            cover_each_bonf=bool(all(c["attackers"][a]["ucb_bonferroni"] >= true12[i] for i, a in enumerate(SUITE12))),
            cover_each_maxt=bool(all(c["attackers"][a]["ucb_maxt"] >= true12[i] for i, a in enumerate(SUITE12))),
            norm_leak=sel["normalised_leakage"], digest=c["digest"][:12])
        ol = {}
        for T in ONE_LINERS:
            d = cf.paired_difference(DECL, test, sel["name"], test, T, n_boot=500, seed=1)
            ol[T] = dict(acc=c["attackers"][T]["test"], diff=d["diff"], diff_se=d["se"], diff_ci=d["ci95"])
        out["oneliner"][str(n)] = ol
        if n in (150, 1425) and dose > 0:
            # unit-level label shuffle of val and test: the same predictions against permuted identities must read as chance
            ysh = _unit_shuffle(yt, ct, rs)
            vsh = _unit_shuffle(val.y, val.cluster, rs)
            cs = cf.certify(DECL, cf.SplitPreds(vsh, val.cluster, val.scores), cf.SplitPreds(ysh, ct, test.scores), split_status="synthetic")
            out["shuffled"][str(n)] = dict(verdict=cs["verdict"]["label"], sel_test=cs["selected"]["test"], baseline=cs["baseline_used"])
        # truth-based selection bias over the 30-attacker pool
        cv = corr["val"]
        ctest = corr["test"][keep]
        out["truth_infl"][str(n)] = {str(K): v for K, v in sb.truth_inflation_rep(cv, val_u, ctest, ct, true_acc, K_LIST, 40, rs).items()}
    out["secs"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--null-reps", type=int, default=300)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true", help="2 replicates of 2 cells, to check the plumbing only (outputs are discarded)")
    a = ap.parse_args()
    jobs = []
    for ci, (ch, d) in enumerate(cells()):
        R = a.null_reps if ch == "null" else a.reps
        if a.smoke:
            if (ch, d) not in (("null", 0.0), ("gain", 0.122)):
                continue
            R = 2
        jobs += [(ch, d, r, ci) for r in range(R)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                j = json.loads(line)
                done.add((j["channel"], j["dose"], j["rep"]))
            except Exception:  # noqa: BLE001
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} jobs to run, {len(done)} already done", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(a.procs) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
