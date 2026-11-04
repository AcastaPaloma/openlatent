"""Coverage re-verification of the certificate's intervals with an accurate truth sample (S.md amendment v1.3). CPU, multiprocessing.

python -m s_suite.run_coverage --out OUT.jsonl --reps 30 --procs 8 [--smoke]

Per replicate: new identities, SUITE12 retrained on the train split, a validation split (150 sessions), a pool of 4,750 sealed sessions from which
n = 150 / 360 / 1,425 / 4,707 sessions are drawn at random, and a TRUTH sample of 25,000 sessions (100,000 windows) that gives each attacker's
true accuracy with a standard error of about 0.1 point (estimated per replicate by a session bootstrap and recorded). The certificate is the frozen
`certify`; this script only adds the truth and the coverage bookkeeping (it also records the Clopper-Pearson interval on effective counts that would
be adopted if the percentile interval failed). Planted data: unit test of the statistics, no power claim.
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
from scipy.stats import beta  # noqa: E402

from . import certificate as cf  # noqa: E402
from . import provenance as prov  # noqa: E402
from . import stats as st  # noqa: E402
from .attackers import SUITE12, fit_predict_pool  # noqa: E402
from .planted import PlantedSpec, features, generate_replicate, raw_view  # noqa: E402
from .run_planted import DECL, DOSES, ENGINE_DOSE_IDX  # noqa: E402

N_TEST = (150, 360, 1425, 4707)
POOL_PER_ID = 190          # 4,750 sealed sessions
TRUTH_PER_ID = 1000        # 25,000 truth sessions


def cells():
    out = [("null", 0.0)]
    for ch, ds in DOSES.items():
        out += [(ch, ds[i]) for i in ENGINE_DOSE_IDX]
    return out


def cp_two_sided(p, se, n_items, n_units, alpha=0.05):
    k, n = st.eff_counts(p, se, n_items, n_units)
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def run_job(job):
    channel, dose, rep, cell_idx = job
    t0 = time.time()
    spec = PlantedSpec(channel=channel, dose=dose, ident_seed=7000 + rep + 100 * cell_idx)
    data = generate_replicate(spec, seed=9000000 + rep + 1000 * cell_idx, units_train=12, units_val=6, units_test=POOL_PER_ID, units_pop=TRUTH_PER_ID)
    vt = {"raw": raw_view(data["train"]["X"]), "feat": features(data["train"]["X"])}
    ve = {s: {"raw": raw_view(data[s]["X"]), "feat": features(data[s]["X"])} for s in ("val", "test", "pop")}
    P = fit_predict_pool(vt, data["train"]["y"], ve, names=SUITE12, n_classes=spec.n_id)
    corr_pop = np.stack([(P["pop"][a].argmax(1) == data["pop"]["y"]) for a in SUITE12], 1).astype(float)
    true12 = corr_pop.mean(0)
    inv_p, m_p = st.unit_index(data["pop"]["unit"])
    S_p = st.unit_sums(corr_pop, inv_p, m_p)
    n_p = np.bincount(inv_p, minlength=m_p).astype(float)
    Wp = st.boot_weights(m_p, 200, 11)
    true_se12 = st.ratio_boot(S_p, n_p, Wp).std(0, ddof=1)
    rs = np.random.RandomState(rep + 31 * cell_idx + 5)
    sess_all = np.unique(data["test"]["unit"])
    out = dict(channel=channel, dose=dose, rep=rep, true=true12.tolist(), true_se=true_se12.tolist(), cert={})
    val = cf.SplitPreds(data["val"]["y"], np.array(["v%d" % u for u in data["val"]["unit"]]), {a: P["val"][a] for a in SUITE12})
    for n in N_TEST:
        pick = rs.choice(sess_all, n, replace=False)
        keep = np.isin(data["test"]["unit"], pick)
        test = cf.SplitPreds(data["test"]["y"][keep], np.array(["t%d" % u for u in data["test"]["unit"][keep]]), {a: P["test"][a][keep] for a in SUITE12})
        c = cf.certify(DECL, val, test, split_status="synthetic")
        sel = c["selected"]
        k = SUITE12.index(sel["name"])
        ts, tse = float(true12[k]), float(true_se12[k])
        n_items = len(test.y)
        lo_cp, hi_cp = cp_two_sided(sel["test"], sel["se"], n_items, n)
        sb = c["suite_bound"]
        out["cert"][str(n)] = dict(
            sel=sel["name"], sel_test=sel["test"], sel_se=sel["se"], ci95=sel["ci95"], ci_cp=[lo_cp, hi_cp], true_sel=ts, true_se_sel=tse,
            z=(sel["test"] - ts) / sel["se"], cover_ci=bool(sel["ci95"][0] <= ts <= sel["ci95"][1]),
            ci_cons=sel["ci95_conservative"], cover_ci_cons=bool(sel["ci95_conservative"][0] <= ts <= sel["ci95_conservative"][1]), cover_cp=bool(lo_cp <= ts <= hi_cp),
            cover_ci_truthadj=bool(abs(sel["test"] - ts) <= 1.96 * np.hypot(sel["se"], tse)), cover_bonf=bool(sb["bonferroni"] >= true12.max()),
            cover_maxt=bool(sb["maxt"] >= true12.max()), cover_ucb1=bool(sel["ucb_onesided"] >= ts), verdict=c["verdict"]["label"],
            true_best12=float(true12.max()), ci_width=sel["ci95"][1] - sel["ci95"][0], cp_width=hi_cp - lo_cp, n_units=c["data"]["n_test_units"])
    out["secs"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rep-start", type=int, default=0, help="first replicate index (fresh identities for out-of-sample verification)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = []
    for rep in range(a.rep_start, a.rep_start + a.reps):                       # replicate-major order: an early stop leaves every cell balanced
        for ci, (ch, d) in enumerate(cells()):
            if a.smoke and (ch, d) not in (("null", 0.0), ("gain", 0.092)):
                continue
            jobs.append((ch, d, rep, ci))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                j = json.loads(line)
                done.add((j["channel"], j["dose"], j["rep"]))
            except Exception:  # noqa: BLE001
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} jobs to run, {len(done)} done; {prov.md_line(__file__)}", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(a.procs) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
