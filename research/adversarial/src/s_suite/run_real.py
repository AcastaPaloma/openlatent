"""Runs on programme 1's saved HAR harness predictions (checks C6 and C8 of results_notes/S.md) and the toy validation of the evidence-budget module.

python -m s_suite.run_real toy    --out DIR
python -m s_suite.run_real cert   --root HARNESS_DIR --out DIR      (certificate on a recording-level val/sealed half split, K = 2 retrained attackers)
python -m s_suite.run_real select --root HARNESS_DIR --out DIR      (selection bias of the maximum, release cells and attacker families)
python -m s_suite.run_real budget --root HARNESS_DIR --out DIR      (evidence budget and independence test)
HARNESS_DIR holds <dataset>/{dataset.json, eval/<release>/{preds.npz,result.json}} downloaded from HF KuanKuanKuan/openlatent-privacy/harness.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import certificate as cf
from . import evidence_budget as eb
from . import harness_io as hio
from . import selection_bias as sb
from . import stats as st

DATASETS = ("kth_ilic", "kth_scenario", "utdmhad", "charades")
FAMILIES = {  # a release and its seed / epoch / DINO attacker variants: same release, same test items
    "kth_ilic": ["pose_canon_down16", "pose_canon_down16_s1", "pose_canon_down16_s2", "pose_canon_down16_e24", "pose_canon_down16_dino"],
    "kth_scenario": ["pose_canon_down16", "pose_canon_down16_s1", "pose_canon_down16_s2", "pose_canon_down16_e24", "pose_canon_down16_dino"],
    "utdmhad": ["motion_down64", "motion_down64_s1", "motion_down64_s2", "motion_down64_e24", "motion_down64_dino"],
    "charades": ["pose", "pose_s1", "pose_s2", "pose_e24", "pose_dino"],
}
K_GRID = [1, 2, 3, 4, 6, 8, 10, 15, 20]
PRIMARY = ("retrained|frame_r50", "retrained|r3d")


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def cmd_toy(a):
    res = {}
    for rho in (0.0, 0.5, 0.9):
        runs = []
        for seed in range(20):
            it = eb.toy_items(rho=rho, n_sessions=8, per_session=12, seed=seed)
            r = eb.budget_curve(it, [1, 2, 3, 5, 8, 10], n_draws=60, n_boot=300, seed=seed)
            ind = r["independence"][-1]
            runs.append(dict(seed=seed, diverse=r["schemes"]["diverse"]["acc"].tolist(), same_session=r["schemes"]["same_session"]["acc"].tolist(),
                             random=r["schemes"]["random"]["acc"].tolist(), detected_K10=ind["dependence_detected"], diff_K10=ind["diff"],
                             mde_K10=ind["mde80"], pred=eb.predicted_n_star(it.probs, it.person)))
        res[str(rho)] = runs
    # shuffled-person control
    ctrl = []
    for seed in range(20):
        it = eb.toy_items(rho=0.0, n_sessions=8, per_session=12, seed=seed)
        rs = np.random.RandomState(seed)
        it2 = eb.Items(it.probs, rs.permutation(it.person), it.session)
        r = eb.budget_curve(it2, [1, 2, 3, 5, 8, 10], schemes=("random", "diverse"), n_draws=60, n_boot=200, seed=seed)
        ctrl.append(dict(seed=seed, random=r["schemes"]["random"]["acc"].tolist(), diverse=r["schemes"]["diverse"]["acc"].tolist()))
    res["shuffled_person_control"] = ctrl
    _dump(res, os.path.join(a.out, "toy_budget.json"))
    print("toy done")


def _halves(units, seed):
    rs = np.random.RandomState(seed)
    u = np.array(sorted(set(units)))
    perm = rs.permutation(len(u))
    return set(u[perm[: len(u) // 2]]), set(u[perm[len(u) // 2:]])


def cmd_cert(a):
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for rel in hio.releases(a.root, ds):
            cell = hio.load_cell(a.root, ds, rel, "within", ("retrained",))
            if not all(k in cell.probs for k in PRIMARY):
                continue
            vset, sset = _halves(cell.record, 0)
            vm = np.array([r in vset for r in cell.record])
            sm = np.array([r in sset for r in cell.record])
            C = cell.probs[PRIMARY[0]].shape[1]
            dec = cf.SuiteDecl(f"harness-{ds}", PRIMARY, "accuracy", 0.05, 2000, 0, 0.02, C)
            val = cf.SplitPreds(cell.y[vm], cell.record[vm], {k: cell.probs[k][vm] for k in PRIMARY})
            test = cf.SplitPreds(cell.y[sm], cell.record[sm], {k: cell.probs[k][sm] for k in PRIMARY})
            c = cf.certify(dec, val, test)
            out[ds][rel] = dict(n_units_sealed=c["data"]["n_test_units"], chance=c["chance"]["used"], selected=c["selected"]["name"],
                                sel_val=c["selected"]["val"], sel_test=c["selected"]["test"], ci95=c["selected"]["ci95"], se=c["selected"]["se"],
                                mde80=c["selected"]["mde80"], excess_lcb=c["selected"]["excess_lcb_onesided"], naive_max=c["naive_max"]["test"],
                                ub_bonf=c["suite_bound"]["bonferroni"], ub_maxt=c["suite_bound"]["maxt"], floor_k2=c["floor"]["ucb_at_baseline_suite"],
                                verdict=c["verdict"]["label"], norm_leak=c["selected"]["normalised_leakage"], digest=c["digest"][:12])
    _dump(out, os.path.join(a.out, "harness_cert_halfsplit.json"))
    print("cert done")


def cmd_select(a):
    res = {"cells": {}, "families": {}}
    t0 = time.time()
    for ds in DATASETS:
        for rel in hio.releases(a.root, ds):
            cell = hio.load_cell(a.root, ds, rel, "within", ("retrained",))
            keys = sorted(cell.probs)
            if len(keys) < 2:
                continue
            Cm = np.stack([(cell.probs[k].argmax(1) == cell.y) for k in keys], 1).astype(float)
            r = sb.split_inflation(Cm, cell.record, list(range(1, len(keys) + 1)), n_splits=400, n_outer=200, inner_per_outer=20, seed=1)
            r["attackers"] = keys
            r["acc_full"] = Cm.mean(0).tolist()
            res["cells"][f"{ds}/{rel}"] = r
        fam = [x for x in FAMILIES[ds] if x in hio.releases(a.root, ds)]
        Cm, names, base = hio.family_pool(a.root, ds, fam, "within", ("retrained",))
        P = Cm.shape[1]
        r = sb.split_inflation(Cm, base.record, [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12) if k <= P] + ([P] if P not in (1, 2, 3, 4, 5, 6, 8, 10, 12) else []),
                               n_splits=400, n_outer=200, inner_per_outer=20, seed=2)
        r["attackers"] = names
        r["acc_full"] = Cm.mean(0).tolist()
        res["families"][ds] = r
        print(ds, "done", f"{time.time() - t0:.0f}s", flush=True)
    _dump(res, os.path.join(a.out, "harness_selection_bias.json"))


def cmd_budget(a):
    res = {}
    t0 = time.time()
    for ds in DATASETS:
        for rel in hio.releases(a.root, ds):
            cell = hio.load_cell(a.root, ds, rel, "within", ("retrained",))
            n_sess = len(np.unique(cell.session)) / max(len(np.unique(cell.y)), 1)
            for key in PRIMARY:
                if key not in cell.probs:
                    continue
                items = eb.Items(cell.probs[key], cell.y, cell.session, cell.record)
                r = eb.budget_curve(items, K_GRID, n_draws=100, n_boot=1000, seed=0, targets=(0.5, 0.9))
                pred = eb.predicted_n_star(cell.probs[key], cell.y)
                res[f"{ds}/{rel}/{key}"] = dict(K=K_GRID, sessions_per_person=n_sess, n_items=len(cell.y), n_persons=len(np.unique(cell.y)), pred=pred,
                                                schemes={s: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()} for s, d in r["schemes"].items()},
                                                independence=r.get("independence"),
                                                acc1=float(np.mean(cell.probs[key].argmax(1) == cell.y)))
        print(ds, "done", f"{time.time() - t0:.0f}s", flush=True)
    _dump(res, os.path.join(a.out, "harness_evidence_budget.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["toy", "cert", "select", "budget"])
    ap.add_argument("--root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"toy": cmd_toy, "cert": cmd_cert, "select": cmd_select, "budget": cmd_budget}[a.cmd](a)


if __name__ == "__main__":
    main()
