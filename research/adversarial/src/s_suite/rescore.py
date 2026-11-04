"""Zero-GPU deliverable (FOCUS amendment A3, audit F8): re-score programme 1's stored harness predictions with the S machinery and state, per
release, whether programme 1's verdict moves by more than the printed MDE.

All inputs are programme-1 test splits: they are DEVELOPMENT/BURNED data (programme 1 selected on them). Nothing here is a sealed number; every
output is labelled `development_burned`.

Per (dataset, release, protocol):
  A1   programme 1's printed headline (report.json: worst attacker over its listed attackers, test top-1 or mAP), recomputed from preds.npz
       ('reproduces' if within 0.1 points; if not, the cell is flagged and excluded from the 'moves' count)
  A2   best-of-K on the full test over the declared suite (retrained frame_r50 and r3d, K = 2): programme 1's selection rule on K = 2
  H    honest value: cross-fit validation-selected attacker. 20 random half-splits of the recordings; the attacker is selected on one half and
       scores the other half, both directions; H = mean over splits. Every recording is scored by an attacker that never saw it.
  D    A2 - H with a paired recording-cluster bootstrap (B resamples of recordings; the selection is redone inside each resample for A2 and for
       the split-selected attacker of H). MDE80(D) = 2.8 x SE(D). moves := |D| > MDE80(D)  (and |D| > 0 when SE(D) = 0).
  UB   Bonferroni upper bound over the K = 2 declared attackers on the full test (Clopper-Pearson on effective counts, alpha / K).
Privacy-scale restatement (programme 1's score): privacy = 1 - clip((a - base) / (raw - base)); `moves_privacy` is the same test in those units.
Adjacent-pair orderings: releases sorted by A1; for each adjacent pair the paired difference of A2 and of H with its MDE80 and whether the 95%
bootstrap interval excludes 0.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import harness_io as hio
from . import provenance as prov
from . import stats as st
from .selection_bias import _subset_cols  # noqa: F401  (kept for symmetry with the other analysis scripts)

SUITE = ("retrained|frame_r50", "retrained|r3d")
N_SPLITS = 20
B_BOOT = 1000
B_BOOT_GENERIC = 300
REPRO_TOL = 0.1  # points
HEADLINE_PAIRS = {  # programme 1 / stress-test headline comparisons (a minus b)
    "kth_ilic": [("pose_canon_down16", "bdq_down64"), ("pose_canon_down16", "pose_canon_down32"), ("pose_canon_down16", "pose_canon_down8"),
                 ("pose_canon_down16", "pose_canon_down16_s1"), ("pose_canon_down16", "pose_canon_down16_e24"), ("bdq_down64", "bdq_down32")],
    "kth_scenario": [("pose_canon_down16", "pose_canon_down32"), ("pose_canon_down16", "pose_canon_down16_e24")],
}


def p1_key(k):  # report 'retrained/frame_r50' -> preds 'retrained|frame_r50'
    return k.replace("/", "|")


def _split_masks(m, R, seed):
    rs = np.random.RandomState(seed)
    M = np.zeros((R, m))
    for s in range(R):
        M[s, rs.permutation(m)[: m // 2]] = 1.0
    return M


def _crossfit_and_best(S, n_c, w, M, chunk=50):
    """S (m,K) unit sums, w (B,m) weights, M (R,m) half masks -> A2 (B,), H (B,)."""
    A2s, Hs = [], []
    for i in range(0, len(w), chunk):
        a, h = _crossfit_chunk(S, n_c, w[i:i + chunk], M)
        A2s.append(a)
        Hs.append(h)
    return np.concatenate(A2s), np.concatenate(Hs)


def _crossfit_chunk(S, n_c, w, M):
    B, m = w.shape
    R = M.shape[0]
    tot_n = w @ n_c
    A2 = (w @ S).max(1) / tot_n
    wA = (w[:, None, :] * M[None]).reshape(B * R, m)
    wB = (w[:, None, :] * (1 - M)[None]).reshape(B * R, m)
    SA, SB = wA @ S, wB @ S
    nA, nB = wA @ n_c, wB @ n_c
    with np.errstate(all="ignore"):
        kA = np.argmax(np.nan_to_num(SA / nA[:, None], nan=-1), axis=1)
        kB = np.argmax(np.nan_to_num(SB / nB[:, None], nan=-1), axis=1)
    r = np.arange(B * R)
    H = ((SB[r, kA] + SA[r, kB]) / (nA + nB)).reshape(B, R).mean(1)
    return A2, H


def _stat(arr, point=None):
    return dict(se=float(arr.std(ddof=1)), ci95=[float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))])


def _diff_block(d, d_point):
    se = float(d.std(ddof=1))
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return dict(diff=float(d_point), se=se, mde80=st.mde80(se), ci95=[lo, hi], resolved=bool(lo > 0 or hi < 0),
                exceeds_mde=bool(abs(d_point) > st.mde80(se) and (se > 0 or abs(d_point) > 0)))


def rescore_dataset(root, ds, protocol, report, seed=0):
    rels = [r for r in hio.releases(root, ds) if r in report["scored"]]
    sec = "identity" if not ds.startswith("hmdb") else "attributes"
    multilabel = ds.startswith("hmdb")
    cells = {}
    for r in rels:
        try:
            c = hio.load_cell(root, ds, r, protocol, ("retrained", "frozen"), secret=sec, multilabel=multilabel)
        except KeyError:
            continue
        if all(k in c.probs for k in SUITE):
            cells[r] = c
    if not cells:
        return {}
    ids0 = next(iter(cells.values())).ids
    cells = {r: c for r, c in cells.items() if c.ids == ids0}
    c0 = next(iter(cells.values()))
    inv, m = st.unit_index(c0.record)
    rs_w = st.boot_weights(m, B_BOOT if not multilabel else B_BOOT_GENERIC, seed + 1)
    M = _split_masks(m, N_SPLITS if not multilabel else 6, seed + 2)
    n_c = np.bincount(inv, minlength=m).astype(float)
    out, boots = {}, {}
    for r, c in cells.items():
        sc = report["scored"][r]["secrets"][sec].get(protocol)
        if sc is None:
            continue
        keys = [p1_key(k) for k in sc["attackers"]]
        keys = [k for k in keys if k in c.probs]
        if multilabel:
            cond = np.array([hio.dataset_index(root, ds)[i]['labels']['action'] for i in c.ids])
            vals, a2, h, a2_b, h_b = _generic_multilabel(c, keys, M, rs_w, inv, m, cond)
        else:
            corr = {k: (c.probs[k].argmax(1) == c.y).astype(float) for k in keys}
            a1_re = max(v.mean() for v in corr.values())
            S = st.unit_sums(np.stack([corr[k] for k in SUITE], 1), inv, m)
            a2_b, h_b = _crossfit_and_best(S, n_c, rs_w, M)
            a2_pt, h_pt = _crossfit_and_best(S, n_c, np.ones((1, m)), M)
            a2, h = float(a2_pt[0]), float(h_pt[0])
            per = st.ratio_point(S, n_c)
            boot = st.ratio_boot(S, n_c, rs_w)
            se_k = boot.std(0, ddof=1)
            single = {k: dict(acc=float(per[i]), h_minus_single=_diff_block(h_b - boot[:, i], h - per[i])) for i, k in enumerate(SUITE)}
            ub = max(st.ucb_effective(per[k], se_k[k], len(c.y), 0.05 / len(SUITE), m) for k in range(len(SUITE)))
        if multilabel:
            a1_re = vals["a1_re"]
            ub = float("nan")
            single = None
        base, raw = sc["base"] / 100.0, report["scored"][r]["secrets"][sec][protocol]["raw"] / 100.0
        a1_p1 = sc["score"] / 100.0
        D_b = a2_b - h_b
        d = _diff_block(D_b, a2 - h)
        gap = raw - base
        out[r] = dict(
            n_units=m, n_items=len(c.y), base=base, raw=raw, a1_printed=a1_p1, a1_recomputed=a1_re, reproduces=bool(abs(a1_re - a1_p1) * 100 <= REPRO_TOL),
            a2=a2, a2_se=float(a2_b.std(ddof=1)), honest=h, honest_se=float(h_b.std(ddof=1)), honest_mde80=st.mde80(h_b.std(ddof=1)),
            optimism_a2_minus_h=d, ub_bonferroni_K2=ub, single_attackers=single, other_family_gain=a1_p1 - a2,
            privacy_printed=sc["privacy"], privacy_honest=(None if gap < 0.02 else 1 - min(1, max(0, (h - base) / gap))),
            privacy_ub=(None if gap < 0.02 or not np.isfinite(ub) else 1 - min(1, max(0, (ub - base) / gap))),
            moves=d["exceeds_mde"], moves_privacy=(None if gap < 0.02 else bool(d["exceeds_mde"])),
            mde80_privacy_units=(None if gap < 0.02 else d["mde80"] / gap))
        boots[r] = (a2_b, h_b)
    # adjacent-pair orderings in programme 1's own ranking
    order = sorted(out, key=lambda r: out[r]["a1_printed"])
    pairs = []
    for r1, r2 in zip(order[:-1], order[1:]):
        a2d = boots[r2][0] - boots[r1][0]
        hd = boots[r2][1] - boots[r1][1]
        pairs.append(dict(lower=r1, higher=r2, printed_gap=out[r2]["a1_printed"] - out[r1]["a1_printed"],
                          a2=_diff_block(a2d, out[r2]["a2"] - out[r1]["a2"]), honest=_diff_block(hd, out[r2]["honest"] - out[r1]["honest"])))
    heads = []
    for r1, r2 in HEADLINE_PAIRS.get(ds, []):
        if r1 in boots and r2 in boots:
            heads.append(dict(a=r1, b=r2, a2=_diff_block(boots[r1][0] - boots[r2][0], out[r1]["a2"] - out[r2]["a2"]),
                              honest=_diff_block(boots[r1][1] - boots[r2][1], out[r1]["honest"] - out[r2]["honest"])))
    return dict(releases=out, adjacent_pairs=pairs, headline_pairs=heads, n_units=m, order=order, boots=boots,
                dropped_releases=sorted(set(rels) - set(cells)))


def _generic_multilabel(c, keys, M, W, inv, m, cond):
    """HMDB51 attributes with programme 1's metric (plan.json: 'conditional'): mean over the 5 attributes of the mean AP within each action
    class where the attribute varies (amendment v1.1: the pre-registered plain mAP did not reproduce programme 1's printed numbers).
    Units = videos; explicit item resampling, fewer resamples."""
    def ap1(scores, labels):
        order = np.argsort(-scores, kind="stable")
        yy = labels[order]
        if yy.sum() == 0:
            return float("nan")
        tp = np.cumsum(yy)
        return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / yy.sum())

    def apk(probs, y, cnd=None):
        cnd = cond_all if cnd is None else cnd
        per_attr = []
        for j in range(y.shape[1]):
            aps = []
            for cls in np.unique(cnd):
                mk = cnd == cls
                if 0 < y[mk, j].sum() < mk.sum():
                    aps.append(ap1(probs[mk, j], y[mk, j]))
            if aps:
                per_attr.append(np.mean(aps))
        return float(np.mean(per_attr)) if per_attr else float("nan")

    cond_all = cond
    y = c.y
    members = [np.flatnonzero(inv == i) for i in range(m)]
    a1_re = max(apk(c.probs[k], y, cond) for k in keys)

    def stats_for(w):
        idx_w = np.repeat(np.arange(m), w.astype(int))
        # item index list with multiplicities
        items = np.concatenate([members[u] for u in idx_w]) if len(idx_w) else np.array([], int)
        mc = {k: apk(c.probs[k][items], y[items], cond[items]) for k in SUITE}
        a2 = max(mc.values())
        Hs = []
        for s in range(M.shape[0]):
            inA = M[s].astype(bool)
            tot, n_tot = 0.0, 0
            for sel_half, score_half in ((inA, ~inA), (~inA, inA)):
                u_sel = np.repeat(np.arange(m), (w * sel_half).astype(int))
                u_sc = np.repeat(np.arange(m), (w * score_half).astype(int))
                if len(u_sel) == 0 or len(u_sc) == 0:
                    continue
                it_sel = np.concatenate([members[u] for u in u_sel])
                it_sc = np.concatenate([members[u] for u in u_sc])
                ks = max(SUITE, key=lambda k: apk(c.probs[k][it_sel], y[it_sel], cond[it_sel]))
                tot += apk(c.probs[ks][it_sc], y[it_sc], cond[it_sc]) * len(it_sc)
                n_tot += len(it_sc)
            Hs.append(tot / n_tot)
        return a2, float(np.mean(Hs))

    a2_pt, h_pt = stats_for(np.ones(m))
    a2_b = np.zeros(len(W))
    h_b = np.zeros(len(W))
    for b in range(len(W)):
        a2_b[b], h_b[b] = stats_for(W[b])
    return dict(a1_re=a1_re), a2_pt, h_pt, a2_b, h_b


def to_markdown(res):
    L = ["# Re-score of programme 1's harness predictions (development/burned data, NOT sealed)\n"]
    for key, r in res.items():
        if not r or key.startswith("_"):
            continue
        L.append(f"\n## {key}  ({r['n_units']} independent units)\n")
        L.append("| release | P1 printed | recomputed | repro | best-of-2 (A2) | honest cross-fit (H) | A2-H [95% CI] (MDE80) | moves | Bonf. UB K=2 | privacy printed / honest | MDE80 in privacy units |")
        L.append("|---|---:|---:|:-:|---:|---:|---|:-:|---:|---|---:|")
        for rel in r["order"]:
            x = r["releases"][rel]
            d = x["optimism_a2_minus_h"]
            pr = "-" if x["privacy_honest"] is None else f"{x['privacy_printed']:.2f} / {x['privacy_honest']:.2f}"
            mp = "-" if x["mde80_privacy_units"] is None else f"{x['mde80_privacy_units']:.2f}"
            L.append(f"| {rel} | {100*x['a1_printed']:.1f} | {100*x['a1_recomputed']:.1f} | {'y' if x['reproduces'] else 'NO'} | {100*x['a2']:.1f} | {100*x['honest']:.1f} "
                     f"| {100*d['diff']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] ({100*d['mde80']:.2f}) | {'MOVES' if x['moves'] else 'no'} "
                     f"| {100*x['ub_bonferroni_K2']:.1f} | {pr} | {mp} |")
        L.append("\nAdjacent orderings in programme 1's ranking (paired, same recordings):\n")
        L.append("| lower leakage | higher leakage | printed gap | best-of-2 gap [CI] (MDE80) resolved | honest gap [CI] (MDE80) resolved |")
        L.append("|---|---|---:|---|---|")
        for p in r["adjacent_pairs"]:
            a, h = p["a2"], p["honest"]
            L.append(f"| {p['lower']} | {p['higher']} | {100*p['printed_gap']:.1f} | {100*a['diff']:+.1f} [{100*a['ci95'][0]:+.1f}, {100*a['ci95'][1]:+.1f}] ({100*a['mde80']:.1f}) {'yes' if a['resolved'] else 'no'} "
                     f"| {100*h['diff']:+.1f} [{100*h['ci95'][0]:+.1f}, {100*h['ci95'][1]:+.1f}] ({100*h['mde80']:.1f}) {'yes' if h['resolved'] else 'no'} |")
        if r.get("headline_pairs"):
            L.append("\nHeadline pairs (a minus b, same recordings):\n")
            L.append("| a | b | best-of-2 diff [CI] (MDE80) resolved | honest diff [CI] (MDE80) resolved |")
            L.append("|---|---|---|---|")
            for p in r["headline_pairs"]:
                a, h = p["a2"], p["honest"]
                L.append(f"| {p['a']} | {p['b']} | {100*a['diff']:+.1f} [{100*a['ci95'][0]:+.1f}, {100*a['ci95'][1]:+.1f}] ({100*a['mde80']:.1f}) {'yes' if a['resolved'] else 'no'} "
                         f"| {100*h['diff']:+.1f} [{100*h['ci95'][0]:+.1f}, {100*h['ci95'][1]:+.1f}] ({100*h['mde80']:.1f}) {'yes' if h['resolved'] else 'no'} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with <dataset>/{dataset.json,eval/...}")
    ap.add_argument("--reports", required=True, help="programme-1 results/harness dir holding <dataset>/report.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="kth_ilic,kth_scenario,utdmhad,charades,hmdb51_vp")
    ap.add_argument("--protocols", default="within", help="primary = within; others are exploratory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, t0 = {}, time.time()
    for ds in a.datasets.split(","):
        report = json.load(open(os.path.join(a.reports, ds, "report.json")))
        for proto in a.protocols.split(","):
            r = rescore_dataset(a.root, ds, proto, report)
            res[f"{ds}/{proto}"] = r
            print(ds, proto, f"{time.time() - t0:.0f}s", flush=True)
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "boots"} for k, v in res.items()}
    json.dump(prov.add(clean, __file__), open(os.path.join(a.out, "rescore_programme1.json"), "w"), indent=1, sort_keys=True, default=float)
    open(os.path.join(a.out, "rescore_programme1.md"), "w").write(prov.md_line(__file__) + "\n" + to_markdown(res))


if __name__ == "__main__":
    main()
