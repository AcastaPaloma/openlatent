"""UNSEAL and analyse N2 (evaluator v3): join the sealed manifest, the sealed oracle report and S's v3 certificates; apply the pre-registered criteria of N.md section 11.

    python -m n_control.analyse2 --manifest ~/sealed_manifest_n2/manifest.json --sealed-dir ~/sealed_manifest_n2 --certs ~/n2_certs --out ~/n2_results \
        [--releases ~/n2_releases --preds ~/n2_preds]     (the last two add the v1-equivalent re-scoring of the SAME predictions, secondary)

Thresholds are constants at the top and are the ones committed in N.md before any release existed (carried over from N1, plus the tightened assurance limit).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import beta

PRIMARY_LEVELS = (3, 4)
DET_RATE, DET_LOWER = 0.80, 0.60
MISS_RATE = 0.50
FA_RATE, FA_UPPER = 0.05, 0.10
ASSURE_MAX = 0.10                      # tightening: BELOW_MARGIN on planted leaks at the primary levels must stay <= 0.10 for every family
CONTROLS = ("bias", "specin")
TARGETS = ("hb2", "phase2", "af2")
NEW = ("pac", "order")
FAMILIES = CONTROLS + TARGETS + NEW
GROUPS = {"hb": ("lr_hb_c0.05", "lda_hb_auto", "rf_hb_200", "knn5_hb"), "phase": ("lr_phase_c0.05", "lda_phase_auto", "rf_phase_200", "knn5_phase"),
          "inter": ("lr_inter_c0.05", "lgbm_lvlspec", "rf_lvlspec_200", "mlp_lvlspec_64")}
SUITE12 = ("lr_raw_c0.1", "lda_raw_auto", "knn5_raw", "rf_raw_100", "lr_feat_c1", "lda_feat_auto", "knn5_feat", "rf_feat_200", "et_feat_200", "lgbm_feat", "mlp_feat_64", "knn1_raw")


def cp(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))


def cp_upper1(k, n, a=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


def rate(rows, key):
    n = len(rows)
    k = sum(1 for r in rows if key(r))
    return dict(k=k, n=n, rate=(k / n if n else float("nan")), ci95=list(cp(k, n)))


def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def load_rows(manifest_path, sealed_dir, certs_dir):
    man = json.loads(open(os.path.expanduser(manifest_path), "rb").read())
    sd = os.path.expanduser(sealed_dir)
    rmap = json.load(open(os.path.join(sd, "release_map.json")))
    orc = json.load(open(os.path.join(sd, "oracle.json")))
    ent = {e["rid"]: e for e in man["entries"]}
    rows = []
    for d, rid in rmap.items():
        p = os.path.join(os.path.expanduser(certs_dir), f"{d}.json")
        e, o = ent[rid], orc[str(rid)]
        r = dict(digest=d, rid=rid, kind=e["kind"], family=e["family"], level=e["level"], base=e["base"], amp=e["amp"], has_cert=os.path.exists(p))
        if e["kind"] != "null":
            r.update(om_w=o["planted"]["win"], om_s=o["planted"]["sess"], omr_w=o["released"]["win"], omr_s=o["released"]["sess"])
        if r["has_cert"]:
            c = json.load(open(p))
            sel, sb, v = c["selected"], c["suite_bound"], c["verdict"]
            per = {a: x["test"] for a, x in c["attackers"].items()}
            r.update(verdict=v["label"], withheld=list(v.get("assurance_withheld", [])), below_bonf=bool(v["below_margin_bonferroni"]), below_maxt=bool(v["below_margin_maxt"]),
                     canaries_run=c["_n_control"]["canaries_run"], sel_name=sel["name"], sel_test=sel["test"], sel_se=sel["se"], excess=sel["excess_over_baseline"],
                     excess_lcb=sel["excess_lcb_onesided"], excess_mde=sel["excess_mde80"], base_used=c["baseline_used"], ub_bonf=sb["bonferroni"], ub_maxt=sb["maxt"],
                     naive_max=c["naive_max"]["test"], n_units=c["data"]["n_test_units"], floor_suite=c["floor"]["ucb_at_baseline_suite"], k=sb["k"],
                     ledger_status=c["split_status"], frozen_match=c["evaluator"]["frozen_match"], evaluator_sha=c["evaluator"]["sha256"], per_attacker=per,
                     group_best={g: max(per[a] for a in names) for g, names in GROUPS.items()}, base12_best=max(per[a] for a in SUITE12),
                     canary_accept={g: (None if x is None else bool(x["accepted"])) for g, x in c["view_scope"]["canaries"].items()})
        rows.append(r)
    return man, rows


def cell(rows):
    out = dict(n=len(rows), detect=rate(rows, lambda r: r["verdict"] == "LEAK"), assure=rate(rows, lambda r: r["verdict"] == "BELOW_MARGIN"),
               unresolved=rate(rows, lambda r: r["verdict"] == "NOT_RESOLVED"), om_w=mean([r.get("om_w") for r in rows]), om_s=mean([r.get("om_s") for r in rows]),
               sel_test=mean([r["sel_test"] for r in rows]), base=mean([r["base_used"] for r in rows]), naive_max=mean([r["naive_max"] for r in rows]),
               excess_mde=mean([r["excess_mde"] for r in rows]), ub_bonf=mean([r["ub_bonf"] for r in rows]))
    res = [r for r in rows if r.get("om_w") is not None and r["om_w"] - r["base_used"] >= r["excess_mde"]]
    out["resolvable_window"] = rate(res, lambda r: r["verdict"] == "LEAK")
    den = out["om_w"] - out["base"]
    out["attain_window"] = (out["sel_test"] - out["base"]) / den if den > 1e-9 else float("nan")
    return out


def status(c):
    d = c["detect"]
    if d["n"] and d["rate"] >= DET_RATE and d["ci95"][0] >= DET_LOWER:
        return "DETECTED"
    if d["n"] and d["rate"] < MISS_RATE:
        return "MISSED"
    return "PARTIAL"


def subverdict(R, fams, need_all):
    st = {f: R["primary"][f]["status"] for f in fams}
    n_det = sum(v == "DETECTED" for v in st.values())
    n_miss = sum(v == "MISSED" for v in st.values())
    assure_ok = all(R["primary"][f]["assure"]["rate"] <= ASSURE_MAX for f in fams)
    if need_all:
        yes, no = n_det == len(fams) and assure_ok, n_miss == len(fams)
    else:
        yes, no = n_det >= 2 and assure_ok, n_miss >= 2
    return dict(status=st, n_detected=n_det, n_missed=n_miss, assurance_ok=assure_ok, label="YES" if yes else ("NO" if no else "MIXED"))


def rescore_v1(rows, rel_dir, pred_dir):
    """v1 suite (SUITE12 alone, the N1 declaration) re-scored on the SAME stored predictions: not a second sealed query (development status, no ledger)."""
    from s_suite import certificate as cf
    from .release import load
    decl = cf.SuiteDecl("n-motionsense-suite12", SUITE12, "accuracy", 0.05, 2000, 0, 0.02, 24)
    for r in rows:
        if not r["has_cert"]:
            continue
        rel = load(os.path.join(os.path.expanduser(rel_dir), f"{r['digest']}.npz"))
        z = np.load(os.path.join(os.path.expanduser(pred_dir), f"{r['digest']}.npz"))
        va, te = rel["split"] == 1, rel["split"] == 2
        val = cf.SplitPreds(rel["y"][va], rel["unit"][va], {a: z[f"val|{a}"] for a in SUITE12})
        test = cf.SplitPreds(rel["y"][te], rel["unit"][te], {a: z[f"test|{a}"] for a in SUITE12})
        c = cf.certify(decl, val, test, split_status="development")
        v = c["verdict"]
        r["v1_label"] = "LEAK" if v["label"] == "LEAK" else ("BELOW_MARGIN" if v["below_margin_bonferroni"] else "NOT_RESOLVED")
        r["v1_sel_test"] = c["selected"]["test"]


def analyse(manifest_path, sealed_dir, certs_dir, rel_dir=None, pred_dir=None):
    man, rows = load_rows(manifest_path, sealed_dir, certs_dir)
    have = [r for r in rows if r["has_cert"]]
    R = dict(n_releases=len(rows), n_certs=len(have), missing=len(rows) - len(have), evaluator_shas=sorted({r["evaluator_sha"] for r in have}),
             evaluator_frozen_match_all=all(r["frozen_match"] for r in have), all_sealed=all(r["ledger_status"] == "sealed" for r in have), k_attackers=sorted({r["k"] for r in have}))
    null0 = [r for r in have if r["kind"] == "null"]
    shuf = [r for r in have if r["kind"] == "shuf"]
    plant = [r for r in have if r["kind"] == "plant"]
    pn = null0 + shuf
    k = sum(r["verdict"] == "LEAK" for r in pn)
    R["G1"] = dict(null0=rate(null0, lambda r: r["verdict"] == "LEAK"), shuf=rate(shuf, lambda r: r["verdict"] == "LEAK"), pooled=rate(pn, lambda r: r["verdict"] == "LEAK"),
                   pooled_upper1=cp_upper1(k, len(pn)), verdict_mix={v: sum(r["verdict"] == v for r in pn) for v in ("LEAK", "BELOW_MARGIN", "NOT_RESOLVED")},
                   null0_mix={v: sum(r["verdict"] == v for r in null0) for v in ("LEAK", "BELOW_MARGIN", "NOT_RESOLVED")},
                   shuf_oracle_released_mean=mean([r["omr_w"] for r in shuf]), shuf_oracle_planted_mean=mean([r["om_w"] for r in shuf]),
                   null_sel_test_mean=mean([r["sel_test"] for r in null0]), null_base_mean=mean([r["base_used"] for r in null0]))
    R["G1"]["pass"] = bool(pn) and R["G1"]["pooled"]["rate"] <= FA_RATE and R["G1"]["pooled_upper1"] <= FA_UPPER
    R["cells"], R["cells_by_base"] = {}, {}
    for f in FAMILIES:
        R["cells"][f], R["cells_by_base"][f] = {}, {}
        for L in (1, 2, 3, 4):
            rr = [r for r in plant if r["family"] == f and r["level"] == L]
            R["cells"][f][L] = cell(rr)
            for b in ("msda", "rep"):
                R["cells_by_base"][f][f"{L}_{b}"] = cell([r for r in rr if r["base"] == b])
    R["shuf_cells"] = {f: {L: cell([r for r in shuf if r["family"] == f and r["level"] == L]) for L in (2, 4)} for f in FAMILIES}
    R["primary"] = {}
    for f in FAMILIES:
        rr = [r for r in plant if r["family"] == f and r["level"] in PRIMARY_LEVELS]
        c = cell(rr)
        c["status"] = status(c)
        c["by_base"] = {b: cell([r for r in rr if r["base"] == b]) for b in ("msda", "rep")}
        c["group_best"] = {g: mean([r["group_best"][g] for r in rr]) for g in GROUPS}
        c["base12_best"] = mean([r["base12_best"] for r in rr])
        R["primary"][f] = c
    R["G2"] = dict(per_family={f: R["primary"][f]["status"] for f in CONTROLS}, pass_=all(R["primary"][f]["status"] == "DETECTED" for f in CONTROLS))
    R["G2"]["pass"] = R["G2"]["pass_"]
    R["mdd"] = {}
    for f in FAMILIES:
        lv = None
        for L in (4, 3, 2, 1):
            c = R["cells"][f][L]
            if c["n"] and c["detect"]["rate"] >= DET_RATE:
                lv = L
            else:
                break
        R["mdd"][f] = dict(level=lv, om_w=(R["cells"][f][lv]["om_w"] if lv else None), om_s=(R["cells"][f][lv]["om_s"] if lv else None))
    va, vb = subverdict(R, TARGETS, False), subverdict(R, NEW, True)
    gates = R["G1"]["pass"] and R["G2"]["pass"]
    R["verdict"] = dict(gates_pass=gates, V_A_targets=va, V_B_new=vb, label_A=(va["label"] if gates else "INCONCLUSIVE"), label_B=(vb["label"] if gates else "INCONCLUSIVE"),
                        assurance_by_family={f: R["primary"][f]["assure"] for f in FAMILIES},
                        assurance_all_ok=all(R["primary"][f]["assure"]["rate"] <= ASSURE_MAX for f in FAMILIES),
                        miss_rate_primary={f: 1 - R["primary"][f]["detect"]["rate"] for f in TARGETS + NEW})
    # canary bookkeeping: among releases whose bounds were both under the margin without a LEAK, how often were canaries accepted
    elig = [r for r in have if r["canaries_run"]]
    R["canaries"] = dict(n_eligible=len(elig), n_assured=sum(r["verdict"] == "BELOW_MARGIN" for r in elig), n_withheld=sum(bool(r["withheld"]) for r in elig),
                         accept_by_group={g: rate([r for r in elig], lambda r, g=g: r["canary_accept"].get(g) is True) for g in GROUPS},
                         eligible_by_family={f: sum(1 for r in elig if r["family"] == f) for f in FAMILIES}, eligible_null=sum(1 for r in elig if r["kind"] == "null"))
    R["reported_vs_planted"] = {f: [dict(level=L, om_w=R["cells"][f][L]["om_w"], om_s=R["cells"][f][L]["om_s"], sel_test=R["cells"][f][L]["sel_test"], base=R["cells"][f][L]["base"],
                                          attain=R["cells"][f][L]["attain_window"]) for L in (1, 2, 3, 4)] for f in FAMILIES}
    if rel_dir and pred_dir:
        rescore_v1(have, rel_dir, pred_dir)
        R["v1_on_n2"] = {}
        for f in FAMILIES:
            rr = [r for r in plant if r["family"] == f and r["level"] in PRIMARY_LEVELS]
            R["v1_on_n2"][f] = dict(detect_v1=rate(rr, lambda r: r["v1_label"] == "LEAK"), assure_v1=rate(rr, lambda r: r["v1_label"] == "BELOW_MARGIN"),
                                    detect_v3=R["primary"][f]["detect"], assure_v3=R["primary"][f]["assure"], sel_v1=mean([r["v1_sel_test"] for r in rr]), sel_v3=R["primary"][f]["sel_test"])
        R["v1_on_n2"]["_null"] = dict(detect_v1=rate(pn, lambda r: r["v1_label"] == "LEAK"), assure_v1=rate(pn, lambda r: r["v1_label"] == "BELOW_MARGIN"),
                                      detect_v3=R["G1"]["pooled"], assure_v3=rate(pn, lambda r: r["verdict"] == "BELOW_MARGIN"))
    return R, rows


def write_tables(R, path):
    L = ["| family | level | n | Omega_w | Omega_s | S selected acc | baseline | detect (CP95) | BELOW_MARGIN | NOT_RESOLVED | printed MDE80 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in FAMILIES:
        for lv in (1, 2, 3, 4):
            c = R["cells"][f][lv]
            d = c["detect"]
            L.append(f"| {f} | {lv} | {c['n']} | {c['om_w']:.3f} | {c['om_s']:.3f} | {c['sel_test']:.3f} | {c['base']:.3f} | {d['k']}/{d['n']} ({d['ci95'][0]:.2f}-{d['ci95'][1]:.2f}) | "
                     f"{c['assure']['k']} | {c['unresolved']['k']} | {c['excess_mde']:.3f} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/sealed_manifest_n2/manifest.json")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest_n2")
    ap.add_argument("--certs", default="~/n2_certs")
    ap.add_argument("--out", default="~/n2_results")
    ap.add_argument("--releases", default=None)
    ap.add_argument("--preds", default=None)
    a = ap.parse_args()
    R, rows = analyse(a.manifest, a.sealed_dir, a.certs, a.releases, a.preds)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    def js(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(R, open(os.path.join(out, "results2.json"), "w"), indent=1, sort_keys=True, default=js)
    keys = ["digest", "rid", "kind", "family", "level", "base", "amp", "om_w", "om_s", "omr_w", "omr_s", "verdict", "withheld", "canaries_run", "sel_name", "sel_test", "excess", "excess_lcb",
            "excess_mde", "base_used", "ub_bonf", "ub_maxt", "naive_max", "n_units", "v1_label", "v1_sel_test"]
    with open(os.path.join(out, "per_release2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_tables(R, os.path.join(out, "tables2.md"))
    print(json.dumps(dict(verdict=R["verdict"], G1=R["G1"], G2=R["G2"], mdd=R["mdd"]), indent=1, sort_keys=True, default=js))


if __name__ == "__main__":
    main()
