"""UNSEAL and analyse: join the sealed manifest, the sealed oracle report and S's certificates; apply the pre-registered criteria of N.md section 5.

    python -m n_control.analyse --manifest ~/sealed_manifest/manifest.json --sealed-dir ~/sealed_manifest --certs ~/n_certs --out ~/n_results

Writes results.json (every number below), per_release.csv (one row per release, digest, family, level, base, planted rates, certificate fields) and
tables.md. Nothing here is tuned after the fact: the thresholds are constants at the top and are the ones committed in N.md before any certificate existed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import beta

# ---- pre-registered constants (N.md section 5)
PRIMARY_LEVELS = (4, 5)
DET_RATE, DET_LOWER = 0.80, 0.60          # DETECTED: pooled primary detection >= 0.80 and two-sided 95% CP lower bound >= 0.60
MISS_RATE = 0.50                           # MISSED: pooled primary detection < 0.50
FA_RATE, FA_UPPER = 0.05, 0.10             # null: LEAK rate <= 0.05 and one-sided 95% CP upper bound <= 0.10
ASSURE_MAX = 0.10                          # unseen family: BELOW_MARGIN rate at the primary levels must be <= 0.10 for a YES
SEEN = ("bias", "specin")
UNSEEN = ("xphase", "ampfreq", "stealth")
REGIME = ("pooled",)
FAMILIES = ("bias", "specin", "xphase", "ampfreq", "stealth", "pooled")


def cp(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def cp_upper1(k, n, a=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


def rate(rows, key):
    n = len(rows)
    k = sum(1 for r in rows if key(r))
    lo, hi = cp(k, n)
    return dict(k=k, n=n, rate=(k / n if n else float("nan")), ci95=[lo, hi])


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
            sel, sb = c["selected"], c["suite_bound"]
            r.update(verdict=c["verdict"]["label"], sel_name=sel["name"], sel_test=sel["test"], sel_se=sel["se"], excess=sel["excess_over_baseline"],
                     excess_lcb=sel["excess_lcb_onesided"], excess_mde=sel["excess_mde80"], base_used=c["baseline_used"], ub_bonf=sb["bonferroni"],
                     ub_maxt=sb["maxt"], naive_max=c["naive_max"]["test"], n_units=c["data"]["n_test_units"], floor_k1=c["floor"]["ucb_at_baseline_k1"],
                     floor_suite=c["floor"]["ucb_at_baseline_suite"], ledger_status=c["split_status"], frozen_match=c["evaluator"]["frozen_match"],
                     per_attacker={a: v["test"] for a, v in c["attackers"].items()})
        rows.append(r)
    return man, rows


def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def cell(rows):
    n = len(rows)
    out = dict(n=n, detect=rate(rows, lambda r: r["verdict"] == "LEAK"), assure=rate(rows, lambda r: r["verdict"] == "BELOW_MARGIN"),
               unresolved=rate(rows, lambda r: r["verdict"] == "NOT_RESOLVED"),
               om_w=mean([r.get("om_w") for r in rows]), om_s=mean([r.get("om_s") for r in rows]), sel_test=mean([r["sel_test"] for r in rows]),
               base=mean([r["base_used"] for r in rows]), naive_max=mean([r["naive_max"] for r in rows]), excess_mde=mean([r["excess_mde"] for r in rows]),
               ub_bonf=mean([r["ub_bonf"] for r in rows]))
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


def analyse(manifest_path, sealed_dir, certs_dir):
    man, rows = load_rows(manifest_path, sealed_dir, certs_dir)
    have = [r for r in rows if r["has_cert"]]
    R = dict(n_releases=len(rows), n_certs=len(have), missing=len(rows) - len(have),
             evaluator_frozen_match_all=all(r["frozen_match"] for r in have), all_sealed=all(r["ledger_status"] == "sealed" for r in have))
    null0 = [r for r in have if r["kind"] == "null"]
    shuf = [r for r in have if r["kind"] == "shuf"]
    plant = [r for r in have if r["kind"] == "plant"]
    # ---- validity gate G1: false alarms
    pooled_null = null0 + shuf
    k = sum(r["verdict"] == "LEAK" for r in pooled_null)
    R["G1"] = dict(null0=rate(null0, lambda r: r["verdict"] == "LEAK"), shuf=rate(shuf, lambda r: r["verdict"] == "LEAK"),
                   pooled=rate(pooled_null, lambda r: r["verdict"] == "LEAK"), pooled_upper1=cp_upper1(k, len(pooled_null)),
                   verdict_mix_null=dict(zip(*np.unique([r["verdict"] for r in pooled_null], return_counts=True))) if pooled_null else {},
                   shuf_oracle_released_mean=mean([r["omr_w"] for r in shuf]), shuf_oracle_planted_mean=mean([r["om_w"] for r in shuf]),
                   null_sel_test_mean=mean([r["sel_test"] for r in null0]), null_base_mean=mean([r["base_used"] for r in null0]))
    R["G1"]["pass"] = bool(pooled_null) and R["G1"]["pooled"]["rate"] <= FA_RATE and R["G1"]["pooled_upper1"] <= FA_UPPER
    # ---- per family x level and base
    R["cells"], R["cells_by_base"] = {}, {}
    for f in FAMILIES:
        R["cells"][f], R["cells_by_base"][f] = {}, {}
        for L in (1, 2, 3, 4, 5):
            rr = [r for r in plant if r["family"] == f and r["level"] == L]
            R["cells"][f][L] = cell(rr)
            for b in ("msda", "rep"):
                R["cells_by_base"][f][f"{L}_{b}"] = cell([r for r in rr if r["base"] == b])
    # shuffled twins per family (detection must collapse)
    R["shuf_cells"] = {f: {L: cell([r for r in shuf if r["family"] == f and r["level"] == L]) for L in (2, 4, 5)} for f in FAMILIES}
    # ---- primary pooled per family
    R["primary"] = {}
    for f in FAMILIES:
        rr = [r for r in plant if r["family"] == f and r["level"] in PRIMARY_LEVELS]
        c = cell(rr)
        c["status"] = status(c)
        c["by_base"] = {b: cell([r for r in rr if r["base"] == b]) for b in ("msda", "rep")}
        R["primary"][f] = c
    # ---- G2 positive controls
    R["G2"] = dict(per_family={f: R["primary"][f]["status"] for f in SEEN})
    R["G2"]["pass"] = all(R["primary"][f]["status"] == "DETECTED" for f in SEEN)
    # ---- MDD per family: lowest level L such that every level >= L has detection rate >= 0.80
    R["mdd"] = {}
    for f in FAMILIES:
        lv = None
        for L in (5, 4, 3, 2, 1):
            c = R["cells"][f][L]
            if c["n"] and c["detect"]["rate"] >= DET_RATE:
                lv = L
            else:
                break
        R["mdd"][f] = dict(level=lv, om_w=(R["cells"][f][lv]["om_w"] if lv else None), om_s=(R["cells"][f][lv]["om_s"] if lv else None),
                           note="lowest tested level from which every higher level also has detection >= 0.80 (point estimates, n per cell in cells)")
    # ---- verdict on the unseen families
    st = {f: R["primary"][f]["status"] for f in UNSEEN}
    n_det = sum(v == "DETECTED" for v in st.values())
    n_miss = sum(v == "MISSED" for v in st.values())
    assure_ok = all(R["primary"][f]["assure"]["rate"] <= ASSURE_MAX for f in UNSEEN)
    if not (R["G1"]["pass"] and R["G2"]["pass"]):
        v = "INCONCLUSIVE"
    elif n_det >= 2 and assure_ok:
        v = "YES"
    elif n_miss >= 2:
        v = "NO"
    else:
        v = "MIXED"
    R["verdict"] = dict(unseen_status=st, n_detected=n_det, n_missed=n_miss, assurance_ok=assure_ok, label=v,
                        false_assurance_rate_primary={f: R["primary"][f]["assure"] for f in FAMILIES},
                        unseen_miss_rate_primary={f: 1 - R["primary"][f]["detect"]["rate"] for f in UNSEEN},
                        pooled_regime_status=R["primary"]["pooled"]["status"])
    # ---- reported vs planted: per family mean selected accuracy against mean planted window rate, by level
    R["reported_vs_planted"] = {f: [dict(level=L, om_w=R["cells"][f][L]["om_w"], om_s=R["cells"][f][L]["om_s"], sel_test=R["cells"][f][L]["sel_test"],
                                          base=R["cells"][f][L]["base"], attain=R["cells"][f][L]["attain_window"]) for L in (1, 2, 3, 4, 5)] for f in FAMILIES}
    # ---- which S attackers carry the signal at the top level
    top = {}
    for f in FAMILIES:
        rr = [r for r in plant if r["family"] == f and r["level"] == 5]
        if rr:
            names = sorted(rr[0]["per_attacker"])
            top[f] = {a: mean([r["per_attacker"][a] for r in rr]) for a in names}
    R["attackers_level5"] = top
    return R, rows


def write_tables(R, path):
    L = []
    L.append("| family | level | n | Omega_w | Omega_s | S selected acc | baseline | detect (CP95) | BELOW_MARGIN | NOT_RESOLVED | printed MDE80 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in FAMILIES:
        for lv in (1, 2, 3, 4, 5):
            c = R["cells"][f][lv]
            d = c["detect"]
            L.append(f"| {f} | {lv} | {c['n']} | {c['om_w']:.3f} | {c['om_s']:.3f} | {c['sel_test']:.3f} | {c['base']:.3f} | {d['k']}/{d['n']} ({d['ci95'][0]:.2f}-{d['ci95'][1]:.2f}) | "
                     f"{c['assure']['k']} | {c['unresolved']['k']} | {c['excess_mde']:.3f} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/sealed_manifest/manifest.json")
    ap.add_argument("--sealed-dir", default="~/sealed_manifest")
    ap.add_argument("--certs", default="~/n_certs")
    ap.add_argument("--out", default="~/n_results")
    a = ap.parse_args()
    R, rows = analyse(a.manifest, a.sealed_dir, a.certs)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    def js(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, sort_keys=True, default=js)
    keys = ["digest", "rid", "kind", "family", "level", "base", "amp", "om_w", "om_s", "omr_w", "omr_s", "verdict", "sel_name", "sel_test", "sel_se", "excess",
            "excess_lcb", "excess_mde", "base_used", "ub_bonf", "ub_maxt", "naive_max", "n_units"]
    with open(os.path.join(out, "per_release.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_tables(R, os.path.join(out, "tables.md"))
    print(json.dumps(dict(verdict=R["verdict"], G1=R["G1"], G2=R["G2"], mdd=R["mdd"]), indent=1, sort_keys=True, default=js))


if __name__ == "__main__":
    main()
