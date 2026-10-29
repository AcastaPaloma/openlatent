"""Report: every evaluated obfuscator scored with the harness definitions (scoring.py), plus checks and a pick.

numpy only, so it runs wherever the eval/ records are (a VM or a laptop).

    python -m privharness.report <root>   ->  <root>/report.json, <root>/report.md, <root>/frontier.svg

Per candidate, secret and protocol:
    A        worst case: the highest score over every attacker (retrained, frozen, skeleton) with its 95% CI
    base     max(chance, task-label attacker) on the same test items
    raw      the raw reference's worst case under the same protocol
    leakage  (A - base) / (raw - base), privacy = 1 - clip(leakage)
Per candidate: utility retention of the primary task (best utility model), privacy = the minimum over secrets
under the headline protocol (within), PUS, f_0.5, and the worst privacy over every protocol.

Checks (pass / fail, reported, never silently dropped):
    measurable        raw footage is >= 2 points above the baseline for every secret (else the secret cannot be
                      scored and is listed as such)
    black control     all-black frames score within their CI of the baseline (else something besides the pixels
                      leaks: labels, file order, a cache)
    frozen gap        how much frozen raw-trained attackers understate the retrained ones (the literature's usual
                      protocol); large gaps are reported, not used
Recommendation: the highest PUS, with the privacy-first (highest privacy at utility >= 0.5) and utility-first
(highest utility at privacy >= 0.5) alternatives.
"""

import glob
import json
import os
import sys

import numpy as np

from . import scoring
from . import spec as S


def load_eval(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "eval", "*", "result.json"))):
        r = json.load(open(p))
        z = np.load(os.path.join(os.path.dirname(p), "preds.npz"))
        r["_preds"] = {k.replace("|", "/"): z[k] for k in z.files}
        out[r["name"]] = r
    return out


def baselines(spec, plan, raw):
    """{secret: {protocol: dict(chance, label, base)}} from the raw reference's train / test item lists."""
    items = {it["id"]: it for it in spec.items}
    prim = spec.primary
    out = {}
    for s in spec.secrets:
        cond = plan["metrics"][s.name] == "conditional"
        out[s.name] = {}
        for pname, r in raw["secrets"][s.name].items():
            if "skip" in r:
                continue
            tr = [items[i] for i in r["train_ids"]]
            te = [items[i] for i in r["test_ids"]]
            yt = lambda its: (np.array([it["labels"][s.name] for it in its]) if s.kind == "multiclass"  # noqa: E731
                              else np.array([it["labels"][s.name] for it in its], float))
            ya, yb = yt(tr), yt(te)
            ta = [it["labels"][prim.name] for it in tr]
            tb = [it["labels"][prim.name] for it in te]
            c = np.array(tb) if cond else None
            chance = scoring.chance(s.kind, yb, ya if s.kind == "multiclass" else None, cond=c)
            probs = scoring.label_attacker(ta, ya, tb, s.kind, s.n)
            label = scoring.metric(s.kind, probs, yb, c)
            out[s.name][pname] = dict(chance=chance, label=label, base=scoring.baseline(chance, label))
    return out


def worst(r):
    """Highest-scoring attacker over retrained / frozen / skeleton -> (name, dict)."""
    cands = [(f"{k}/{a}", v) for k in ("retrained", "frozen") for a, v in r.get(k, {}).items()]
    return max(cands, key=lambda kv: kv[1]["score"]) if cands else (None, None)


def score_candidate(spec, plan, res, raw, base):
    prim = spec.primary.name
    ut = {a: v for a, v in res["utility"][prim].items() if a != "test_ids"}
    ur = {a: v for a, v in raw["utility"][prim].items() if a != "test_ids"}
    ua, ub = max(ut.values(), key=lambda v: v["score"]), max(ur.values(), key=lambda v: v["score"])
    t = spec.primary
    te = {it["id"]: it for it in spec.items}
    y = [te[i]["labels"][prim] for i in res["utility"][prim]["test_ids"]]
    u_chance = scoring.chance(t.kind, np.array(y) if t.kind == "multiclass" else np.array(y, float),
                              np.array([it["labels"][prim] for it in spec.split("train") if prim in it["labels"]])
                              if t.kind == "multiclass" else None)
    U = scoring.retention(ua["score"], u_chance, ub["score"])
    out = dict(utility=dict(score=ua["score"], lo=ua["lo"], hi=ua["hi"], raw=ub["score"], chance=u_chance, retention=U,
                            retention_lo=scoring.retention(ua["lo"], u_chance, ub["score"]),
                            retention_hi=scoring.retention(ua["hi"], u_chance, ub["score"])), secrets={})
    privs_within, privs_all = [], []
    for s in spec.secrets:
        out["secrets"][s.name] = {}
        for pname, r in res["secrets"][s.name].items():
            if "skip" in r or pname not in base[s.name]:
                out["secrets"][s.name][pname] = dict(skip=r.get("skip", "no baseline"))
                continue
            b = base[s.name][pname]["base"]
            att, w = worst(r)
            _, wr = worst(raw["secrets"][s.name][pname])
            L = scoring.leakage(w["score"], b, wr["score"])
            P = scoring.privacy(L)
            row = dict(attacker=att, score=w["score"], lo=w["lo"], hi=w["hi"], base=b, raw=wr["score"],
                       chance=base[s.name][pname]["chance"], label_only=base[s.name][pname]["label"],
                       leakage=L, privacy=P,
                       privacy_lo=scoring.privacy(scoring.leakage(w["hi"], b, wr["score"])),
                       privacy_hi=scoring.privacy(scoring.leakage(w["lo"], b, wr["score"])),
                       attackers={f"{k}/{a}": round(v["score"], 2) for k in ("retrained", "frozen") for a, v in r.get(k, {}).items()})
            fr = [v["score"] for v in r.get("frozen", {}).values()]
            rt = [v["score"] for v in r.get("retrained", {}).values()]
            if fr and rt:
                row["frozen_understates_by"] = max(rt) - max(fr)
            out["secrets"][s.name][pname] = row
            if P is not None:
                privs_all.append(P)
                if pname == "within":
                    privs_within.append(P)
    out["privacy"] = min(privs_within) if privs_within else None
    out["privacy_worst_protocol"] = min(privs_all) if privs_all else None
    out["pus"] = scoring.pus(out["privacy"], U)
    out["f_lambda"] = scoring.f_lambda(out["privacy"], U)
    return out


def checks(spec, scored, base, raw):
    ch = []
    for s in spec.secrets:
        for pname, b in base[s.name].items():
            _, wr = worst(raw["secrets"][s.name][pname])
            ok = wr["score"] - b["base"] >= 2.0
            ch.append(dict(check="measurable", secret=s.name, protocol=pname, ok=ok,
                           detail=f"raw {wr['score']:.1f} vs baseline {b['base']:.1f} (chance {b['chance']:.1f}, task label alone {b['label']:.1f})"))
    if "black" in scored:
        for s in spec.secrets:
            row = scored["black"]["secrets"][s.name].get("within", {})
            if "score" in row:
                ok = row["lo"] - 1.0 <= row["base"] <= row["hi"] + 1.0 or row["score"] <= row["base"] + 1.0
                ch.append(dict(check="black control", secret=s.name, protocol="within", ok=ok,
                               detail=f"black frames {row['score']:.1f} [{row['lo']:.1f}, {row['hi']:.1f}] vs baseline {row['base']:.1f}"))
    return ch


def frontier_svg(points, path, W=520, H=420, pad=56):
    """Utility retention (x) vs privacy (y), Pareto front highlighted. Self-contained SVG, no plotting library."""
    xs = lambda u: pad + max(0, min(1, u)) * (W - 2 * pad)  # noqa: E731
    ys = lambda p: H - pad - max(0, min(1, p)) * (H - 2 * pad)  # noqa: E731
    front = set(scoring.pareto([(u, p) for _, u, p, _ in points]))
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
          f'<rect width="{W}" height="{H}" fill="white"/>',
          f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#999"/>',
          f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" stroke="#999"/>',
          f'<text x="{W / 2}" y="{H - 16}" text-anchor="middle">utility retention</text>',
          f'<text x="16" y="{H / 2}" text-anchor="middle" transform="rotate(-90 16 {H / 2})">privacy (worst attacker)</text>']
    for v in (0, 0.5, 1):
        el.append(f'<text x="{xs(v)}" y="{H - pad + 16}" text-anchor="middle" fill="#666">{v}</text>')
        el.append(f'<text x="{pad - 8}" y="{ys(v) + 4}" text-anchor="end" fill="#666">{v}</text>')
    fpts = sorted([(points[i][1], points[i][2]) for i in front])
    if fpts:
        d = " ".join(f"{xs(u):.1f},{ys(p):.1f}" for u, p in fpts)
        el.append(f'<polyline points="{d}" fill="none" stroke="#2a6fdb" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for i, (name, u, p, fam) in enumerate(points):
        col = "#2a6fdb" if i in front else "#888"
        el.append(f'<circle cx="{xs(u):.1f}" cy="{ys(p):.1f}" r="4" fill="{col}"/>')
        el.append(f'<text x="{xs(u) + 6:.1f}" y="{ys(p) - 6:.1f}" fill="#333">{name}</text>')
    el.append("</svg>")
    open(path, "w").write("\n".join(el))


def fmt(v, d=1, pct=False):
    if v is None:
        return "–"
    return f"{100 * v:.{d}f}" if pct else f"{v:.{d}f}"


def build(root):
    spec = S.load(root)
    plan = json.load(open(os.path.join(root, "plan.json")))
    profile = json.load(open(os.path.join(root, "profile.json")))
    ingest = json.load(open(os.path.join(root, "ingest_report.json"))) if os.path.exists(os.path.join(root, "ingest_report.json")) else {}
    screened = json.load(open(os.path.join(root, "screen.json"))) if os.path.exists(os.path.join(root, "screen.json")) else {}
    ev = load_eval(root)
    assert "raw" in ev, "the raw reference must be evaluated first"
    raw = ev["raw"]
    base = baselines(spec, plan, raw)
    scored = {n: score_candidate(spec, plan, r, raw, base) for n, r in ev.items()}
    for n in scored:
        scored[n]["family"] = ev[n]["candidate"]["family"]
        scored[n]["learned"] = ev[n]["candidate"]["learned"]
        scored[n]["minutes"] = ev[n]["minutes"]
    main = {n: v for n, v in scored.items() if n not in ("raw", "black") and n == ev[n]["candidate"]["name"]}
    pts = [(n, v["utility"]["retention"], v["privacy"], v["family"]) for n, v in main.items()
           if v["utility"]["retention"] is not None and v["privacy"] is not None]
    front = [pts[i][0] for i in scoring.pareto([(u, p) for _, u, p, _ in pts])]
    hv = scoring.hypervolume([(u, p) for _, u, p, _ in pts]) if pts else None
    ok = [n for n, u, p, _ in pts]
    rec = max(ok, key=lambda n: main[n]["pus"]) if ok else None
    priv_first = max([n for n in ok if main[n]["utility"]["retention"] >= 0.5], key=lambda n: main[n]["privacy"], default=None)
    util_first = max([n for n in ok if main[n]["privacy"] >= 0.5], key=lambda n: main[n]["utility"]["retention"], default=None)
    robust = {n: v for n, v in scored.items() if rec and n != rec and n.startswith(rec + "_")}
    rep = dict(dataset=spec.name, generated_from=sorted(ev), checks=checks(spec, scored, base, raw), baselines=base,
               scored={n: v for n, v in scored.items()}, frontier=front, hypervolume=hv,
               recommendation=dict(best=rec, privacy_first=priv_first, utility_first=util_first, robustness=sorted(robust),
                                   # certified = the worst privacy over the recommendation and every robustness re-run
                                   # (more attacker seeds, longer training, DINOv2): an attacker that does better on any
                                   # re-run counts, the same worst-case rule as within one run
                                   certified_privacy=min([main[rec]["privacy"]] + [v["privacy"] for v in robust.values()
                                                                                  if v["privacy"] is not None]) if rec else None,
                                   certified_utility=min([main[rec]["utility"]["retention"]] + [v["utility"]["retention"] for v in robust.values()
                                                                                                 if v["utility"]["retention"] is not None]) if rec else None),
               screen={n: {k: v for k, v in r.items() if k != "probes"} for n, r in screened.items()})
    if screened:
        common = [n for n in main if n in screened and screened[n].get("privacy") is not None]
        if len(common) >= 4:
            a = np.argsort(np.argsort([screened[n]["privacy"] for n in common]))
            b = np.argsort(np.argsort([main[n]["privacy"] for n in common]))
            rep["screen_rank_correlation"] = float(np.corrcoef(a, b)[0, 1])
    json.dump(rep, open(os.path.join(root, "report.json"), "w"), indent=1)
    if pts:
        frontier_svg(pts, os.path.join(root, "frontier.svg"))
    open(os.path.join(root, "report.md"), "w").write(markdown(spec, plan, profile, ingest, rep, main, scored))
    return rep


def markdown(spec, plan, profile, ingest, rep, main, scored):
    st = profile["stats"]
    L = [f"# Privacy report: {spec.name}", "",
         "Scores (0-100): **privacy** = 100 x (1 - leakage), leakage = (best attacker - baseline) / (raw footage - "
         "baseline), worst case over every attacker; baseline = max(chance, the secret predicted from the task label "
         "alone). **Utility** = task score retained relative to raw footage, above chance. **PUS** = harmonic mean of "
         "the two. Intervals: 95% bootstrap over test recordings. Definitions: privharness/scoring.py.", "",
         "## Dataset", "",
         f"- {st['n_items']} items ({st['n_train']} train / {st['n_test']} test), {spec.modality}, {spec.size[0]}x{spec.size[1]}, "
         f"{spec.channels} channel(s), {spec.fps or '-'} fps, median {st['frames_per_item']:.0f} frames per item",
         f"- people visible in {st['person_rate']:.0%} of sampled frames (mean {st['mean_people']:.1f}, median area "
         f"{st['person_area']:.1%}), skeletons found in {st['skeleton_rate']:.0%}, camera motion {st['camera_motion']:.1f}",
         "- targets: " + "; ".join(f"{t.name} ({t.role}, {t.kind}, {t.n} classes)" for t in spec.targets)]
    for w in ingest.get("warnings", []) + plan["warnings"]:
        L.append(f"- warning: {w}")
    L += ["", "## Where the task and the secrets live (profile)", "",
          "Linear probes on frozen features of cue-isolating views; 1.00 = the view predicts the target as well as raw "
          "footage. Probes rank cues, they do not certify privacy.", "",
          "| view | " + " | ".join(t.name for t in spec.targets) + " |", "|---|" + "---:|" * len(spec.targets)]
    for v in ("colour", "lowres", "scene", "person", "body", "posture", "motion", "label"):
        L.append(f"| {v} | " + " | ".join(fmt(profile["predictability"][t.name].get(v), 2) for t in spec.targets) + " |")
    L += ["", "## Plan", ""] + [f"- {r}" for r in plan["reasons"]]
    L += ["", "## Results", "",
          f"Headline protocol: within (the dataset's split). Attackers: {', '.join(plan['attackers'])} retrained on the "
          "obfuscated training set, the same architectures trained on raw footage (frozen), and a skeleton/gait "
          "attacker for identities. Frontier hypervolume: " + fmt(rep["hypervolume"], 3) + ".", "",
          "| obfuscator | family | utility | privacy | PUS | f_0.5 | worst-protocol privacy | worst attacker (within) |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    rows = sorted(main.items(), key=lambda kv: -(kv[1]["pus"] or -1))
    for n in ["raw", "black"] + [n for n, _ in rows]:
        v = scored.get(n)
        if v is None:
            continue
        u = v["utility"]
        w = [f"{s}: {r['attacker']} {r['score']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]" for s, sr in v["secrets"].items()
             for p, r in sr.items() if p == "within" and "score" in r]
        star = " **(front)**" if n in rep["frontier"] else ""
        L.append(f"| {n}{star} | {v['family']} | {fmt(u['retention'], 1, True)} [{fmt(u['retention_lo'], 0, True)}, "
                 f"{fmt(u['retention_hi'], 0, True)}] | {fmt(v['privacy'], 1, True)} | {fmt(v['pus'])} | {fmt(v['f_lambda'])} | "
                 f"{fmt(v['privacy_worst_protocol'], 1, True)} | {'; '.join(w)} |")
    L += ["", "### Per secret and protocol", ""]
    for s in spec.secrets:
        L += [f"**{s.name}** ({plan['metrics'][s.name]})", "",
              "| obfuscator | protocol | worst attacker | score [95% CI] | baseline (chance / task label) | raw | privacy |",
              "|---|---|---|---:|---:|---:|---:|"]
        for n in ["raw", "black"] + [n for n, _ in rows]:
            v = scored.get(n)
            if v is None:
                continue
            for p, r in v["secrets"][s.name].items():
                if "score" not in r:
                    L.append(f"| {n} | {p} | skipped: {r.get('skip')} | | | | |")
                    continue
                L.append(f"| {n} | {p} | {r['attacker']} | {r['score']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] | "
                         f"{r['base']:.1f} ({r['chance']:.1f} / {r['label_only']:.1f}) | {r['raw']:.1f} | {fmt(r['privacy'], 1, True)} |")
        L.append("")
    L += ["## Checks", ""] + [f"- {'PASS' if c['ok'] else 'FAIL'} {c['check']} ({c['secret']}, {c['protocol']}): {c['detail']}" for c in rep["checks"]]
    gaps = [(n, s, r["frozen_understates_by"]) for n, v in main.items() for s, sr in v["secrets"].items()
            for p, r in sr.items() if p == "within" and "frozen_understates_by" in r]
    if gaps:
        med = float(np.median([g for _, _, g in gaps]))
        L.append(f"- frozen raw-trained attackers understate the retrained ones by a median {med:.1f} points "
                 f"(max {max(g for _, _, g in gaps):.1f}, {max(gaps, key=lambda x: x[2])[0]})")
    if "screen_rank_correlation" in rep:
        L.append(f"- screening vs full evaluation, privacy rank correlation: {rep['screen_rank_correlation']:.2f}")
    r = rep["recommendation"]
    L += ["", "## Recommendation", ""]
    if r["best"]:
        b = main[r["best"]]
        L.append(f"- **{r['best']}** ({b['family']}): utility {fmt(b['utility']['retention'], 1, True)}, privacy "
                 f"{fmt(b['privacy'], 1, True)}, PUS {fmt(b['pus'])}")
        L.append(f"- privacy first (utility >= 50): {r['privacy_first']}; utility first (privacy >= 50): {r['utility_first']}")
        if r["robustness"]:
            cp, cu = r["certified_privacy"], r["certified_utility"]
            L.append(f"- **certified** (worst over the recommendation and its {len(r['robustness'])} robustness re-runs): "
                     f"privacy {fmt(cp, 1, True)}, utility {fmt(cu, 1, True)}, PUS {fmt(scoring.pus(cp, cu))}")
        for n in r["robustness"]:
            v = scored[n]
            L.append(f"- robustness {n}: privacy {fmt(v['privacy'], 1, True)}, utility {fmt(v['utility']['retention'], 1, True)}")
        L.append("- config: `" + json.dumps(plan["candidates"][r["best"]]["config"]) + "`")
    else:
        L.append("- no candidate has both a utility and a privacy score")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rep = build(sys.argv[1])
    print(json.dumps(rep["recommendation"], indent=1))
