"""Build results/dashboard.json: everything the live dashboard (kuant.space/privacy) renders.

    runs     every finished run: config, metrics (strong/weak, frame/video), leakage, adversarial
             training history, per-model training curves parsed from run.log, matching figures
    tree     experiment tree: phase -> method family -> runs (with live/failed runs included)
    live     per-VM running/pending jobs with their latest log line (optional, from --live JSON)
    journal  docs/journal.md as markdown (hypotheses and findings)
    figures  results/figures/*.png (served from the repo's raw URL)

Usage: python3 scripts/dashboard.py [--live live.json]
"""

import argparse
import glob
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW = "https://raw.githubusercontent.com/AcastaPaloma/openlatent/privacy-research/research/privacy-obfuscation"

PHASES = {
    "p1": "Phase 1: fixed obfuscators and controls (KTH)",
    "p2": "Phase 2: learned obfuscators (KTH)",
    "p3": "Phase 3: VP-UCF101 (5 privacy attributes)",
    "p4": "Phase 4: DINOv2 attacker (attack-only)",
    "p5": "Phase 5: sanity controls",
    "p6": "Phase 6: re-evaluations after the release fix",
}


def family(cfg):
    a = cfg["anonymizer"]
    t = a["type"]
    if t == "chain":
        return "chained"
    if t in ("bg_median", "fg_only", "person_box", "raw"):
        return {"bg_median": "controls", "fg_only": "controls", "person_box": "person removal", "raw": "raw"}[t]
    if t in ("blur", "down", "pixelate", "diff", "bdq_fixed", "hold"):
        return "fixed filters"
    return {"bdq": "BDQ (learned)", "unet3d": "3D U-Net (learned)", "vae_latent": "frozen VAE + latent edit"}.get(t, t)


CURVE = re.compile(r"\[(action|person|attr)\] epoch (\d+)/(\d+) loss ([\d.]+) train_acc ([\d.]+)")
ADV = re.compile(r"\[adv\] epoch (\d+)/(\d+) action_acc ([\d.]+) adv_acc ([\d.]+) adv_entropy ([\d.]+)/([\d.]+)")


def curves(log_path):
    """Per-model training curves from run.log: {"strong action/r2p1d": [{epoch, loss, acc}], ...}."""
    out, cur = {}, None
    if not os.path.exists(log_path):
        return out
    for line in open(log_path, errors="replace"):
        m = re.search(r"strong eval: training (\w+) (\S+) on anonymised data", line)
        if m:
            cur = f"strong {m.group(1)}/{m.group(2)}"
            continue
        m = re.search(r"training clean (\w+) model (\S+)", line)
        if m:
            cur = f"clean {m.group(1)}/{m.group(2)}"
            continue
        m = CURVE.search(line)
        if m and cur:
            out.setdefault(cur, []).append(dict(epoch=int(m.group(2)), loss=float(m.group(4)), acc=float(m.group(5))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=None)
    args = ap.parse_args()

    runs = {}
    raw_priv, floor = {}, {"kth": 4.0}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*", "result.json"))):
        r = json.load(open(p))
        if r["name"].startswith("smoke"):
            continue
        ds = r.get("dataset", "kth")
        priv = r.get("privacy_main", r.get("privacy_top1"))
        if r["config"]["anonymizer"]["type"] == "raw" and r.get("split", "ilic") in ("ilic", "s1") and r.get("seed", r["config"].get("seed", 0)) == 0:
            raw_priv[ds] = priv
            for d in r.get("details", {}).values():
                if d.get("prevalence"):
                    floor[ds] = sum(d["prevalence"]) / len(d["prevalence"])
        runs[r["name"]] = dict(
            name=r["name"], dataset=ds, split=r.get("split"), config=r["config"], family=family(r["config"]),
            phase=r["name"].split("_")[0], status="done", minutes=round(r.get("minutes", 0), 1),
            action=r["action_top1"], privacy=priv, privacy_metric=r.get("privacy_metric", "top1"),
            strong=r.get("strong", {}), weak=r.get("weak", {}),
            details={k: {kk: vv for kk, vv in v.items() if kk in ("ap", "f1", "by_action")} for k, v in r.get("details", {}).items()},
            adv_history=r.get("adv_history", []),
            curves=curves(os.path.join(os.path.dirname(p), "run.log")),
        )
    for r in runs.values():
        ds = r["dataset"]
        r["leakage"] = (round(100 * (r["privacy"] - floor[ds]) / (raw_priv[ds] - floor[ds]), 1)
                        if ds in raw_priv and ds in floor and raw_priv[ds] != floor[ds] else None)

    try:  # a partial snapshot (a VM timed out mid-write) must not block the results
        live = json.load(open(args.live)) if args.live and os.path.exists(args.live) else {}
    except json.JSONDecodeError:
        live = {}
    for host in live.get("hosts", []):
        for job in host.get("running", []):
            if job["name"] not in runs:
                c = job.get("config") or {}
                runs[job["name"]] = dict(name=job["name"], status="running", host=host["name"], last_line=job.get("last_line", ""),
                                         phase=job["name"].split("_")[0], family=family(c) if c.get("anonymizer") else "running",
                                         dataset=c.get("dataset", "kth"), config=c)

    tree = {}
    for r in runs.values():
        ph = PHASES.get(r["phase"], r["phase"])
        tree.setdefault(ph, {}).setdefault(r.get("family") or "other", []).append(r["name"])
    tree = [dict(phase=ph, families=[dict(family=f, runs=sorted(rs)) for f, rs in sorted(fams.items())])
            for ph, fams in sorted(tree.items())]

    figures = [dict(url=f"{RAW}/results/figures/{os.path.basename(f)}", name=os.path.basename(f))
               for f in sorted(glob.glob(os.path.join(ROOT, "results", "figures", "*.png")))]
    gifs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "gifs", "*", "index.json"))):
        ds = os.path.basename(os.path.dirname(p))
        idx = json.load(open(p))
        gifs[ds] = dict(clips=idx.get("clips", []),
                        runs=[dict(e, url=f"{RAW}/results/gifs/{ds}/{e['file']}") for e in idx.get("runs", [])])
    journal = open(os.path.join(ROOT, "docs", "journal.md")).read()

    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), raw_base=RAW,
               reference=dict(raw=raw_priv, floor=floor), runs=runs, tree=tree, live=live, figures=figures, gifs=gifs, journal=journal)
    path = os.path.join(ROOT, "results", "dashboard.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    print(f"wrote {path}: {len(runs)} runs, {len(figures)} figures, {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
