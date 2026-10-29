"""Screening: a cheap first pass over every feasible fixed obfuscator (GPU).

Each candidate obfuscates one 16-frame clip per profiled item; two frozen-feature probes attack the output (DINOv2
on 8 of its frames: appearance; Kinetics R(2+1)D on the clip: appearance + motion), and the same probes predict the
task. Proxy scores use the harness definitions (scoring.py) against the raw candidate screened the same way.

Shortlist for full evaluation (at most `budget`): the proxy Pareto front of (utility, privacy), then the highest
proxy trade-off scores, then the planner's priorities; the raw reference and the black control always.
Linear probes under-estimate trained attackers, so screening only orders candidates; it certifies nothing.
"""

import json
import os
import time

import numpy as np
import torch

from . import scoring
from . import spec as S
from .probes import probe
from .profile import labels, sample_items

DEV = "cuda"


def obfuscated_features(spec, root, items, config, batch=32, log=print):
    from ppv import anonymizers
    from ppv.clips import LazyFrames
    from . import views as V
    f = anonymizers.build(config, in_ch=spec.channels).to(DEV).eval()
    dino, vid = V.Dino(), V.VideoFeat()
    frames = LazyFrames(S.frames_index(spec, root))
    A, B = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        clips = [V.sample_clip(frames[it["frames"]], it["start"], it["end"]) for it in chunk]
        x = torch.from_numpy(np.stack(clips)).to(DEV).permute(0, 4, 1, 2, 3).float() / 255  # (b, 3, t, h, w)
        if spec.channels == 1:
            x = x[:, :1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = f(x).float().clamp(0, 1)
        y = (y * 255).round().byte()
        y = y.expand(-1, 3, -1, -1, -1) if y.shape[1] == 1 else y
        out = y.permute(0, 2, 3, 4, 1).cpu().numpy()  # (b, t, h, w, 3)
        idx = np.linspace(0, out.shape[1] - 1, 8).round().astype(int)
        A.append(dino([o[idx] for o in out]))
        B.append(vid(list(out)))
    del f
    torch.cuda.empty_cache()
    return np.concatenate(A), np.concatenate(B)


def probe_all(spec, plan, items, feats):
    """-> {target: dict(score, lo, hi, chance)} with the best of the two probes per target."""
    prim = spec.primary
    idx_tr = [i for i, it in enumerate(items) if it["split"] == "train"]
    idx_te = [i for i, it in enumerate(items) if it["split"] == "test"]
    out = {}
    for t in spec.targets:
        a = [i for i in idx_tr if t.name in items[i]["labels"]]
        b = [i for i in idx_te if t.name in items[i]["labels"]]
        ya, yb = labels([items[i] for i in a], t), labels([items[i] for i in b], t)
        if t.kind == "multiclass":
            seen = set(ya.tolist())
            keep = np.array([y in seen for y in yb])
            b, yb = [i for i, k in zip(b, keep) if k], yb[keep]
        cond = None
        if plan["metrics"].get(t.name) == "conditional":
            cond = np.array([items[i]["labels"][prim.name] for i in b])
        best = None
        for X in feats:
            r = probe(X[a], ya, X[b], yb, t.kind, t.n, groups_tr=[items[i]["groups"]["source"] for i in a], cond_te=cond)
            if best is None or r["score"] > best["score"]:
                best = r
        out[t.name] = dict(score=best["score"], lo=best["lo"], hi=best["hi"],
                           chance=scoring.chance(t.kind, yb, ya if t.kind == "multiclass" else None, cond=cond))
    return out


def proxy_scores(spec, res, raw):
    prim = spec.primary.name
    u = scoring.retention(res[prim]["score"], res[prim]["chance"], raw[prim]["score"])
    ps = [scoring.privacy(scoring.leakage(res[s.name]["score"], res[s.name]["chance"], raw[s.name]["score"])) for s in spec.secrets]
    ps = [p for p in ps if p is not None]
    p = min(ps) if ps else None
    return dict(utility=u, privacy=p, pus=scoring.pus(p, u))


def screen(root, plan, log=print, seed=0):
    spec = S.load(root)
    items = sample_items(spec, max_train=1500, max_test=800, seed=seed)
    out_path = os.path.join(root, "screen.json")
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    names = ["raw"] + [n for n in plan["screen"] if n != "raw"]
    for n in names:
        if n in done:
            continue
        t0 = time.time()
        feats = obfuscated_features(spec, root, items, plan["candidates"][n]["config"], log=log)
        done[n] = dict(probes=probe_all(spec, plan, items, feats), minutes=(time.time() - t0) / 60)
        if n != "raw":
            done[n].update(proxy_scores(spec, done[n]["probes"], done["raw"]["probes"]))
        log(f"  screened {n}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in done[n].items() if k != 'probes'})}")
        json.dump(done, open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)
    return done


def shortlist(plan, screened, budget=None):
    budget = budget or plan["shortlist_size"]
    pts = {n: (r["utility"], r["privacy"]) for n, r in screened.items()
           if n not in ("raw", "black") and r.get("utility") is not None and r.get("privacy") is not None}
    names = list(pts)
    front = [names[i] for i in scoring.pareto([pts[n] for n in names])]
    by_pus = sorted(names, key=lambda n: -(screened[n].get("pus") or 0))
    by_prio = [n for n in plan["priority"] if n in pts]
    picked = []
    for n in sorted(front, key=lambda n: -(screened[n].get("pus") or 0)) + by_pus + by_prio:
        if n not in picked and len(picked) < budget:
            picked.append(n)
    return ["raw", "black"] + picked + plan["learned"]
