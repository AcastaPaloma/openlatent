"""Profiling: where do the task and each secret live in this dataset? (GPU)

For a sample of items (all test items up to `max_test`, training items up to `max_train`, recordings kept whole)
the profiler runs person and keypoint detectors, builds the cue views of views.py, embeds them, and fits a linear
probe per (target, view) under the dataset's own split. Output <root>/profile.json:

    stats         modality, size, fps, channels, items, frames per item, class balance, identities and items per
                  identity, person presence / count / area, skeleton detection rate, camera motion
    probes        {target: {view: {score, lo, hi}}} plus chance and the raw reference per target
    predictability{target: {view: (score - chance) / (raw - chance)}}  1 = the view carries as much as raw data
    implied       {secret: label-only predictability}: how much of a secret the task label alone gives away
    transfer      {secret: {view: predictability under cross_utility}} for multiclass secrets: does the secret
                  survive on activities the probe never saw (style) or not
    cues          {target: views ranked by predictability}

Probes are a screening instrument: linear on frozen features, so they under-estimate what a trained attacker
recovers. They rank cues; they never certify privacy. Certification is evaluate.py's job.
"""

import json
import os
import time

import numpy as np

from . import protocols, scoring
from . import spec as S
from .probes import probe

VIEWS_DINO = ("raw", "lowres", "scene", "person")
VIEWS_HAND = ("colour", "body", "posture", "motion")


def sample_items(spec, max_train=2500, max_test=1500, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for sp, cap in (("train", max_train), ("test", max_test)):
        its = spec.split(sp)
        if len(its) > cap:
            by_src = {}
            for it in its:
                by_src.setdefault(it["groups"]["source"], []).append(it)
            srcs = list(by_src)
            rng.shuffle(srcs)
            picked = []
            for s in srcs:
                if len(picked) >= cap:
                    break
                picked += by_src[s]
            its = picked
        out += its
    return out


def extract(spec, root, items, k=8, batch=64, log=print):
    """-> {view: (n, d) features}, per-item stats list."""
    from ppv.clips import LazyFrames
    from . import views as V
    det, dino, vid = V.Detectors(), V.Dino(), V.VideoFeat()
    frames = LazyFrames(S.frames_index(spec, root))
    feats = {v: [] for v in VIEWS_DINO + VIEWS_HAND + ("video",)}
    stats = []
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        fr = [V.sample_frames(frames[it["frames"]], it["start"], it["end"], k) for it in chunk]
        clips = [V.sample_clip(frames[it["frames"]], it["start"], it["end"]) for it in chunk] if spec.modality == "video" else fr
        dets = det(np.concatenate(fr))
        dets = [dets[j * k:(j + 1) * k] for j in range(len(chunk))]
        feats["raw"].append(dino(fr))
        feats["lowres"].append(dino([V.lowres(f) for f in fr]))
        feats["scene"].append(dino([V.mask_people(f, d) for f, d in zip(fr, dets)]))
        feats["person"].append(dino([V.crop_people(f, d) for f, d in zip(fr, dets)]))
        feats["colour"].append(np.stack([V.colour_feats(f) for f in fr]))
        feats["body"].append(np.stack([V.body_feats(d) for d in dets]))
        feats["posture"].append(np.stack([V.posture_feats(d) for d in dets]))
        feats["motion"].append(np.stack([V.motion_feats(c) for c in clips]))
        feats["video"].append(vid([c if len(c) == 16 else np.repeat(c[:1], 16, 0) for c in clips]))
        for f, d, c in zip(fr, dets, clips):
            st = V.scene_stats(f, d)
            g = c.astype(np.float32).mean(-1)
            h, w = g.shape[1:]
            border = np.concatenate([g[:, : h // 10].reshape(len(g), -1), g[:, -h // 10:].reshape(len(g), -1)], 1)
            st["camera_motion"] = float(np.abs(np.diff(border, axis=0)).mean()) if len(g) > 1 else 0.0
            stats.append(st)
        if (i // batch) % 10 == 0:
            log(f"  features {min(i + batch, len(items))}/{len(items)} ({time.time() - t0:.0f}s)")
    return {v: np.concatenate(x).astype(np.float32) for v, x in feats.items()}, stats


def labels(items, t):
    y = [it["labels"][t.name] for it in items]
    return np.array(y) if t.kind == "multiclass" else np.array(y, dtype=np.float32)


def run_probes(spec, items, feats, log=print):
    prim = spec.primary
    idx_tr = [i for i, it in enumerate(items) if it["split"] == "train"]
    idx_te = [i for i, it in enumerate(items) if it["split"] == "test"]
    tr, te = [items[i] for i in idx_tr], [items[i] for i in idx_te]
    if prim.kind == "multiclass":
        onehot = np.eye(prim.n, dtype=np.float32)
        feats = dict(feats, label=np.stack([onehot[it["labels"][prim.name]] for it in items]))
    else:
        feats = dict(feats, label=np.stack([np.array(it["labels"][prim.name], np.float32) for it in items]))
    out = dict(probes={}, chance={}, raw={}, predictability={}, implied={}, transfer={}, cues={})
    for t in spec.targets:
        lab_tr = [i for i in idx_tr if t.name in items[i]["labels"]]
        lab_te = [i for i in idx_te if t.name in items[i]["labels"]]
        y_tr, y_te = labels([items[i] for i in lab_tr], t), labels([items[i] for i in lab_te], t)
        if t.kind == "multiclass":  # only test classes the probe has seen in training
            seen = set(y_tr.tolist())
            m = np.array([y in seen for y in y_te])
            lab_te, y_te = [i for i, k in zip(lab_te, m) if k], y_te[m]
        chance = scoring.chance(t.kind, y_te, y_tr if t.kind == "multiclass" else None)
        out["chance"][t.name] = chance
        res = {}
        view_names = list(feats) if t.role == "secret" else [v for v in feats if v != "label"]
        for v in view_names:
            r = probe(feats[v][lab_tr], y_tr, feats[v][lab_te], y_te, t.kind, t.n,
                      groups_tr=[items[i]["groups"]["source"] for i in lab_tr])
            res[v] = {k: r[k] for k in ("score", "lo", "hi")}
        raw = max(res["raw"]["score"], res["video"]["score"])
        out["probes"][t.name], out["raw"][t.name] = res, raw
        pred = {v: (r["score"] - chance) / (raw - chance) if raw - chance > 1e-6 else None for v, r in res.items()}
        out["predictability"][t.name] = pred
        out["cues"][t.name] = sorted([v for v in pred if v not in ("raw", "video", "label") and pred[v] is not None],
                                     key=lambda v: -pred[v])
        if t.role == "secret":
            out["implied"][t.name] = pred.get("label")
            if t.kind == "multilabel" and prim.kind == "multiclass":  # conditional metric: beyond what the task implies
                cond_te = np.array([items[i]["labels"][prim.name] for i in lab_te])
                cres = {}
                for v in ("raw", "video", "scene", "person", "body", "posture", "colour"):
                    r = probe(feats[v][lab_tr], y_tr, feats[v][lab_te], y_te, t.kind, t.n,
                              groups_tr=[items[i]["groups"]["source"] for i in lab_tr], cond_te=cond_te)
                    cres[v] = {k: r[k] for k in ("score", "lo", "hi")}
                out.setdefault("conditional", {})[t.name] = dict(
                    chance=scoring.chance(t.kind, y_te, cond=cond_te), probes=cres)
            if t.kind == "multiclass":
                p = protocols.build(S.Spec(**{**spec.__dict__, "items": items}), t, "cross_utility")
                if "skip" not in p:
                    pos = {it["id"]: i for i, it in enumerate(items)}
                    a = [pos[it["id"]] for it in p["train"]]
                    b = [pos[it["id"]] for it in p["test"]]
                    ya, yb = labels(p["train"], t), labels(p["test"], t)
                    c = scoring.chance(t.kind, yb, ya)
                    tr_res = {}
                    for v in ("raw", "video", "colour", "scene", "person", "body", "posture", "motion"):
                        r = probe(feats[v][a], ya, feats[v][b], yb, t.kind, t.n,
                                  groups_tr=[items[i]["groups"]["source"] for i in a])
                        tr_res[v] = dict(score=r["score"], lo=r["lo"], hi=r["hi"],
                                         predictability=(r["score"] - c) / (raw - chance) if raw - chance > 1e-6 else None)
                    out["transfer"][t.name] = dict(chance=c, note=p["note"], probes=tr_res)
        log(f"  {t.name}: chance {chance:.1f}, raw {raw:.1f}, cues {[(v, round(pred[v], 2)) for v in out['cues'][t.name][:4]]}")
    return out


def dataset_stats(spec, items, stats):
    st = dict(modality=spec.modality, size=spec.size, fps=spec.fps, channels=spec.channels, n_items=len(spec.items),
              n_train=len(spec.split("train")), n_test=len(spec.split("test")), n_profiled=len(items),
              frames_per_item=float(np.median([it["end"] - it["start"] + 1 for it in spec.items])))
    for key in ("person_rate", "mean_people", "person_area", "skeleton_rate", "camera_motion"):
        st[key] = float(np.mean([s[key] for s in stats]))
    st["targets"] = {}
    for t in spec.targets:
        ys = [it["labels"][t.name] for it in spec.items if t.name in it["labels"]]
        if t.kind == "multiclass":
            c = np.bincount(ys, minlength=t.n)
            st["targets"][t.name] = dict(kind=t.kind, role=t.role, classes=t.n, min_per_class=int(c.min()),
                                         max_per_class=int(c.max()), imbalance=float(c.max() / max(1, c.min())))
            if t.role == "secret":
                per_src = {}
                for it in spec.items:
                    if t.name in it["labels"]:
                        per_src.setdefault(it["labels"][t.name], set()).add(it["groups"]["source"])
                st["targets"][t.name]["recordings_per_class"] = float(np.median([len(v) for v in per_src.values()]))
        else:
            a = np.array(ys)
            st["targets"][t.name] = dict(kind=t.kind, role=t.role, classes=t.n,
                                         positive_rate=[round(float(v), 3) for v in a.mean(0)])
    return st


def profile(root, log=print, max_train=2500, max_test=1500, seed=0):
    spec = S.load(root)
    items = sample_items(spec, max_train, max_test, seed)
    log(f"profiling {spec.name}: {len(items)} of {len(spec.items)} items")
    cache = os.path.join(root, "profile_features.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        ids = [str(x) for x in z["ids"]]
        if ids == [it["id"] for it in items]:
            feats = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
            stats = json.load(open(os.path.join(root, "profile_stats.json")))
        else:
            feats = None
    else:
        feats = None
    if feats is None:
        feats, stats = extract(spec, root, items, log=log)
        np.savez(cache, ids=np.array([it["id"] for it in items]), **{f"f_{k}": v for k, v in feats.items()})
        json.dump(stats, open(os.path.join(root, "profile_stats.json"), "w"))
    out = dict(dataset=spec.name, stats=dataset_stats(spec, items, stats))
    out.update(run_probes(spec, items, feats, log=log))
    out["generated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(out, open(os.path.join(root, "profile.json"), "w"), indent=1)
    return out
