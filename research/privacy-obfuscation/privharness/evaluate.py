"""Full evaluation of one obfuscator on a harness dataset (GPU).

    python -m privharness.evaluate <root> <candidate> [--seed 0] [--epochs 12] [--attackers a,b] [--tag x]

1. Learned candidates are trained adversarially (ppv.train.train_anonymizer): the obfuscator and a utility model
   play against restarted adversaries for every secret at once (a composite head, ppv.train.segments), with
   collapse guards.
2. The whole dataset is obfuscated (8-bit output, as released).
3. Utility: fresh models per utility target train on obfuscated training items and are tested on obfuscated test
   items (SPAct's protocol).
4. Secrets, per protocol in the plan: retrained attackers (fresh models on obfuscated training items, the
   headline), frozen attackers (trained on raw footage, the protocol most published numbers use), and for
   multiclass secrets with detectable people a skeleton/gait attacker.
5. Every prediction is saved per item (eval/<name>/preds.npz) so report.py can bootstrap every number over
   recordings and compute baselines; eval/<name>/result.json holds the summary.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from . import attackers as A
from . import protocols, scoring
from . import spec as S

DEFAULT_ADV = dict(epochs=10, restart_every=3, batch=8, repeats=1, utility="r2p1d", adversary=["frame_r18", "r3d"],
                   lr_f=1e-3, min_action=10.0, b_steps=1)


@torch.no_grad()
def predict(model, dl, n_rows, kind):
    """Per-row probabilities, averaged over the row's test clips."""
    from ppv.train import DEV, to_input
    model.to(DEV).eval()
    acc, cnt = None, np.zeros(n_rows)
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(batch[0])).float()
        p = (logits.softmax(1) if kind == "multiclass" else logits.sigmoid()).cpu().numpy()
        if acc is None:
            acc = np.zeros((n_rows, p.shape[1]))
        idx = batch[3].numpy()
        np.add.at(acc, idx, p)
        np.add.at(cnt, idx, 1)
    return acc / np.maximum(cnt, 1)[:, None]


def composite(spec):
    """Privacy kind and label builder for adversarial training against every secret at once."""
    secs = spec.secrets
    if len(secs) == 1:
        s = secs[0]
        return s.kind, s.n, (lambda it: it["labels"][s.name])
    kind = tuple((s.kind, s.n) for s in secs)
    return kind, sum(s.n for s in secs), (lambda it: [float(v) for s in secs for v in
                                                      ([it["labels"][s.name]] if s.kind == "multiclass" else it["labels"][s.name])])


def evaluate(root, cand_name, seed=0, epochs=12, attackers=None, tag="", log=print, keep_frames=False):
    from ppv import anonymizers, models, train
    from ppv.clips import LazyFrames
    spec = S.load(root)
    plan = json.load(open(os.path.join(root, "plan.json")))
    cand = plan["candidates"][cand_name]
    name = cand_name + (f"_{tag}" if tag else "") + (f"_s{seed}" if seed else "") + (f"_e{epochs}" if epochs != 12 else "")
    out = os.path.join(root, "eval", name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    video = spec.modality == "video"
    stride = 2 if video and (spec.fps or 25) >= 20 else 1
    T = 16 if video else 1
    frames = LazyFrames(S.frames_index(spec, root))
    prim = spec.primary
    res = dict(name=name, candidate=cand, seed=seed, epochs=epochs, dataset=spec.name, utility={}, secrets={})

    # 1. obfuscator
    f = anonymizers.build(cand["config"], in_ch=spec.channels)
    if cand["learned"]:
        kind, n_priv, lab = composite(spec)
        tr_items = [it for it in spec.split("train") if prim.name in it["labels"] and all(s.name in it["labels"] for s in spec.secrets)]
        rows = [dict(video=it["frames"], start=it["start"], end=it["end"], action_id=it["labels"][prim.name], priv=lab(it))
                for it in tr_items]
        adv = {**DEFAULT_ADV, **cand.get("adv", {})}
        if not video:
            adv.update(utility="frame_r50", adversary=["frame_r18", "frame_dino_probe"])
        train.ACTION_KIND = prim.kind
        dl = train.loader(rows, frames, True, adv["batch"], repeats=adv["repeats"], seed=seed, stride=stride)
        dl.dataset.T = T
        f, hist = train.train_anonymizer(f, dl, adv, prim.n, n_priv, priv_kind=kind, log=log)
        res["adv_history"] = hist
        torch.save({k: v for k, v in f.state_dict().items() if not anonymizers.frozen_key(k)}, os.path.join(out, "anonymizer.pt"))

    # 2. obfuscated dataset
    tmp = None
    if cand["config"]["type"] == "raw":
        anon = frames
    else:
        nbytes = sum(np.load(p, mmap_mode="r").nbytes for p in frames.paths.values())
        tmp = os.path.join(os.path.expanduser("~/work/tmp"), f"harness_{spec.name}_{name}") if nbytes > 4e9 else None
        t1 = time.time()
        anon = train.materialize(f, frames, out_dir=tmp, batch=32)
        log(f"obfuscated {len(frames)} recordings ({time.time() - t1:.0f}s)")
    f.cpu()
    del f
    torch.cuda.empty_cache()

    preds = {}

    def fit_predict(arch, tr_items, te_items, target, data, ep):
        kind = target.kind
        train.ACTION_KIND = kind
        rtr, rte = S.rows_for(tr_items, target), S.rows_for(te_items, target)
        m = models.build(arch, target.n)
        dl_tr = train.loader(rtr, data, True, 16, repeats=2, seed=seed, stride=stride)
        dl_tr.dataset.T = T
        train.train_classifier(m, dl_tr, "action", ep, frames_per_clip=4 if arch.startswith("frame") and video else None,
                               priv_kind=kind, log=log)
        dl_te = train.loader(rte, data, False, 32, stride=stride)
        dl_te.dataset.T = T
        return m, predict(m, dl_te, len(rte), kind), rte

    def score(target, probs, te_items, cond_metric):
        y = (np.array([it["labels"][target.name] for it in te_items]) if target.kind == "multiclass"
             else np.array([it["labels"][target.name] for it in te_items], float))
        cond = np.array([it["labels"][prim.name] for it in te_items]) if cond_metric else None
        g = [it["groups"]["source"] for it in te_items]
        point, lo, hi = scoring.bootstrap(target.kind, probs, y, cond, groups=g)
        return dict(score=point, lo=lo, hi=hi, n_test=len(y))

    # 3. utility
    for u in spec.utility:
        tr_items = [it for it in spec.split("train") if u.name in it["labels"]]
        te_items = [it for it in spec.split("test") if u.name in it["labels"]]
        res["utility"][u.name] = {}
        for arch in plan["utility_models"]:
            m, probs, _ = fit_predict(arch, tr_items, te_items, u, anon, epochs)
            del m
            preds[f"utility/{u.name}/{arch}"] = probs
            res["utility"][u.name][arch] = score(u, probs, te_items, False)
            log(f"  utility {u.name}/{arch}: {res['utility'][u.name][arch]['score']:.2f}")
        res["utility"][u.name]["test_ids"] = [it["id"] for it in te_items]

    # 4. secrets
    att = attackers or plan["attackers"]
    for s in spec.secrets:
        cond_metric = plan["metrics"][s.name] == "conditional"
        res["secrets"][s.name] = {}
        for pname in plan["protocols"][s.name]:
            p = protocols.build(spec, s, pname, seed=0)
            if "skip" in p:
                res["secrets"][s.name][pname] = dict(skip=p["skip"])
                continue
            r = dict(note=p["note"], retrained={}, frozen={}, test_ids=[it["id"] for it in p["test"]],
                     train_ids=[it["id"] for it in p["train"]])
            for arch in att:
                m, probs, _ = fit_predict(arch, p["train"], p["test"], s, anon, epochs)
                del m
                preds[f"secret/{s.name}/{pname}/retrained/{arch}"] = probs
                r["retrained"][arch] = score(s, probs, p["test"], cond_metric)
                log(f"  {s.name}/{pname}/retrained {arch}: {r['retrained'][arch]['score']:.2f}")
                if cand["config"]["type"] != "raw" and arch in plan["attackers"]:
                    cache = os.path.join(root, "cache", f"{s.name}_{pname}_{arch}_e{epochs}_s{seed}.pt")
                    mc = models.build(arch, s.n)
                    if os.path.exists(cache):
                        mc.load_state_dict(torch.load(cache, map_location="cpu"))
                    else:
                        mc, _, _ = fit_predict(arch, p["train"], p["test"], s, frames, epochs)
                        os.makedirs(os.path.dirname(cache), exist_ok=True)
                        torch.save(mc.state_dict(), cache + f".{os.getpid()}.tmp")
                        os.replace(cache + f".{os.getpid()}.tmp", cache)
                    rte = S.rows_for(p["test"], s)
                    dl_te = train.loader(rte, anon, False, 32, stride=stride)
                    dl_te.dataset.T = T
                    probs = predict(mc, dl_te, len(rte), s.kind)
                    del mc
                    preds[f"secret/{s.name}/{pname}/frozen/{arch}"] = probs
                    r["frozen"][arch] = score(s, probs, p["test"], cond_metric)
                torch.cuda.empty_cache()
            profile = json.load(open(os.path.join(root, "profile.json")))
            if s.kind == "multiclass" and video and profile["stats"]["skeleton_rate"] >= 0.5 and not (attackers and "skeleton" not in attackers):
                t1 = time.time()
                Xa = A.keypoint_sequences(anon, p["train"], log=log)
                Xb = A.keypoint_sequences(anon, p["test"], log=log)
                ya = np.array([it["labels"][s.name] for it in p["train"]])
                probs = A.train_skeleton(Xa, ya, Xb, s.kind, s.n, seed=seed)
                preds[f"secret/{s.name}/{pname}/retrained/skeleton"] = probs
                r["retrained"]["skeleton"] = score(s, probs, p["test"], cond_metric)
                r["retrained"]["skeleton"]["detected"] = float((np.abs(Xb).sum(-1) > 0).mean())
                log(f"  {s.name}/{pname}/retrained skeleton: {r['retrained']['skeleton']['score']:.2f} "
                    f"(people detected in {r['retrained']['skeleton']['detected']:.0%} of frames, {time.time() - t1:.0f}s)")
            res["secrets"][s.name][pname] = r

    res["minutes"] = (time.time() - t0) / 60
    np.savez_compressed(os.path.join(out, "preds.npz"), **{k.replace("/", "|"): v for k, v in preds.items()})
    with open(os.path.join(out, "result.json.tmp"), "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    if tmp and not keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"done {name} ({res['minutes']:.1f} min)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--attackers", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    evaluate(a.root, a.candidate, a.seed, a.epochs, a.attackers.split(",") if a.attackers else None, a.tag)
