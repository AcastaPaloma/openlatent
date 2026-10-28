"""One experiment: (train anonymizer) -> materialise anonymised data -> attack evaluation.

Usage: python -m ppv.run <config.json>

Config keys:
    name        run name (output dir ~/work/runs/<name>)
    dataset     "kth" (split: "ilic" | "seq" | "scenario") or "ucf101_vp" (official split 1)
    anonymizer  {"type": ..., params}  (see anonymizers.py)
    adv         adversarial training settings, for learned anonymizers:
                {"epochs", "alpha", "restart_every", "lr_f", "lr_t", "lr_b", "utility", "adversary", "batch"}
    eval        {"epochs": 12, "batch": 16, "action": ["r2p1d"], "privacy": ["frame_r50", "r3d"],
                 "clean_attacker": true}
    seed

Evaluation protocols (both reported):
    strong  fresh models trained on anonymised train data, tested on anonymised test data.
            Privacy = best identity top-1 over the attackers; this is the number that matters.
    weak    models trained on raw data, tested on anonymised test data (Ilic et al. protocol).
"""

import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")  # Lambda images ship TensorFlow; keep it out of transformers

import resource

import numpy as np
import torch

from . import anonymizers, charades, hmdb51, kth, models, train, ucf101, utdmhad

WORK = os.path.expanduser("~/work")
HF_REPO = os.environ.get("PPV_HF_REPO", "KuanKuanKuan/openlatent-privacy")
DEFAULT_EVAL = dict(epochs=12, batch=16, repeats=2, action=["r2p1d"], privacy=["frame_r50", "r3d"],
                    clean_attacker=True, frames_per_clip=4)


def git_head():
    try:
        return subprocess.run(["git", "-C", os.path.dirname(__file__), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def clean_model(arch, task, n_classes, rows_tr, frames, ev, tag, priv_kind, stride, log):
    """Model trained on raw frames, cached per (dataset/split, arch, task, epochs) for the weak protocol."""
    path = os.path.join(WORK, "cache", f"{tag}_{task}_{arch}_e{ev['epochs']}.pt")
    m = models.build(arch, n_classes)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
    log(f"training clean {task} model {arch} (cached at {path})")
    dl = train.loader(rows_tr, frames, True, ev["batch"], repeats=ev["repeats"], stride=stride)
    train.train_classifier(m, dl, task, ev["epochs"], frames_per_clip=ev["frames_per_clip"] if arch.startswith("frame") else None,
                           priv_kind=priv_kind, log=log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(m.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)
    return m


def upload(out, name, log):
    try:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=out, path_in_repo=f"runs/{name}", repo_id=HF_REPO,
                              allow_patterns=["*.json", "*.pt", "*.log"])
        log(f"uploaded to hf://{HF_REPO}/runs/{name}")
    except Exception as e:  # never lose a finished run over an upload hiccup
        log(f"HF upload failed: {e!r}")


def load_dataset(cfg, log):
    """-> dict(rows_tr, rows_te, frames, n_act, n_priv, priv_kind, priv_task, tag, stride, big)."""
    ds = cfg.get("dataset", "kth")
    if ds == "kth":
        split = cfg.get("split", "ilic")
        root = os.path.join(WORK, "data", "kth")
        rows = kth.load_meta(root)
        rows_tr, rows_te = kth.split(rows, split)
        frames = kth.load_frames(root, [r["video"] for r in rows])
        log(f"kth split={split}: {len(rows_tr)} train / {len(rows_te)} test subsequences, {len(frames)} videos")
        return dict(rows_tr=rows_tr, rows_te=rows_te, frames=frames, n_act=len(kth.ACTIONS), n_priv=25,
                    priv_kind="multiclass", priv_task="person", tag=f"kth_{split}" + (f"_st{cfg['clip_stride']}" if cfg.get("clip_stride", 2) != 2 else ""), stride=cfg.get("clip_stride", 2), big=False, in_ch=1)
    if ds == "ucf101_vp":
        rows_tr, rows_te, frames = ucf101.load(os.path.join(WORK, "data", "ucf101"))
        log(f"VP-UCF101 split 1: {len(rows_tr)} train / {len(rows_te)} test videos")
        return dict(rows_tr=rows_tr, rows_te=rows_te, frames=frames, n_act=101, n_priv=len(ucf101.ATTRS),
                    priv_kind="multilabel", priv_task="attr", tag="ucf101_vp_s1", stride=cfg.get("clip_stride", 1), big=True, in_ch=3)
    if ds == "hmdb51_vp":
        rows_tr, rows_te, frames = hmdb51.load(os.path.join(WORK, "data", "hmdb51"))
        log(f"VP-HMDB51 split 1: {len(rows_tr)} train / {len(rows_te)} test videos")
        return dict(rows_tr=rows_tr, rows_te=rows_te, frames=frames, n_act=51, n_priv=len(hmdb51.ATTRS),
                    priv_kind="multilabel", priv_task="attr", tag="hmdb51_vp_s1", stride=cfg.get("clip_stride", 1), big=True, in_ch=3)
    if ds == "charades":
        rows_tr, rows_te, frames, n_subj = charades.load(os.path.join(WORK, "data", "charades"))
        log(f"Charades identity split: {len(rows_tr)} train / {len(rows_te)} test videos, {n_subj} subjects")
        return dict(rows_tr=rows_tr, rows_te=rows_te, frames=frames, n_act=charades.N_ACTIONS, n_priv=n_subj,
                    priv_kind="multiclass", priv_task="person", tag="charades_id", stride=cfg.get("clip_stride", 1), big=True,
                    in_ch=3, act_kind="multilabel")
    if ds == "utdmhad":
        rows_tr, rows_te, frames = utdmhad.load(os.path.join(WORK, "data", "utdmhad"))
        log(f"UTD-MHAD: {len(rows_tr)} train / {len(rows_te)} test videos, {utdmhad.N_SUBJECTS} subjects")
        return dict(rows_tr=rows_tr, rows_te=rows_te, frames=frames, n_act=utdmhad.N_ACTIONS, n_priv=utdmhad.N_SUBJECTS,
                    priv_kind="multiclass", priv_task="person", tag="utdmhad", stride=cfg.get("clip_stride", 1), big=False, in_ch=3)
    raise ValueError(ds)


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    name = cfg["name"]
    out = os.path.join(WORK, "runs", name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # large datasets map many files
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True
    ev = {**DEFAULT_EVAL, **cfg.get("eval", {})}
    t_start = time.time()
    log(f"run {name} @ {git_head()}: {json.dumps(cfg)}")
    D = load_dataset(cfg, log)
    train.ACTION_KIND = D.get("act_kind", "multiclass")
    rows_tr, rows_te, frames, stride = D["rows_tr"], D["rows_te"], D["frames"], D["stride"]
    if cfg.get("shuffle_train_privacy"):
        # sanity control: permute the privacy labels of the TRAIN rows only. A correct pipeline must then score at
        # chance on test, because nothing links train labels to test identities any more.
        import random
        labels = [r["priv"] for r in rows_tr]
        random.Random(cfg.get("seed", 0)).shuffle(labels)
        rows_tr = [dict(r, priv=l) for r, l in zip(rows_tr, labels)]
        log("SANITY: train privacy labels shuffled")
    assert not {r["video"] for r in rows_tr} & {r["video"] for r in rows_te}, "train/test share videos"
    pk, pt = D["priv_kind"], D["priv_task"]
    result = dict(name=name, config=cfg, git=git_head(), dataset=cfg.get("dataset", "kth"), split=cfg.get("split", "ilic") if cfg.get("dataset", "kth") == "kth" else "s1",
                  n_train=len(rows_tr), n_test=len(rows_te), privacy_metric="top1" if pk == "multiclass" else "cmap")

    f = anonymizers.build(cfg["anonymizer"], in_ch=D["in_ch"])
    if cfg.get("anonymizer_from"):
        # attack-only: reuse a trained anonymizer (its anonymizer.pt from the HF checkpoint repo), skip training
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, f"runs/{cfg['anonymizer_from']}/anonymizer.pt")
        sd = torch.load(path, map_location="cpu")
        target = f
        if isinstance(f, anonymizers.Chain) and not any(k.startswith("stages.") for k in sd):
            # a single-stage checkpoint goes into the chain's (only) learned stage; a chain checkpoint loads as is
            learned = [m for m in f.stages if getattr(m, "learned", False)]
            assert len(learned) == 1, "anonymizer_from with a chain needs exactly one learned stage"
            target = learned[0]
        missing, unexpected = target.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("vae.")] and not unexpected, (missing, unexpected)
        log(f"loaded anonymizer from {cfg['anonymizer_from']} (missing {len([k for k in missing if not k.startswith('vae.')])}, unexpected {len(unexpected)})")
    elif getattr(f, "learned", False) and cfg.get("adv"):  # a learned f without "adv" runs untrained (e.g. plain VAE reconstruction)
        adv = cfg["adv"]
        dl = train.loader(rows_tr, frames, True, adv.get("batch", 8), repeats=adv.get("repeats", 2), seed=cfg.get("seed", 0), stride=stride)
        f, hist = train.train_anonymizer(f, dl, adv, D["n_act"], D["n_priv"], priv_kind=pk, log=log)
        result["adv_history"] = hist
        torch.save({k: v for k, v in f.state_dict().items() if not k.startswith("vae.")}, os.path.join(out, "anonymizer.pt"))

    tmp_dir = None
    if cfg["anonymizer"]["type"] == "raw":
        anon = frames
    elif cfg["anonymizer"]["type"] == "bg_median":
        # scene-only control: every frame becomes the video's per-pixel temporal median, which
        # removes people who move and keeps the background. Identity read from this output comes
        # from the scene / recording session, not the person.
        anon = {v: np.repeat(np.median(frames[v], 0).astype(np.uint8)[None], len(frames[v]), 0) for v in frames}
    elif cfg["anonymizer"]["type"] == "fg_only":
        # person-only control (counterpart of bg_median): keep pixels that differ from the video's
        # temporal median by more than `thresh` (the moving person), black out the static scene
        thr = cfg["anonymizer"].get("thresh", 20)
        anon = {}
        for v in frames:
            a = frames[v]
            med = np.median(a, 0)
            anon[v] = np.where(np.abs(a.astype(np.int16) - med) > thr, a, 0).astype(np.uint8)
    else:
        t0 = time.time()
        if D["big"]:
            tmp_dir = os.path.join(WORK, "tmp", name)
        # the VAE decodes every frame at full resolution; 2 windows (68 frames) per call fit in memory
        mb = cfg.get("materialize_batch", 2 if cfg["anonymizer"]["type"] == "vae_latent" else 32)
        anon = train.materialize(f, frames, out_dir=tmp_dir, batch=mb)
        log(f"materialised anonymised dataset ({time.time() - t0:.0f}s)")
    f.cpu()
    del f
    torch.cuda.empty_cache()

    # Privacy attackers normally use the action split. priv_split "xaction" trains them on two thirds of the actions
    # and tests them on the rest (all trials/videos of those actions): identity must transfer to actions the attacker
    # never saw, so a leak that is only "how this person does this action" no longer counts.
    rows_tr_p, rows_te_p, tag_p = rows_tr, rows_te, D["tag"]
    if cfg.get("priv_split") == "xaction":
        everything = rows_tr + rows_te
        rows_tr_p = [r for r in everything if r["action_id"] % 3 != 2]
        rows_te_p = [r for r in everything if r["action_id"] % 3 == 2]
        tag_p = D["tag"] + "_xaction"
        log(f"privacy split xaction: {len(rows_tr_p)} train / {len(rows_te_p)} test rows (actions with id % 3 == 2 held out)")
    elif cfg.get("priv_split") == "xaction_heldout":
        # unseen actions AND the dataset's own held-out videos (e.g. KTH's unseen scenario): attackers train on the
        # train rows of the seen actions and are tested on the test rows of the held-out actions
        rows_tr_p = [r for r in rows_tr if r["action_id"] % 3 != 2]
        rows_te_p = [r for r in rows_te if r["action_id"] % 3 == 2]
        tag_p = D["tag"] + "_xaction_heldout"
        log(f"privacy split xaction_heldout: {len(rows_tr_p)} train / {len(rows_te_p)} test rows")
    dl_te = train.loader(rows_te, anon, False, ev["batch"] * 2, stride=stride)
    dl_te_p = train.loader(rows_te_p, anon, False, ev["batch"] * 2, stride=stride) if rows_te_p is not rows_te else dl_te
    strong, weak, details = {}, {}, {}
    for task, archs, n_cls in (("action", ev["action"], D["n_act"]), (pt, ev["privacy"], D["n_priv"])):
        is_priv = task != "action"
        rtr, rte, dte, tag = (rows_tr_p, rows_te_p, dl_te_p, tag_p) if is_priv else (rows_tr, rows_te, dl_te, D["tag"])
        for arch in archs:
            key = f"{task}/{arch}"
            m = models.build(arch, n_cls)
            dl_tr = train.loader(rtr, anon, True, ev["batch"], repeats=ev["repeats"], seed=cfg.get("seed", 0), stride=stride)
            log(f"strong eval: training {task} {arch} on anonymised data")
            train.train_classifier(m, dl_tr, task, ev["epochs"],
                                   frames_per_clip=ev["frames_per_clip"] if arch.startswith("frame") else None, priv_kind=pk, log=log)
            r = train.evaluate(m, dte, task, len(rte), priv_kind=pk)
            strong[key], details[f"strong/{key}"] = r["main"], r
            log(f"  strong {key}: {r['main']:.2f}" + (f"  (AP per attribute {[round(a, 1) for a in r['ap']]}, F1 {r['f1']:.1f})" if "ap" in r else ""))
            del m
            if ev["clean_attacker"]:
                mc = clean_model(arch, task, n_cls, rtr, frames, ev, tag, pk, stride, log)
                r = train.evaluate(mc, dte, task, len(rte), priv_kind=pk)
                weak[key], details[f"weak/{key}"] = r["main"], r
                log(f"  weak {key}: {r['main']:.2f}")
                del mc
            torch.cuda.empty_cache()

    result["strong"], result["weak"], result["details"] = strong, weak, details
    result["action_top1"] = max([v for k, v in strong.items() if k.startswith("action/")], default=None)  # None: attack-only
    result["action_metric"] = "map" if D.get("act_kind") == "multilabel" else "top1"  # Charades: mAP over 157 actions
    result["privacy_main"] = max(v for k, v in strong.items() if not k.startswith("action/"))
    if pk == "multiclass":
        result["privacy_top1"] = result["privacy_main"]
        result["chance"] = dict(action=100 / D["n_act"], person=100 / D["n_priv"])
    else:
        result["privacy_cmap"] = result["privacy_main"]
    result["minutes"] = (time.time() - t_start) / 60
    with open(os.path.join(out, "result.json.tmp"), "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    log(f"DONE {name}: action {result['action_top1'] if result['action_top1'] is None else round(result['action_top1'], 2)}  privacy {result['privacy_metric']} (strong, best attacker) "
        f"{result['privacy_main']:.2f}  ({result['minutes']:.1f} min)")
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.environ.get("PPV_MIRROR"):  # Colab workers: copy results to Drive for the controller
        dst = os.path.join(os.environ["PPV_MIRROR"], name)
        os.makedirs(dst, exist_ok=True)
        for fn in ("result.json", "run.log", "anonymizer.pt"):
            if os.path.exists(os.path.join(out, fn)):
                shutil.copy(os.path.join(out, fn), dst)
    upload(out, name, log)


if __name__ == "__main__":
    main(sys.argv[1])
