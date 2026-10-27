"""One experiment: (train anonymizer) -> materialise anonymised data -> attack evaluation.

Usage: python -m ppv.run <config.json>

Config keys:
    name        run name (output dir ~/work/runs/<name>)
    dataset     "kth";  split: "seq" | "scenario"
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

import torch

from . import anonymizers, kth, models, train

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


def clean_model(arch, task, n_classes, rows_tr, frames, ev, split, log):
    """Model trained on raw frames, cached per (split, arch, task, epochs) for the weak protocol."""
    path = os.path.join(WORK, "cache", f"kth_{split}_{task}_{arch}_e{ev['epochs']}.pt")
    m = models.build(arch, n_classes)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
    log(f"training clean {task} model {arch} (cached at {path})")
    dl = train.loader(rows_tr, frames, True, ev["batch"], repeats=ev["repeats"])
    train.train_classifier(m, dl, task, ev["epochs"], frames_per_clip=ev["frames_per_clip"] if arch.startswith("frame") else None, log=log)
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

    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True
    ev = {**DEFAULT_EVAL, **cfg.get("eval", {})}
    split = cfg.get("split", "ilic")
    t_start = time.time()
    log(f"run {name} @ {git_head()}: {json.dumps(cfg)}")

    root = os.path.join(WORK, "data", "kth")
    rows = kth.load_meta(root)
    rows_tr, rows_te = kth.split(rows, split)
    frames = kth.load_frames(root, [r["video"] for r in rows])
    n_act, n_per = len(kth.ACTIONS), 25
    log(f"kth split={split}: {len(rows_tr)} train / {len(rows_te)} test subsequences, {len(frames)} videos")

    result = dict(name=name, config=cfg, git=git_head(), split=split, n_train=len(rows_tr), n_test=len(rows_te))

    f = anonymizers.build(cfg["anonymizer"])
    if getattr(f, "learned", False):
        adv = cfg["adv"]
        dl = train.loader(rows_tr, frames, True, adv.get("batch", 8), repeats=adv.get("repeats", 2), seed=cfg.get("seed", 0))
        f, hist = train.train_anonymizer(f, dl, adv, n_act, n_per, log=log)
        result["adv_history"] = hist
        torch.save({k: v for k, v in f.state_dict().items() if not k.startswith("vae.")}, os.path.join(out, "anonymizer.pt"))

    if cfg["anonymizer"]["type"] == "raw":
        anon = frames
    else:
        t0 = time.time()
        anon = train.materialize(f, frames)
        log(f"materialised anonymised dataset ({time.time() - t0:.0f}s)")
    f.cpu()
    del f
    torch.cuda.empty_cache()

    dl_te = train.loader(rows_te, anon, False, ev["batch"] * 2)
    strong, weak = {}, {}
    for task, archs, n_cls in (("action", ev["action"], n_act), ("person", ev["privacy"], n_per)):
        for arch in archs:
            m = models.build(arch, n_cls)
            dl_tr = train.loader(rows_tr, anon, True, ev["batch"], repeats=ev["repeats"], seed=cfg.get("seed", 0))
            log(f"strong eval: training {task} {arch} on anonymised data")
            train.train_classifier(m, dl_tr, task, ev["epochs"],
                                   frames_per_clip=ev["frames_per_clip"] if arch.startswith("frame") else None, log=log)
            strong[f"{task}/{arch}"] = train.evaluate(m, dl_te, task, len(rows_te))
            log(f"  strong {task}/{arch}: {strong[f'{task}/{arch}']:.2f}")
            del m
            if ev["clean_attacker"]:
                mc = clean_model(arch, task, n_cls, rows_tr, frames, ev, split, log)
                weak[f"{task}/{arch}"] = train.evaluate(mc, dl_te, task, len(rows_te))
                log(f"  weak {task}/{arch}: {weak[f'{task}/{arch}']:.2f}")
                del mc
            torch.cuda.empty_cache()

    result["strong"], result["weak"] = strong, weak
    result["action_top1"] = max(v for k, v in strong.items() if k.startswith("action/"))
    result["privacy_top1"] = max(v for k, v in strong.items() if k.startswith("person/"))
    result["chance"] = dict(action=100 / n_act, person=100 / n_per)
    result["minutes"] = (time.time() - t_start) / 60
    with open(os.path.join(out, "result.json.tmp"), "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(os.path.join(out, "result.json.tmp"), os.path.join(out, "result.json"))
    log(f"DONE {name}: action {result['action_top1']:.2f}  privacy (strong, best attacker) {result['privacy_top1']:.2f}  "
        f"({result['minutes']:.1f} min)")
    if os.environ.get("PPV_MIRROR"):  # Colab workers: copy results to Drive for the controller
        dst = os.path.join(os.environ["PPV_MIRROR"], name)
        os.makedirs(dst, exist_ok=True)
        for fn in ("result.json", "run.log", "anonymizer.pt"):
            if os.path.exists(os.path.join(out, fn)):
                shutil.copy(os.path.join(out, fn), dst)
    upload(out, name, log)


if __name__ == "__main__":
    main(sys.argv[1])
