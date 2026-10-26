"""Crash-safe epoch loop for long training runs on preemptible VMs.

A Colab VM can disappear mid-run. This module makes each run resumable at epoch
granularity:

- Every checkpoint is written to a temporary file, fsync'd, then renamed over the
  real path. A crash mid-write leaves the previous checkpoint intact.
- After every epoch, `last.pt` holds model, optimizer, epoch counter, best score,
  history and RNG state (python, numpy, torch, CUDA). Restarting resumes from the
  next epoch and reproduces the uninterrupted run on CPU (see tests).
- `best.pt` keeps the upstream format ({"epoch", "state_dict", "optimizer"}) so the
  upstream eval scripts can load it unchanged.
- A finished run writes `DONE.json`; re-running it is a no-op.

Resume is at epoch boundaries only, so a crash loses at most one epoch of work.
"""

import json
import math
import os
import random

import numpy as np
import torch

LAST = "last.pt"
BEST = "best.pt"
DONE = "DONE.json"


def atomic_torch_save(obj, path):
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json_save(obj, path):
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_best(run_dir, model, optimizer, epoch):
    atomic_torch_save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_dir, BEST),
    )


def save_last(run_dir, model, optimizer, epoch, best_top1, history):
    atomic_torch_save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "history": history,
            "rng": capture_rng_state(),
        },
        os.path.join(run_dir, LAST),
    )


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, DONE))


def mark_done(run_dir, payload):
    atomic_json_save(payload, os.path.join(run_dir, DONE))


def run_epochs(
    *,
    run_dir,
    model,
    optimizer,
    n_epochs,
    train_epoch,
    eval_epoch=None,
    eval_every=5,
    log=print,
):
    """Run epochs [resume_point, n_epochs) with checkpoints after each one.

    train_epoch(epoch) -> dict of training metrics (must include "loss" if desired).
    eval_epoch(epoch) -> top-1 accuracy (float, percent). Called every `eval_every`
    epochs and on the last epoch, as upstream does. Best-scoring weights go to best.pt.

    Returns (best_top1, history). Skips work entirely if DONE.json exists.
    """
    if is_done(run_dir):
        log(f"[resume] {run_dir} already DONE; nothing to do")
        return None, None

    last_path = os.path.join(run_dir, LAST)
    start_epoch, best_top1, history = 0, -math.inf, []

    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        restore_rng_state(ckpt["rng"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt["best_top1"]
        history = ckpt["history"]
        log(f"[resume] restored {last_path}: continuing at epoch {start_epoch}")

    for epoch in range(start_epoch, n_epochs):
        train_metrics = dict(train_epoch(epoch) or {})
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if eval_epoch is not None and (epoch % eval_every == 0 or epoch == n_epochs - 1):
            top1 = float(eval_epoch(epoch))
            record["test_top1"] = top1
            if top1 > best_top1:
                best_top1 = top1
                save_best(run_dir, model, optimizer, epoch)
                record["new_best"] = True

        history.append(record)
        save_last(run_dir, model, optimizer, epoch, best_top1, history)
        log(f"[epoch {epoch}] {record}")

    best_out = None if best_top1 == -math.inf else best_top1
    mark_done(run_dir, {"best_top1": best_out, "epochs": n_epochs, "history": history})
    return best_top1, history
