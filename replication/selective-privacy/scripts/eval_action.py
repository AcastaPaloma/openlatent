"""Evaluate a trained action model on the test split, with or without obfuscation.

Mirrors upstream action_eval.py: the model is trained on original frames (train_action.py)
and obfuscation is applied only at test time. One invocation scores one condition:

  original  : python eval_action.py ... (no -selectively_mask)
  ours      : python eval_action.py ... -selectively_mask --obfuscate torso arm leg hair

Reads  <out>/action/<ds>/<model_name>/best.pt  (written by train_action.py)
Writes <out>/eval/action/<ds>/<model_name>/<tag>.json

Run from the upstream checkout, like train_action.py.
"""

import os
import sys
from os.path import join

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.getcwd()
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, HERE)

import compat  # noqa: E402,F401  (must precede pytorchvideo imports)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import build_cfg, parser  # noqa: E402

parser.add_argument("--architecture", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
cfg = build_cfg()

import resumable  # noqa: E402
from dataset.db_factory import DBfactory  # noqa: E402
from models.valid_models import action_models  # noqa: E402
from utils.model_utils import build_model, build_model_name, load_weights, set_seed  # noqa: E402
from utils.Trainer import Trainer  # noqa: E402


def condition_tag(cfg):
    if not cfg["selectively_mask"]:
        return "original"
    return "obf_" + "_".join(sorted(cfg["obfuscate"]))


def main():
    arch = cfg["architecture"]
    if arch not in action_models:
        raise SystemExit(f"{arch} is not an action architecture")
    if cfg["selectively_mask"] and not cfg["obfuscate"]:
        raise SystemExit("-selectively_mask needs --obfuscate <templates>")

    set_seed()
    ds = cfg["datasetname"]
    test_ds = DBfactory(ds, set_split="test", config=cfg)
    test_dl = DataLoader(
        test_ds,
        cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    model = build_model(arch, cfg["pretrained"], test_ds.num_classes, cfg["train_backbone"])
    model.name = build_model_name(cfg)
    weights = join(cfg["out"], "action", ds, model.name, "best.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"missing trained weights: {weights}")
    load_weights(model, None, weights)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # Trainer only needs save_data for diagnostics; a null sink keeps eval out of the run dir.
    class _NullSim:
        def save_data(self, *args, **kwargs):
            pass

    top1 = Trainer(_NullSim()).do(
        "test", model, test_dl, 0, criterion, None, writer=_NullWriter(), log_video=False
    )

    tag = condition_tag(cfg)
    out_dir = join(cfg["out"], "eval", "action", ds, model.name)
    result = {
        "dataset": ds,
        "architecture": arch,
        "condition": tag,
        "obfuscate": cfg["obfuscate"] if cfg["selectively_mask"] else [],
        "top1": top1,
        "weights": weights,
        "upstream_commit": resumable.git_head(UPSTREAM),
        "openlatent_commit": resumable.git_head(HERE),
    }
    resumable.atomic_json_save(result, join(out_dir, f"{tag}.json"))
    print(f"{ds} {arch} {tag}: top1 = {top1:.2f}")


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_video(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    main()
