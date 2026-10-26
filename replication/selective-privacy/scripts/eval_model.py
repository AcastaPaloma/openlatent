"""Evaluate a trained action or privacy model on the test split, with or without obfuscation.

Mirrors upstream action_eval.py and privacy_eval.py. The model is trained on original
frames (train_model.py) and obfuscation is applied only at test time. One invocation
scores one condition:

  original : no -selectively_mask
  ours     : -selectively_mask --obfuscate torso arm leg hair

Reads  <out>/<kind>/<ds>/<model_name>/best.pt   (written by train_model.py)
Writes <out>/eval/<kind>/<ds>/<model_name>/<tag>.json

Note on privacy: upstream privacy_eval.py scores one randomly sampled frame per clip
(the single-frame loader). It does not average over 32 frames as the paper's text
describes. We follow the code; the README records the difference.

Run from the upstream checkout, like train_model.py.
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

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import build_cfg, parser  # noqa: E402

parser.add_argument("--architecture", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
cfg = build_cfg()

import resumable  # noqa: E402
from dataset.db_factory import DBfactory  # noqa: E402
from models.valid_models import action_models, privacy_models  # noqa: E402
from utils.model_utils import (  # noqa: E402
    build_model,
    build_model_name,
    build_model_name_singleframe,
    build_model_privacy,
    load_weights,
    set_seed,
)
from utils.Trainer import Trainer  # noqa: E402


class _NullSim:
    """Trainer only needs save_data for diagnostics. A null sink keeps eval out of the run dir."""

    def save_data(self, *args, **kwargs):
        pass


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_video(self, *args, **kwargs):
        pass


def condition_tag(cfg):
    if not cfg["selectively_mask"]:
        return "original"
    return "obf_" + "_".join(sorted(cfg["obfuscate"]))


def main():
    arch = cfg["architecture"]
    privacy = cfg["privacy"]
    kind = "privacy" if privacy else "action"
    allowed = privacy_models if privacy else action_models
    if arch not in allowed:
        raise SystemExit(f"{arch} is not a {kind} architecture")
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

    if privacy:
        model = build_model_privacy(arch, cfg["pretrained"], test_ds.num_classes, cfg["train_backbone"])
        model.name = build_model_name_singleframe(cfg)
    else:
        model = build_model(arch, cfg["pretrained"], test_ds.num_classes, cfg["train_backbone"])
        model.name = build_model_name(cfg)

    weights = join(cfg["out"], kind, ds, model.name, "best.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"missing trained weights: {weights}")
    load_weights(model, None, weights)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    top1 = Trainer(_NullSim()).do(
        "test", model, test_dl, 0, criterion, None, writer=_NullWriter(), log_video=False
    )

    tag = condition_tag(cfg)
    out_dir = join(cfg["out"], "eval", kind, ds, model.name)
    result = {
        "kind": kind,
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
    print(f"{kind} {ds} {arch} {tag}: top1 = {top1:.2f}")


if __name__ == "__main__":
    main()
