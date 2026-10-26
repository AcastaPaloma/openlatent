"""Early read: evaluate a trained model only on test clips whose similarity maps are complete.

While a similarity stage is still running, this scores the same subset of finished clips
twice: original frames and obfuscated frames. The comparison is like-for-like, but it is a
subset of the test split, so the final number comes from eval_model.py on all clips.

Usage (from the upstream checkout):
    python partial_eval.py --architecture resnet18 --datasetname kth -pretrained -train_backbone \
        -privacy --batch_size 128 --out <runs> --obfuscate torsoT armT legT hairT
"""

import os
import sys
from os.path import join

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.getcwd()
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, HERE)

import compat  # noqa: E402,F401
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import build_cfg, parser  # noqa: E402

parser.add_argument("--architecture", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
cfg = build_cfg()

from dataset.db_factory import DBfactory  # noqa: E402
from utils.model_utils import (  # noqa: E402
    build_model,
    build_model_name,
    build_model_name_singleframe,
    build_model_privacy,
    load_weights,
    set_seed,
)


def complete_clips(templates):
    frames = {v: len(os.listdir(join("data/kth", v))) for v in os.listdir("data/kth")}
    done = set(frames)
    for t in templates:
        d = f"data/kth_{t}_sim"
        done &= {v for v in (os.listdir(d) if os.path.isdir(d) else [])
                 if len(os.listdir(join(d, v))) == frames.get(v, -1)}
    return done


def top1(model, ds):
    set_seed()
    dl = DataLoader(ds, cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for inputs, _, _, labels in dl:
            pred = model(inputs.cuda()).argmax(1).cpu()
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / total, total


def main():
    templates = cfg["obfuscate"]
    done = complete_clips(templates)
    privacy = cfg["privacy"]
    kind = "privacy" if privacy else "action"

    results = {}
    for label, sel in (("original", False), ("obfuscated", True)):
        c = dict(cfg, selectively_mask=sel, obfuscate=templates if sel else [])
        ds = DBfactory(cfg["datasetname"], set_split="test", config=c)
        ds.annot = ds.annot[ds.annot.video.isin(done)]
        if label == "original":
            if privacy:
                model = build_model_privacy(cfg["architecture"], True, ds.num_classes, True)
                model.name = build_model_name_singleframe(cfg)
            else:
                model = build_model(cfg["architecture"], True, ds.num_classes, True)
                model.name = build_model_name(cfg)
            load_weights(model, None, join(cfg["out"], kind, cfg["datasetname"], model.name, "best.pt"))
            model = model.cuda().eval()
        results[label] = top1(model, ds)

    n = results["original"][1]
    print(f"PARTIAL {kind} {cfg['architecture']} on {len(done)} complete test clips ({n} samples): "
          f"original {results['original'][0]:.2f}  obfuscated {results['obfuscated'][0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
