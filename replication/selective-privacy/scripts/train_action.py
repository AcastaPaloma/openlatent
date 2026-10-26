"""Resumable action-model training on original frames, reusing upstream code.

Same datasets, models, optimizer, schedule and Trainer as upstream action_train.py,
but the epoch loop goes through resumable.run_epochs and the output directory is
fixed, so a VM restart resumes instead of starting over.

Output: <out>/action/<dataset>/<model_name>/
    last.pt       checkpoint after every epoch (resume point)
    best.pt       best test top-1 so far, upstream format (used by eval)
    DONE.json     written when the run finishes; re-running is a no-op
    provenance.json, tensorboard/, suspend.log

Run from the upstream checkout, because upstream dataset paths are relative to it:
    cd SelectivePrivacyPreservation
    python <repo>/replication/selective-privacy/scripts/train_action.py \
        --architecture x3d_s --datasetname kth -pretrained -train_backbone --out <dir>
"""

import os
import subprocess
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
from torch import nn, optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from config import build_cfg, parser  # noqa: E402

parser.add_argument("--architecture", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
cfg = build_cfg()

import resumable  # noqa: E402
from dataset.db_factory import DBfactory  # noqa: E402
from models.valid_models import action_models  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from utils.model_utils import build_model, build_model_name, set_seed  # noqa: E402
from utils.Trainer import Trainer  # noqa: E402


def git_head(path):
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    if cfg["privacy"]:
        raise SystemExit("train_action.py trains action models; privacy needs train_privacy.py")
    arch = cfg["architecture"]
    if arch not in action_models:
        raise SystemExit(f"{arch} is not an action architecture in models/valid_models.py")

    set_seed()
    torch.backends.cudnn.benchmark = True
    ds = cfg["datasetname"]

    train_ds = DBfactory(ds, set_split="train", config=cfg)
    test_ds = DBfactory(ds, set_split="test", config=cfg)
    train_dl = DataLoader(
        train_ds,
        cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    test_dl = DataLoader(
        test_ds,
        cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    model = build_model(arch, cfg["pretrained"], train_ds.num_classes, cfg["train_backbone"])
    model.name = build_model_name(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    # numbered=False and a fixed sim_name give a stable directory across restarts.
    # suspend=True appends to suspend.log instead of truncating it and skips the
    # git snapshot that a fresh numbered run would take (provenance.json covers it).
    with Simulation(
        sim_name=f"action/{ds}/{model.name}",
        output_root=cfg["out"],
        numbered=False,
        suspend=True,
    ) as sim:
        run_dir = sim.outdir
        if resumable.is_done(run_dir):
            print(f"[done] {run_dir} already finished")
            return

        resumable.atomic_json_save(
            {
                "upstream_commit": git_head(UPSTREAM),
                "openlatent_commit": git_head(HERE),
                "cfg": cfg,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            join(run_dir, "provenance.json"),
        )

        writer = SummaryWriter(join(run_dir, "tensorboard"))
        trainer = Trainer(sim)

        def train_epoch(epoch):
            top1 = trainer.do(
                "train",
                model,
                train_dl,
                epoch,
                criterion,
                optimizer,
                writer,
                log_video=False,
                accumulate_grad_batches=cfg["accumulate_grad_batches"],
            )
            return {"top1": top1}

        def eval_epoch(epoch):
            return trainer.do(
                "test",
                model,
                test_dl,
                epoch,
                criterion,
                None,
                writer,
                log_video=False,
            )

        best, _ = resumable.run_epochs(
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            n_epochs=cfg["num_epochs"],
            train_epoch=train_epoch,
            eval_epoch=eval_epoch,
            eval_every=5,
        )
        writer.close()
        print(f"Run {run_dir} finished; best test top1 = {best}")


if __name__ == "__main__":
    main()
