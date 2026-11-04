"""Plumbing test of the harness runners on a FAKE harness directory (synthetic probabilities); no programme-1 file is read."""
import argparse
import json
import os

import numpy as np

from s_suite import harness_io as hio
from s_suite import run_real as rr


def make_fake(root, ds="kth_ilic", n_p=6, n_scen=3, acts=("boxing", "walking"), clips=3, releases=("pose_canon_down16", "pose_canon_down16_s1", "black", "pose_canon_down16_dino")):
    rs = np.random.RandomState(0)
    items, ids = [], []
    for p in range(n_p):
        for sc in range(n_scen):
            for a in acts:
                for c in range(clips):
                    i = f"person{p + 1:02d}_{a}_d{sc + 1}_{c + 1}"
                    items.append(dict(id=i, labels=dict(identity=p), groups=dict(source=f"person{p + 1:02d}_{a}_d{sc + 1}", scenario=f"d{sc + 1}"), split="test"))
                    ids.append(i)
    os.makedirs(os.path.join(root, ds), exist_ok=True)
    json.dump(dict(items=items), open(os.path.join(root, ds, "dataset.json"), "w"))
    y = np.array([it["labels"]["identity"] for it in items])
    for rel in releases:
        d = os.path.join(root, ds, "eval", rel)
        os.makedirs(d, exist_ok=True)
        archs = ["frame_dino_probe", "frame_dino_ft"] if rel.endswith("dino") else ["frame_r50", "r3d"]
        sig = 0.0 if rel == "black" else 1.2
        preds = {}
        for arch in archs:
            P = rs.randn(len(y), n_p) + sig * np.eye(n_p)[y]
            P = np.exp(P) / np.exp(P).sum(1, keepdims=True)
            preds[f"secret|identity|within|retrained|{arch}"] = P
        np.savez_compressed(os.path.join(d, "preds.npz"), **preds)
        json.dump(dict(secrets=dict(identity=dict(within=dict(test_ids=ids)))), open(os.path.join(d, "result.json"), "w"))


def test_runners_on_fake_harness(tmp_path, monkeypatch):
    root, out = str(tmp_path / "h"), str(tmp_path / "o")
    make_fake(root)
    os.makedirs(out)
    monkeypatch.setattr(rr, "DATASETS", ("kth_ilic",))
    monkeypatch.setattr(rr, "FAMILIES", {"kth_ilic": ["pose_canon_down16", "pose_canon_down16_s1", "pose_canon_down16_dino"]})
    monkeypatch.setattr(rr, "K_GRID", [1, 2, 4])
    a = argparse.Namespace(root=root, out=out)
    rr.cmd_select(a)
    rr.cmd_budget(a)
    sel = json.load(open(os.path.join(out, "harness_selection_bias.json")))
    assert sel["_provenance"]["evaluator"]["sha256"] and sel["_provenance"]["script"] == "run_real.py"
    assert "kth_ilic/black" in sel["cells"] and sel["families"]["kth_ilic"]["P"] == 6
    bud = json.load(open(os.path.join(out, "harness_evidence_budget.json")))
    k = "kth_ilic/pose_canon_down16/retrained|r3d"
    assert bud[k]["sessions_per_person"] == 3 and bud[k]["independence"] is not None
    acc = bud[k]["schemes"]["diverse"]["acc"]
    assert acc[-1] >= acc[0] - 0.05


def test_toy_command(tmp_path):
    a = argparse.Namespace(root=None, out=str(tmp_path))
    rr.cmd_toy(a)
    d = json.load(open(os.path.join(str(tmp_path), "toy_budget.json")))
    assert set(d) >= {"0.0", "0.5", "0.9", "shuffled_person_control"}
