"""Plumbing test of rescore.py on a FAKE harness and FAKE report (no programme-1 file is read)."""
import json
import os

import numpy as np

from s_suite import rescore as rs
from .test_run_real_smoke import make_fake


def fake_report(root, ds, rels, n_p=6):
    scored = {}
    for rel in rels:
        z = np.load(os.path.join(root, ds, "eval", rel, "preds.npz"))
        r = json.load(open(os.path.join(root, ds, "eval", rel, "result.json")))
        idx = {it["id"]: it for it in json.load(open(os.path.join(root, ds, "dataset.json")))["items"]}
        y = np.array([idx[i]["labels"]["identity"] for i in r["secrets"]["identity"]["within"]["test_ids"]])
        att = {k.split("|", 3)[3].replace("|", "/") if False else "retrained/" + k.split("|")[4]: 100 * float((z[k].argmax(1) == y).mean()) for k in z.files}
        sc = max(att.values())
        scored[rel] = dict(secrets=dict(identity=dict(within=dict(attacker="x", score=sc, base=100 / n_p, raw=99.0, attackers=att,
                                                                      privacy=1 - min(1, max(0, (sc - 100 / n_p) / (99 - 100 / n_p)))))))
    return dict(scored=scored)


def test_rescore_accuracy_path(tmp_path, monkeypatch):
    root = str(tmp_path / "h")
    rels = ("pose_canon_down16", "pose_canon_down16_s1", "black")
    make_fake(root, releases=rels)
    rep = fake_report(root, "kth_ilic", rels)
    monkeypatch.setattr(rs, "B_BOOT", 100)
    monkeypatch.setattr(rs, "N_SPLITS", 4)
    r = rs.rescore_dataset(root, "kth_ilic", "within", rep)
    assert set(r["releases"]) == set(rels) and len(r["adjacent_pairs"]) == 2
    x = r["releases"]["pose_canon_down16"]
    assert x["reproduces"] is True
    assert 0 <= x["honest"] <= 1 and x["optimism_a2_minus_h"]["mde80"] > 0
    assert r["releases"]["black"]["a2"] < x["a2"]
    assert "pose_canon_down16" in rs.to_markdown({"kth_ilic/within": r})


def test_rescore_multilabel_path(tmp_path, monkeypatch):
    root = str(tmp_path / "h")
    ds = "hmdb51_vp"
    rs0 = np.random.RandomState(0)
    n = 80
    ids = [f"c/v{i}" for i in range(n)]
    Y = (rs0.rand(n, 5) < 0.4).astype(int)
    os.makedirs(os.path.join(root, ds), exist_ok=True)
    json.dump(dict(items=[dict(id=i, labels=dict(attributes=Y[k].tolist()), groups=dict(source=i)) for k, i in enumerate(ids)]), open(os.path.join(root, ds, "dataset.json"), "w"))
    d = os.path.join(root, ds, "eval", "bdq")
    os.makedirs(d)
    np.savez(os.path.join(d, "preds.npz"), **{f"secret|attributes|within|{kd}|{a}": rs0.rand(n, 5) + 0.5 * Y for kd in ("retrained", "frozen") for a in ("frame_r50", "r3d")})
    json.dump(dict(secrets=dict(attributes=dict(within=dict(test_ids=ids)))), open(os.path.join(d, "result.json"), "w"))
    atts = {f"{kd}/{a}": 60.0 for kd in ("retrained", "frozen") for a in ("frame_r50", "r3d")}
    rep = dict(scored=dict(bdq=dict(secrets=dict(attributes=dict(within=dict(score=60.0, base=40.0, raw=70.0, attackers=atts, privacy=0.5))))))
    monkeypatch.setattr(rs, "B_BOOT_GENERIC", 10)
    r = rs.rescore_dataset(root, ds, "within", rep)
    x = r["releases"]["bdq"]
    assert 0.3 < x["a2"] < 1 and np.isfinite(x["honest"])
