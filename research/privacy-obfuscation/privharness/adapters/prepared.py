"""Convert datasets already decoded by ppv (KTH, VP-UCF101, VP-HMDB51, Charades, UTD-MHAD) into harness datasets.

Frames are not re-decoded: <out>/frames is a symlink to the ppv frames directory. Splits are the ones every ppv
result used, so harness numbers are directly comparable with results/SUMMARY.md.

Usage: python -m privharness.adapters.prepared <kth|ucf101_vp|hmdb51_vp|charades|utdmhad> <out> [--split ilic|scenario]
"""

import argparse
import json
import os

from .. import spec as S

WORK = os.path.expanduser("~/work")
VP_ATTRS = ["face", "skin_color", "gender", "nudity", "relationship"]


def _link_frames(src_root, out):
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "frames")
    if not os.path.islink(link):
        os.symlink(os.path.join(src_root, "frames"), link)


def kth(out, split="ilic"):
    from ppv import kth as K
    root = os.path.join(WORK, "data", "kth")
    rows = K.load_meta(root)
    tr, te = K.split(rows, split)
    test_ids = {(r["video"], r["start"]) for r in te}
    items = [dict(id=f"{r['video']}_{r['start']}", frames=f"frames/{r['video']}.npy", start=r["start"], end=r["end"],
                  labels=dict(action=r["action_id"], identity=r["person_id"]),
                  groups=dict(source=r["video"], scenario=f"d{r['scenario']}"),
                  split="test" if (r["video"], r["start"]) in test_ids else "train") for r in rows]
    targets = [S.Target("action", "utility", "multiclass", list(K.ACTIONS)),
               S.Target("identity", "secret", "multiclass", [f"person{i:02d}" for i in range(1, 26)])]
    return S.Spec(name=f"kth_{split}", meta=dict(source="https://www.csc.kth.se/cvap/actions/", citation="Schuldt, Laptev, Caputo, ICPR 2004", split_note="ilic: Ilic et al. CVPR 2024 re-split (every person in train and test)" if split == "ilic" else f"split {split}"), modality="video", size=[K.H, K.W], channels=1, fps=25.0, targets=targets,
                  groups=["source", "scenario"], items=items), root


def _vp(name, sub, split_key, out):
    root = os.path.join(WORK, "data", sub)
    rows = json.load(open(os.path.join(root, "meta.json")))
    classes = sorted({r["video"].split("/")[0] for r in rows})
    items = []
    for r in rows:
        cls, vid = r["video"].split("/")
        if name == "ucf101_vp":  # v_<Class>_g<NN>_c<NN>: clips of one group are cut from one long recording
            source = f"{cls}/{vid.rsplit('_', 1)[0]}"
        else:
            source = r["video"]
        items.append(dict(id=r["video"], frames=f"frames/{r['video']}.npy", start=r["start"], end=r["end"],
                          labels=dict(action=classes.index(cls), attributes=list(r["priv"])),
                          groups=dict(source=source), split=r[split_key]))
    targets = [S.Target("action", "utility", "multiclass", classes),
               S.Target("attributes", "secret", "multilabel", list(VP_ATTRS))]
    return S.Spec(name=name, modality="video", size=[112, 150], channels=3, fps=12.5, targets=targets,
                  groups=["source"], items=items,
                  meta=dict(source="UCF101 (CRCV) / HMDB51 (Serre lab, via the MichiganNLP/hmdb mirror)",
                            labels="STPrivacy video-level privacy attributes (Li et al., ICCV 2023), github.com/ming1993li/stprivacy",
                            note="attribute labels are largely implied by the action class; see the conditional metric")), root


def charades(out):
    root = os.path.join(WORK, "data", "charades")
    meta = json.load(open(os.path.join(root, "meta.json")))
    items = [dict(id=r["video"], frames=f"frames/{r['video']}.npy", start=r["start"], end=r["end"],
                  labels=dict(actions=list(r["action_id"]), identity=r["priv"]),
                  groups=dict(source=r["video"], scene=r["scene"]), split=r["split"]) for r in meta["rows"]]
    targets = [S.Target("actions", "utility", "multilabel", [f"c{i:03d}" for i in range(157)]),
               S.Target("identity", "secret", "multiclass", list(meta["subjects"]))]
    return S.Spec(name="charades", meta=dict(source="https://prior.allenai.org/projects/charades", licence="non-commercial use (AI2 Charades licence); do not redistribute derived data", citation="Sigurdsson et al., ECCV 2016"), modality="video", size=[112, 150], channels=3, fps=6.25, targets=targets,
                  groups=["source", "scene"], items=items), root


def utdmhad(out):
    root = os.path.join(WORK, "data", "utdmhad")
    rows = json.load(open(os.path.join(root, "meta.json")))
    items = [dict(id=r["video"], frames=f"frames/{r['video']}.npy", start=r["start"], end=r["end"],
                  labels=dict(action=r["action_id"], identity=r["priv"]),
                  groups=dict(source=r["video"], trial=f"t{r['trial']}"), split=r["split"]) for r in rows]
    targets = [S.Target("action", "utility", "multiclass", [f"a{i:02d}" for i in range(1, 28)]),
               S.Target("identity", "secret", "multiclass", [f"s{i}" for i in range(1, 9)])]
    return S.Spec(name="utdmhad", meta=dict(source="https://personal.utdallas.edu/~kehtar/UTD-MHAD.html", citation="Chen, Jafari, Kehtarnavaz, ICIP 2015"), modality="video", size=[112, 150], channels=3, fps=15.0, targets=targets,
                  groups=["source", "trial"], items=items), root


def convert(name, out, split="ilic"):
    if name == "kth":
        spec, root = kth(out, split)
    elif name == "ucf101_vp":
        spec, root = _vp(name, "ucf101", "split1", out)
    elif name == "hmdb51_vp":
        spec, root = _vp(name, "hmdb51", "split1", out)
    elif name == "charades":
        spec, root = charades(out)
    elif name == "utdmhad":
        spec, root = utdmhad(out)
    else:
        raise ValueError(name)
    warnings = spec.validate()
    _link_frames(root, out)
    spec.save(out)
    from ..ingest import audit
    report = audit(spec, out)
    report.update(source=f"ppv:{name}", warnings=warnings + report["warnings"])
    json.dump(report, open(os.path.join(out, "ingest_report.json"), "w"), indent=1)
    return spec, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--split", default="ilic")
    a = ap.parse_args()
    s, r = convert(a.name, a.out, a.split)
    print(f"{s.name}: {len(s.items)} items, warnings: {r['warnings']}")
