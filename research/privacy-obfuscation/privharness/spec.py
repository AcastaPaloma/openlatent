"""The dataset specification every inbound dataset is converted to.

A harness dataset lives in one directory:

    <root>/dataset.json      this spec (targets, groups, items)
    <root>/frames/<id>.npy   uint8 frames, (T, H, W) grayscale or (T, H, W, 3) RGB; images are T = 1

dataset.json:

    name, modality ("video" | "image"), fps (decoded frame rate; null for images), size [H, W], channels (1 | 3)
    meta:    free-form provenance carried into the report: source URL, licence / access terms, citation
    targets: [{name, role ("utility" | "secret"), kind ("multiclass" | "multilabel"), classes: [names]}]
             the first utility target is the primary task; every secret is protected and scored
    groups:  metadata keys carried by items (e.g. "source", "scene", "session", "subject"); "source" always exists
             and names the recording an item was cut from, so no recording is ever on both sides of a split
    items:   [{id, frames (path relative to root), start, end (1-based inclusive frame range),
               labels {target: class index | [0/1 per class]}, groups {key: value}, split ("train" | "test")}]

Several items may share one frames file (KTH's labelled subsequences of one recording); they share a "source" group.
"""

import json
import os
from dataclasses import asdict, dataclass, field

ROLES = ("utility", "secret")
KINDS = ("multiclass", "multilabel")


@dataclass
class Target:
    name: str
    role: str
    kind: str
    classes: list

    def __post_init__(self):
        assert self.role in ROLES, f"target {self.name}: role must be one of {ROLES}"
        assert self.kind in KINDS, f"target {self.name}: kind must be one of {KINDS}"
        assert len(self.classes) >= 2 or self.kind == "multilabel", f"target {self.name}: needs >= 2 classes"

    @property
    def n(self):
        return len(self.classes)


@dataclass
class Spec:
    name: str
    modality: str
    size: list
    channels: int
    targets: list
    items: list
    groups: list = field(default_factory=lambda: ["source"])
    fps: float = None
    meta: dict = field(default_factory=dict)

    @property
    def utility(self):
        return [t for t in self.targets if t.role == "utility"]

    @property
    def secrets(self):
        return [t for t in self.targets if t.role == "secret"]

    @property
    def primary(self):
        return self.utility[0]

    def target(self, name):
        return next(t for t in self.targets if t.name == name)

    def split(self, which):
        return [it for it in self.items if it["split"] == which]

    def validate(self):
        """Raise on anything a later stage would trip over; return a list of non-fatal warnings."""
        assert self.modality in ("video", "image"), self.modality
        assert self.utility, "a dataset needs at least one utility target"
        assert self.secrets, "a dataset needs at least one secret target"
        assert len({t.name for t in self.targets}) == len(self.targets), "target names must be unique"
        assert "source" in self.groups, "every dataset carries a 'source' group"
        ids = [it["id"] for it in self.items]
        assert len(set(ids)) == len(ids), "item ids must be unique"
        warnings = []
        for it in self.items:
            assert it["split"] in ("train", "test"), f"{it['id']}: split must be train or test"
            assert 1 <= it["start"] <= it["end"], f"{it['id']}: bad frame range"
            for g in self.groups:
                assert g in it["groups"], f"{it['id']}: missing group {g}"
            for t in self.targets:
                y = it["labels"].get(t.name)
                if y is None:
                    continue  # unlabelled for this target: the item is skipped for it
                if t.kind == "multiclass":
                    assert isinstance(y, int) and 0 <= y < t.n, f"{it['id']}: {t.name}={y}"
                else:
                    assert len(y) == t.n and all(v in (0, 1) for v in y), f"{it['id']}: {t.name} must be {t.n} 0/1 values"
        by_source = {}
        for it in self.items:
            by_source.setdefault(it["groups"]["source"], set()).add(it["split"])
        mixed = [s for s, sp in by_source.items() if len(sp) > 1]
        if mixed:
            raise AssertionError(f"{len(mixed)} sources have items on both sides of the split, e.g. {mixed[:3]}")
        for t in self.targets:
            n_lab = sum(t.name in it["labels"] for it in self.items)
            if n_lab < len(self.items):
                warnings.append(f"{t.name}: {len(self.items) - n_lab} of {len(self.items)} items unlabelled")
        return warnings

    def to_json(self):
        d = asdict(self)
        d["targets"] = [asdict(t) for t in self.targets]
        return d

    def save(self, root):
        tmp = os.path.join(root, "dataset.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, os.path.join(root, "dataset.json"))


def load(root):
    d = json.load(open(os.path.join(root, "dataset.json")))
    d["targets"] = [Target(**t) for t in d["targets"]]
    s = Spec(**d)
    s.validate()
    return s


def rows_for(items, target):
    """Items labelled for `target` as rows for ppv.clips.Clips (keyed by frames file, so items cut from one recording
    share one array); both action_id and priv carry the target's label."""
    return [dict(video=it["frames"], start=it["start"], end=it["end"], action_id=it["labels"][target.name],
                 priv=it["labels"][target.name], id=it["id"]) for it in items if target.name in it["labels"]]


def frames_index(spec, root):
    """{frames path: absolute .npy path} for ppv.clips.LazyFrames."""
    return {it["frames"]: os.path.join(root, it["frames"]) for it in spec.items}
