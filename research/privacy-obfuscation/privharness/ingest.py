"""Ingestion: raw media + a description of its labels -> a validated harness dataset (spec.py) with an audit.

The description (JSON) says where the media are, which columns are utility targets, secrets and groups, and how to
split. Two sources of rows are supported:

    "table": "items.csv" | "items.jsonl"          one row per item; "path_column" names the media column
    "folder": {"root": ..., "pattern": regex}     every file under root whose path relative to root matches the
                                                  regex; named groups of the regex become columns
                                                  (e.g. "(?P<action>[^/]+)/(?P<subject>person\\d+)_.*")

Example:
    {"name": "kth_raw", "folder": {"root": "/data/kth", "pattern": "(?P<subject>person\\d+)_(?P<action>[a-z]+)_(?P<scene>d\\d)_uncomp\\.avi"},
     "targets": [{"name": "action", "role": "utility", "kind": "multiclass", "column": "action"},
                 {"name": "identity", "role": "secret", "kind": "multiclass", "column": "subject"}],
     "groups": {"scene": "scene"},
     "split": {"mode": "stratified", "by": "identity", "test_fraction": 0.25},
     "decode": {"short_side": 120, "max_long_side": 160, "fps": null, "gray": "auto"}}

Targets: multiclass from one column (class names sorted unless "classes" is given); multilabel from "columns"
(one 0/1 column per class) or "column" holding class names separated by "sep" (default ";").
Segments: optional "segments": {"start": col, "end": col, "unit": "frame" | "second"}; otherwise the whole file.
Split: {"column": name} (values train/test), {"mode": "stratified", "by": target, "test_fraction": f} (recordings
assigned whole, per class of `by`), or {"mode": "group", "key": group, "test_values": [...]} (hold out group values).
Decode: frames are resized so the short side is `short_side` (long side capped at `max_long_side`, both even),
optionally resampled to `fps`; "gray": "auto" stores one channel when the media carry no colour.

Output under <out>: dataset.json, frames/*.npy, ingest_report.json.
"""

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import spec as S

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v", ".wmv", ".flv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def media_kind(path):
    if os.path.isdir(path):
        return "frame_dir"
    ext = os.path.splitext(path)[1].lower()
    return "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else None


def probe_video(path):
    """(width, height, fps, n_frames) via ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / max(1.0, float(den)), int(s.get("nb_read_packets", 0))


def target_size(w, h, short_side, max_long):
    scale = short_side / min(w, h)
    if max_long and max(w, h) * scale > max_long:
        scale = max_long / max(w, h)
    even = lambda v: max(2, int(round(v * scale / 2)) * 2)  # noqa: E731
    return even(w), even(h)


def _read_image(path):
    import cv2
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f"unreadable image {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def _resize(arr, w, h):
    import cv2
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)


def decode_one(job):
    """job = (src, out, decode cfg) -> dict(out, n, fps, w, h, colour, error). Idempotent: reuses an existing out."""
    src, out, dec = job
    kind = media_kind(src)
    try:
        if os.path.exists(out):
            a = np.load(out, mmap_mode="r")
            return dict(src=src, out=out, n=a.shape[0], fps=dec.get("fps"), h=a.shape[1], w=a.shape[2],
                        colour=a.ndim == 4, error=None)
        if kind == "video":
            w0, h0, fps0, _ = probe_video(src)
            w, h = target_size(w0, h0, dec["short_side"], dec.get("max_long_side"))
            vf = (f"fps={dec['fps']}," if dec.get("fps") else "") + f"scale={w}:{h}:flags=area"
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 check=True, capture_output=True).stdout
            arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
            fps = dec.get("fps") or fps0
        elif kind in ("image", "frame_dir"):
            files = ([src] if kind == "image" else
                     sorted(os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMAGE_EXT))
            first = _read_image(files[0])
            w, h = target_size(first.shape[1], first.shape[0], dec["short_side"], dec.get("max_long_side"))
            arr = np.stack([_resize(_read_image(f), w, h) for f in files])
            fps = dec.get("frame_dir_fps") if kind == "frame_dir" else None
        else:
            raise ValueError(f"unsupported media {src}")
        if len(arr) == 0:
            raise ValueError("no frames decoded")
        # colour test on up to 8 frames: a gray video decoded as RGB has three identical channels
        sample = arr[np.linspace(0, len(arr) - 1, min(8, len(arr))).astype(int)].astype(np.int16)
        colour = bool(np.abs(sample[..., 0] - sample[..., 1]).max() > 3 or np.abs(sample[..., 1] - sample[..., 2]).max() > 3)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out + ".tmp.npy", arr)
        os.replace(out + ".tmp.npy", out)
        return dict(src=src, out=out, n=len(arr), fps=fps, h=arr.shape[1], w=arr.shape[2], colour=colour, error=None)
    except Exception as e:  # recorded in the report; the item is dropped
        return dict(src=src, out=out, error=f"{type(e).__name__}: {e}"[:300])


def to_gray(path):
    a = np.load(path)
    if a.ndim == 4:
        g = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).round().astype(np.uint8)
        np.save(path + ".tmp.npy", g)
        os.replace(path + ".tmp.npy", path)


def ahash(frame, size=8):
    """64-bit average hash of a frame (perceptual near-duplicate test)."""
    import cv2
    g = frame if frame.ndim == 2 else frame.mean(-1).astype(np.uint8)
    small = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.packbits(small > small.mean())


def motion_trace(arr, start, end, n=64):
    """Mean |frame difference| per frame over up to n + 1 frames from `start`, on 32x24 thumbnails; None when the
    item is too short (< 17 frames) or too still to correlate."""
    import cv2
    s, e = start - 1, min(end, start - 1 + n + 1)
    if e - s < 17:
        return None
    x = np.asarray(arr[s:e]).astype(np.float32)
    x = x.mean(-1) if x.ndim == 4 else x
    small = np.stack([cv2.resize(f, (32, 24), interpolation=cv2.INTER_AREA) for f in x])
    t = np.abs(np.diff(small, axis=0)).mean((1, 2))
    return t if t.std() > 1e-3 else None


def trace_corr(a, b, maxlag=4):
    best = -1.0
    for lag in range(-maxlag, maxlag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        m = min(len(x), len(y))
        if m < 16:
            continue
        x, y = x[:m], y[:m]
        if x.std() > 1e-6 and y.std() > 1e-6:
            best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def read_rows(desc, base):
    if "table" in desc:
        path = os.path.join(base, desc["table"])
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in open(path) if line.strip()]
        else:
            rows = list(csv.DictReader(open(path, newline="")))
        col = desc.get("path_column", "path")
        root = os.path.join(base, desc.get("media_root", ""))
        for r in rows:
            r["_path"] = r[col] if os.path.isabs(r[col]) else os.path.join(root, r[col])
        return rows
    f = desc["folder"]
    root, pat = os.path.join(base, f["root"]), re.compile(f["pattern"])
    rows = []
    for dirpath, dirnames, files in os.walk(root):
        names = files + (dirnames if f.get("frame_dirs") else [])
        for name in names:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            m = pat.fullmatch(rel)
            if m:
                rows.append(dict(m.groupdict(), _path=os.path.join(root, rel)))
    return sorted(rows, key=lambda r: r["_path"])


def build_targets(desc, rows):
    targets, labels = [], defaultdict(dict)
    for td in desc["targets"]:
        if td["kind"] == "multiclass":
            vals = [r.get(td["column"]) for r in rows]
            classes = td.get("classes") or sorted({str(v) for v in vals if v not in (None, "")}, key=natural_key)
            idx = {c: i for i, c in enumerate(classes)}
            for i, v in enumerate(vals):
                if v not in (None, "") and str(v) in idx:
                    labels[i][td["name"]] = idx[str(v)]
        else:
            if "columns" in td:
                classes = td.get("classes") or list(td["columns"])
                for i, r in enumerate(rows):
                    if all(r.get(c) not in (None, "") for c in td["columns"]):
                        labels[i][td["name"]] = [int(float(r[c]) > 0) for c in td["columns"]]
            else:
                sep = td.get("sep", ";")
                split = lambda v: [s.strip() for s in str(v).split(sep) if s.strip()] if v not in (None, "") else []  # noqa: E731
                classes = td.get("classes") or sorted({c for r in rows for c in split(r.get(td["column"]))}, key=natural_key)
                for i, r in enumerate(rows):
                    if r.get(td["column"]) is not None:
                        present = set(split(r[td["column"]]))
                        labels[i][td["name"]] = [int(c in present) for c in classes]
        targets.append(S.Target(name=td["name"], role=td["role"], kind=td["kind"], classes=classes))
    return targets, labels


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def make_split(desc, items, targets, seed=0):
    sp = desc.get("split", {"mode": "stratified"})
    if "column" in sp:
        return {it["id"]: it["_row"][sp["column"]].strip().lower() for it in items}
    by_source = defaultdict(list)
    for it in items:
        by_source[it["groups"]["source"]].append(it)
    rng = np.random.default_rng(seed)
    if sp.get("mode") == "group":
        held = {str(v) for v in sp["test_values"]}
        return {it["id"]: "test" if str(it["groups"][sp["key"]]) in held else "train" for it in items}
    # stratified: recordings assigned whole; each class of `by` gets ~test_fraction of its recordings in test and
    # keeps at least one recording on each side whenever it has two or more
    by = sp.get("by") or next((t.name for t in targets if t.role == "secret" and t.kind == "multiclass"),
                              next(t.name for t in targets if t.role == "utility"))
    frac = sp.get("test_fraction", 0.3)
    groups = defaultdict(list)
    for src, its in by_source.items():
        key = its[0]["labels"].get(by)
        groups[key if isinstance(key, int) else -1].append(src)
    assign = {}
    for key, srcs in sorted(groups.items()):
        srcs = sorted(srcs)
        rng.shuffle(srcs)
        n_test = int(round(frac * len(srcs)))
        if len(srcs) >= 2:
            n_test = min(max(1, n_test), len(srcs) - 1)
        for i, s in enumerate(srcs):
            assign[s] = "test" if i < n_test else "train"
    return {it["id"]: assign[it["groups"]["source"]] for it in items}


def ingest(desc_path, out, workers=16, seed=0):
    desc = json.load(open(desc_path))
    base = os.path.dirname(os.path.abspath(desc_path))
    dec = {"short_side": 112, "max_long_side": 200, "fps": None, "gray": "auto", **desc.get("decode", {})}
    rows = read_rows(desc, base)
    assert rows, "no rows matched"
    targets, labels = build_targets(desc, rows)
    media = sorted({r["_path"] for r in rows})
    kinds = Counter(media_kind(p) for p in media)
    assert None not in kinds, f"unsupported media: {[p for p in media if media_kind(p) is None][:5]}"
    modality = desc.get("modality") or ("image" if set(kinds) == {"image"} else "video")
    name_of = {p: hashlib.sha1(p.encode()).hexdigest()[:16] for p in media}
    jobs = [(p, os.path.join(out, "frames", name_of[p] + ".npy"), dec) for p in media]
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        dec_out = {d["src"]: d for d in ex.map(decode_one, jobs, chunksize=4)}
    failed = {p: d["error"] for p, d in dec_out.items() if d["error"]}
    ok = {p: d for p, d in dec_out.items() if not d["error"]}
    assert ok, f"nothing decoded; first errors: {list(failed.items())[:3]}"
    colour_share = np.mean([d["colour"] for d in ok.values()])
    channels = 3 if dec["gray"] == "never" or (dec["gray"] == "auto" and colour_share > 0.05) else 1
    if channels == 1:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(to_gray, [d["out"] for d in ok.values()]))

    seg = desc.get("segments")
    group_cols = {"source": None, **desc.get("groups", {})}
    items, dropped = [], Counter()
    for i, r in enumerate(rows):
        p = r["_path"]
        if p not in ok:
            dropped["decode failed"] += 1
            continue
        n, fps = ok[p]["n"], ok[p]["fps"]
        start, end = 1, n
        if seg:
            a, b = float(r[seg["start"]]), float(r[seg["end"]])
            if seg.get("unit", "frame") == "second":
                assert fps, f"{p}: seconds need a known frame rate"
                a, b = a * fps + 1, b * fps
            start, end = max(1, int(round(a))), min(n, int(round(b)))
        if end < start:
            dropped["empty segment"] += 1
            continue
        if not labels.get(i):
            dropped["no labels"] += 1
            continue
        groups = {g: (os.path.relpath(p, base) if col is None else str(r.get(col, ""))) for g, col in group_cols.items()}
        items.append(dict(id=f"{name_of[p]}_{start}_{end}", frames=f"frames/{name_of[p]}.npy", start=start, end=end,
                          labels=labels[i], groups=groups, _row=r))
    dup_ids = [k for k, v in Counter(it["id"] for it in items).items() if v > 1]
    assert not dup_ids, f"duplicate items (same file and range): {dup_ids[:3]}"
    split = make_split(desc, items, targets, seed)
    for it in items:
        it["split"] = split[it["id"]]
    sizes = Counter((d["h"], d["w"]) for d in ok.values())
    fps_vals = [d["fps"] for d in ok.values() if d["fps"]]
    spec = S.Spec(name=desc["name"], modality=modality, size=list(sizes.most_common(1)[0][0]), channels=channels,
                  targets=targets, groups=list(group_cols), fps=float(np.median(fps_vals)) if fps_vals else None,
                  items=[{k: v for k, v in it.items() if k != "_row"} for it in items], meta=desc.get("meta", {}))
    warnings = spec.validate()
    report = audit(spec, out)
    report.update(source=desc_path, n_rows=len(rows), n_media=len(media), media_kinds=dict(kinds), decode=dec,
                  decode_failed=len(failed), decode_errors=dict(list(failed.items())[:20]), dropped=dict(dropped),
                  channels=channels, colour_share=float(colour_share), sizes={f"{h}x{w}": c for (h, w), c in sizes.items()},
                  warnings=warnings + report["warnings"])
    os.makedirs(out, exist_ok=True)
    spec.save(out)
    json.dump(report, open(os.path.join(out, "ingest_report.json"), "w"), indent=1)
    return spec, report


def audit(spec, root):
    """Label balance per split, coverage of secret classes on both sides, near-duplicate footage across the split."""
    rep, warnings = dict(n_items=len(spec.items), targets={}), []
    for t in spec.targets:
        per = {}
        for sp in ("train", "test"):
            ys = [it["labels"][t.name] for it in spec.items if it["split"] == sp and t.name in it["labels"]]
            if t.kind == "multiclass":
                c = np.bincount(ys, minlength=t.n) if ys else np.zeros(t.n, int)
                per[sp] = dict(n=len(ys), per_class_min=int(c.min()), per_class_max=int(c.max()), classes_present=int((c > 0).sum()))
            else:
                a = np.array(ys) if ys else np.zeros((0, t.n))
                per[sp] = dict(n=len(ys), positive_rate=[round(float(v), 3) for v in (a.mean(0) if len(a) else np.zeros(t.n))])
        rep["targets"][t.name] = per
        if t.kind == "multiclass" and t.role == "secret":
            missing = t.n - per["test"]["classes_present"]
            if per["train"]["classes_present"] < t.n:
                warnings.append(f"{t.name}: {t.n - per['train']['classes_present']} classes have no training items")
            if missing:
                warnings.append(f"{t.name}: {missing} classes have no test items")
            if t.n < 10:
                warnings.append(f"{t.name}: only {t.n} classes; identity among few people is easy to recover from body "
                                f"shape and motion alone, so expect low privacy scores for every obfuscator")
    # near duplicates: the same footage on both sides of the split (Barz & Denzler 2019 found 3-10% of CIFAR test
    # images duplicated in training). Appearance alone cannot decide it for static cameras: on KTH's gray, static
    # scenes, different recordings of one person hash alike. So appearance is only a gate (average hashes of three
    # frames within 6 of 64 bits); a pair is confirmed when the per-frame motion-energy traces of the two items (64
    # frames, thumbnails 32x24) correlate >= 0.99 at some lag within +-4 frames. Calibrated on KTH: the same footage
    # shifted 2-3 frames correlates 1.00 (5 of 5 planted copies found); among 1,793 other recordings the best match
    # is 0.95 for 99% of test items and 0.98 for short running clips (people crossing the frame alike).
    # Items too short or too still for a trace, and images, need a strict appearance match (<= 2 bits) instead.
    sigs = {}
    for it in spec.items:
        a = np.load(os.path.join(root, it["frames"]), mmap_mode="r")
        idx = np.linspace(it["start"] - 1, max(it["start"] - 1, it["end"] - 2), 3).astype(int)
        hashes = np.stack([ahash(np.asarray(a[i])) for i in idx])
        sigs[it["id"]] = (np.unpackbits(hashes, axis=-1), motion_trace(a, it["start"], it["end"]))
    tr = [it for it in spec.items if it["split"] == "train"]
    te = [it for it in spec.items if it["split"] == "test"]
    if tr and te:
        bits = np.stack([sigs[it["id"]][0] for it in tr])  # (Ntr, 3, 64)
        near = []
        for it in te:
            b, tb = sigs[it["id"]]
            d = (bits != b[None]).sum(-1).max(1)
            for j in np.flatnonzero(d <= 6):
                ta = sigs[tr[j]["id"]][1]
                if ta is not None and tb is not None:
                    c = trace_corr(ta, tb)
                    if c >= 0.99:
                        near.append((it["id"], tr[j]["id"], round(c, 3)))
                        break
                elif d[j] <= 2:
                    near.append((it["id"], tr[j]["id"], int(d[j])))
                    break
        rep["near_duplicates_across_split"] = len(near)
        rep["near_duplicate_examples"] = near[:20]
        if len(near) > 0.02 * len(te):
            warnings.append(f"{len(near)} test items have near-identical footage in training (e.g. {near[:2]}); "
                            "attackers can match footage instead of recognising the secret")
    rep["warnings"] = warnings
    return rep
