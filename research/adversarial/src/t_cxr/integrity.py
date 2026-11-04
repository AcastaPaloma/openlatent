"""Independent recomputation of the bank integrity (T.md A2).  CPU, single thread; reads zip central directories (names), the bank, and a small random sample
of PNGs for a byte-exact re-decode comparison.  `ok` of record = names_ok and bank_ok; it is recomputed, never set by hand.
Outputs: integrity.json and exclude_dups.json (the exact-duplicate groups and the D1 / D2 classification of T.md A2 b).
   python -m t_cxr.integrity --data DIR --zips DIR --bank cxr256.npy --prep-json prep.json --salt-file F --pri DIR --out DIR"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--zips", required=True); ap.add_argument("--bank", required=True); ap.add_argument("--prep-json", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sample", type=int, default=600)
    a = ap.parse_args()
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt")
    csv_ids = set(d["img"]); N = len(d)
    # ---- (1) names: classify every zip entry, exact set equality
    cat = collections.Counter(); valid = collections.Counter(); where = {}
    for k in range(1, 13):
        p = Path(a.zips) / f"images_{k:03d}.zip"
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                base = Path(n).name
                if n.endswith("/"):
                    cat["directory"] += 1
                elif n.startswith("__MACOSX/") or base.startswith("._"):
                    cat["macosx_resource_fork"] += 1; cat["macosx_resource_fork_png"] += int(base.endswith(".png"))
                elif base.endswith(".png") and base in csv_ids:
                    cat["real_png_in_csv"] += 1; valid[base] += 1; where[base] = (p.name, n)
                else:
                    cat["unexpected"] += 1
    names_ok = bool(set(valid) == csv_ids and all(v == 1 for v in valid.values()) and cat["unexpected"] == 0)
    rep = dict(entry_categories=dict(cat), ids_in_zips=len(valid), ids_in_csv=N, missing=sorted(csv_ids - set(valid))[:20], extras=0, each_id_once=all(v == 1 for v in valid.values()), names_ok=names_ok)
    # ---- (2) bank: no all-zero row, byte-exact re-decode of a random sample
    bank = np.load(a.bank, mmap_mode="r")
    assert bank.shape == (N, 256, 256), bank.shape
    h = [hashlib.sha256(bank[i].tobytes()).hexdigest() for i in range(N)]
    zero_rows = int(sum(1 for i in range(N) if not bank[i].any()))
    rng = np.random.default_rng(20261104)
    sample = rng.choice(N, size=min(a.sample, N), replace=False)
    by_zip = collections.defaultdict(list)
    for i in sample:
        by_zip[where[d["img"].iat[i]][0]].append(i)
    mism = 0
    for zname, rows in by_zip.items():
        with zipfile.ZipFile(Path(a.zips) / zname) as z:
            for i in rows:
                im = Image.open(io.BytesIO(z.read(where[d["img"].iat[i]][1])))
                assert im.size == (1024, 1024)
                mism += int(not np.array_equal(np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR)), bank[i]))
    prep = json.loads(Path(a.prep_json).read_text())
    rep.update(zero_rows=zero_rows, sample_n=int(len(sample)), sample_mismatches=mism, prep_bad_dims=prep["bad_dims"], prep_filled=prep["filled"],
               per_zip_sha256={k: v["sha256"] for k, v in prep["zips"].items()}, per_zip_images={k: v.get("images") for k, v in prep["zips"].items()})
    bank_ok = bool(zero_rows == 0 and mism == 0 and prep["bad_dims"] == 0 and prep["filled"] == N)
    # ---- (3) exact duplicates at 256 px, by patient and by role
    grp = collections.defaultdict(list)
    for i, x in enumerate(h):
        grp[x].append(i)
    groups = [v for v in grp.values() if len(v) > 1]
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d)
    salt = Path(a.salt_file).read_text().strip()
    roles = {rot: meta.assign_roles(d, seen, salt, rot)["role"].to_numpy() for rot in (0, 1, 2)}
    out_groups, d2 = [], []
    for g in groups:
        pids = d["pid"].iloc[g].tolist(); imgs = d["img"].iloc[g].tolist()
        cross_patient = len(set(pids)) > 1
        cross_role = {rot: len(set(roles[rot][g])) > 1 for rot in roles}
        out_groups.append(dict(images=imgs, patients=[int(p) for p in pids], cross_patient=cross_patient, roles_rot0=roles[0][g].tolist(), cross_role_any_rotation=bool(any(cross_role.values())),
                               class_=("D2" if cross_patient else "D1")))
        if cross_patient:
            d2 += imgs
    dup = dict(n_groups=len(groups), n_images_in_groups=int(sum(len(g) for g in groups)), n_cross_patient_groups=int(sum(g["cross_patient"] for g in out_groups)),
               n_cross_role_groups=int(sum(g["cross_role_any_rotation"] for g in out_groups)), groups=out_groups)
    rep.update(duplicates={k: v for k, v in dup.items() if k != "groups"}, bank_ok=bank_ok, ok=bool(names_ok and bank_ok))
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "integrity.json").write_text(json.dumps(rep, indent=1))
    (o / "exclude_dups.json").write_text(json.dumps(dict(rule="T.md A2(b): D2 groups (identical 256-px arrays across patients) excluded entirely; D1 groups (same patient) only never form a positive pair",
                                                         D2_exclude=sorted(set(d2)), D1_groups=[g["images"] for g in out_groups if g["class_"] == "D1"], groups=out_groups), indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_zip_sha256", "per_zip_images")}, indent=1))
    print("D1/D2:", [(g["class_"], g["images"], g["patients"], g["roles_rot0"]) for g in out_groups])


if __name__ == "__main__":
    main()
