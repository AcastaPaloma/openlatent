"""Command line: openlatent {splits,pixels,score,compare}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from openlatent.data.chestxray14 import check_splits, load_metadata, make_splits, read_id_list
from openlatent.features import load_features, pixel_features, save_features
from openlatent.harness import HarnessConfig, compare_reports, cross_dependence, run_report


def _write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _read_splits(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)


def cmd_splits(args: argparse.Namespace) -> None:
    meta = load_metadata(args.meta)
    test_ids = read_id_list(args.test_list) if args.test_list else None
    splits = make_splits(meta, test_ids=test_ids, val_frac=args.val_frac, salt=args.salt)
    check_splits(splits)
    splits.to_csv(args.out, index=False)
    counts = splits.groupby("split").agg(images=("image_id", "size"), patients=("patient_id", "nunique"))
    print(counts.to_string())
    print(splits["reid_role"].replace("", "-").value_counts().to_string())


def cmd_pixels(args: argparse.Namespace) -> None:
    splits = _read_splits(args.splits)
    rows = splits if args.all else splits[splits["split"] == "eval"]
    ids = sorted(rows["image_id"])
    save_features(args.out, pixel_features(args.images, ids, args.size), ids)
    print(f"wrote {len(ids)} x {args.size * args.size} pixel features to {args.out}")


def cmd_score(args: argparse.Namespace) -> None:
    features, ids = load_features(args.features, args.ids)
    report = run_report(features, ids, load_metadata(args.meta), _read_splits(args.splits), HarnessConfig(seed=args.seed))
    _write_json(args.out, report)
    r = report["reid"]
    print(f"re-ID top1 {r['top1']:.4f} (chance {r['chance_top1']:.4f})  verif AUC {r['verification_auc']:.4f}")
    print(f"sex AUC {report['sex_probe']['auc']}  age R2 {report['age_probe']['r2']:.4f}")
    print(f"findings mean AUC {report['findings_probe']['mean_auc']}")


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    result = compare_reports(before, after)
    if args.x_features and args.z_features:
        x, x_ids = load_features(args.x_features)
        z, z_ids = load_features(args.z_features)
        cfg = HarnessConfig(seed=before["config"]["seed"])
        result["x_vs_z"] = cross_dependence(x, x_ids, z, z_ids, load_metadata(args.meta), _read_splits(args.splits), cfg)
    if args.out:
        _write_json(args.out, result)
    fmt = lambda v: "   n/a" if v is None else f"{v:6.3f}"  # noqa: E731
    print(f"{'privacy metric':32} before  after chance removed")
    for name, m in result["privacy"].items():
        print(f"{name:32} {fmt(m['before'])} {fmt(m['after'])} {fmt(m['chance'])} {fmt(m['leakage_removed'])}")
    print(f"{'utility metric':32} before  after chance   kept")
    for name, m in result["utility"].items():
        print(f"{name:32} {fmt(m['before'])} {fmt(m['after'])} {fmt(m['chance'])} {fmt(m['utility_kept'])}")
    if "x_vs_z" in result:
        d = result["x_vs_z"]
        print(f"x vs z: linear CKA {d['linear_cka']:.4f}  dCor {d['distance_correlation']:.4f}  (n={d['n']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="openlatent")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("splits", help="patient-disjoint splits and eval roles from Data_Entry_2017*.csv")
    p.add_argument("--meta", required=True)
    p.add_argument("--test-list", help="official test_list.txt; otherwise eval patients are chosen by hash")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--salt", default="openlatent-v1")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("pixels", help="raw-pixel baseline features")
    p.add_argument("--splits", required=True)
    p.add_argument("--images", required=True, help="directory containing the PNGs (searched recursively)")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--all", action="store_true", help="every image, not just the eval split")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pixels)

    p = sub.add_parser("score", help="run every deterministic scorer on one feature file")
    p.add_argument("--meta", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--features", required=True, help=".npy with a sibling .ids.txt")
    p.add_argument("--ids", help="override the .ids.txt path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="leakage removed and utility kept between two reports")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--x-features", help="with --z-features, --meta and --splits: add CKA and dCor between x and z")
    p.add_argument("--z-features")
    p.add_argument("--meta")
    p.add_argument("--splits")
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
