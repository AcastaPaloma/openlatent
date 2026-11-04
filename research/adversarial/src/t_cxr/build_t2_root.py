"""Assemble the T2 aggregation root from the per-job output directories on the VM (symlinks/copies, no recomputation).
For every release dir that has `preds_f1_match_s0.json` the F1 entry used by the envelope is the SINGLE-seed `f1_match_s0` (A4 iv); mismatch / anon2 ensembles are dropped.
   python -m t_cxr.build_t2_root --runs ~/work/runs --out ~/work/t_agg_t2"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = 0
    for job in sorted(Path(a.runs).glob("t_*")):
        for d in job.iterdir():
            if d.is_dir() and ("_rot" in d.name) and (d / "summary.json").exists() or (d.is_dir() and d.name.startswith("labels14_rot")):
                dst = out / d.name
                if dst.exists():
                    continue
                shutil.copytree(d, dst)
                if (dst / "preds_f1_match_s0.json").exists():
                    shutil.copy(dst / "preds_f1_match_s0.json", dst / "preds_f1_match_ens.json")
                    for f in list(dst.glob("preds_f1_mismatch*")) + list(dst.glob("preds_f1_anon2*")) + list(dst.glob("preds_f1_match_s[1-9].json")):
                        f.unlink()
                n += 1
            elif d.is_file() and d.name.startswith("utility_") and d.suffix == ".json":
                if not (out / d.name).exists():
                    shutil.copy(d, out / d.name)
    print("assembled", n, "release dirs in", out)


if __name__ == "__main__":
    main()
