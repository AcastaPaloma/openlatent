"""Evaluator freeze (audit F9 / FOCUS A3).

The evaluator is the set of files in EVALUATOR_FILES. FREEZE.json (next to this file, excluded from the hash) records the git tag, the sha256 of
the evaluator and of every analysis script at the moment the pre-registration was committed. Every certificate prints the CURRENT sha256, the
EXPECTED one from FREEZE.json and whether they match; a certificate whose evaluator differs from the frozen one says so on its face.

    python -m s_suite.freeze --write s-suite-eval-v1     (re-hash and write FREEZE.json; done once, at the pre-registration commit)
    python -m s_suite.freeze --check                     (exit 1 if the evaluator or an analysis script differs from FREEZE.json)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVALUATOR_FILES = ("stats.py", "certificate.py", "selection_bias.py", "evidence_budget.py", "harness_io.py", "locked_eval.py", "ledger.py", "freeze.py")
ANALYSIS_FILES = ("rescore.py", "run_real.py", "run_planted.py", "planted.py", "attackers.py", "oracle.py")
FREEZE_PATH = os.path.join(HERE, "FREEZE.json")


def file_sha(name: str) -> str:
    with open(os.path.join(HERE, name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluator_sha256() -> str:
    h = hashlib.sha256()
    for n in EVALUATOR_FILES:
        h.update(f"{n}:{file_sha(n)}\n".encode())
    return h.hexdigest()


def load_freeze() -> dict:
    try:
        with open(FREEZE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def evaluator_block(version: str) -> dict:
    cur, rec = evaluator_sha256(), load_freeze()
    exp = rec.get("evaluator_sha256")
    return dict(version=version, sha256=cur, expected_sha256=exp, tag=rec.get("tag"), frozen_match=bool(exp is not None and exp == cur))


def write_freeze(tag: str):
    rec = dict(tag=tag, evaluator_sha256=evaluator_sha256(), evaluator_files={n: file_sha(n) for n in EVALUATOR_FILES},
               analysis_files={n: file_sha(n) for n in ANALYSIS_FILES if os.path.exists(os.path.join(HERE, n))})
    with open(FREEZE_PATH, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def check() -> list:
    rec, bad = load_freeze(), []
    if not rec:
        return ["FREEZE.json missing"]
    if rec.get("evaluator_sha256") != evaluator_sha256():
        bad.append("evaluator sha256 differs from FREEZE.json: " + ", ".join(n for n in EVALUATOR_FILES if rec["evaluator_files"].get(n) != file_sha(n)))
    for n, s in rec.get("analysis_files", {}).items():
        if file_sha(n) != s:
            bad.append(f"analysis script {n} differs from FREEZE.json")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        print(json.dumps(write_freeze(sys.argv[2]), indent=1))
    elif sys.argv[1:] == ["--check"]:
        problems = check()
        print("\n".join(problems) if problems else "evaluator and analysis scripts match FREEZE.json")
        sys.exit(1 if problems else 0)
    else:
        print(__doc__)
