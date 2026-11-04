"""Provenance stamps for every analysis output (S.md amendment v1.3, closes E6(c)).

stamp(script) -> {evaluator: {version, sha256, expected_sha256, tag, frozen_match}, script, script_sha256}
add(obj, script)      adds it under the key "_provenance" (readers skip keys that start with "_")
md_line(script)       one line for markdown outputs
CLI: python -m s_suite.provenance FILE --script run_planted.py     writes FILE.provenance.json (sha256 of FILE + the stamp), for files whose
generating script cannot be edited (frozen) or that are line-oriented (jsonl).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from . import certificate, freeze

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stamp(script) -> dict:
    script = os.path.join(HERE, os.path.basename(script))
    return dict(evaluator=freeze.evaluator_block(certificate.VERSION), script=os.path.basename(script), script_sha256=_sha(script))


def add(obj: dict, script) -> dict:
    out = dict(obj)
    out["_provenance"] = stamp(script)
    return out


def md_line(script) -> str:
    s = stamp(script)
    e = s["evaluator"]
    return (f"<!-- provenance: evaluator sha256 {e['sha256']} (frozen match: {e['frozen_match']}, tag {e['tag']}), "
            f"script {s['script']} sha256 {s['script_sha256']} -->")


def sidecar(path, script):
    side = path + ".provenance.json"
    with open(side, "w") as f:
        json.dump(dict(file=os.path.basename(path), file_sha256=_sha(path), **stamp(script)), f, indent=1, sort_keys=True)
    return side


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--script":
        print(sidecar(sys.argv[1], sys.argv[3]))
    else:
        print(__doc__)
