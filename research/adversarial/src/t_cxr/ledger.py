"""Sealed-query ledger (Auditor-2 F9): one sealed look per (release, config) digest.  The ledger lives OUTSIDE the repo and OUTSIDE $OUT
(default ~/work/t_ledger/sealed_queries.jsonl) so a job cannot rewrite it.  A second sealed query for the same digest is refused unless the
environment names a declared amendment (T_AMEND=<id> that appears in T.md section 6)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

LEDGER = Path(os.environ.get("T_LEDGER", str(Path.home() / "work" / "t_ledger" / "sealed_queries.jsonl")))


def evaluator_sha(src_dir: str | Path | None = None) -> str:
    """sha256 over the evaluator sources (t_cxr/*.py) so every result.json names the exact code that produced it."""
    d = Path(src_dir) if src_dir else Path(__file__).parent
    h = hashlib.sha256()
    for f in sorted(d.glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def digest(release: str, config: dict) -> str:
    return hashlib.sha256((release + json.dumps(config, sort_keys=True)).encode()).hexdigest()[:16]


def sealed_query(release: str, config: dict, role: str, rot: int, ledger: Path = LEDGER) -> dict:
    """Record a sealed look at role T.  Raises if the same digest was already queried and no amendment is declared."""
    dg = digest(release, dict(config, role=role, rot=rot))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    amend = os.environ.get("T_AMEND", "")
    if any(r["digest"] == dg for r in seen) and not amend:
        raise PermissionError(f"sealed split already queried for {release} ({dg}); declare an amendment (T_AMEND=...) in T.md first")
    rec = dict(digest=dg, release=release, role=role, rot=rot, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), amend=amend, evaluator=evaluator_sha()[:16], look=len(seen) + 1)
    with open(ledger, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
