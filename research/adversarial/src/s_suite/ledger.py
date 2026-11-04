"""Sealed-query ledger (audit F9 / FOCUS A3).

Rules enforced here:
  * a sealed split of a given RELEASE (digest of the released representation, not of the attacker's predictions) may be queried once; a
    second query for the same (release_digest, split_id) is refused unless the custodian has listed a declared amendment for it in
    amendments.json in the ledger directory;
  * every entry stores the evaluator sha256, the git HEAD and whether the work tree was dirty, so `check` can verify afterwards that the
    pre-registration tag is an ancestor of the commit that produced the first sealed number and that the evaluator was the frozen one;
  * the ledger directory must not live inside the git checkout (builder-writable).

Limit, stated plainly: a process that runs as the builder can still write wherever the builder can write. This module enforces the protocol and
makes violations detectable; real separation needs the ledger directory to be owned by the sealed-split custodian (another user or host).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from . import freeze


class LedgerError(RuntimeError):
    pass


def _git(args, cwd):
    try:
        r = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def repo_root(start=None):
    return _git(["rev-parse", "--show-toplevel"], start or freeze.HERE)


def _entries(ledger_dir):
    p = os.path.join(ledger_dir, "ledger.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def _amendments(ledger_dir):
    p = os.path.join(ledger_dir, "amendments.json")
    return json.load(open(p)) if os.path.exists(p) else []


def record_sealed_query(ledger_dir, release_digest, split_id, amendment=None, allow_inside_repo=False):
    if not release_digest or not split_id:
        raise LedgerError("a sealed query needs a release_digest and a split_id")
    ledger_dir = os.path.realpath(ledger_dir)
    root = repo_root()
    if root and not allow_inside_repo and (ledger_dir + os.sep).startswith(os.path.realpath(root) + os.sep):
        raise LedgerError("ledger directory is inside the git checkout (builder-writable); use a custodian-owned directory")
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "ledger.jsonl")
    with open(path, "a+") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        prior = [json.loads(x) for x in f if x.strip()]
        dup = [e for e in prior if e["release_digest"] == release_digest and e["split_id"] == split_id]
        if dup:
            ok = amendment and any(a.get("id") == amendment and a.get("release_digest") == release_digest and a.get("split_id") == split_id
                                   for a in _amendments(ledger_dir))
            if not ok:
                raise LedgerError(f"sealed split {split_id} of release {release_digest[:12]} was already queried; a declared amendment is required")
        rec = freeze.load_freeze()
        entry = dict(release_digest=release_digest, split_id=split_id, evaluator_sha256=freeze.evaluator_sha256(), tag=rec.get("tag"),
                     git_head=_git(["rev-parse", "HEAD"], root) if root else None,
                     git_dirty=bool(_git(["status", "--porcelain", "--untracked-files=no"], root)) if root else None,
                     utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), amendment=amendment, repeat=bool(dup))
        f.seek(0, 2)
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def check(ledger_dir, repo=None):
    """Return a list of violations (empty = clean): prereg tag not an ancestor of the producing commit, dirty tree, evaluator differs from the
    frozen one, repeated sealed query without an amendment."""
    bad, rec, root = [], freeze.load_freeze(), repo or repo_root()
    tag = rec.get("tag")
    tag_commit = _git(["rev-parse", f"{tag}^{{commit}}"], root) if tag and root else None
    if not tag_commit:
        bad.append(f"pre-registration tag {tag!r} not found in the repository")
    seen = {}
    for e in _entries(ledger_dir):
        key = (e["release_digest"], e["split_id"])
        if e.get("git_head") and tag_commit:
            r = subprocess.run(["git", "merge-base", "--is-ancestor", tag_commit, e["git_head"]], cwd=root, capture_output=True)
            if r.returncode != 0:
                bad.append(f"{key[1]}: prereg tag commit is not an ancestor of {e['git_head'][:10]}")
        elif not e.get("git_head"):
            bad.append(f"{key[1]}: entry has no git HEAD")
        if e.get("git_dirty"):
            bad.append(f"{key[1]}: produced from a dirty work tree")
        if e.get("evaluator_sha256") != rec.get("evaluator_sha256") and not e.get("amendment"):
            bad.append(f"{key[1]}: evaluator sha256 differs from the frozen one with no declared amendment")
        if key in seen and not e.get("amendment"):
            bad.append(f"{key[1]}: repeated sealed query without an amendment")
        seen[key] = True
    return bad


if __name__ == "__main__":
    import sys
    problems = check(sys.argv[1])
    print("\n".join(problems) if problems else "ledger clean")
    sys.exit(1 if problems else 0)
