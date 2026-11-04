import json
import os
import subprocess

import numpy as np
import pytest

from s_suite import certificate as cf
from s_suite import freeze, ledger
from .test_certificate import build, decl


def test_certificate_prints_evaluator_block_and_status():
    val, test = build(0, [0.3, 0.2, 0.1])
    c = cf.certify(decl(3), val, test)
    ev = c["evaluator"]
    assert ev["sha256"] == freeze.evaluator_sha256() and set(ev) >= {"version", "sha256", "expected_sha256", "tag", "frozen_match"}
    assert c["split_status"] == "development"
    with pytest.raises(ValueError):
        cf.certify(decl(3), val, test, split_status="sealed")      # cannot claim 'sealed' without the ledger


def test_sealed_query_once_per_release(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "repo_root", lambda start=None: None)           # tmp dir is outside any repo
    val, test = build(1, [0.3, 0.2, 0.1])
    d = str(tmp_path / "ledger")
    c = cf.certify(decl(3), val, test, sealed_ledger_dir=d, release_digest="r" * 64, split_id="kth/test")
    assert c["split_status"] == "sealed" and c["ledger_entry"]["repeat"] is False
    with pytest.raises(ledger.LedgerError):                                       # same release, same split, other seed: refused
        cf.certify(cf.SuiteDecl("t", decl(3).attackers, n_boot=500, seed=9, n_classes=10), val, test, sealed_ledger_dir=d,
                   release_digest="r" * 64, split_id="kth/test")
    cf.certify(decl(3), val, test, sealed_ledger_dir=d, release_digest="s" * 64, split_id="kth/test")   # another release is fine
    with pytest.raises(ledger.LedgerError):                                       # an amendment id the custodian has not listed does nothing
        cf.certify(decl(3), val, test, sealed_ledger_dir=d, release_digest="r" * 64, split_id="kth/test", amendment="A1")
    json.dump([dict(id="A1", release_digest="r" * 64, split_id="kth/test", reason="declared re-run")], open(os.path.join(d, "amendments.json"), "w"))
    c3 = cf.certify(decl(3), val, test, sealed_ledger_dir=d, release_digest="r" * 64, split_id="kth/test", amendment="A1")
    assert c3["ledger_entry"]["repeat"] is True
    with pytest.raises(ledger.LedgerError):
        cf.certify(decl(3), val, test, sealed_ledger_dir=d, split_id="x")        # no release digest


def test_ledger_refuses_directory_inside_repo():
    root = ledger.repo_root()
    if root is None:
        pytest.skip("not a git checkout")
    with pytest.raises(ledger.LedgerError):
        ledger.record_sealed_query(os.path.join(root, "research", "adversarial", "_ledger_must_not_exist"), "d" * 64, "s")


def _repo(tmp_path):
    r = str(tmp_path / "repo")
    os.makedirs(r)
    run = lambda *a: subprocess.run(["git", *a], cwd=r, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    open(os.path.join(r, "a"), "w").write("1")
    run("add", "a")
    run("commit", "-qm", "c1")
    return r, run


def test_check_flags_prereg_not_ancestor(tmp_path, monkeypatch):
    r, run = _repo(tmp_path)
    monkeypatch.setattr(freeze, "load_freeze", lambda: dict(tag="prereg", evaluator_sha256=freeze.evaluator_sha256(), evaluator_files={}))
    d = str(tmp_path / "led")
    monkeypatch.setattr(ledger, "repo_root", lambda start=None: r)
    head0 = run("rev-parse", "HEAD")
    # a sealed number produced BEFORE the pre-registration commit exists
    e0 = ledger.record_sealed_query(d, "a" * 64, "s0")
    assert e0["git_head"] == head0
    open(os.path.join(r, "b"), "w").write("2")
    run("add", "b")
    run("commit", "-qm", "prereg")
    run("tag", "prereg")
    # a sealed number produced AFTER it
    open(os.path.join(r, "c"), "w").write("3")
    run("add", "c")
    run("commit", "-qm", "after")
    ledger.record_sealed_query(d, "b" * 64, "s1")
    bad = ledger.check(d, r)
    assert any("s0" in b and "ancestor" in b for b in bad) and not any(b.startswith("s1") for b in bad)


def test_freeze_check_detects_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "FREEZE_PATH", str(tmp_path / "FREEZE.json"))   # never touch the real record
    freeze.write_freeze("tmp-tag")
    assert freeze.check() == []
    monkeypatch.setattr(freeze, "file_sha", lambda n: "0" * 64)
    assert freeze.check()
