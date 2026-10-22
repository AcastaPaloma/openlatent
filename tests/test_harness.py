import json

import numpy as np
import pytest

from openlatent.cli import main
from openlatent.data.synthetic import fake_features
from openlatent.features import save_features
from openlatent.harness import compare_reports, cross_dependence, run_report

from conftest import write_nih_csv


@pytest.fixture(scope="module")
def ids(meta):
    return list(meta["image_id"])


@pytest.fixture(scope="module")
def raw(meta):
    return fake_features(meta, identity=1.0, seed=0)


@pytest.fixture(scope="module")
def private(meta):
    return fake_features(meta, identity=0.0, seed=0)


def test_report_is_deterministic(raw, ids, meta, splits):
    a = run_report(raw, ids, meta, splits)
    b = run_report(raw.copy(), list(ids), meta.copy(), splits.copy())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_report_ignores_feature_row_order(raw, ids, meta, splits):
    perm = np.random.default_rng(3).permutation(len(ids))
    a = run_report(raw, ids, meta, splits)
    b = run_report(raw[perm], [ids[i] for i in perm], meta, splits)
    for section in ("reid", "sex_probe", "age_probe", "findings_probe", "identity_alignment"):
        assert a[section] == b[section]


def test_missing_eval_features_raise(raw, ids, meta, splits):
    with pytest.raises(ValueError, match="missing"):
        run_report(raw[:10], ids[:10], meta, splits)


def test_before_after_comparison(raw, private, ids, meta, splits):
    before = run_report(raw, ids, meta, splits)
    after = run_report(private, ids, meta, splits)
    cmp = compare_reports(before, after)
    assert before["reid"]["top1"] > 0.9
    assert cmp["privacy"]["reid.top1"]["leakage_removed"] > 0.9
    # Sex and age are still encoded in `private`, so their leakage is not removed.
    assert cmp["privacy"]["sex_probe.auc"]["leakage_removed"] < 0.2

    dep = cross_dependence(raw, ids, private, ids, meta, splits)
    assert 0.0 <= dep["linear_cka"] <= 1.0 and dep["n"] > 0


def test_compare_rejects_mismatched_configs(raw, ids, meta, splits):
    a = run_report(raw, ids, meta, splits)
    b = json.loads(json.dumps(a))
    b["config"]["seed"] = 1
    with pytest.raises(ValueError, match="config"):
        compare_reports(a, b)


def test_cli_end_to_end(raw, private, ids, meta, tmp_path, capsys):
    csv = tmp_path / "Data_Entry_2017_v2020.csv"
    write_nih_csv(meta, csv)
    save_features(tmp_path / "x.npy", raw, ids)
    save_features(tmp_path / "z.npy", private, ids)
    main(["splits", "--meta", str(csv), "--out", str(tmp_path / "splits.csv")])
    for name in ("x", "z"):
        main(["score", "--meta", str(csv), "--splits", str(tmp_path / "splits.csv"),
              "--features", str(tmp_path / f"{name}.npy"), "--out", str(tmp_path / f"{name}.json")])
    main(["compare", "--before", str(tmp_path / "x.json"), "--after", str(tmp_path / "z.json"),
          "--x-features", str(tmp_path / "x.npy"), "--z-features", str(tmp_path / "z.npy"),
          "--meta", str(csv), "--splits", str(tmp_path / "splits.csv"), "--out", str(tmp_path / "cmp.json")])
    out = capsys.readouterr().out
    assert "reid.top1" in out and "linear CKA" in out
    assert json.loads((tmp_path / "cmp.json").read_text())["privacy"]["reid.top1"]["leakage_removed"] > 0.5
