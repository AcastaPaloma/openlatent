import pandas as pd
import pytest

from openlatent.data.chestxray14 import _parse_age, check_splits, findings_matrix, load_metadata, make_splits

from conftest import write_nih_csv


def test_splits_are_patient_disjoint(splits):
    check_splits(splits)
    assert set(splits["split"]) == {"encoder_train", "encoder_val", "eval"}


def test_splits_ignore_row_order(meta, splits):
    shuffled = make_splits(meta.sample(frac=1.0, random_state=1), test_frac=0.5)
    a = splits.sort_values("image_id", ignore_index=True)
    b = shuffled.sort_values("image_id", ignore_index=True)
    pd.testing.assert_frame_equal(a, b)


def test_reid_roles(splits):
    eval_rows = splits[splits["split"] == "eval"]
    assert (splits.loc[splits["split"] != "eval", "reid_role"] == "").all()
    gallery = eval_rows[eval_rows["reid_role"] == "gallery"]
    query = eval_rows[eval_rows["reid_role"] == "query"]
    assert gallery["patient_id"].is_unique
    assert set(query["patient_id"]) <= set(gallery["patient_id"])
    # The gallery item is each patient's earliest follow-up.
    assert gallery["image_id"].str.endswith("_000.png").all()


def test_official_test_list(meta):
    test_patients = set(meta["patient_id"].unique()[:50])
    test_ids = list(meta.loc[meta["patient_id"].isin(test_patients), "image_id"])
    splits = make_splits(meta, test_ids=test_ids)
    assert set(splits.loc[splits["split"] == "eval", "image_id"]) == set(test_ids)

    leaky = test_ids + [meta.loc[~meta["patient_id"].isin(test_patients), "image_id"].iloc[0]]
    other = meta.loc[meta["image_id"] == leaky[-1], "patient_id"].iloc[0]
    one_of_theirs = meta.loc[meta["patient_id"] == other, "image_id"]
    if len(one_of_theirs) > 1:  # only leaky if that patient has images left outside the list
        with pytest.raises(ValueError, match="patient-disjoint"):
            make_splits(meta, test_ids=leaky)


def test_load_metadata_roundtrip(meta, tmp_path):
    path = tmp_path / "Data_Entry_2017_v2020.csv"
    write_nih_csv(meta, path)
    loaded = load_metadata(path)
    assert list(loaded["image_id"]) == list(meta["image_id"])
    assert (loaded["sex"] == meta["sex"]).all()


def test_parse_age():
    assert _parse_age("058Y") == 58
    assert _parse_age("024M") == 2
    assert _parse_age(45) == 45
    assert _parse_age(414) != _parse_age(414)  # implausible ages become NaN


def test_findings_matrix():
    m = findings_matrix(pd.Series(["Effusion|Hernia", "No Finding"]))
    assert m.shape == (2, 14)
    assert m[0].sum() == 2 and m[1].sum() == 0
