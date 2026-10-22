import pandas as pd
import pytest

from openlatent.data.chestxray14 import make_splits
from openlatent.data.synthetic import fake_metadata


@pytest.fixture(scope="session")
def meta() -> pd.DataFrame:
    return fake_metadata(n_patients=300, seed=0)


@pytest.fixture(scope="session")
def splits(meta) -> pd.DataFrame:
    return make_splits(meta, test_frac=0.5)


def write_nih_csv(meta: pd.DataFrame, path) -> None:
    """Write normalized metadata back out with the v2020 NIH headers."""
    meta.rename(
        columns={
            "image_id": "Image Index",
            "findings": "Finding Labels",
            "followup": "Follow-up #",
            "patient_id": "Patient ID",
            "age": "Patient Age",
            "sex": "Patient Sex",
            "view": "View Position",
        }
    ).assign(**{"Patient Age": meta["age"].astype(int)}).to_csv(path, index=False)
