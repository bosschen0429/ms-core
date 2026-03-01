"""Basic tests for MSDataset."""

import pandas as pd
import pytest

from ms_core.dataset import MSDataset


@pytest.fixture
def sample_dataset():
    matrix = pd.DataFrame(
        {"feat_1": [100.0, 200.0, 150.0], "feat_2": [300.0, 400.0, 350.0]},
        index=["S1", "S2", "QC1"],
    )
    labels = pd.Series(["Control", "Exposed", "QC"], index=matrix.index, name="Group")
    return MSDataset(matrix=matrix, labels=labels)


def test_properties(sample_dataset):
    ds = sample_dataset
    assert ds.n_samples == 3
    assert ds.n_features == 2
    assert list(ds.sample_ids) == ["S1", "S2", "QC1"]
    assert list(ds.feature_ids) == ["feat_1", "feat_2"]
    assert ds.groups == ["Control", "Exposed", "QC"]


def test_replace_matrix(sample_dataset):
    ds = sample_dataset
    new_matrix = ds.matrix.iloc[:2, :]  # drop QC1
    ds2 = ds.replace_matrix(new_matrix, msg="Dropped QC")
    assert ds2.n_samples == 2
    assert "QC1" not in ds2.sample_ids
    assert ds2.processing_log[-1] == "Dropped QC"
    # Original unchanged
    assert ds.n_samples == 3
    assert len(ds.processing_log) == 0


def test_copy_is_independent(sample_dataset):
    ds = sample_dataset
    ds2 = ds.copy()
    ds2.matrix.iloc[0, 0] = -999.0
    ds2.log("modified copy")
    assert ds.matrix.iloc[0, 0] == 100.0
    assert len(ds.processing_log) == 0
