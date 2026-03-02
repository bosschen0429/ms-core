"""Tests for MSPipeline orchestrator.

Focuses on the processing steps (4, 8, 10, 11) which are pure DataFrame
operations. Calibration steps (5-7, 9) require file I/O and are tested
separately.
"""

import numpy as np
import pandas as pd
import pytest

from ms_core.dataset import MSDataset
from ms_core.pipeline import MSPipeline, PipelineConfig, STEPS, TOTAL_STEPS


@pytest.fixture
def sample_dataset():
    """A small dataset with some zeros and varying variance."""
    np.random.seed(42)
    n_samples, n_features = 20, 50
    matrix = pd.DataFrame(
        np.random.exponential(scale=1000, size=(n_samples, n_features)),
        index=[f"S{i:02d}" for i in range(n_samples)],
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Inject some zeros (will become NaN in step 4)
    matrix.iloc[0, 0] = 0
    matrix.iloc[1, 1] = 0
    matrix.iloc[2, 2] = 0

    labels = pd.Series(
        ["Control"] * 10 + ["Exposed"] * 10,
        index=matrix.index,
        name="Group",
    )
    return MSDataset(matrix=matrix, labels=labels)


class TestStepRegistry:
    def test_all_steps_defined(self):
        assert len(STEPS) == TOTAL_STEPS

    def test_step_numbers_sequential(self):
        assert list(STEPS.keys()) == list(range(1, TOTAL_STEPS + 1))

    def test_list_steps(self):
        steps = MSPipeline.list_steps()
        assert len(steps) == TOTAL_STEPS
        assert steps[0].number == 1
        assert steps[-1].number == 11


class TestStep4MissingValue:
    def test_zeros_replaced_and_imputed(self, sample_dataset):
        pipeline = MSPipeline()
        ds = pipeline.run_step(4, sample_dataset, impute_method="min")

        # No NaN or zeros should remain
        assert not ds.matrix.isna().any().any()
        assert (ds.matrix != 0).all().all()

        # Log should mention Step 4
        assert any("[Step 4]" in msg for msg in ds.processing_log)

    def test_missing_percent_removal(self, sample_dataset):
        ds = sample_dataset
        # Make one feature 60% missing
        ds.matrix.iloc[:12, 3] = 0

        pipeline = MSPipeline()
        result = pipeline.run_step(4, ds, missing_threshold=0.5)
        assert "feat_3" not in result.matrix.columns

    def test_snapshot_created(self, sample_dataset):
        pipeline = MSPipeline()
        pipeline.run_step(4, sample_dataset)
        assert 4 in pipeline.snapshots
        assert pipeline.current_step == 4


class TestStep8FeatureFilter:
    def test_filters_low_variance(self, sample_dataset):
        """Explicit cutoff=0.20 removes the bottom 20% by IQR."""
        ds = sample_dataset
        # Make 5 features constant (zero IQR)
        for i in range(5):
            ds.matrix[f"feat_{i}"] = 1.0

        pipeline = MSPipeline()
        # Force 20% cutoff so constant features (IQR=0) are below threshold
        result = pipeline.run_step(8, ds, filter_method="iqr", filter_cutoff=0.20)

        assert result.n_features < ds.n_features
        # All constant features should be gone
        for i in range(5):
            assert f"feat_{i}" not in result.matrix.columns
        assert any("[Step 8]" in msg for msg in result.processing_log)


class TestStep10Transform:
    def test_glog_transform(self, sample_dataset):
        pipeline = MSPipeline()
        ds = pipeline.run_step(10, sample_dataset, transform_method="LogNorm")
        # Transformed values should be much smaller
        assert ds.matrix.max().max() < sample_dataset.matrix.max().max()
        assert any("[Step 10]" in msg for msg in ds.processing_log)

    def test_skip_none(self, sample_dataset):
        pipeline = MSPipeline()
        ds = pipeline.run_step(10, sample_dataset, transform_method="None")
        pd.testing.assert_frame_equal(ds.matrix, sample_dataset.matrix)


class TestStep11Scaling:
    def test_auto_scale(self, sample_dataset):
        pipeline = MSPipeline()
        ds = pipeline.run_step(11, sample_dataset, scale_method="AutoNorm")
        # Mean should be ~0 for each feature
        means = ds.matrix.mean()
        assert (means.abs() < 1e-10).all()
        assert any("[Step 11]" in msg for msg in ds.processing_log)

    def test_skip_none(self, sample_dataset):
        pipeline = MSPipeline()
        ds = pipeline.run_step(11, sample_dataset, scale_method="None")
        pd.testing.assert_frame_equal(ds.matrix, sample_dataset.matrix)


class TestUndo:
    def test_undo_restores_snapshot(self, sample_dataset):
        pipeline = MSPipeline()
        ds1 = pipeline.run_step(4, sample_dataset)
        ds2 = pipeline.run_step(10, ds1)

        # Undo step 10 → back to post-step-4 state
        restored = pipeline.undo(10)
        pd.testing.assert_frame_equal(restored.matrix, ds1.matrix)

    def test_undo_missing_step_raises(self, sample_dataset):
        pipeline = MSPipeline()
        with pytest.raises(KeyError, match="No snapshot for step 7"):
            pipeline.undo(7)


class TestRunAll:
    def test_skip_steps(self, sample_dataset):
        config = PipelineConfig(
            skip_steps={1, 2, 3, 5, 6, 7, 9},  # Only run 4, 8, 10, 11
            transform_method="LogNorm",
            scale_method="AutoNorm",
        )
        pipeline = MSPipeline()
        ds = pipeline.run_all(sample_dataset, config)

        # Should have processed through steps 4, 8, 10, 11
        assert pipeline.current_step == 11
        assert ds.n_features <= sample_dataset.n_features

        # Log should mention skipped steps
        assert any("skipped" in msg for msg in ds.processing_log)

    def test_invalid_step_raises(self, sample_dataset):
        pipeline = MSPipeline()
        with pytest.raises(ValueError, match="Invalid step"):
            pipeline.run_step(99, sample_dataset)
