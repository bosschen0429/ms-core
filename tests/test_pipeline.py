"""Tests for MSPipeline orchestrator.

Focuses on the processing steps (4, 8, 10, 11) which are pure DataFrame
operations.  The ``TestCalibrationInMemory`` class validates the new
in-memory calibration path added to eliminate temp-file round-trips.
"""

import numpy as np
import pandas as pd
import pytest

from ms_core.dataset import MSDataset
from ms_core.pipeline import (
    MSPipeline,
    PipelineConfig,
    STEPS,
    TOTAL_STEPS,
    _dataset_to_calibration_df,
    _calibration_df_to_dataset,
)


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


# ======================================================================
# Calibration in-memory helpers and round-trip tests
# ======================================================================


class TestCalibrationInMemory:
    """Validate in-memory calibration helpers and module entry points."""

    @pytest.fixture
    def calibration_dataset(self):
        """MSDataset with m/z-rt style feature IDs and sample info."""
        np.random.seed(99)
        n_samples, n_features = 15, 30
        feature_ids = [f"{100 + i * 0.5:.4f}/{10 + i * 0.3:.1f}" for i in range(n_features)]
        sample_ids = [f"Sample_{i:02d}" for i in range(n_samples)]

        matrix = pd.DataFrame(
            np.random.exponential(scale=5000, size=(n_samples, n_features)),
            index=sample_ids,
            columns=feature_ids,
        )
        labels = pd.Series(
            ["Control"] * 7 + ["Exposed"] * 8,
            index=matrix.index,
            name="Group",
        )
        sample_info = pd.DataFrame(
            {
                "Sample_Name": sample_ids,
                "Sample_Type": (["Control"] * 5 + ["QC"] * 3 + ["Exposed"] * 5 + ["QC"] * 2),
                "Injection_Order": list(range(1, n_samples + 1)),
                "Batch": ["Batch1"] * 8 + ["Batch2"] * 7,
            },
            index=sample_ids,
        )
        return MSDataset(
            matrix=matrix,
            labels=labels,
            sample_info=sample_info,
        )

    # ------------------------------------------------------------------
    # Round-trip: MSDataset ↔ calibration DataFrame
    # ------------------------------------------------------------------

    def test_roundtrip_preserves_shape(self, calibration_dataset):
        """_dataset_to_calibration_df → _calibration_df_to_dataset preserves shape."""
        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)

        # cal_df should be features×samples with FeatureID first column
        assert cal_df.columns[0] == "FeatureID"
        assert cal_df.shape[0] == ds.n_features
        # sample columns = all columns except FeatureID
        assert len(cal_df.columns) - 1 == ds.n_samples

        # Convert back
        restored = _calibration_df_to_dataset(cal_df, ds, step_num=5)
        assert restored.matrix.shape == ds.matrix.shape
        # Values should match (within float tolerance)
        pd.testing.assert_frame_equal(
            restored.matrix.sort_index(axis=0).sort_index(axis=1),
            ds.matrix.sort_index(axis=0).sort_index(axis=1),
            atol=1e-10,
        )

    def test_roundtrip_preserves_labels(self, calibration_dataset):
        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)
        restored = _calibration_df_to_dataset(cal_df, ds, step_num=5)
        pd.testing.assert_series_equal(restored.labels, ds.labels)

    def test_calibration_df_orientation(self, calibration_dataset):
        """Calibration df should have features as rows, samples as columns."""
        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)

        # Row count = number of features
        assert cal_df.shape[0] == ds.n_features
        # First column contains original column names (feature IDs)
        assert set(cal_df["FeatureID"]) == set(ds.matrix.columns)
        # Remaining columns are sample IDs
        sample_cols = cal_df.columns[1:]
        assert set(sample_cols) == set(ds.matrix.index)

    # ------------------------------------------------------------------
    # ISTD process_in_memory
    # ------------------------------------------------------------------

    def test_istd_process_in_memory(self, calibration_dataset):
        """ISTD process_in_memory with explicitly marked ISTDs."""
        from ms_core.calibration.istd_correction import process_in_memory

        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)

        # Mark first 3 features as ISTD
        istd_ids = cal_df["FeatureID"].iloc[:3].tolist()

        result = process_in_memory(
            cal_df,
            ds.sample_info.copy(),
            istd_feature_ids=istd_ids,
        )

        assert result is not None
        assert "FeatureID" in result.columns
        # Result should have fewer features (ISTDs are removed from output)
        assert len(result) == ds.n_features - len(istd_ids)
        # Sample columns should be present
        sample_cols = [c for c in result.columns if c != "FeatureID"]
        assert len(sample_cols) > 0

    def test_istd_reads_feature_info_is_istd(self, calibration_dataset):
        """_step_istd_correction extracts ISTD IDs from feature_info.is_istd."""
        from unittest.mock import patch, MagicMock
        from ms_core.pipeline import _step_istd_correction

        ds = calibration_dataset
        # Populate feature_info with is_istd column (simulating ISTDMarker output)
        feature_ids = ds.matrix.columns.tolist()
        istd_mask = [True] * 3 + [False] * (len(feature_ids) - 3)
        ds.feature_info = pd.DataFrame(
            {"is_istd": istd_mask},
            index=feature_ids,
        )

        # Mock the calibration module so we can inspect what was passed
        mock_mod = MagicMock()
        mock_result_df = _dataset_to_calibration_df(ds)
        mock_mod.process_in_memory.return_value = mock_result_df

        with patch("importlib.import_module", return_value=mock_mod):
            _step_istd_correction(ds)

        # Verify istd_feature_ids was passed through
        call_kwargs = mock_mod.process_in_memory.call_args
        assert "istd_feature_ids" in call_kwargs.kwargs
        assert set(call_kwargs.kwargs["istd_feature_ids"]) == set(feature_ids[:3])

    def test_istd_returns_none_without_istd_info(self, calibration_dataset):
        """Without ISTD markers, should return None for fallback."""
        from ms_core.calibration.istd_correction import process_in_memory

        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)
        result = process_in_memory(cal_df, ds.sample_info.copy())
        assert result is None

    # ------------------------------------------------------------------
    # Batch effect process_in_memory
    # ------------------------------------------------------------------

    def test_batch_effect_single_batch_passthrough(self, calibration_dataset):
        """Single batch → returns data unchanged."""
        from ms_core.calibration.batch_effect import process_in_memory

        ds = calibration_dataset
        cal_df = _dataset_to_calibration_df(ds)

        # Force single batch
        single_batch_info = ds.sample_info.copy()
        single_batch_info["Batch"] = "Batch1"

        result = process_in_memory(cal_df, single_batch_info)
        assert result is not None
        # Should be the same DataFrame (single batch = no correction)
        pd.testing.assert_frame_equal(result, cal_df)

    # ------------------------------------------------------------------
    # Wrapper integration
    # ------------------------------------------------------------------

    def test_calibration_wrapper_uses_inmemory_path(self, calibration_dataset):
        """_calibration_wrapper dispatches to process_in_memory when available."""
        from unittest.mock import patch, MagicMock
        from ms_core.pipeline import _calibration_wrapper

        ds = calibration_dataset

        # Create a mock module with process_in_memory
        mock_mod = MagicMock()
        mock_result_df = _dataset_to_calibration_df(ds)  # pass-through
        mock_mod.process_in_memory.return_value = mock_result_df

        with patch("importlib.import_module", return_value=mock_mod):
            result = _calibration_wrapper(
                step_num=7,
                calibration_module="mock.module",
                ds=ds,
            )

        # process_in_memory should have been called
        mock_mod.process_in_memory.assert_called_once()
        # main() should NOT have been called (no fallback)
        mock_mod.main.assert_not_called()
        assert isinstance(result, MSDataset)
        assert any("[Step 7]" in msg for msg in result.processing_log)
