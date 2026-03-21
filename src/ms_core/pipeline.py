"""Unified 12-step pipeline orchestrator.

Pipeline order (NEVER reorder):
    1.  Data organization         (preprocessing)
    2.  Duplicate signal removal  (preprocessing)
    3.  MS quality filter         (preprocessing)  — ratio-based, per-group
    4.  Missing value imputation  (processing)     — 8 methods
    5.  ISTD marking + correction (preprocessing + calibration)
    6.  QC-LOWESS trend correction(calibration)
    7.  Batch effect ComBat       (calibration)
    8.  Feature filtering IQR/RSD (processing)     — overall variance
    9.  PQN normalization         (calibration)     — smart QC assessment
    10. Generalized log transform (processing)     — glog, NOT log2(x+1)
    11. Scaling                   (processing)     — MeanCenter/Auto/Pareto/Range
    12. Statistical analysis      (analysis + visualization) — not in pipeline

Usage::

    from ms_core.pipeline import MSPipeline, PipelineConfig

    pipeline = MSPipeline()
    ds = pipeline.run_step(4, dataset, impute_method="min")
    ds = pipeline.run_step(10, ds, transform_method="LogNorm")

    # Or run all at once:
    config = PipelineConfig(skip_steps={1, 2, 3, 5})
    ds = pipeline.run_all(dataset, config)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ms_core.dataset import MSDataset

logger = logging.getLogger(__name__)

# ======================================================================
# Configuration
# ======================================================================


@dataclass
class PipelineConfig:
    """All parameters for the 12-step pipeline.

    Steps in ``skip_steps`` will be silently skipped by ``run_all()``.
    """

    # Step 1: Data organization
    # (no user-facing params — handled by DataOrganizer config)

    # Step 2: Duplicate removal
    mz_tolerance_ppm: float = 20.0
    rt_tolerance: float = 1.0

    # Step 3: MS quality filter (ratio-based)
    quality_bg_threshold: float = 0.33
    quality_diff_threshold: float = 0.30
    quality_intensity_fc_threshold: float = 2.0

    # Step 4: Missing values
    missing_threshold: float = 0.5
    impute_method: str = "min"

    # Step 5: ISTD
    istd_mz_list: list[float] | None = None

    # Step 6-7: QC-LOWESS / Batch effect
    # (use defaults from calibration modules)

    # Step 8: Feature filtering (IQR/RSD)
    filter_method: str = "iqr"
    filter_cutoff: float | None = None  # auto

    # Step 9: Normalization
    norm_method: str = "PQN"

    # Step 10: Transformation
    transform_method: str = "LogNorm"

    # Step 11: Scaling
    scale_method: str = "AutoNorm"

    # Steps to skip
    skip_steps: set[int] = field(default_factory=set)


# ======================================================================
# Step registry
# ======================================================================


@dataclass
class StepDef:
    """Definition of a single pipeline step."""

    number: int
    name: str
    label: str
    source: str  # "preprocessing" | "calibration" | "processing"


# Canonical step order
STEPS: dict[int, StepDef] = {
    1: StepDef(1, "data_organize", "Data Organization", "preprocessing"),
    2: StepDef(2, "duplicate_remove", "Duplicate Signal Removal", "preprocessing"),
    3: StepDef(3, "ms_quality_filter", "MS Quality Filter", "preprocessing"),
    4: StepDef(4, "missing_value", "Missing Value Imputation", "processing"),
    5: StepDef(5, "istd_correction", "ISTD Marking + Correction", "calibration"),
    6: StepDef(6, "qc_lowess", "QC-LOWESS Trend Correction", "calibration"),
    7: StepDef(7, "batch_effect", "Batch Effect ComBat", "calibration"),
    8: StepDef(8, "feature_filter", "Feature Filtering (IQR/RSD)", "processing"),
    9: StepDef(9, "pqn_normalize", "PQN Normalization", "calibration"),
    10: StepDef(10, "transform", "Generalized Log Transform", "processing"),
    11: StepDef(11, "scaling", "Scaling", "processing"),
}

TOTAL_STEPS = 11  # Step 12 (analysis) is not part of the pipeline


# ======================================================================
# Step executors
# ======================================================================


def _step_data_organize(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 1: m/z+RT merge, column renaming, SampleInfo extraction."""
    from ms_core.preprocessing.data_organizer import DataOrganizer

    processor = DataOrganizer()
    result = processor.process(ds.matrix, **params)
    if not result.success:
        raise RuntimeError(f"Step 1 failed: {result.message}")
    return ds.replace_matrix(result.data, msg=f"[Step 1] {result.message}")


def _step_duplicate_remove(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 2: Remove duplicate signals by m/z+RT tolerance."""
    from ms_core.preprocessing.duplicate_remover import DuplicateRemover

    processor = DuplicateRemover()
    kw = {}
    if "mz_tolerance_ppm" in params:
        kw["mz_tolerance_ppm"] = params["mz_tolerance_ppm"]
    if "rt_tolerance" in params:
        kw["rt_tolerance"] = params["rt_tolerance"]
    result = processor.process(ds.matrix, **kw)
    if not result.success:
        raise RuntimeError(f"Step 2 failed: {result.message}")
    return ds.replace_matrix(result.data, msg=f"[Step 2] {result.message}")


def _step_ms_quality_filter(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 3: Ratio-based per-group quality filtering."""
    from ms_core.preprocessing.ms_quality_filter import FeatureFilter

    processor = FeatureFilter()
    kw = {}
    for key in ("background_threshold", "diff_threshold", "intensity_fc_threshold"):
        if key in params:
            kw[key] = params[key]
    result = processor.process(ds.matrix, **kw)
    if not result.success:
        raise RuntimeError(f"Step 3 failed: {result.message}")
    return ds.replace_matrix(result.data, msg=f"[Step 3] {result.message}")


def _step_missing_value(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 4: Zero→NaN, remove high-missing features, impute."""
    from ms_core.processing.missing_values import (
        impute_missing,
        remove_missing_percent,
        replace_zero_with_nan,
    )

    threshold = params.get("missing_threshold", 0.5)
    method = params.get("impute_method", "min")

    df = replace_zero_with_nan(ds.matrix)
    n_before = df.shape[1]
    df = remove_missing_percent(df, threshold=threshold)
    n_removed = n_before - df.shape[1]
    df = impute_missing(df, method=method)

    msg = (
        f"[Step 4] Zero→NaN, removed {n_removed} features "
        f"(>{threshold:.0%} missing), imputed with '{method}'"
    )
    return ds.replace_matrix(df, msg=msg)


def _step_istd_correction(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 5: ISTD marking + correction.

    ISTD feature IDs are resolved from (in priority order):
    1. ``istd_feature_ids`` in *params*  (explicit override)
    2. ``is_istd`` column in ``ds.feature_info``  (set by upstream ISTDMarker)
    3. Red-font detection in Excel  (legacy temp-file fallback)
    """
    if "istd_feature_ids" not in params:
        # Try to extract from feature_info (populated by ISTDMarker / Step 2)
        fi = ds.feature_info
        if not fi.empty and "is_istd" in fi.columns:
            istd_ids = fi.index[fi["is_istd"].astype(bool)].tolist()
            if istd_ids:
                params = {**params, "istd_feature_ids": istd_ids}

    return _calibration_wrapper(
        step_num=5,
        calibration_module="ms_core.calibration.istd_correction",
        ds=ds,
        **params,
    )


def _step_qc_lowess(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 6: QC-LOWESS trend correction via tempfile round-trip."""
    return _calibration_wrapper(
        step_num=6,
        calibration_module="ms_core.calibration.qc_lowess",
        ds=ds,
        **params,
    )


def _step_batch_effect(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 7: ComBat batch effect correction via tempfile round-trip."""
    return _calibration_wrapper(
        step_num=7,
        calibration_module="ms_core.calibration.batch_effect",
        ds=ds,
        **params,
    )


def _step_feature_filter(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 8: IQR/SD/MAD/RSD variance-based filtering."""
    from ms_core.processing.feature_filter import filter_features

    method = params.get("filter_method", "iqr")
    cutoff = params.get("filter_cutoff", None)

    n_before = ds.n_features
    df = filter_features(ds.matrix, method=method, cutoff=cutoff)
    n_removed = n_before - df.shape[1]

    msg = f"[Step 8] Feature filter ({method}): {n_removed} removed, {df.shape[1]} retained"
    return ds.replace_matrix(df, msg=msg)


def _step_pqn_normalize(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 9: PQN normalization via tempfile round-trip."""
    return _calibration_wrapper(
        step_num=9,
        calibration_module="ms_core.calibration.pqn",
        ds=ds,
        **params,
    )


def _step_transform(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 10: Generalized log transform (glog, NOT log2(x+1))."""
    from ms_core.processing.transformation import apply_transform

    method = params.get("transform_method", "LogNorm")
    if method == "None":
        ds.log("[Step 10] Transform: skipped (None)")
        return ds

    df = apply_transform(ds.matrix, method=method)
    return ds.replace_matrix(df, msg=f"[Step 10] Transform: {method}")


def _step_scaling(ds: MSDataset, **params: Any) -> MSDataset:
    """Step 11: Column-wise scaling."""
    from ms_core.processing.scaling import apply_scaling

    method = params.get("scale_method", "AutoNorm")
    if method == "None":
        ds.log("[Step 11] Scaling: skipped (None)")
        return ds

    df = apply_scaling(ds.matrix, method=method)
    return ds.replace_matrix(df, msg=f"[Step 11] Scaling: {method}")


# Map step number → executor
_EXECUTORS: dict[int, Callable[..., MSDataset]] = {
    1: _step_data_organize,
    2: _step_duplicate_remove,
    3: _step_ms_quality_filter,
    4: _step_missing_value,
    5: _step_istd_correction,
    6: _step_qc_lowess,
    7: _step_batch_effect,
    8: _step_feature_filter,
    9: _step_pqn_normalize,
    10: _step_transform,
    11: _step_scaling,
}


# ======================================================================
# Calibration helpers — in-memory data conversion
# ======================================================================


def _dataset_to_calibration_df(ds: MSDataset) -> pd.DataFrame:
    """Transpose MSDataset (samples×features) → calibration format (features×samples).

    Calibration modules expect rows=features, columns=samples, with a leading
    ``FeatureID`` column.  MSDataset stores the transpose of that.
    """
    transposed = ds.matrix.T  # index=feature_ids, columns=sample_ids
    df = transposed.reset_index()
    df.rename(columns={df.columns[0]: "FeatureID"}, inplace=True)
    return df


def _calibration_df_to_dataset(
    result_df: pd.DataFrame,
    original_ds: MSDataset,
    step_num: int,
) -> MSDataset:
    """Convert calibration result (features×samples) → MSDataset.

    Only sample columns present in the original dataset are kept; extra
    statistical columns produced by calibration modules are dropped.
    """
    feature_col = result_df.columns[0]
    sample_names = list(original_ds.matrix.index)
    available = [s for s in sample_names if s in result_df.columns]

    matrix = result_df.set_index(feature_col)[available].T
    matrix.columns.name = None  # clear index name artifact from set_index
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    return MSDataset(
        matrix=matrix,
        labels=original_ds.labels.reindex(matrix.index),
        sample_info=original_ds.sample_info,
        feature_info=original_ds.feature_info,
        processing_log=list(original_ds.processing_log),
    )


# ======================================================================
# Calibration wrapper (in-memory preferred, temp-file fallback)
# ======================================================================


def _write_dataset_to_excel(ds: MSDataset, path: str) -> None:
    """Write MSDataset to a multi-sheet Excel file for calibration modules."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ds.matrix.to_excel(writer, sheet_name="RawIntensity")
        if not ds.sample_info.empty:
            ds.sample_info.to_excel(writer, sheet_name="SampleInfo")


def _read_result_excel(output_path: str, original_ds: MSDataset) -> MSDataset:
    """Read calibration output Excel back into MSDataset."""
    xls = pd.ExcelFile(output_path, engine="openpyxl")

    # Find the result sheet — calibration modules use various names
    result_sheet = None
    priority = [
        "PQN_SampleSpecific_Result",
        "Batch_effect_result",
        "QC LOWESS result",
        "ISTD_Correction_Result",
        "RawIntensity",
    ]
    for name in priority:
        if name in xls.sheet_names:
            result_sheet = name
            break
    if result_sheet is None:
        result_sheet = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=result_sheet, index_col=0)

    # Read updated SampleInfo if present
    sample_info = original_ds.sample_info
    if "SampleInfo" in xls.sheet_names:
        sample_info = pd.read_excel(xls, sheet_name="SampleInfo", index_col=0)

    xls.close()

    return MSDataset(
        matrix=df,
        labels=original_ds.labels.reindex(df.index),
        sample_info=sample_info,
        feature_info=original_ds.feature_info,
        processing_log=list(original_ds.processing_log),
    )


def _calibration_wrapper(
    step_num: int,
    calibration_module: str,
    ds: MSDataset,
    **params: Any,
) -> MSDataset:
    """Run a calibration module, preferring the in-memory path.

    If the module exposes ``process_in_memory(data_df, sample_info_df, **kw)``
    the wrapper converts MSDataset ↔ calibration DataFrame directly, avoiding
    all Excel serialization (≈10× faster).  Otherwise it falls back to the
    legacy temp-file round-trip.
    """
    import importlib

    mod = importlib.import_module(calibration_module)
    step_label = STEPS[step_num].label

    # ── fast path: in-memory ──────────────────────────────────────────
    if hasattr(mod, "process_in_memory"):
        data_df = _dataset_to_calibration_df(ds)
        logger.info("Running %s (in-memory) ...", step_label)
        result_df = mod.process_in_memory(data_df, ds.sample_info.copy(), **params)
        if result_df is not None:
            new_ds = _calibration_df_to_dataset(result_df, ds, step_num)
            new_ds.log(f"[Step {step_num}] {step_label}: completed")
            return new_ds
        # None → module cannot run in-memory (e.g., missing ISTD info)
        logger.info("%s: in-memory unavailable, falling back to temp-file", step_label)

    # ── slow path: temp-file round-trip (legacy) ─────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "input.xlsx")
        _write_dataset_to_excel(ds, input_path)

        logger.info("Running %s ...", step_label)
        result = mod.main(input_file=input_path)

        if result is None:
            raise RuntimeError(f"Step {step_num} ({step_label}) returned None")

        # result is a ProcessingResult with .output_path
        output_path = result.output_path if hasattr(result, "output_path") else None
        if output_path is None or not Path(output_path).exists():
            raise RuntimeError(
                f"Step {step_num} ({step_label}) did not produce an output file"
            )

        new_ds = _read_result_excel(output_path, ds)

    new_ds.log(f"[Step {step_num}] {step_label}: completed")
    return new_ds


# ======================================================================
# Pipeline orchestrator
# ======================================================================


class MSPipeline:
    """12-step pipeline with snapshots and step-by-step execution.

    Example::

        pipeline = MSPipeline()

        # Step-by-step
        ds = pipeline.run_step(4, dataset, impute_method="min")
        ds = pipeline.run_step(10, ds)

        # One-shot
        config = PipelineConfig(skip_steps={1, 2, 3})
        ds = pipeline.run_all(dataset, config)

        # Undo
        ds = pipeline.undo(10)  # back to pre-step-10 state
    """

    def __init__(self) -> None:
        self._snapshots: dict[int, MSDataset] = {}
        self._current_step: int = 0
        self._config = PipelineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_step(self, step: int, ds: MSDataset, **params: Any) -> MSDataset:
        """Execute a single pipeline step.

        Args:
            step: Step number (1-11).
            ds: Input dataset.
            **params: Step-specific parameters (override config defaults).

        Returns:
            New MSDataset with the step applied.

        Raises:
            ValueError: If step number is invalid.
            RuntimeError: If the step fails.
        """
        if step not in STEPS:
            raise ValueError(f"Invalid step {step}. Valid: 1-{TOTAL_STEPS}")

        executor = _EXECUTORS[step]
        step_def = STEPS[step]

        # Snapshot before execution
        self._snapshots[step] = ds.copy()

        logger.info("Pipeline step %d: %s", step, step_def.label)
        result = executor(ds, **params)

        self._current_step = step
        return result

    def run_all(
        self,
        ds: MSDataset,
        config: PipelineConfig | None = None,
    ) -> MSDataset:
        """Execute all pipeline steps sequentially.

        Steps listed in ``config.skip_steps`` are skipped.

        Args:
            ds: Input dataset.
            config: Pipeline configuration. Uses defaults if None.

        Returns:
            Final MSDataset after all steps.
        """
        cfg = config or self._config

        # Build params dict from config for each step
        step_params: dict[int, dict[str, Any]] = {
            2: {
                "mz_tolerance_ppm": cfg.mz_tolerance_ppm,
                "rt_tolerance": cfg.rt_tolerance,
            },
            3: {
                "background_threshold": cfg.quality_bg_threshold,
                "diff_threshold": cfg.quality_diff_threshold,
                "intensity_fc_threshold": cfg.quality_intensity_fc_threshold,
            },
            4: {
                "missing_threshold": cfg.missing_threshold,
                "impute_method": cfg.impute_method,
            },
            5: {"istd_mz_list": cfg.istd_mz_list} if cfg.istd_mz_list else {},
            8: {
                "filter_method": cfg.filter_method,
                "filter_cutoff": cfg.filter_cutoff,
            },
            9: {"norm_method": cfg.norm_method},
            10: {"transform_method": cfg.transform_method},
            11: {"scale_method": cfg.scale_method},
        }

        for step_num in range(1, TOTAL_STEPS + 1):
            if step_num in cfg.skip_steps:
                ds.log(f"[Step {step_num}] {STEPS[step_num].label}: skipped")
                logger.info("Skipping step %d: %s", step_num, STEPS[step_num].label)
                continue

            params = step_params.get(step_num, {})
            ds = self.run_step(step_num, ds, **params)

        return ds

    def undo(self, step: int) -> MSDataset:
        """Return the dataset snapshot from before the given step.

        Args:
            step: The step number to undo (returns the pre-step state).

        Returns:
            The MSDataset as it was before the step executed.

        Raises:
            KeyError: If no snapshot exists for the given step.
        """
        if step not in self._snapshots:
            raise KeyError(
                f"No snapshot for step {step}. "
                f"Available: {sorted(self._snapshots.keys())}"
            )
        self._current_step = step - 1
        return self._snapshots[step]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """The last successfully completed step number (0 = none)."""
        return self._current_step

    @property
    def log(self) -> list[str]:
        """Aggregated processing log from all snapshots."""
        logs: list[str] = []
        for step_num in sorted(self._snapshots.keys()):
            logs.extend(self._snapshots[step_num].processing_log)
        return logs

    @property
    def snapshots(self) -> dict[int, MSDataset]:
        """Read-only access to step snapshots."""
        return dict(self._snapshots)

    def get_step_info(self, step: int) -> StepDef:
        """Get step definition by number."""
        if step not in STEPS:
            raise ValueError(f"Invalid step {step}. Valid: 1-{TOTAL_STEPS}")
        return STEPS[step]

    @staticmethod
    def list_steps() -> list[StepDef]:
        """Return all step definitions in order."""
        return [STEPS[i] for i in range(1, TOTAL_STEPS + 1)]
