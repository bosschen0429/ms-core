"""
Feature Filter Module - Step 4 of the preprocessing pipeline.

This module handles feature filtering and missing value imputation:
- Dynamic sample type detection
- Ratio calculation for each group
- Multi-criteria feature filtering
- Intelligent missing value imputation

Based on: Feature_barrier_V3.bas
"""

from typing import Optional, Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.settings import FeatureFilterConfig
from ms_core.utils.validators import detect_fixed_columns


class FeatureFilter(BaseProcessor):
    """
    Filters features and imputes missing values.

    This processor:
    1. Automatically detects sample types from row 2
    2. Calculates signal ratio for each group
    3. Filters features based on multiple criteria:
       - Stable: >=2 groups with ratio >= background threshold
       - Skewed: Any group with ratio >= skew threshold
       - Different: Any two groups with ratio difference >= diff threshold
    4. Removes features with QC_ratio = 0 or below threshold
    5. Imputes missing values using group-specific minimum/2
    """

    def __init__(self, config: Optional[FeatureFilterConfig] = None):
        """
        Initialize the Feature Filter.

        Args:
            config: Configuration options for feature filtering
        """
        super().__init__("Feature Filter")
        self.config = config or FeatureFilterConfig()

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for feature filtering.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df) < 2:
            return False, "Data must have at least 2 rows (Sample_Type + data)"

        if len(df.columns) < 3:
            return False, "Data must have at least 3 columns (feature + samples)"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        background_threshold: Optional[float] = None,
        skew_threshold: Optional[float] = None,
        diff_threshold: Optional[float] = None,
        qc_ratio_threshold: Optional[float] = None,
        protected_rows: Optional[Set[int]] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process data for feature filtering and missing value imputation.

        Args:
            df: Input DataFrame
            background_threshold: Threshold for stable features (0-1)
            skew_threshold: Threshold for skewed features (0-1)
            diff_threshold: Threshold for differential features (0-1)
            qc_ratio_threshold: Minimum QC_ratio to keep a feature (0-1)
            protected_rows: Set of row indices (red font) to protect from removal
            **kwargs: Additional parameters

        Returns:
            ProcessingResult with filtered data and imputed values
        """
        self.reset()

        # Use config defaults if not specified
        bg_thresh = background_threshold if background_threshold is not None else self.config.default_background_threshold
        skew_thresh = skew_threshold if skew_threshold is not None else self.config.default_skew_threshold
        diff_thresh = diff_threshold if diff_threshold is not None else self.config.default_diff_threshold
        qc_ratio_thresh = (
            qc_ratio_threshold
            if qc_ratio_threshold is not None
            else self.config.default_qc_ratio_threshold
        )

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(5, "Starting feature filtering...")

        try:
            # Create a copy
            result_df = df.copy()
            deleted_features = []

            # Step 1: Detect sample types
            self.update_progress(10, "Detecting sample types...")
            group_info = self._detect_sample_types(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Calculate ratios for each group
            self.update_progress(25, "Calculating group ratios...")
            result_df, ratio_cols = self._calculate_ratios(result_df, group_info)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Filter features
            self.update_progress(50, "Filtering features...")
            result_df, deleted_features, filter_stats = self._filter_features(
                result_df,
                group_info,
                ratio_cols,
                bg_thresh,
                skew_thresh,
                diff_thresh,
                qc_ratio_thresh,
                protected_rows or set(),
            )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 4: Impute missing values
            self.update_progress(75, "Imputing missing values...")
            result_df, impute_stats = self._impute_missing_values(
                result_df,
                group_info,
                ratio_cols,
            )

            self.update_progress(100, "Feature filtering complete")

            # Compile statistics
            stats = {
                **filter_stats,
                **impute_stats,
                "final_features": len(result_df) - 1,
                "groups_detected": len(group_info["groups"]),
                "has_qc": group_info["has_qc"],
            }

            # Create deleted features DataFrame
            deleted_df = None
            if deleted_features:
                # Reconstruct deleted rows
                pass  # Handled in metadata

            return ProcessingResult(
                success=True,
                data=result_df,
                message=f"Feature filtering completed. Kept {filter_stats.get('kept_count', 0)}, "
                        f"removed {filter_stats.get('deleted_count', 0)} features.",
                statistics=stats,
                metadata={
                    "group_info": group_info,
                    "ratio_columns": ratio_cols,
                    "thresholds": {
                        "background": bg_thresh,
                        "skew": skew_thresh,
                        "diff": diff_thresh,
                        "qc_ratio": qc_ratio_thresh,
                    },
                    "deleted_features": deleted_features,
                    "blue_font_cells": impute_stats.get("imputed_cells", []),
                    "imputation_stats": {
                        "cells_imputed": int(impute_stats.get("cells_imputed", 0)),
                        "cells_imputed_from_nan": int(impute_stats.get("cells_imputed_from_nan", 0)),
                        "cells_imputed_from_zero": int(impute_stats.get("cells_imputed_from_zero", 0)),
                    },
                    "red_font_rows": filter_stats.get("red_font_rows", []),
                    "protected_rows": filter_stats.get("red_font_rows", []),
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during feature filtering: {str(e)}",
            )

    def _detect_sample_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect sample types from row index 1 (Sample_Type row).

        Returns dict with group information.
        """
        info = {
            "groups": {},  # group_name -> list of column indices
            "qc_cols": [],
            "excluded_cols": [],
            "unknown_types": set(),
            "has_qc": False,
        }

        excluded_types = set(t.lower() for t in self.config.excluded_types)

        # Row 0 contains sample types (Sample_Type row)
        sample_type_row = 0

        fixed_cols, start_idx = detect_fixed_columns(df)
        if not fixed_cols:
            start_idx = 1

        for col_idx in range(start_idx, len(df.columns)):
            col_name = df.columns[col_idx]
            sample_type = str(df.iat[sample_type_row, col_idx]).lower().strip()

            if sample_type in ['', 'nan', 'na', 'none']:
                continue

            if sample_type == 'qc':
                info["qc_cols"].append(col_idx)
                info["has_qc"] = True
            elif sample_type in excluded_types:
                info["excluded_cols"].append(col_idx)
            else:
                # Analysis group
                if sample_type not in info["groups"]:
                    info["groups"][sample_type] = []
                info["groups"][sample_type].append(col_idx)

        return info

    def _calculate_ratios(
        self,
        df: pd.DataFrame,
        group_info: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Calculate signal ratio for each group.

        Returns DataFrame with ratio columns and dict mapping group to ratio column name.
        """
        ratio_cols = {}
        signal_threshold = self.config.signal_threshold

        # Build numeric block once for all groups/QC
        all_cols_set = set()
        for cols in group_info["groups"].values():
            all_cols_set.update(cols)
        all_cols_set.update(group_info.get("qc_cols", []))
        all_cols = sorted(all_cols_set)
        col_pos = {col_idx: pos for pos, col_idx in enumerate(all_cols)}
        if all_cols:
            block_all = df.iloc[1:, all_cols].apply(pd.to_numeric, errors="coerce")
            block_all_values = block_all.to_numpy()
        else:
            block_all_values = np.zeros((len(df) - 1, 0))

        # Add ratio columns for each group (vectorized)
        for group_name, col_indices in group_info["groups"].items():
            ratio_col = f"{group_name}_ratio"
            ratio_cols[group_name] = ratio_col

            if not col_indices:
                df[ratio_col] = ["na"] + [0] * (len(df) - 1)
                continue

            pos = [col_pos[c] for c in col_indices]
            block = block_all_values[:, pos]
            signal_count = (block >= signal_threshold).sum(axis=1)
            total_count = len(pos)
            ratios = signal_count / total_count if total_count > 0 else np.zeros(len(signal_count))
            df[ratio_col] = ["na"] + ratios.tolist()

        # Add QC ratio if QC samples exist (vectorized)
        if group_info["has_qc"]:
            qc_ratio_col = "QC_ratio"
            ratio_cols["QC"] = qc_ratio_col

            qc_cols = group_info["qc_cols"]
            if qc_cols:
                pos = [col_pos[c] for c in qc_cols]
                block = block_all_values[:, pos]
                signal_count = (block >= signal_threshold).sum(axis=1)
                total_count = len(pos)
                qc_ratios = signal_count / total_count if total_count > 0 else np.zeros(len(signal_count))
                df[qc_ratio_col] = ["na"] + qc_ratios.tolist()
            else:
                df[qc_ratio_col] = ["na"] + [0] * (len(df) - 1)

        return df, ratio_cols

    def _filter_features(
        self,
        df: pd.DataFrame,
        group_info: Dict[str, Any],
        ratio_cols: Dict[str, str],
        bg_threshold: float,
        skew_threshold: float,
        diff_threshold: float,
        qc_ratio_threshold: float,
        protected_rows: Set[int],
    ) -> Tuple[pd.DataFrame, List[pd.Series], Dict[str, Any]]:
        """
        Filter features based on ratio criteria.

        Returns filtered DataFrame, deleted rows, and statistics.
        """
        stats = {
            "kept_count": 0,
            "deleted_count": 0,
            "skew_kept": 0,
            "stable_kept": 0,
            "diff_kept": 0,
            "qc_zero_deleted": 0,
            "qc_low_deleted": 0,
            "protected_kept": 0,
        }

        deleted_features = []
        rows_to_keep = [0]  # Always keep Sample_Type row

        group_names = list(group_info["groups"].keys())
        has_qc = group_info["has_qc"]
        qc_ratio_col = ratio_cols.get("QC")

        # Build ratio matrix (rows: features, cols: groups)
        ratio_matrix = []
        for group_name in group_names:
            ratio_col = ratio_cols[group_name]
            ratio_series = pd.to_numeric(df[ratio_col].iloc[1:], errors="coerce").fillna(0)
            ratio_matrix.append(ratio_series.to_numpy())

        if ratio_matrix:
            ratio_matrix = np.vstack(ratio_matrix).T  # shape: (n_rows, n_groups)
        else:
            ratio_matrix = np.zeros((len(df) - 1, 0))

        # QC ratios
        if has_qc and qc_ratio_col:
            qc_ratio = pd.to_numeric(df[qc_ratio_col].iloc[1:], errors="coerce").fillna(0).to_numpy()
        else:
            qc_ratio = np.ones(len(df) - 1)

        protected_mask = np.zeros(len(df) - 1, dtype=bool)
        for idx in protected_rows:
            if idx > 0 and idx < len(df):
                protected_mask[idx - 1] = True

        # Rule: QC ratio == 0
        qc_zero = (qc_ratio == 0)
        # Optional stricter gate: remove non-zero values below threshold
        qc_low = (
            (qc_ratio < qc_ratio_threshold) & (qc_ratio > 0)
            if has_qc and qc_ratio_threshold > 0
            else np.zeros(len(df) - 1, dtype=bool)
        )

        # Conditions
        if ratio_matrix.shape[1] > 0:
            skew_keep = (ratio_matrix >= skew_threshold).any(axis=1)
            # Max diff across groups
            max_diff = ratio_matrix.max(axis=1) - ratio_matrix.min(axis=1)
            diff_keep = max_diff >= diff_threshold if ratio_matrix.shape[1] >= 2 else np.zeros(len(df) - 1, dtype=bool)
            stable_keep = (ratio_matrix >= bg_threshold).sum(axis=1) >= 2
        else:
            skew_keep = np.zeros(len(df) - 1, dtype=bool)
            diff_keep = np.zeros(len(df) - 1, dtype=bool)
            stable_keep = np.zeros(len(df) - 1, dtype=bool)

        keep_mask = protected_mask | skew_keep | diff_keep | stable_keep
        # If QC ratio fails gate and not protected, force delete
        keep_mask = np.where((qc_zero | qc_low) & ~protected_mask, False, keep_mask)

        # Update stats
        stats["protected_kept"] = int(protected_mask.sum())
        stats["skew_kept"] = int((skew_keep & ~protected_mask).sum())
        stats["diff_kept"] = int((diff_keep & ~protected_mask).sum())
        stats["stable_kept"] = int((stable_keep & ~protected_mask).sum())
        stats["qc_zero_deleted"] = int((qc_zero & ~protected_mask).sum())
        stats["qc_low_deleted"] = int((qc_low & ~protected_mask).sum())

        # Build keep rows
        for i, keep in enumerate(keep_mask, start=1):
            if keep:
                rows_to_keep.append(i)
                stats["kept_count"] += 1
            else:
                deleted_features.append(df.iloc[i].copy())
                stats["deleted_count"] += 1

        # Build mapping for kept rows (for protected row updates)
        row_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(rows_to_keep)}
        stats["red_font_rows"] = sorted(
            row_mapping[idx] for idx in protected_rows if idx in row_mapping
        )

        # Filter DataFrame — .copy() ensures writable backing array (avoids numpy read-only error)
        result_df = df.iloc[rows_to_keep].reset_index(drop=True).copy()

        return result_df, deleted_features, stats

    def _get_max_ratio_diff(self, ratios: List[float]) -> float:
        """Calculate maximum difference between any two ratios."""
        if not ratios:
            return 0.0
        return float(max(ratios) - min(ratios))

    def _impute_missing_values(
        self,
        df: pd.DataFrame,
        group_info: Dict[str, Any],
        ratio_cols: Dict[str, str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing values using group-specific minimum/2.

        Returns DataFrame with imputed values and statistics.
        """
        stats = {
            "cells_imputed": 0,
            "cells_imputed_from_nan": 0,
            "cells_imputed_from_zero": 0,
            "imputation_method": "group_min_half",
        }

        # Track imputed cells for blue font marking
        imputed_cells = []

        signal_threshold = self.config.signal_threshold

        # Precompute group ratios for special cases
        group_ratios = {}
        for group_name in group_info["groups"].keys():
            ratio_col = ratio_cols.get(group_name)
            if ratio_col:
                group_ratios[group_name] = pd.to_numeric(df[ratio_col].iloc[1:], errors="coerce").fillna(0).to_numpy()
            else:
                group_ratios[group_name] = np.zeros(len(df) - 1)

        # Precompute special-case mask per group
        special_case = {}
        if len(group_info["groups"]) > 1:
            for group_name in group_info["groups"].keys():
                grp_ratio = group_ratios.get(group_name, np.zeros(len(df) - 1))
                other_all_one = np.ones(len(df) - 1, dtype=bool)
                for other_name in group_info["groups"].keys():
                    if other_name == group_name:
                        continue
                    other_all_one &= (group_ratios.get(other_name, np.zeros(len(df) - 1)) == 1.0)
                special_case[group_name] = (grp_ratio == 0) & other_all_one
        else:
            for group_name in group_info["groups"].keys():
                special_case[group_name] = np.zeros(len(df) - 1, dtype=bool)

        # Build numeric block once for all group/QC columns
        all_cols_set = set()
        for cols in group_info["groups"].values():
            all_cols_set.update(cols)
        all_cols_set.update(group_info.get("qc_cols", []))
        all_cols = sorted(all_cols_set)
        col_pos = {col_idx: pos for pos, col_idx in enumerate(all_cols)}
        if all_cols:
            block_all = df.iloc[1:, all_cols].apply(pd.to_numeric, errors="coerce")
            block_values = block_all.to_numpy(copy=True)
        else:
            block_values = np.zeros((len(df) - 1, 0))

        # Impute group columns in blocks
        for group_name, col_indices in group_info["groups"].items():
            if not col_indices:
                continue
            pos = [col_pos[c] for c in col_indices]
            block = block_values[:, pos]
            if block.shape[0] == 0:
                continue
            missing_nan_mask = np.isnan(block)
            missing_zero_mask = (block == 0)
            missing_mask = missing_nan_mask | missing_zero_mask
            if not missing_mask.any():
                continue

            block_positive = np.where(block > 0, block, np.nan)
            # Avoid RuntimeWarning on all-NaN rows by using +inf sentinel.
            mins = np.min(np.where(np.isnan(block_positive), np.inf, block_positive), axis=1)
            mins = np.where(np.isinf(mins), signal_threshold, mins)

            special = special_case[group_name]
            fill_values = np.where(special, signal_threshold, mins / 2)

            filled = np.where(missing_mask, fill_values[:, None], block)
            block_values[:, pos] = filled

            idx = np.argwhere(missing_mask)
            if idx.size > 0:
                rows = (idx[:, 0] + 1).astype(int).tolist()
                cols = [col_indices[j] for j in idx[:, 1].tolist()]
                imputed_cells.extend(list(zip(rows, cols)))
                nan_count = int(missing_nan_mask.sum())
                zero_count = int(missing_zero_mask.sum())
                stats["cells_imputed"] += (nan_count + zero_count)
                stats["cells_imputed_from_nan"] += nan_count
                stats["cells_imputed_from_zero"] += zero_count

        # Impute QC columns in blocks
        qc_cols = group_info.get("qc_cols", [])
        if qc_cols:
            pos = [col_pos[c] for c in qc_cols]
            block = block_values[:, pos]
            if block.shape[0] > 0:
                missing_nan_mask = np.isnan(block)
                missing_zero_mask = (block == 0)
                missing_mask = missing_nan_mask | missing_zero_mask
                if missing_mask.any():
                    block_positive = np.where(block > 0, block, np.nan)
                    qc_mins = np.min(np.where(np.isnan(block_positive), np.inf, block_positive), axis=1)
                    qc_mins = np.where(np.isinf(qc_mins), signal_threshold, qc_mins)
                    fill_values = (qc_mins / 2)[:, None]
                    filled = np.where(missing_mask, fill_values, block)
                    block_values[:, pos] = filled

                    idx = np.argwhere(missing_mask)
                    if idx.size > 0:
                        rows = (idx[:, 0] + 1).astype(int).tolist()
                        cols = [qc_cols[j] for j in idx[:, 1].tolist()]
                        imputed_cells.extend(list(zip(rows, cols)))
                        nan_count = int(missing_nan_mask.sum())
                        zero_count = int(missing_zero_mask.sum())
                        stats["cells_imputed"] += (nan_count + zero_count)
                        stats["cells_imputed_from_nan"] += nan_count
                        stats["cells_imputed_from_zero"] += zero_count

        # Write back to DataFrame
        if all_cols:
            df.iloc[1:, all_cols] = block_values

        stats["imputed_cells"] = imputed_cells

        return df, stats

    def get_group_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get a summary of detected groups and their statistics.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with group summary information
        """
        group_info = self._detect_sample_types(df)

        summary = {
            "groups": {},
            "qc_count": len(group_info["qc_cols"]),
            "has_qc": group_info["has_qc"],
            "excluded_count": len(group_info["excluded_cols"]),
            "unknown_types": list(group_info["unknown_types"]),
        }

        for group_name, col_indices in group_info["groups"].items():
            summary["groups"][group_name] = {
                "sample_count": len(col_indices),
                "columns": [df.columns[i] for i in col_indices],
            }

        return summary
