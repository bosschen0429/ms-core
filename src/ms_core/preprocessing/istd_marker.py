"""
ISTD Marker Module - Step 2 of the preprocessing pipeline.

This module handles Internal Standard (ISTD) marking and duplicate detection:
- Sort data by m/z values
- Mark ISTD features (yellow background)
- Detect and mark duplicate features based on m/z and RT tolerance

Based on: FindSTDs_mzRT_Jia_Simplified.bas
"""

from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.settings import ISTDConfig
from ms_core.utils.file_handler import parse_mz_rt_string


class ISTDMarker(BaseProcessor):
    """
    Marks Internal Standards (ISTD) and detects duplicate features.

    This processor:
    1. Renames the first sheet to 'RawIntensity'
    2. Sorts data by m/z values (ascending)
    3. Detects duplicate features based on m/z (ppm) and RT tolerance
    4. Tracks duplicates (for reporting)
    5. Marks ISTD features with red font (duplicates are not colored)
    6. Removes ISTD-marked rows (optional; default keeps them)
    """

    def __init__(self, config: Optional[ISTDConfig] = None):
        """
        Initialize the ISTD Marker.

        Args:
            config: Configuration options for ISTD marking
        """
        super().__init__("ISTD Marker")
        self.config = config or ISTDConfig()

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for ISTD marking.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df) < 2:
            return False, "Data must have at least 2 rows (Sample_Type + data)"

        # Check if first column contains m/z/RT format
        first_col = df.columns[0]
        sample_values = df[first_col].iloc[1:11]
        valid_format_count = sum(1 for v in sample_values if self._is_valid_mz_rt(v))

        if valid_format_count < len(sample_values) * 0.5:
            return False, "First column should contain m/z/RT format (e.g., '123.456/1.23')"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        istd_features: Optional[Set[str]] = None,
        custom_tolerances: Optional[Dict[str, Tuple[float, float]]] = None,
        istd_mz_list: Optional[List[float]] = None,
        istd_record_file: Optional[Path] = None,
        istd_record_date: Optional[str] = None,
        keep_istd_rows: bool = True,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process data for ISTD marking and duplicate detection.

        Args:
            df: Input DataFrame (data starts from row index 1)
            istd_features: Set of feature IDs to mark as ISTD (will be removed)
            custom_tolerances: Dict mapping feature ID to (ppm_tol, rt_tol) tuple
            **kwargs: Additional parameters

        Returns:
            ProcessingResult with processed data
        """
        self.reset()

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(10, "Starting ISTD marking...")

        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()

            # Step 1: Standardize header structure
            self.update_progress(20, "Standardizing headers...")
            result_df = self._standardize_headers(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Sort by m/z values
            self.update_progress(40, "Sorting by m/z values...")
            result_df, sort_stats = self._sort_by_mz(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Determine ISTD features
            self.update_progress(60, "Detecting duplicate features...")
            istd_features_set: Set[str] = set(istd_features or set())
            metadata: Dict[str, Any] = {}

            # If ISTD record file provided, infer ISTD features from record
            if istd_record_file:
                inferred, infer_meta = self.infer_istd_from_record(
                    result_df,
                    istd_record_file,
                    istd_record_date,
                    istd_mz_list or self.config.default_istd_mz,
                )
                istd_features_set = inferred
                metadata.update(infer_meta)
            elif not istd_features_set and istd_mz_list:
                # Fallback: m/z-only matching
                istd_features_set = set(
                    self.find_potential_istd(
                        result_df,
                        istd_mz_list,
                        self.config.default_ppm_tolerance,
                    )
                )
                metadata["istd_mz_list"] = list(istd_mz_list)

            duplicate_indices = self._detect_istd_duplicates(
                result_df,
                istd_features_set,
                custom_tolerances or {},
            )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 4: Remove ISTD rows (optional)
            self.update_progress(80, "Removing ISTD features...")
            pre_removal_df = result_df
            result_df, removal_stats, row_mapping = self._remove_istd_rows(
                result_df, istd_features_set, keep_istd_rows=keep_istd_rows
            )

            self.update_progress(100, "ISTD marking complete")

            # Compile statistics
            stats = {
                **sort_stats,
                "duplicates_marked": len(duplicate_indices),
                **removal_stats,
            }

            # Remap duplicate indices after ISTD removal
            remapped_duplicates = {
                row_mapping[idx] for idx in duplicate_indices if idx in row_mapping
            }
            istd_rows = {
                row_mapping[idx]
                for idx in range(1, len(pre_removal_df))
                if str(pre_removal_df.iat[idx, 0]) in istd_features_set and idx in row_mapping
            }
            # Red font should represent ISTD only (duplicates are tracked separately)
            red_rows = sorted(istd_rows)

            return ProcessingResult(
                success=True,
                data=result_df,
                message=f"ISTD marking completed. Marked {len(duplicate_indices)} duplicates, "
                        f"removed {removal_stats.get('rows_removed', 0)} ISTD features.",
                statistics=stats,
                metadata={
                    "duplicate_indices": sorted(remapped_duplicates),
                    "istd_features": list(istd_features_set),
                    "istd_rows": sorted(istd_rows),
                    "protected_rows": sorted(istd_rows),
                    "red_font_rows": red_rows,
                    **metadata,
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during ISTD marking: {str(e)}",
            )

    def _standardize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the header structure."""
        # Drop tolerance column if it exists (not needed after Step 2)
        columns = list(df.columns)
        tolerance_cols = [c for c in columns if "tolerance" in str(c).lower()]
        if tolerance_cols:
            df = df.drop(columns=tolerance_cols)
            columns = list(df.columns)

        df.columns = columns

        # Ensure Sample_Type row values
        if len(df) > 0:
            df.iat[0, 0] = self.config.sample_type_col
            # Only set tolerance column to 'na' if it exists as second column
            if len(df.columns) > 1 and "tolerance" in str(df.columns[1]).lower():
                df.iat[0, 1] = "na"

        return df

    def _sort_by_mz(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Sort data rows by m/z values."""
        stats = {"original_order_changed": False}

        # Data rows start from index 1 (after Sample_Type row)
        if len(df) <= 1:
            return df, stats

        # Extract header rows
        header_rows = df.iloc[:1].copy()
        data_rows = df.iloc[1:].copy()

        # Extract m/z values for sorting (vectorized)
        mz_arr, _ = self._extract_mz_rt_arrays(data_rows)
        mz_arr = np.where(np.isnan(mz_arr), np.inf, mz_arr)
        data_rows["_sort_mz"] = mz_arr

        # Sort by m/z
        data_rows_sorted = data_rows.sort_values('_sort_mz', ascending=True)
        data_rows_sorted = data_rows_sorted.drop('_sort_mz', axis=1)

        # Check if order changed
        if not data_rows_sorted.index.equals(data_rows.index):
            stats["original_order_changed"] = True

        # Reconstruct DataFrame
        data_rows_sorted = data_rows_sorted.reset_index(drop=True)
        result_df = pd.concat([header_rows.reset_index(drop=True), data_rows_sorted], ignore_index=True)

        stats["rows_sorted"] = len(data_rows)

        return result_df, stats

    def _detect_istd_duplicates(
        self,
        df: pd.DataFrame,
        istd_features: Set[str],
        custom_tolerances: Dict[str, Tuple[float, float]],
    ) -> Set[int]:
        """
        Detect non-ISTD features that fall within m/z (ppm) and RT tolerance
        of a known ISTD feature.

        Only compares non-ISTD rows against ISTD rows; does NOT perform
        general-purpose deduplication (that is Step 3's responsibility).

        Returns set of row indices that are duplicates of ISTD features.
        """
        duplicate_indices = set()

        if len(df) <= 1 or not istd_features:
            return duplicate_indices

        first_col = df.columns[0]
        tolerance_col = None
        if len(df.columns) > 1:
            col1 = str(df.columns[1])
            if "tolerance" in col1.lower():
                tolerance_col = df.columns[1]

        # Get ISTD row indices and their m/z/RT values
        mz_arr, rt_arr = self._extract_mz_rt_arrays(df)
        istd_data = {}  # feature_id -> (row_idx, mz, rt, ppm_tol, rt_tol)

        for row_idx in range(1, len(df)):
            feature_id = str(df.iat[row_idx, 0])
            if feature_id in istd_features:
                mz = mz_arr[row_idx]
                rt = rt_arr[row_idx]
                if not np.isnan(mz) and not np.isnan(rt):
                    # Get tolerance from column B or custom or default
                    ppm_tol = self.config.default_ppm_tolerance
                    rt_tol = self.config.default_rt_tolerance

                    if feature_id in custom_tolerances:
                        ppm_tol, rt_tol = custom_tolerances[feature_id]
                    elif tolerance_col:
                        tol_value = str(df.iat[row_idx, 1])
                        if tol_value.lower() not in ['na', 'nan', '']:
                            try:
                                tol_parts = tol_value.split('/')
                                if len(tol_parts) == 2:
                                    ppm_tol = float(tol_parts[0])
                                    rt_tol = float(tol_parts[1])
                            except ValueError:
                                pass

                    istd_data[feature_id] = (row_idx, mz, rt, ppm_tol, rt_tol)

        # Check all rows for duplicates with ISTD features
        for row_idx in range(1, len(df)):
            feature_id = str(df.iat[row_idx, 0])

            # Skip ISTD features themselves
            if feature_id in istd_features:
                continue

            mz = mz_arr[row_idx]
            rt = rt_arr[row_idx]
            if np.isnan(mz) or np.isnan(rt):
                continue

            # Check against each ISTD
            for istd_id, (istd_row, istd_mz, istd_rt, ppm_tol, rt_tol) in istd_data.items():
                # Calculate ppm difference
                if istd_mz != 0:
                    ppm_diff = abs((mz - istd_mz) / istd_mz * 1_000_000)
                else:
                    ppm_diff = float('inf')

                rt_diff = abs(rt - istd_rt)

                # Check if within tolerance
                if ppm_diff <= ppm_tol and rt_diff <= rt_tol:
                    duplicate_indices.add(row_idx)
                    break  # No need to check other ISTDs

        return duplicate_indices

    def _remove_istd_rows(
        self,
        df: pd.DataFrame,
        istd_features: Set[str],
        keep_istd_rows: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[int, int]]:
        """Remove rows marked as ISTD."""
        stats = {"rows_removed": 0, "features_removed": []}

        if keep_istd_rows or not istd_features or len(df) <= 1:
            return df, stats, {i: i for i in range(len(df))}

        first_col = df.columns[0]
        rows_to_keep = []
        row_mapping: Dict[int, int] = {}
        new_idx = 0

        for row_idx in range(len(df)):
            if row_idx == 0:
                # Keep Sample_Type row
                rows_to_keep.append(row_idx)
                row_mapping[row_idx] = new_idx
                new_idx += 1
                continue

            feature_id = str(df.iat[row_idx, 0])
            if feature_id not in istd_features:
                rows_to_keep.append(row_idx)
                row_mapping[row_idx] = new_idx
                new_idx += 1
            else:
                stats["rows_removed"] += 1
                stats["features_removed"].append(feature_id)

        result_df = df.iloc[rows_to_keep].reset_index(drop=True)

        return result_df, stats, row_mapping

    @staticmethod
    def _is_valid_mz_rt(value) -> bool:
        """Check if value is in valid m/z/RT format."""
        mz, rt = parse_mz_rt_string(str(value))
        return mz is not None and rt is not None

    def find_potential_istd(
        self,
        df: pd.DataFrame,
        known_istd_mz: List[float],
        mz_tolerance_ppm: float = 20.0,
    ) -> List[str]:
        """
        Find potential ISTD features based on known m/z values.

        Args:
            df: Input DataFrame
            known_istd_mz: List of known ISTD m/z values
            mz_tolerance_ppm: m/z tolerance in ppm

        Returns:
            List of feature IDs that match known ISTD m/z values
        """
        potential_istd = []

        if len(df) <= 1:
            return potential_istd

        mz_arr, _ = self._extract_mz_rt_arrays(df)
        for row_idx in range(1, len(df)):
            feature_id = str(df.iat[row_idx, 0])
            mz = mz_arr[row_idx]

            if np.isnan(mz):
                continue

            for istd_mz in known_istd_mz:
                if istd_mz != 0:
                    ppm_diff = abs((mz - istd_mz) / istd_mz * 1_000_000)
                    if ppm_diff <= mz_tolerance_ppm:
                        potential_istd.append(feature_id)
                        break

        return potential_istd

    def infer_istd_from_record(
        self,
        df: pd.DataFrame,
        record_path: Path,
        record_date: Optional[str],
        known_istd_mz: List[float],
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Infer ISTD features using ISTD record file (QC RT mean).

        Args:
            df: Input DataFrame
            record_path: Path to ISTD record Excel file
            record_date: Target date string (YYYYMMDD) or None
            known_istd_mz: List of ISTD m/z values

        Returns:
            Tuple of (istd_features set, metadata dict)
        """
        metadata: Dict[str, Any] = {}
        record_path = Path(record_path)
        if not record_path.exists():
            return set(), {"warning": f"ISTD record not found: {record_path}"}

        target_rt_by_mz, record_meta = self._load_istd_record_targets(
            record_path,
            record_date,
            known_istd_mz,
            self.config.default_ppm_tolerance,
        )
        metadata.update(record_meta)

        if not target_rt_by_mz:
            metadata["warning"] = "No ISTD RT targets found from record file"
            return set(), metadata

        # Determine fixed columns (Mz/RT + optional tolerance)
        fixed_cols = 1
        if len(df.columns) > 1 and "tolerance" in str(df.columns[1]).lower():
            fixed_cols = 2

        # Identify QC columns from Sample_Type row if present (for metadata only)
        qc_cols = []
        if len(df) > 0 and str(df.iat[0, 0]).lower().strip() == "sample_type":
            for col_idx in range(fixed_cols, len(df.columns)):
                st = str(df.iat[0, col_idx]).lower().strip()
                if st == "qc":
                    qc_cols.append(col_idx)
        metadata["qc_columns"] = qc_cols

        # Use all sample columns for occurrence counting (requirement: all samples)
        all_sample_cols = list(range(fixed_cols, len(df.columns)))
        if not all_sample_cols:
            return set(), {"warning": "No sample columns found for ISTD matching"}

        # Build candidates for each ISTD m/z
        mz_arr, rt_arr = self._extract_mz_rt_arrays(df)
        candidates: Dict[float, List[Tuple[int, float, int, float]]] = {}
        for row_idx in range(1, len(df)):
            feature_id = str(df.iat[row_idx, 0])
            mz = mz_arr[row_idx]
            rt = rt_arr[row_idx]
            if np.isnan(mz) or np.isnan(rt):
                continue

            for target_mz, target_rt in target_rt_by_mz.items():
                ppm_diff = abs((mz - target_mz) / target_mz * 1_000_000) if target_mz else float("inf")
                rt_diff = abs(rt - target_rt)
                if ppm_diff <= self.config.default_ppm_tolerance and rt_diff <= self.config.default_rt_tolerance:
                    # Occurrence count: intensity > 0 across all samples
                    count = 0
                    total_intensity = 0.0
                    for col_idx in all_sample_cols:
                        val = pd.to_numeric(df.iat[row_idx, col_idx], errors="coerce")
                        if pd.notna(val) and val > 0:
                            count += 1
                            total_intensity += float(val)
                    candidates.setdefault(target_mz, []).append(
                        (row_idx, rt_diff, count, total_intensity)
                    )

        istd_features: Set[str] = set()
        chosen_rows: Dict[float, int] = {}

        for target_mz, rows in candidates.items():
            if not rows:
                continue
            # Choose by occurrence count, then total intensity, then RT closeness
            rows_sorted = sorted(
                rows,
                key=lambda x: (x[2], x[3], -x[1]),
                reverse=True,
            )
            best_row = rows_sorted[0][0]
            chosen_rows[target_mz] = best_row
            istd_features.add(str(df.iat[best_row, 0]))

        metadata["istd_target_rt"] = target_rt_by_mz
        metadata["istd_candidates"] = {k: [r[0] for r in v] for k, v in candidates.items()}
        metadata["istd_chosen_rows"] = chosen_rows
        metadata["istd_record_path"] = str(record_path)

        return istd_features, metadata

    def _extract_mz_rt_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized extraction of m/z and RT arrays from Mz/RT column."""
        col = df.iloc[:, 0].astype(str)
        parts = col.str.split("/", n=1, expand=True)
        mz = pd.to_numeric(parts[0], errors="coerce").to_numpy()
        if parts.shape[1] >= 2:
            rt = pd.to_numeric(parts[1], errors="coerce").to_numpy()
        else:
            rt = np.full(len(df), np.nan)
        return mz, rt

    def _load_istd_record_targets(
        self,
        record_path: Path,
        record_date: Optional[str],
        known_istd_mz: List[float],
        mz_tolerance_ppm: float,
    ) -> Tuple[Dict[float, float], Dict[str, Any]]:
        """
        Load ISTD record file and compute QC RT mean per ISTD m/z.
        """
        metadata: Dict[str, Any] = {}
        df = pd.read_excel(record_path, sheet_name=0, header=None)

        header_row = 2
        mz_col = None
        for col_idx in range(df.shape[1]):
            val = str(df.iat[header_row, col_idx]).strip().lower()
            if val == "m/z":
                mz_col = col_idx
                break

        # Find RT/Intensity column pairs by header row
        rt_cols = []
        for col_idx in range(df.shape[1] - 1):
            rt_header = str(df.iat[header_row, col_idx]).strip().lower()
            int_header = str(df.iat[header_row, col_idx + 1]).strip().lower()
            if rt_header == "rt" and int_header == "intensity":
                rt_cols.append(col_idx)

        if not rt_cols:
            return {}, {"warning": "No RT/Intensity columns found in ISTD record"}

        # Normalize target date
        def normalize_date(value) -> Optional[str]:
            if pd.isna(value):
                return None
            try:
                if isinstance(value, (int, float)) and not pd.isna(value):
                    return str(int(value))
            except Exception:
                pass
            text = str(value).strip()
            digits = "".join(ch for ch in text if ch.isdigit())
            return digits if digits else None

        available_dates = []
        for col in rt_cols:
            date_val = normalize_date(df.iat[0, col])
            if date_val:
                available_dates.append(date_val)

        available_dates = list(dict.fromkeys(available_dates))

        target_date = normalize_date(record_date) if record_date else None
        if target_date and available_dates and target_date not in available_dates:
            if len(available_dates) == 1:
                metadata["warning"] = (
                    f"Requested date {target_date} not found; using {available_dates[0]}"
                )
                target_date = available_dates[0]
            else:
                return {}, {
                    "warning": "ISTD record has multiple dates; please specify target date",
                    "available_dates": available_dates,
                }

        if not target_date:
            if len(available_dates) == 1:
                target_date = available_dates[0]
            else:
                return {}, {
                    "warning": "ISTD record has multiple dates; please specify target date",
                    "available_dates": available_dates,
                }

        metadata["istd_record_date"] = target_date

        # Filter columns by target date and QC samples
        target_cols = []
        for col in rt_cols:
            date_val = normalize_date(df.iat[0, col])
            if date_val != target_date:
                continue
            sample_name = str(df.iat[1, col]).strip()
            if "qc" not in sample_name.lower():
                continue
            target_cols.append(col)

        metadata["istd_record_qc_columns"] = target_cols
        if not target_cols:
            return {}, {"warning": f"No QC RT columns found for date {target_date}"}

        # Determine best m/z column by match count (exclude RT/Intensity columns)
        excluded_cols = set(target_cols)
        excluded_cols.update(col + 1 for col in target_cols)
        candidate_cols = [c for c in range(df.shape[1]) if c not in excluded_cols]

        def match_count_for_col(col_idx: int) -> int:
            count = 0
            for row_idx in range(header_row + 1, len(df)):
                val = pd.to_numeric(df.iat[row_idx, col_idx], errors="coerce")
                if pd.isna(val):
                    continue
                for known_mz in known_istd_mz:
                    ppm_diff = abs((val - known_mz) / known_mz * 1_000_000) if known_mz else None
                    if ppm_diff is not None and ppm_diff <= mz_tolerance_ppm:
                        count += 1
                        break
            return count

        best_col = None
        best_count = 0
        for col_idx in candidate_cols:
            c = match_count_for_col(col_idx)
            if c > best_count:
                best_count = c
                best_col = col_idx

        # Prefer detected best column; fallback to header 'm/z'
        if best_col is not None and best_count > 0:
            mz_col = best_col
        if mz_col is None:
            return {}, {"warning": "ISTD record missing usable m/z column"}

        metadata["istd_mz_column"] = mz_col
        metadata["istd_mz_match_count"] = best_count

        # Collect RT values per ISTD m/z from QC samples
        rt_values_by_mz: Dict[float, List[float]] = {mz: [] for mz in known_istd_mz}

        for row_idx in range(header_row + 1, len(df)):
            mz_val = df.iat[row_idx, mz_col]
            try:
                mz_val = float(mz_val)
            except (TypeError, ValueError):
                continue

            # Match to known ISTD m/z by ppm
            best_match = None
            best_ppm = None
            for known_mz in known_istd_mz:
                ppm_diff = abs((mz_val - known_mz) / known_mz * 1_000_000) if known_mz else None
                if ppm_diff is not None and ppm_diff <= mz_tolerance_ppm:
                    if best_ppm is None or ppm_diff < best_ppm:
                        best_ppm = ppm_diff
                        best_match = known_mz

            if best_match is None:
                continue

            for col in target_cols:
                rt_val = pd.to_numeric(df.iat[row_idx, col], errors="coerce")
                intensity_val = pd.to_numeric(df.iat[row_idx, col + 1], errors="coerce")
                if pd.notna(rt_val) and pd.notna(intensity_val) and intensity_val > 0:
                    rt_values_by_mz[best_match].append(float(rt_val))

        # Compute mean RT per ISTD
        target_rt_by_mz: Dict[float, float] = {}
        for mz, rt_vals in rt_values_by_mz.items():
            if rt_vals:
                target_rt_by_mz[mz] = float(np.mean(rt_vals))

        metadata["istd_target_rt_count"] = {mz: len(rt_vals) for mz, rt_vals in rt_values_by_mz.items()}

        return target_rt_by_mz, metadata
