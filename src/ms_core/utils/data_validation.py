"""
Data validation utilities (防呆 checks) for metabolomics data processing.

This module consolidates the validation checks that were scattered across
the processing modules into a centralized, reusable set of utilities.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field

from .constants import VALIDATION_THRESHOLDS, SHEET_NAMES, NON_SAMPLE_COLUMNS, FEATURE_ID_COLUMN


@dataclass
class ValidationResult:
    """
    Structured validation result with errors, warnings, and metadata.

    Attributes:
        is_valid: Whether validation passed (no errors)
        errors: List of critical error messages
        warnings: List of non-critical warning messages
        info: Additional metadata about the validation
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        """Add a warning (does not affect validity)."""
        self.warnings.append(msg)

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)
        if not other.is_valid:
            self.is_valid = False


class DataValidator:
    """
    Comprehensive data validation for metabolomics Excel files.

    Consolidates the scattered "防呆" (fool-proofing) checks into a
    single, reusable validator class.

    Usage:
        validator = DataValidator()
        result = validator.validate_sample_info(sample_info_df)
        if not result.is_valid:
            for error in result.errors:
                print(f"Error: {error}")
    """

    def __init__(self, strict: bool = False):
        """
        Initialize the validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict

    def validate_file_path(self, file_path: str) -> ValidationResult:
        """
        Validate that a file path is valid and accessible.

        Checks:
        - File exists
        - File is Excel format (.xlsx or .xls)
        - File is not empty
        - File is large enough to be valid

        Args:
            file_path: Path to the file to validate

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()

        # Check existence
        if not os.path.exists(file_path):
            result.add_error(f"找不到檔案: {file_path}")
            return result

        # Check format
        file_lower = file_path.lower()
        if not (file_lower.endswith('.xlsx') or file_lower.endswith('.xls')):
            result.add_error(f"輸入檔案必須是 Excel 格式 (.xlsx 或 .xls)")
            return result

        # Check size
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            result.add_error(f"無法讀取檔案大小: {e}")
            return result

        if file_size == 0:
            result.add_error("檔案是空的 (0 bytes)")
            return result

        min_size = VALIDATION_THRESHOLDS.get('min_file_size_bytes', 1024)
        if file_size < min_size:
            result.add_warning(f"檔案較小: {file_size} bytes (最小建議 {min_size} bytes)")

        result.info['file_size'] = file_size
        return result

    def validate_sample_info(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate SampleInfo sheet structure and content.

        Checks:
        - DataFrame is not empty
        - Required columns exist (Sample_Name, Sample_Type)
        - No duplicate sample names
        - QC samples are present
        - Sufficient QC samples for analysis

        Args:
            df: SampleInfo DataFrame

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()

        # Check empty
        if df is None or df.empty:
            result.add_error("SampleInfo 為空")
            return result

        result.info['total_samples'] = len(df)

        # Check required columns
        required_cols = ['Sample_Name', 'Sample_Type']
        for col in required_cols:
            if col not in df.columns:
                # Try to find similar column names
                similar = [c for c in df.columns if col.lower() in c.lower()]
                if similar:
                    result.add_warning(
                        f"缺少 '{col}' 欄位，但找到類似欄位: {similar}"
                    )
                else:
                    result.add_error(f"缺少必要欄位: {col}")

        if not result.is_valid:
            return result

        # Check for duplicates
        duplicates = df[df['Sample_Name'].duplicated(keep=False)]
        if not duplicates.empty:
            dup_names = duplicates['Sample_Name'].unique()[:5].tolist()
            msg = f"發現重複的樣本名稱: {dup_names}"
            if len(duplicates['Sample_Name'].unique()) > 5:
                msg += f" ... 還有 {len(duplicates['Sample_Name'].unique()) - 5} 個"
            result.add_warning(msg)

        # Check QC samples
        qc_mask = df['Sample_Type'].astype(str).str.upper().str.contains('QC', na=False)
        qc_count = qc_mask.sum()
        result.info['qc_count'] = qc_count

        min_qc = VALIDATION_THRESHOLDS.get('min_qc_samples', 3)
        if qc_count == 0:
            result.add_warning("未找到 QC 樣本 (Sample_Type 中無 'QC')")
        elif qc_count < min_qc:
            result.add_warning(
                f"QC 樣本數量 ({qc_count}) 低於建議最小值 ({min_qc})"
            )

        # Collect sample type distribution
        type_counts = df['Sample_Type'].value_counts().to_dict()
        result.info['sample_types'] = type_counts

        return result

    def validate_raw_intensity(
        self,
        df: pd.DataFrame,
        sample_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate RawIntensity sheet structure and content.

        Checks:
        - DataFrame is not empty
        - FeatureID column exists
        - No duplicate FeatureIDs
        - Sample columns match SampleInfo (if provided)
        - Numeric values are valid

        Args:
            df: RawIntensity DataFrame
            sample_names: Optional list of expected sample names

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()

        # Check empty
        if df is None or df.empty:
            result.add_error("RawIntensity 為空")
            return result

        result.info['feature_count'] = len(df)

        # Check feature ID column (supports both 'Mz/RT' and 'FeatureID')
        if FEATURE_ID_COLUMN not in df.columns and 'FeatureID' not in df.columns:
            if len(df.columns) > 0:
                result.add_warning(
                    f"未找到 '{FEATURE_ID_COLUMN}' 欄位，使用第一欄 '{df.columns[0]}' 作為特徵ID"
                )
            else:
                result.add_error("RawIntensity 沒有任何欄位")
                return result

        # Check for duplicate FeatureIDs
        feature_col = (FEATURE_ID_COLUMN if FEATURE_ID_COLUMN in df.columns
                       else 'FeatureID' if 'FeatureID' in df.columns
                       else df.columns[0])
        duplicates = df[df[feature_col].duplicated(keep=False)]
        if not duplicates.empty:
            dup_count = len(duplicates[feature_col].unique())
            result.add_warning(f"發現 {dup_count} 個重複的 FeatureID")

        # Check sample column matching
        if sample_names:
            data_cols = set(df.columns) - {feature_col}
            sample_set = set(sample_names)

            missing_in_data = sample_set - data_cols
            if missing_in_data:
                sample_list = list(missing_in_data)[:5]
                msg = f"SampleInfo 中的樣本在 RawIntensity 中找不到: {sample_list}"
                if len(missing_in_data) > 5:
                    msg += f" ... 還有 {len(missing_in_data) - 5} 個"
                result.add_warning(msg)

        # Check for extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            max_val = df[numeric_cols].max().max()
            min_val = df[numeric_cols].min().min()
            result.info['value_range'] = (min_val, max_val)

            max_threshold = VALIDATION_THRESHOLDS.get('max_intensity_value', 1e15)
            if max_val > max_threshold:
                result.add_warning(f"發現極端值: {max_val:.2e} (超過 {max_threshold:.2e})")

            if min_val < 0:
                neg_count = (df[numeric_cols] < 0).sum().sum()
                result.add_warning(f"發現 {neg_count} 個負值 (最小: {min_val:.2e})")

        return result

    def validate_istd_signals(
        self,
        istd_df: pd.DataFrame,
        sample_columns: List[str]
    ) -> ValidationResult:
        """
        Validate Internal Standard (ISTD) signals.

        Checks:
        - At least one ISTD is present
        - ISTD signals have valid intensity values
        - ISTD CV% is within acceptable range

        Args:
            istd_df: DataFrame containing ISTD signals
            sample_columns: List of sample column names

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()

        istd_count = len(istd_df)
        result.info['istd_count'] = istd_count

        if istd_count == 0:
            result.add_error("未找到 ISTD 訊號 (無紅色標記的 FeatureID)")
            return result

        min_istd = VALIDATION_THRESHOLDS.get('min_istd_count', 1)
        if istd_count < min_istd:
            result.add_warning(
                f"ISTD 數量 ({istd_count}) 低於建議值 ({min_istd})"
            )

        # Check ISTD intensities
        valid_cols = [c for c in sample_columns if c in istd_df.columns]
        if valid_cols:
            numeric_data = istd_df[valid_cols].apply(pd.to_numeric, errors='coerce')

            # Check for missing values
            missing_pct = numeric_data.isna().sum().sum() / numeric_data.size * 100
            if missing_pct > 50:
                result.add_warning(f"ISTD 訊號有 {missing_pct:.1f}% 的缺失值")

            # Check CV% for each ISTD
            max_cv = VALIDATION_THRESHOLDS.get('max_cv_percent', 30.0)
            high_cv_istds = []

            for idx, row in istd_df.iterrows():
                values = pd.to_numeric(row[valid_cols], errors='coerce')
                clean_values = values[values > 0].dropna()

                if len(clean_values) >= 2:
                    cv = (clean_values.std() / clean_values.mean()) * 100
                    if cv > max_cv:
                        fid = row.get('FeatureID', f'Row {idx}')
                        high_cv_istds.append((fid, cv))

            if high_cv_istds:
                msg = f"有 {len(high_cv_istds)} 個 ISTD 的 CV% 超過 {max_cv}%"
                result.add_warning(msg)
                result.info['high_cv_istds'] = high_cv_istds[:5]

        return result

    def validate_batch_info(
        self,
        sample_info_df: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate batch information for batch effect correction.

        Checks:
        - Batch column exists
        - At least 2 batches present
        - Each batch has sufficient samples

        Args:
            sample_info_df: SampleInfo DataFrame

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()

        if 'Batch' not in sample_info_df.columns:
            result.add_error("SampleInfo 缺少 'Batch' 欄位")
            return result

        # Count batches
        batch_counts = sample_info_df['Batch'].value_counts()
        result.info['batch_counts'] = batch_counts.to_dict()

        # Check NA batches
        na_count = sample_info_df['Batch'].isna().sum()
        if na_count > 0:
            result.add_warning(f"{na_count} 個樣本缺少 Batch 資訊")

        # Check batch count
        valid_batches = batch_counts[batch_counts >= 3]
        if len(valid_batches) < 2:
            result.add_error("需要至少 2 個批次才能進行批次效應校正")

        # Check batch sizes
        small_batches = batch_counts[batch_counts < 3]
        if not small_batches.empty:
            result.add_warning(
                f"有 {len(small_batches)} 個批次樣本數少於 3: {small_batches.to_dict()}"
            )

        return result


def validate_dataframe_numeric(
    df: pd.DataFrame,
    columns: List[str],
    context: str = "DataFrame"
) -> ValidationResult:
    """
    Validate that specified columns contain valid numeric data.

    Args:
        df: DataFrame to validate
        columns: Columns that should be numeric
        context: Context string for error messages

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    for col in columns:
        if col not in df.columns:
            continue

        # Try to convert to numeric
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        non_numeric = df[col][numeric_col.isna() & df[col].notna()]

        if not non_numeric.empty:
            sample = non_numeric.head(3).tolist()
            result.add_warning(
                f"{context} 欄位 '{col}' 包含非數值資料: {sample}"
            )

    return result


def quick_validate_excel(file_path: str) -> Tuple[bool, List[str]]:
    """
    Quick validation of an Excel file for metabolomics processing.

    This is a convenience function that performs basic validation
    and returns a simple pass/fail result.

    Args:
        file_path: Path to the Excel file

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    validator = DataValidator()

    # Validate file
    result = validator.validate_file_path(file_path)
    if not result.is_valid:
        return False, result.errors

    # Try to load and validate sheets
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        # Check required sheets
        required = ['RawIntensity', 'SampleInfo']
        missing = [s for s in required if s not in sheet_names]
        if missing:
            return False, [f"缺少必要的工作表: {', '.join(missing)}"]

        # Validate SampleInfo
        sample_info = pd.read_excel(excel_file, sheet_name='SampleInfo')
        sample_result = validator.validate_sample_info(sample_info)
        if not sample_result.is_valid:
            return False, sample_result.errors

        # Validate RawIntensity
        raw_intensity = pd.read_excel(excel_file, sheet_name='RawIntensity')
        sample_names = sample_info['Sample_Name'].tolist() if 'Sample_Name' in sample_info.columns else None
        raw_result = validator.validate_raw_intensity(raw_intensity, sample_names)
        if not raw_result.is_valid:
            return False, raw_result.errors

        return True, []

    except Exception as e:
        return False, [f"讀取 Excel 檔案時發生錯誤: {e}"]
