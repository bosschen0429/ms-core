"""
Safe mathematical operations for metabolomics calculations.

This module provides functions that handle edge cases like division by zero,
empty arrays, and NaN values that are common in metabolomics data processing.
"""
import numpy as np
import pandas as pd
from typing import Union, List

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray, pd.Series, List[float]]


def safe_divide(
    numerator: ArrayLike,
    denominator: ArrayLike,
    fill_value: float = np.nan
) -> Union[float, np.ndarray]:
    """
    Safe division that handles zero denominators.

    Args:
        numerator: Dividend (scalar or array)
        denominator: Divisor (scalar or array)
        fill_value: Value to use when denominator is zero (default: np.nan)

    Returns:
        Result of division, or fill_value where denominator was zero

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        nan
        >>> safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]))
        array([ 5., nan,  6.])
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    # Use numpy's where for vectorized conditional
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            fill_value
        )

    # Return scalar if inputs were scalar
    if result.ndim == 0:
        return float(result)
    return result


def safe_cv_percent(
    values: ArrayLike,
    min_samples: int = 2,
    ddof: int = 1
) -> float:
    """
    Calculate coefficient of variation (CV%) safely.

    Handles edge cases:
    - Fewer than min_samples values: returns NaN
    - Mean is zero: returns NaN
    - All values are NaN: returns NaN
    - Negative values: filtered out (intensities should be positive)

    Args:
        values: Array of values to calculate CV% for
        min_samples: Minimum number of valid values required (default: 2)
        ddof: Delta degrees of freedom for std calculation (default: 1)

    Returns:
        CV% as float, or np.nan if calculation not possible

    Examples:
        >>> safe_cv_percent([100, 110, 90, 105])
        8.165...
        >>> safe_cv_percent([100])  # Too few samples
        nan
        >>> safe_cv_percent([0, 0, 0])  # Zero mean
        nan
    """
    values = np.asarray(values, dtype=float)

    # Filter to valid positive values
    valid_mask = np.isfinite(values) & (values > 0)
    clean_values = values[valid_mask]

    if len(clean_values) < min_samples:
        return np.nan

    mean_val = np.mean(clean_values)
    if mean_val == 0:
        return np.nan

    std_val = np.std(clean_values, ddof=ddof)
    return (std_val / mean_val) * 100


def safe_cv_percent_vectorized(
    data_matrix: np.ndarray,
    axis: int = 1,
    min_samples: int = 2,
    ddof: int = 1
) -> np.ndarray:
    """
    Calculate CV% for each row/column of a matrix (vectorized).

    This is much faster than calling safe_cv_percent in a loop.

    Args:
        data_matrix: 2D array of values
        axis: Axis along which to calculate (1=rows, 0=columns)
        min_samples: Minimum valid samples required per row/column
        ddof: Delta degrees of freedom for std

    Returns:
        1D array of CV% values

    Examples:
        >>> data = np.array([[100, 110, 90], [200, 200, 200]])
        >>> safe_cv_percent_vectorized(data)
        array([10.0, 0.0])
    """
    data_matrix = np.asarray(data_matrix, dtype=float)

    # Replace non-positive values with NaN for calculation
    data_clean = np.where(data_matrix > 0, data_matrix, np.nan)

    # Count valid values per row/column
    valid_counts = np.sum(np.isfinite(data_clean), axis=axis)

    # Calculate mean and std using nanmean/nanstd
    with np.errstate(all='ignore'):
        means = np.nanmean(data_clean, axis=axis)
        stds = np.nanstd(data_clean, axis=axis, ddof=ddof)

        # Calculate CV%
        cv_percent = np.where(
            (valid_counts >= min_samples) & (means != 0),
            (stds / means) * 100,
            np.nan
        )

    return cv_percent


def safe_log_transform(
    values: ArrayLike,
    base: float = 2,
    offset: float = 1.0
) -> np.ndarray:
    """
    Safe log transformation with offset to handle zeros.

    Args:
        values: Values to transform
        base: Log base (default: 2 for log2)
        offset: Offset to add before log (default: 1)

    Returns:
        Log-transformed values
    """
    values = np.asarray(values, dtype=float)

    # Add offset and take log
    with np.errstate(divide='ignore', invalid='ignore'):
        if base == 2:
            result = np.log2(values + offset)
        elif base == 10:
            result = np.log10(values + offset)
        elif base == np.e:
            result = np.log(values + offset)
        else:
            result = np.log(values + offset) / np.log(base)

    # Replace -inf with NaN
    result = np.where(np.isfinite(result), result, np.nan)
    return result


def safe_normalize(
    values: ArrayLike,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Safe normalization that handles edge cases.

    Args:
        values: Values to normalize
        method: 'minmax' for 0-1 scaling, 'zscore' for standardization

    Returns:
        Normalized values
    """
    values = np.asarray(values, dtype=float)

    if method == 'minmax':
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        range_val = max_val - min_val

        if range_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / range_val

    elif method == 'zscore':
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)

        if std_val == 0:
            return np.zeros_like(values)
        return (values - mean_val) / std_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_valid_numeric_values(
    row: pd.Series,
    columns: List[str]
) -> np.ndarray:
    """
    Extract valid numeric values from a row for specified columns.

    Faster alternative to iterating with get_valid_values() in loops.

    Args:
        row: pandas Series (a row from a DataFrame)
        columns: List of column names to extract

    Returns:
        numpy array of valid (finite, positive) values
    """
    values = []
    for col in columns:
        if col in row.index:
            try:
                val = float(row[col])
                if np.isfinite(val) and val > 0:
                    values.append(val)
            except (ValueError, TypeError):
                pass
    return np.array(values)


def extract_numeric_matrix(
    df: pd.DataFrame,
    columns: List[str],
    replace_invalid: float = np.nan
) -> np.ndarray:
    """
    Extract a numeric matrix from DataFrame columns.

    Much faster than row-by-row extraction with iterrows().

    Args:
        df: Source DataFrame
        columns: Columns to extract
        replace_invalid: Value to use for invalid entries (default: NaN)

    Returns:
        2D numpy array (rows x columns)
    """
    # Select columns and convert to numeric
    subset = df[columns].apply(pd.to_numeric, errors='coerce')

    # Get numpy array
    matrix = subset.values.astype(float)

    # Replace non-positive values if needed
    if not np.isnan(replace_invalid):
        matrix = np.where(matrix > 0, matrix, replace_invalid)

    return matrix
