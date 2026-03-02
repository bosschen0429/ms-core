"""
Variable filtering utilities following MetaboAnalyst behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import iqr, median_abs_deviation


def get_auto_cutoff(n_features: int) -> float:
    """
    Adaptive removal proportion based on feature count.

    < 250: 5%
    250-499: 10%
    500-999: 25%
    >= 1000: 40%
    """
    if n_features < 250:
        return 0.05
    if n_features < 500:
        return 0.10
    if n_features < 1000:
        return 0.25
    return 0.40


FILTER_METHODS = {
    "None": "No filtering",
    "iqr": "Interquartile range (IQR)",
    "sd": "Standard deviation (SD)",
    "mad": "Median absolute deviation (MAD)",
    "rsd": "Relative standard deviation (RSD)",
    "nrsd": "Normalized RSD (MAD / median)",
}


def compute_filter_scores(df: pd.DataFrame, method: str = "iqr") -> pd.Series:
    """
    Compute per-feature dispersion score.
    """
    if method in (None, "None"):
        return pd.Series(1.0, index=df.columns)
    if method == "iqr":
        return df.apply(iqr)
    if method == "sd":
        return df.std()
    if method == "mad":
        return df.apply(median_abs_deviation)
    if method == "rsd":
        means = df.mean().replace(0, np.nan)
        return df.std() / means
    if method == "nrsd":
        medians = df.median().replace(0, np.nan)
        return df.apply(median_abs_deviation) / medians
    raise ValueError(f"未知過濾方法: {method}")


def filter_features(
    df: pd.DataFrame,
    method: str = "iqr",
    cutoff: float | None = None,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Keep high-dispersion features after quantile thresholding.

    `cutoff` is removal quantile in [0, 1].
    """
    if method in (None, "None"):
        return df.copy()

    n_features = df.shape[1]
    if cutoff is None:
        cutoff = get_auto_cutoff(n_features)

    scores = compute_filter_scores(df, method=method)
    threshold = scores.quantile(cutoff)
    filtered = df.loc[:, scores >= threshold]

    if filtered.shape[1] > max_features:
        top_idx = scores[filtered.columns].nlargest(max_features).index
        filtered = filtered[top_idx]

    return filtered


def filter_by_qc_rsd(
    df: pd.DataFrame,
    qc_mask: np.ndarray,
    rsd_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Filter features using RSD calculated from QC samples only.
    QC rows are removed from output.
    """
    qc_data = df[qc_mask]
    means = qc_data.mean().replace(0, np.nan)
    rsd = qc_data.std() / means
    keep = rsd[rsd.abs() <= rsd_threshold].index
    return df.loc[~qc_mask, keep]
