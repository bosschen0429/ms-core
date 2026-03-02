"""
PCA-based outlier detection (Hotelling's T2 + DModX).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.decomposition import PCA


@dataclass
class OutlierResult:
    scores: np.ndarray
    t2_values: np.ndarray
    t2_threshold: float
    dmodx: np.ndarray
    dmodx_threshold: float
    outlier_mask_t2: np.ndarray
    outlier_mask_dmodx: np.ndarray
    sample_names: list
    explained_variance: np.ndarray
    pca_model: PCA

    def get_outlier_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sample": self.sample_names,
                "T2": self.t2_values,
                "T2_Outlier": self.outlier_mask_t2,
                "DModX": self.dmodx,
                "DModX_Outlier": self.outlier_mask_dmodx,
                "Any_Outlier": self.outlier_mask_t2 | self.outlier_mask_dmodx,
            }
        )


def run_outlier_detection(
    df: pd.DataFrame,
    n_components: int = 2,
    alpha: float = 0.05,
) -> OutlierResult:
    """
    Detect outliers from PCA scores and reconstruction residuals.
    """
    n, p = df.shape
    if n < 2 or p < 1:
        raise ValueError("Outlier detection requires at least 2 samples and 1 feature.")

    k = min(n_components, n - 1, p)
    if k < 1:
        raise ValueError("No valid PCA component available for outlier detection.")

    pca = PCA(n_components=k)
    scores = pca.fit_transform(df.values)

    eigenvalues = np.where(pca.explained_variance_ > 0, pca.explained_variance_, np.nan)
    t2_values = np.nansum((scores**2) / eigenvalues, axis=1)
    t2_values = np.nan_to_num(t2_values, nan=0.0, posinf=0.0, neginf=0.0)

    denom = max(n - k, 1)
    f_crit = f_dist.ppf(1 - alpha, k, denom)
    t2_threshold = k * (n - 1) / denom * f_crit if np.isfinite(f_crit) else np.inf
    outlier_t2 = t2_values > t2_threshold

    reconstructed = scores @ pca.components_ + pca.mean_
    residuals = df.values - reconstructed
    dmodx_denom = max(p - k, 1)
    dmodx = np.sqrt(np.sum(residuals**2, axis=1) / dmodx_denom)
    dmodx_threshold = float(np.percentile(dmodx, (1 - alpha) * 100))
    outlier_dmodx = dmodx > dmodx_threshold

    return OutlierResult(
        scores=scores,
        t2_values=t2_values,
        t2_threshold=float(t2_threshold),
        dmodx=dmodx,
        dmodx_threshold=dmodx_threshold,
        outlier_mask_t2=outlier_t2,
        outlier_mask_dmodx=outlier_dmodx,
        sample_names=list(df.index),
        explained_variance=pca.explained_variance_ratio_,
        pca_model=pca,
    )
