"""
階層式分群模組 — 用於 Heatmap 的 dendrogram 計算

對應 R: hclust() + pheatmap::pheatmap()
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


def compute_linkage(data: np.ndarray, method: str = "ward", metric: str = "euclidean"):
    """
    計算階層式分群的 linkage matrix

    Parameters
    ----------
    data : 2D array
        資料矩陣
    method : str
        連結方法: ward, complete, average, single
    metric : str
        距離指標: euclidean, correlation, cosine 等
    """
    if method == "ward":
        # ward 方法只支援 euclidean
        return linkage(data, method="ward", metric="euclidean")
    else:
        dist = pdist(data, metric=metric)
        return linkage(dist, method=method)


def select_top_features(df: pd.DataFrame, max_features: int = 2000,
                        by: str = "var") -> pd.DataFrame:
    """
    選擇 top N 變異最大的特徵

    Parameters
    ----------
    df : DataFrame
    max_features : int
        最大特徵數
    by : str
        排序依據: "var" (變異數), "mad" (中位數絕對偏差)
    """
    if df.shape[1] <= max_features:
        return df

    if by == "mad":
        from scipy.stats import median_abs_deviation
        scores = df.apply(median_abs_deviation)
    else:
        scores = df.var()

    top_cols = scores.nlargest(max_features).index
    return df[top_cols]
