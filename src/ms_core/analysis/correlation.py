"""
相關性分析模組 — Pearson / Spearman 相關矩陣

用途：
  - 特徵間相關性分析
  - 高度相關特徵偵測（冗餘移除）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrelationResult:
    """相關性分析結果"""
    corr_matrix: pd.DataFrame
    method: str
    high_corr_pairs: pd.DataFrame  # 高相關特徵對

    def get_top_pairs(self, n: int = 20) -> pd.DataFrame:
        """取得相關性最高的 n 對特徵"""
        return self.high_corr_pairs.head(n)


def run_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.9,
    top_features: Optional[int] = 50,
) -> CorrelationResult:
    """
    計算特徵間相關矩陣

    Parameters
    ----------
    df : DataFrame
        數值資料框（樣本 × 特徵）
    method : str
        'pearson' 或 'spearman'
    threshold : float
        高相關性門檻，用於找出冗餘特徵對
    top_features : int or None
        僅使用變異最大的 N 個特徵（避免矩陣過大）

    Returns
    -------
    CorrelationResult
    """
    # 限制特徵數
    if top_features and df.shape[1] > top_features:
        top_cols = df.var().nlargest(top_features).index
        df_sub = df[top_cols]
    else:
        df_sub = df

    # 計算相關矩陣
    corr_matrix = df_sub.corr(method=method)

    # 找出高相關特徵對
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({
                    "Feature_1": cols[i],
                    "Feature_2": cols[j],
                    "Correlation": r,
                    "Abs_Correlation": abs(r),
                })

    high_corr_df = pd.DataFrame(pairs)
    if len(high_corr_df) > 0:
        high_corr_df = high_corr_df.sort_values("Abs_Correlation", ascending=False)
        high_corr_df = high_corr_df.reset_index(drop=True)

    return CorrelationResult(
        corr_matrix=corr_matrix,
        method=method,
        high_corr_pairs=high_corr_df,
    )
