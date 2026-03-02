"""
Row-wise 標準化模組 — 複製 MetaboAnalyst 6.0 的樣本間校正

對應 R 函式:
  Normalization(mSetObj, rowNorm, transNorm, scaleNorm, ref=NULL)
  （此模組只處理 rowNorm 部分）
"""

import numpy as np
import pandas as pd

try:
    import qnorm
    HAS_QNORM = True
except ImportError:
    HAS_QNORM = False


ROW_NORM_METHODS = {
    "None": "不處理",
    "SumNorm": "總量標準化 (Sum)",
    "MedianNorm": "中位數標準化 (Median)",
    "SamplePQN": "PQN — 參考樣本",
    "GroupPQN": "PQN — 參考組",
    "CompNorm": "內標校正 (ISTD)",
    "QuantileNorm": "分位數標準化",
    "SpecNorm": "自訂校正因子",
}


class RowNormalizer:
    """MetaboAnalyst Row-wise Normalization"""

    @staticmethod
    def sum_norm(df: pd.DataFrame) -> pd.DataFrame:
        """每個樣本縮放至總強度 1000"""
        row_sums = df.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        return df.div(row_sums, axis=0) * 1000

    @staticmethod
    def median_norm(df: pd.DataFrame) -> pd.DataFrame:
        """每個樣本除以該樣本中位數"""
        row_medians = df.median(axis=1)
        row_medians = row_medians.replace(0, np.nan)
        return df.div(row_medians, axis=0)

    @staticmethod
    def pqn_sample(df: pd.DataFrame, ref_sample: pd.Series) -> pd.DataFrame:
        """PQN：以參考樣本為基準"""
        ref_sample = ref_sample.replace(0, np.nan)
        quotients = df.div(ref_sample, axis=1)
        factors = quotients.median(axis=1)
        factors = factors.replace(0, np.nan)
        return df.div(factors, axis=0)

    @staticmethod
    def pqn_group(df: pd.DataFrame, group_mask: np.ndarray) -> pd.DataFrame:
        """PQN：以參考組（如 QC 或 control）的均值為基準"""
        ref_profile = df[group_mask].mean(axis=0)
        return RowNormalizer.pqn_sample(df, ref_profile)

    @staticmethod
    def comp_norm(df: pd.DataFrame, ref_feature: str) -> pd.DataFrame:
        """內標校正：除以參考特徵後 x1000，並移除參考特徵"""
        ref_values = df[ref_feature].replace(0, np.nan)
        result = df.div(ref_values, axis=0) * 1000
        return result.drop(columns=[ref_feature])

    @staticmethod
    def quantile_norm(df: pd.DataFrame) -> pd.DataFrame:
        """分位數標準化"""
        if not HAS_QNORM:
            raise ImportError("需要安裝 qnorm: pip install qnorm")
        return qnorm.quantile_normalize(df, axis=0)

    @staticmethod
    def spec_norm(df: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
        """以使用者提供的校正因子（如組織重量、體積）除之"""
        aligned = factors.reindex(df.index)
        aligned = aligned.replace(0, np.nan)
        return df.div(aligned, axis=0)


def apply_row_norm(
    df: pd.DataFrame,
    method: str = "None",
    ref_sample: pd.Series = None,
    ref_feature: str = None,
    group_mask: np.ndarray = None,
    factors: pd.Series = None,
) -> pd.DataFrame:
    """
    統一的 row-wise 標準化入口

    Parameters
    ----------
    df : DataFrame
    method : str
        None, SumNorm, MedianNorm, SamplePQN, GroupPQN, CompNorm, QuantileNorm, SpecNorm
    ref_sample, ref_feature, group_mask, factors
        依方法需要傳入的參數
    """
    norm = RowNormalizer()

    if method == "None" or method is None:
        return df.copy()
    elif method == "SumNorm":
        return norm.sum_norm(df)
    elif method == "MedianNorm":
        return norm.median_norm(df)
    elif method == "SamplePQN":
        if ref_sample is None:
            raise ValueError("SamplePQN 需要提供 ref_sample")
        return norm.pqn_sample(df, ref_sample)
    elif method == "GroupPQN":
        if group_mask is None:
            raise ValueError("GroupPQN 需要提供 group_mask")
        return norm.pqn_group(df, group_mask)
    elif method == "CompNorm":
        if ref_feature is None:
            raise ValueError("CompNorm 需要提供 ref_feature")
        return norm.comp_norm(df, ref_feature)
    elif method == "QuantileNorm":
        return norm.quantile_norm(df)
    elif method == "SpecNorm":
        if factors is None:
            raise ValueError("SpecNorm 需要提供 factors")
        return norm.spec_norm(df, factors)
    else:
        raise ValueError(f"未知的標準化方法: {method}")
