"""
Column-wise 縮放模組 — 複製 MetaboAnalyst 6.0 的特徵間縮放

對應 R 函式:
  Normalization() 中的 scaleNorm 參數
"""

import numpy as np
import pandas as pd


SCALING_METHODS = {
    "None": "不縮放",
    "MeanCenter": "均值中心化 (Mean Centering)",
    "AutoNorm": "自動縮放 (Auto Scaling / Z-score)",
    "ParetoNorm": "帕累托縮放 (Pareto Scaling)",
    "RangeNorm": "範圍縮放 (Range Scaling)",
}


class ColumnScaler:
    """MetaboAnalyst Column-wise Scaling"""

    @staticmethod
    def mean_center(df: pd.DataFrame) -> pd.DataFrame:
        """均值中心化: x' = x - mean"""
        return df - df.mean()

    @staticmethod
    def auto_scale(df: pd.DataFrame) -> pd.DataFrame:
        """Auto scaling (Z-score): x' = (x - mean) / sd"""
        std = df.std()
        std = std.replace(0, np.nan)
        return (df - df.mean()) / std

    @staticmethod
    def pareto_scale(df: pd.DataFrame) -> pd.DataFrame:
        """Pareto scaling: x' = (x - mean) / sqrt(sd)"""
        std = df.std()
        std = std.replace(0, np.nan)
        return (df - df.mean()) / np.sqrt(std)

    @staticmethod
    def range_scale(df: pd.DataFrame) -> pd.DataFrame:
        """Range scaling: x' = (x - mean) / (max - min)"""
        r = df.max() - df.min()
        r = r.replace(0, np.nan)
        return (df - df.mean()) / r


def apply_scaling(df: pd.DataFrame, method: str = "None") -> pd.DataFrame:
    """
    統一的 column-wise 縮放入口

    Parameters
    ----------
    df : DataFrame
    method : str
        None, MeanCenter, AutoNorm, ParetoNorm, RangeNorm
    """
    s = ColumnScaler()
    methods = {
        "None": lambda x: x.copy(),
        "MeanCenter": s.mean_center,
        "AutoNorm": s.auto_scale,
        "ParetoNorm": s.pareto_scale,
        "RangeNorm": s.range_scale,
    }
    if method not in methods and method is not None:
        raise ValueError(f"未知的縮放方法: {method}")
    if method is None:
        return df.copy()
    return methods[method](df)
