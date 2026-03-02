"""
資料轉換模組 — 複製 MetaboAnalyst 6.0 的 Data Transformation

重要：MetaboAnalyst 的 "LogNorm" 是 Generalized Log（廣義對數），
不是標準的 log2(x+1)。公式為:
  glog2(x) = log2((x + sqrt(x^2 + lambda^2)) / 2)

其中 lambda = min(非零絕對值) / 10

性質：
- x >> lambda 時，glog2(x) ≈ log2(x)
- x = 0 時，glog2(0) = log2(lambda/2)，為有限值
- x < 0 時仍產生實數
"""

import numpy as np
import pandas as pd


TRANSFORM_METHODS = {
    "None": "不轉換",
    "LogNorm": "廣義 Log₂ (glog2)",
    "Log10Norm": "廣義 Log₁₀ (glog10)",
    "SrNorm": "廣義平方根 (gsqrt)",
    "CrNorm": "立方根 (Cube Root)",
}


class DataTransformer:
    """MetaboAnalyst Data Transformation — 使用 Generalized Log"""

    @staticmethod
    def _get_lambda(df: pd.DataFrame) -> float:
        """計算 glog 所需的 lambda 常數: min(非零絕對值) / 10"""
        nonzero = df[df != 0].abs()
        min_val = nonzero.min().min()
        if np.isnan(min_val) or min_val == 0:
            return 1e-10
        return min_val / 10

    @staticmethod
    def glog2(df: pd.DataFrame) -> pd.DataFrame:
        """廣義 log2 轉換（MetaboAnalyst 預設）"""
        lam = DataTransformer._get_lambda(df)
        return np.log2((df + np.sqrt(df**2 + lam**2)) / 2)

    @staticmethod
    def glog10(df: pd.DataFrame) -> pd.DataFrame:
        """廣義 log10 轉換"""
        lam = DataTransformer._get_lambda(df)
        return np.log10((df + np.sqrt(df**2 + lam**2)) / 2)

    @staticmethod
    def gsqrt(df: pd.DataFrame) -> pd.DataFrame:
        """廣義平方根轉換"""
        lam = DataTransformer._get_lambda(df)
        return np.sqrt((df + np.sqrt(df**2 + lam**2)) / 2)

    @staticmethod
    def cube_root(df: pd.DataFrame) -> pd.DataFrame:
        """立方根轉換（保留符號，不使用 glog）"""
        return np.sign(df) * np.abs(df) ** (1 / 3)


def apply_transform(df: pd.DataFrame, method: str = "None") -> pd.DataFrame:
    """
    統一的資料轉換入口

    Parameters
    ----------
    df : DataFrame
    method : str
        None, LogNorm, Log10Norm, SrNorm, CrNorm
    """
    t = DataTransformer()
    methods = {
        "None": lambda x: x.copy(),
        "LogNorm": t.glog2,
        "Log10Norm": t.glog10,
        "SrNorm": t.gsqrt,
        "CrNorm": t.cube_root,
    }
    if method not in methods and method is not None:
        raise ValueError(f"未知的轉換方法: {method}")
    if method is None:
        return df.copy()
    return methods[method](df)
