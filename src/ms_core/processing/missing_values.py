"""
缺失值處理模組 — 複製 MetaboAnalyst 6.0 的 Missing Value Handling

對應 R 函式:
  - SanityCheckData() 中的零值轉 NA
  - RemoveMissingPercent(mSetObj, percent=0.5)
  - ImputeMissingVar(mSetObj, method="min")
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """將所有 0 值轉為 NaN（MetaboAnalyst SanityCheckData 行為）"""
    return df.replace(0, np.nan)


def remove_missing_percent(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    移除缺失比例 >= threshold 的特徵欄位

    Parameters
    ----------
    df : DataFrame
        僅含數值特徵的資料框
    threshold : float
        缺失比例門檻，預設 0.5（50%）

    Returns
    -------
    DataFrame
        過濾後的資料框
    """
    missing_ratio = df.isna().sum() / len(df)
    return df.loc[:, missing_ratio < threshold]


def _impute_min_lod(df: pd.DataFrame) -> pd.DataFrame:
    """每個特徵的 LoD = min(正值) / 5，所有 NA 以 LoD 替代"""
    df_out = df.copy()
    for col in df_out.columns:
        pos_vals = df_out[col][df_out[col] > 0]
        if len(pos_vals) > 0:
            lod = pos_vals.min() / 5
        else:
            lod = 1e-10
        df_out[col] = df_out[col].fillna(lod)
    return df_out


def _impute_knn(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """KNN 填補，k=10"""
    imputer = KNNImputer(n_neighbors=k)
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)


def _impute_svd(df: pd.DataFrame, rank: int = 2) -> pd.DataFrame:
    """SVD 填補（需 fancyimpute）"""
    try:
        from fancyimpute import IterativeSVD
    except ImportError:
        raise ImportError("需要安裝 fancyimpute: pip install fancyimpute")
    imputer = IterativeSVD(rank=rank)
    imputed = imputer.fit_transform(df.values)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)


def _impute_ppca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """PPCA 填補（需 pyppca）"""
    try:
        import pyppca
    except ImportError:
        raise ImportError("需要安裝 pyppca: pip install pyppca")
    result = pyppca.ppca(df.values, d=n_components, dia=False)
    imputed = result[4]  # C matrix (completed data)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)


def _impute_bpca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """BPCA 填補（使用 BayesianRidge + IterativeImputer）"""
    try:
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
    except ImportError:
        raise ImportError("需要 scikit-learn >= 0.21 的 IterativeImputer")
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=100,
        random_state=0
    )
    imputed = imputer.fit_transform(df.values)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)


# 方法名稱到顯示標籤的對應（進階方法始終列出，缺少套件時在呼叫時報錯）
IMPUTE_METHODS = {
    "min": "最小值/5 (LoD)",
    "mean": "平均值",
    "median": "中位數",
    "exclude": "移除含缺失特徵",
    "knn": "KNN (k=10)",
    "ppca": "PPCA (nPcs=2)",
    "bpca": "BPCA (nPcs=2)",
    "svd": "SVD (rank=2)",
}


def impute_missing(df: pd.DataFrame, method: str = "min", **kwargs) -> pd.DataFrame:
    """
    統一的缺失值填補入口

    Parameters
    ----------
    df : DataFrame
        含有 NaN 的數值資料框
    method : str
        填補方法: min, mean, median, exclude, knn, ppca, bpca, svd
    **kwargs
        傳給特定填補方法的額外參數（如 k, n_components, rank）

    Returns
    -------
    DataFrame
        填補後的資料框
    """
    if method == "min" or method == "lod":
        return _impute_min_lod(df)
    elif method == "mean":
        return df.fillna(df.mean())
    elif method == "median":
        return df.fillna(df.median())
    elif method == "exclude":
        return df.dropna(axis=1)
    elif method == "knn":
        return _impute_knn(df, k=kwargs.get("k", 10))
    elif method == "ppca":
        return _impute_ppca(df, n_components=kwargs.get("n_components", 2))
    elif method == "bpca":
        return _impute_bpca(df, n_components=kwargs.get("n_components", 2))
    elif method == "svd":
        return _impute_svd(df, rank=kwargs.get("rank", 2))
    else:
        raise ValueError(f"未知的填補方法: {method}")
