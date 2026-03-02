"""
PCA 分析模組 — 複製 MetaboAnalyst 6.0 的 PCA.Anal()

使用 sklearn PCA (SVD-based)，等同 R 的 prcomp()
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAResult:
    """PCA 分析結果容器"""

    def __init__(self, scores, loadings, explained_variance_ratio, feature_names, sample_names, labels):
        self.scores = scores                                  # n_samples x n_components
        self.loadings = loadings                              # n_features x n_components
        self.explained_variance_ratio = explained_variance_ratio
        self.feature_names = feature_names
        self.sample_names = sample_names
        self.labels = labels
        self.n_components = scores.shape[1]

    def get_score_df(self) -> pd.DataFrame:
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        df = pd.DataFrame(self.scores, index=self.sample_names, columns=cols)
        df["Group"] = self.labels.values if hasattr(self.labels, "values") else self.labels
        return df

    def get_loading_df(self) -> pd.DataFrame:
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(self.loadings, index=self.feature_names, columns=cols)

    def get_variance_text(self, pc: int) -> str:
        return f"PC{pc+1} ({self.explained_variance_ratio[pc]*100:.1f}%)"


def run_pca(data: pd.DataFrame, labels, n_components: int = 5) -> PCAResult:
    """
    執行 PCA 分析

    Parameters
    ----------
    data : DataFrame
        已標準化的數值資料（樣本 x 特徵）
    labels : Series or array
        分組標籤
    n_components : int
        主成分數量，預設 5

    Returns
    -------
    PCAResult
    """
    n_components = min(n_components, min(data.shape))

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data.values)
    loadings = pca.components_.T  # 轉為 features x components

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=pca.explained_variance_ratio_,
        feature_names=list(data.columns),
        sample_names=list(data.index),
        labels=labels,
    )
