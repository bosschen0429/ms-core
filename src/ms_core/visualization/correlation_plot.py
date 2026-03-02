"""
相關性矩陣可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def plot_correlation_heatmap(
    corr_result,
    fig: Figure = None,
    cmap: str = "RdBu_r",
    annot: bool = False,
    max_features: int = 30,
):
    """
    繪製相關性熱圖

    Parameters
    ----------
    corr_result : CorrelationResult
    fig : matplotlib Figure (可選)
    cmap : 色盤
    annot : 是否標註數值
    max_features : 最多顯示的特徵數
    """
    corr = corr_result.corr_matrix
    if corr.shape[0] > max_features:
        corr = corr.iloc[:max_features, :max_features]

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    fig.clf()
    ax = fig.add_subplot(111)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        square=True, linewidths=0.5,
        annot=annot, fmt=".2f" if annot else "",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=-1, vmax=1,
    )
    method_name = corr_result.method.capitalize()
    ax.set_title(f"{method_name} 相關矩陣 (Top {corr.shape[0]} 特徵)")
    fig.tight_layout()
    return fig


def plot_correlation_network(
    corr_result,
    fig: Figure = None,
    threshold: float = 0.8,
    top_n: int = 30,
):
    """
    繪製高相關性特徵對 (橫條圖)

    Parameters
    ----------
    corr_result : CorrelationResult
    fig : matplotlib Figure
    threshold : 相關性門檻
    top_n : 最多顯示的特徵對數
    """
    pairs = corr_result.high_corr_pairs
    if len(pairs) == 0:
        if fig is None:
            fig = plt.figure(figsize=(8, 4))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"沒有相關性 >= {threshold} 的特徵對",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig

    plot_df = pairs.head(top_n).copy()
    plot_df["Label"] = plot_df["Feature_1"] + " vs " + plot_df["Feature_2"]

    if fig is None:
        fig = plt.figure(figsize=(8, max(4, len(plot_df) * 0.35)))
    fig.clf()
    ax = fig.add_subplot(111)

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in plot_df["Correlation"]]
    ax.barh(range(len(plot_df)), plot_df["Correlation"].values, color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Label"].values, fontsize=8)
    ax.set_xlabel("相關係數")
    ax.set_title(f"高相關特徵對 (|r| >= {threshold})")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
