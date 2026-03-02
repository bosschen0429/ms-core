"""
Heatmap with Hierarchical Clustering — MetaboAnalyst 風格

對應 R: pheatmap::pheatmap() + hclust()
預設: euclidean 距離, ward 連結, row scaling, RdBu_r 色盤
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ms_core.analysis.clustering import select_top_features


def plot_heatmap(
    df: pd.DataFrame,
    labels,
    method: str = "ward",
    metric: str = "euclidean",
    scale: str = "row",
    max_features: int = 2000,
    top_by: str = "var",
    fig=None,
):
    """
    繪製 MetaboAnalyst 風格的 Heatmap

    Parameters
    ----------
    df : DataFrame
        標準化後的數值資料
    labels : Series or array
        分組標籤
    method : str
        連結方法: ward, complete, average, single
    metric : str
        距離指標: euclidean, correlation, cosine
    scale : str
        縮放方式: "row", "col", None
    max_features : int
        最大顯示特徵數
    top_by : str
        選擇 top 特徵的依據: "var", "mad"
    """
    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    # 限制特徵數
    plot_df = select_top_features(df, max_features=max_features, by=top_by)

    # seaborn standard_scale: 0=row, 1=col
    if scale == "row":
        standard_scale = 0
    elif scale == "col":
        standard_scale = 1
    else:
        standard_scale = None

    # 組別色帶
    groups = sorted(set(labels_arr))
    palette = dict(zip(groups, sns.color_palette("Set1", len(groups))))
    row_colors = pd.Series(labels_arr, index=plot_df.index).map(palette)

    # ward 強制 euclidean
    if method == "ward":
        metric = "euclidean"

    try:
        g = sns.clustermap(
            plot_df,
            method=method,
            metric=metric,
            standard_scale=standard_scale,
            cmap="RdBu_r",
            figsize=(12, 8),
            row_colors=row_colors,
            dendrogram_ratio=(0.12, 0.12),
            linewidths=0,
            xticklabels=False if plot_df.shape[1] > 50 else True,
            yticklabels=False if plot_df.shape[0] > 50 else True,
        )
        g.fig.suptitle("Heatmap with Hierarchical Clustering", y=1.01, fontsize=12)

        # 加圖例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=palette[g_name], label=str(g_name)) for g_name in groups]
        g.ax_heatmap.legend(
            handles=legend_elements, loc="upper left",
            bbox_to_anchor=(1.02, 1), fontsize=8, title="Group",
        )
        return g.fig
    except Exception as e:
        # fallback: 如果 clustermap 失敗，用簡單 heatmap
        if fig is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig.clear()
            ax = fig.add_subplot(111)
        sns.heatmap(plot_df, cmap="RdBu_r", ax=ax, xticklabels=False)
        ax.set_title(f"Heatmap (clustering 失敗: {e})")
        fig.tight_layout()
        return fig
