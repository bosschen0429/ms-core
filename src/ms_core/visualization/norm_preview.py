"""
標準化前後對比預覽圖

左: 標準化前 (boxplot + density)
右: 標準化後 (boxplot + density)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_norm_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    labels,
    fig=None,
):
    """
    繪製標準化前後對比 (2x2: boxplot + density)

    Parameters
    ----------
    before_df : DataFrame
        標準化前的資料
    after_df : DataFrame
        標準化後的資料
    labels : array-like
        分組標籤
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    else:
        fig.clear()

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    groups = sorted(set(labels_arr))
    cmap = plt.cm.Set1
    color_map = {g: cmap(i / max(len(groups) - 1, 1)) for i, g in enumerate(groups)}

    # ── 左上: Before Boxplot ──
    ax1 = fig.add_subplot(2, 2, 1)
    _draw_group_box(ax1, before_df, labels_arr, groups, color_map, "標準化前")

    # ── 右上: After Boxplot ──
    ax2 = fig.add_subplot(2, 2, 2)
    _draw_group_box(ax2, after_df, labels_arr, groups, color_map, "標準化後")

    # ── 左下: Before Density ──
    ax3 = fig.add_subplot(2, 2, 3)
    _draw_density(ax3, before_df, labels_arr, groups, color_map, "標準化前")

    # ── 右下: After Density ──
    ax4 = fig.add_subplot(2, 2, 4)
    _draw_density(ax4, after_df, labels_arr, groups, color_map, "標準化後")

    fig.suptitle("標準化前後比較", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _draw_group_box(ax, df, labels_arr, groups, color_map, title):
    """在指定 axes 上繪製分組 boxplot"""
    data_by_group = []
    colors = []
    for g in groups:
        vals = df[labels_arr == g].values.flatten()
        vals = vals[~np.isnan(vals)]
        data_by_group.append(vals)
        colors.append(color_map[g])

    bp = ax.boxplot(data_by_group, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels([str(g)[:10] for g in groups], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("數值")


def _draw_density(ax, df, labels_arr, groups, color_map, title):
    """在指定 axes 上繪製分組 density"""
    all_vals = df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) < 2:
        ax.set_title(title)
        return

    x_min, x_max = np.percentile(all_vals, [1, 99])
    x_range = np.linspace(x_min, x_max, 200)

    for g in groups:
        vals = df[labels_arr == g].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        try:
            kde = gaussian_kde(vals)
            ax.plot(x_range, kde(x_range), color=color_map[g], alpha=0.7, label=str(g))
        except Exception:
            continue

    ax.set_xlabel("數值")
    ax.set_ylabel("密度")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
