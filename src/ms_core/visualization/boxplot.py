"""
Boxplot — 按組別顯示特徵分佈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_group_boxplot(df: pd.DataFrame, labels, title="Feature Distribution", fig=None):
    """
    按組別顯示整體強度分佈 boxplot

    每個樣本計算所有特徵的 summary 後按組別繪製
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    plot_df = df.copy()
    plot_df["_Group"] = labels_arr
    melted = plot_df.melt(id_vars="_Group", var_name="Feature", value_name="Value")

    sns.boxplot(data=melted, x="_Group", y="Value", hue="_Group", ax=ax,
                palette="Set1", fliersize=2, linewidth=0.8, legend=False)
    ax.set_xlabel("組別")
    ax.set_ylabel("數值")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sample_boxplot(df: pd.DataFrame, labels, title="Sample Distribution", fig=None):
    """
    每個樣本一個 boxplot，顏色依組別
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.4), 5))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    groups = sorted(set(labels_arr))
    palette = dict(zip(groups, sns.color_palette("Set1", len(groups))))
    colors = [palette[g] for g in labels_arr]

    bp = ax.boxplot(
        [df.iloc[i].values for i in range(len(df))],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(
        [str(s)[:12] for s in df.index],
        rotation=90, fontsize=7,
    )
    ax.set_ylabel("數值")
    ax.set_title(title)

    # 圖例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[g], label=str(g)) for g in groups]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    fig.tight_layout()
    return fig
