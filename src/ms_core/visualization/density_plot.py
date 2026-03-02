"""
Density Plot — 每個樣本的強度分佈密度圖
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_density(df: pd.DataFrame, labels, title="Density Plot", fig=None):
    """
    每個樣本繪製一條密度曲線，顏色依組別
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

    groups = sorted(set(labels_arr))
    cmap = plt.cm.Set1
    color_map = {g: cmap(i / max(len(groups) - 1, 1)) for i, g in enumerate(groups)}

    # 找出全域 x 範圍
    all_vals = df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        return fig
    x_min, x_max = np.percentile(all_vals, [1, 99])
    x_range = np.linspace(x_min, x_max, 300)

    plotted_groups = set()
    for idx in range(len(df)):
        values = df.iloc[idx].dropna().values
        if len(values) < 2:
            continue
        group = labels_arr[idx]
        try:
            kde = gaussian_kde(values)
            label = str(group) if group not in plotted_groups else None
            ax.plot(x_range, kde(x_range), color=color_map[group], alpha=0.3, label=label)
            plotted_groups.add(group)
        except Exception:
            continue

    ax.set_xlabel("強度")
    ax.set_ylabel("密度")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig
