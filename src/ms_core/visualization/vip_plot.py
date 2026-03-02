"""
VIP Score Plot — MetaboAnalyst 風格

VIP > 1 紅色, VIP < 1 灰色
水平條形圖，VIP=1 虛線
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_vip(plsda_result, top_n: int = 25, fig=None):
    """
    繪製 VIP Score Plot

    Parameters
    ----------
    plsda_result : PLSDAResult
    top_n : int
        顯示 top N 特徵
    fig : Figure or None
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    vip_df = plsda_result.get_vip_df().head(top_n)
    vip_df = vip_df.iloc[::-1]  # 反轉以讓最高 VIP 在上方

    colors = ["#e74c3c" if v >= 1 else "#95a5a6" for v in vip_df["VIP"]]
    ax.barh(range(len(vip_df)), vip_df["VIP"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(vip_df)))
    ax.set_yticklabels([str(f)[:25] for f in vip_df["Feature"]], fontsize=8)
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="VIP = 1")
    ax.set_xlabel("VIP Score")
    ax.set_title("VIP Scores from PLS-DA")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
