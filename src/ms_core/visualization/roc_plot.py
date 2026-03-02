"""
ROC 曲線可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_roc_curves(
    roc_result,
    fig: Figure = None,
    show_multi: bool = True,
    top_n: int = 5,
):
    """
    繪製 ROC 曲線圖

    Parameters
    ----------
    roc_result : ROCResult
    fig : matplotlib Figure
    show_multi : 是否顯示多特徵 Logistic Regression ROC
    top_n : 顯示前 N 個最佳特徵的曲線
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 7))
    fig.clf()
    ax = fig.add_subplot(111)

    # 對角線
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")

    # 單特徵 ROC
    colors = plt.cm.Set1(np.linspace(0, 1, min(top_n, len(roc_result.single_rocs))))
    for i, roc in enumerate(roc_result.single_rocs[:top_n]):
        label = f"{roc.feature[:20]} (AUC={roc.auc_score:.3f})"
        ax.plot(roc.fpr, roc.tpr, color=colors[i], linewidth=1.5, label=label)
        # 標記最佳 cutoff 點
        best_idx = np.argmax(roc.tpr - roc.fpr)
        ax.plot(roc.fpr[best_idx], roc.tpr[best_idx], "o",
                color=colors[i], markersize=6)

    # 多特徵 ROC
    if show_multi and roc_result.multi_fpr is not None:
        ax.plot(roc_result.multi_fpr, roc_result.multi_tpr,
                color="black", linewidth=2.5,
                label=f"Multi-feature LR (AUC={roc_result.multi_auc:.3f})")

    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC 曲線分析")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    return fig


def plot_auc_ranking(
    roc_result,
    fig: Figure = None,
    top_n: int = 15,
):
    """
    AUC 排序橫條圖

    Parameters
    ----------
    roc_result : ROCResult
    fig : matplotlib Figure
    top_n : 顯示的特徵數
    """
    summary = roc_result.summary_df.head(top_n)
    if len(summary) == 0:
        return fig

    if fig is None:
        fig = plt.figure(figsize=(8, max(4, len(summary) * 0.35)))
    fig.clf()
    ax = fig.add_subplot(111)

    colors = ["#27ae60" if a >= 0.7 else "#f39c12" if a >= 0.5 else "#e74c3c"
              for a in summary["AUC"]]
    ax.barh(range(len(summary)), summary["AUC"].values, color=colors)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["Feature"].values, fontsize=8)
    ax.set_xlabel("AUC")
    ax.set_title("生物標記物 AUC 排序")
    ax.axvline(x=0.5, color="grey", linestyle="--", alpha=0.5, label="Random")
    ax.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim([0, 1])
    fig.tight_layout()
    return fig
