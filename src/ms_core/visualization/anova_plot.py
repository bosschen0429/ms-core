"""
ANOVA 結果可視化

- 重要特徵排名圖 (bar chart by -log10(p))
- 單一特徵的多組 boxplot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_anova_importance(anova_result, top_n=25, fig=None):
    """
    繪製 ANOVA 重要特徵排名圖

    Parameters
    ----------
    anova_result : ANOVAResult
    top_n : int
        顯示 top N 特徵
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    df = anova_result.result_df.sort_values("neg_log10p", ascending=False).head(top_n)
    df = df.iloc[::-1]  # 反轉

    colors = ["#e74c3c" if s else "#95a5a6" for s in df["significant"]]
    ax.barh(range(len(df)), df["neg_log10p"].values, color=colors, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([str(f)[:25] for f in df["Feature"]], fontsize=8)
    ax.axvline(
        x=-np.log10(anova_result.p_thresh),
        color="red", linestyle="--", alpha=0.5, linewidth=1,
        label=f"p = {anova_result.p_thresh}",
    )
    ax.set_xlabel("-log10(adj. p-value)")
    ax.set_title("ANOVA: 重要特徵排名")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_boxplot(
    df: pd.DataFrame, labels, feature_name: str, fig=None,
):
    """
    繪製單一特徵的多組 boxplot

    Parameters
    ----------
    df : DataFrame
    labels : array-like
    feature_name : str
        特徵名稱
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    plot_data = pd.DataFrame({
        "Group": labels_arr,
        "Value": df[feature_name].values,
    })

    sns.boxplot(data=plot_data, x="Group", y="Value", hue="Group",
                palette="Set1", ax=ax, legend=False)
    sns.stripplot(data=plot_data, x="Group", y="Value",
                  color="black", alpha=0.4, size=4, ax=ax)

    ax.set_title(f"Feature: {feature_name}", fontsize=10)
    ax.set_xlabel("組別")
    ax.set_ylabel("數值")
    fig.tight_layout()
    return fig
