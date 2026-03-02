"""
PCA Score Plot / Loading Plot — MetaboAnalyst 風格

- 95% 信賴橢圓 (per group)
- 軸標籤格式: "PC1 (42.3%)"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_pca_score(pca_result, pc_x=0, pc_y=1, fig=None):
    """
    繪製 PCA Score Plot (2D)

    Parameters
    ----------
    pca_result : PCAResult
    pc_x, pc_y : int
        要顯示的主成分索引 (0-based)
    fig : Figure or None
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    scores = pca_result.scores
    labels = pca_result.labels
    var_ratio = pca_result.explained_variance_ratio

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    groups = sorted(set(labels_arr))
    cmap = plt.cm.Set1
    colors = [cmap(i / max(len(groups) - 1, 1)) for i in range(len(groups))]

    for i, group in enumerate(groups):
        mask = labels_arr == group
        x = scores[mask, pc_x]
        y = scores[mask, pc_y]
        ax.scatter(x, y, c=[colors[i]], label=str(group), s=50, alpha=0.8, edgecolors="white", linewidth=0.5)

        # 95% 信賴橢圓
        if len(x) > 2:
            cov = np.cov(x, y)
            if np.linalg.det(cov) > 1e-10:
                vals, vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
                n_std = np.sqrt(chi2.ppf(0.95, 2))
                w, h = 2 * n_std * np.sqrt(np.abs(vals))
                ell = Ellipse(
                    xy=(x.mean(), y.mean()), width=w, height=h,
                    angle=angle, edgecolor=colors[i],
                    facecolor="none", linestyle="--", linewidth=1.5,
                )
                ax.add_patch(ell)

    ax.set_xlabel(f"PC{pc_x+1} ({var_ratio[pc_x]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc_y+1} ({var_ratio[pc_y]*100:.1f}%)")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("PCA Score Plot")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    return fig


def plot_pca_scree(pca_result, fig=None):
    """繪製 Scree Plot（解釋變異比例）"""
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    var = pca_result.explained_variance_ratio
    n = len(var)
    x = np.arange(1, n + 1)

    ax.bar(x, var * 100, color="steelblue", alpha=0.8, label="個別")
    ax.plot(x, np.cumsum(var) * 100, "ro-", markersize=5, label="累積")
    ax.set_xlabel("主成分")
    ax.set_ylabel("解釋變異 (%)")
    ax.set_title("Scree Plot")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pca_loading(pca_result, pc=0, top_n=20, fig=None):
    """繪製 Loading Plot（特徵貢獻度）"""
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    loading_df = pca_result.get_loading_df()
    col = loading_df.columns[pc]
    vals = loading_df[col].abs().nlargest(top_n).sort_values(ascending=True)

    colors = ["#e74c3c" if loading_df.loc[f, col] > 0 else "#3498db" for f in vals.index]
    actual_vals = [loading_df.loc[f, col] for f in vals.index]

    ax.barh(range(len(vals)), actual_vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([str(f)[:25] for f in vals.index], fontsize=8)
    ax.set_xlabel(f"Loading ({col})")
    ax.set_title(f"PCA Loading Plot — Top {top_n} Features")
    ax.axvline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    return fig
