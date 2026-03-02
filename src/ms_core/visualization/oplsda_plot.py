"""
OPLS-DA 可視化 — Score Plot + S-Plot (Loading)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_oplsda_score(oplsda_result, fig: Figure = None):
    """
    OPLS-DA Score Plot: T_predictive vs T_orthogonal
    含 95% 信賴橢圓
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    score_df = oplsda_result.get_score_df()
    groups = sorted(score_df['Group'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    for i, g in enumerate(groups):
        mask = score_df['Group'] == g
        x = score_df.loc[mask, 'T_predictive'].values
        y = score_df.loc[mask, 'T_orthogonal'].values
        ax.scatter(x, y, c=[colors[i]], label=str(g), s=50, alpha=0.8)

        # 95% 信賴橢圓
        if len(x) > 2:
            cov = np.cov(x, y)
            if np.all(np.isfinite(cov)):
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 0)
                angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
                n_std = np.sqrt(chi2.ppf(0.95, 2))
                w, h = 2 * n_std * np.sqrt(vals)
                ell = Ellipse(
                    xy=(x.mean(), y.mean()), width=w, height=h,
                    angle=angle, edgecolor=colors[i],
                    facecolor='none', linestyle='--', linewidth=1.5,
                )
                ax.add_patch(ell)

    ax.set_xlabel("T predictive [1]")
    ax.set_ylabel("T orthogonal [1]")
    ax.set_title(
        f"OPLS-DA Score Plot\n"
        f"R²Y={oplsda_result.r2y:.3f} | Q²={oplsda_result.q2:.3f}"
    )
    ax.legend()
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-')
    fig.tight_layout()
    return fig


def plot_oplsda_splot(oplsda_result, fig: Figure = None, top_n: int = 10):
    """
    OPLS-DA S-Plot: Loading 散佈圖
    標記最重要的特徵
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)

    imp_df = oplsda_result.get_importance_df()
    if imp_df.empty:
        ax.set_title("OPLS-DA S-Plot (no data)")
        fig.tight_layout()
        return fig

    loadings = imp_df['Loading'].values
    importance = imp_df['Importance'].values
    features = imp_df['Feature'].values

    # 散佈圖
    ax.scatter(loadings, importance, c='steelblue', s=30, alpha=0.6)

    # 標記 top N
    top_idx = np.argsort(importance)[-top_n:]
    for idx in top_idx:
        ax.annotate(
            features[idx][:20],
            (loadings[idx], importance[idx]),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
        )

    ax.set_xlabel("Predictive Loading p[1]")
    ax.set_ylabel("|p[1]| (Importance)")
    ax.set_title("OPLS-DA S-Plot")
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-')
    fig.tight_layout()
    return fig
