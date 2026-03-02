"""
Outlier visualization for Hotelling's T2 and DModX.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_outlier_score(
    outlier_result,
    labels=None,
    fig: Figure = None,
):
    """
    Plot PCA score scatter + T2 bar chart.
    Handles low-dimensional score spaces safely.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 5))
    fig.clf()

    scores = outlier_result.scores
    t2 = outlier_result.t2_values
    outlier_mask = outlier_result.outlier_mask_t2
    var_ratio = outlier_result.explained_variance

    x = scores[:, 0]
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(x)

    ax1 = fig.add_subplot(121)
    normal = ~outlier_mask
    ax1.scatter(x[normal], y[normal], c="#3498db", alpha=0.7, s=40, label="Normal")
    ax1.scatter(
        x[outlier_mask],
        y[outlier_mask],
        c="#e74c3c",
        marker="x",
        s=80,
        linewidths=2,
        label="Outlier",
    )

    names = outlier_result.sample_names
    for i in np.where(outlier_mask)[0]:
        ax1.annotate(str(names[i]), (x[i], y[i]), fontsize=7, color="red", alpha=0.8)

    if len(x) > 2 and scores.shape[1] > 1:
        cov = np.cov(x, y)
        if np.all(np.isfinite(cov)) and np.linalg.det(cov) > 0:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0)
            angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
            n_std = np.sqrt(chi2.ppf(0.95, 2))
            w, h = 2 * n_std * np.sqrt(vals)
            ell = Ellipse(
                xy=(x.mean(), y.mean()),
                width=w,
                height=h,
                angle=angle,
                edgecolor="grey",
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
            )
            ax1.add_patch(ell)

    pc1 = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
    pc2 = var_ratio[1] * 100 if len(var_ratio) > 1 else 0
    ax1.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pc2:.1f}%)" if len(var_ratio) > 1 else "Pseudo-PC2")
    ax1.set_title("PCA Score Plot - Outlier Detection")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(122)
    colors = ["#e74c3c" if o else "#3498db" for o in outlier_mask]
    ax2.bar(range(len(t2)), t2, color=colors, alpha=0.7)
    ax2.axhline(
        y=outlier_result.t2_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"95% threshold ({outlier_result.t2_threshold:.2f})",
    )
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Hotelling T2")
    ax2.set_title("Hotelling's T2")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_dmodx(
    outlier_result,
    fig: Figure = None,
):
    """
    Plot DModX bar chart.
    """
    if fig is None:
        fig = plt.figure(figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)

    dmodx = outlier_result.dmodx
    outlier_mask = outlier_result.outlier_mask_dmodx
    colors = ["#e74c3c" if o else "#2ecc71" for o in outlier_mask]

    ax.bar(range(len(dmodx)), dmodx, color=colors, alpha=0.7)
    ax.axhline(
        y=outlier_result.dmodx_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"95% threshold ({outlier_result.dmodx_threshold:.4f})",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("DModX")
    ax.set_title("DModX (Distance to Model)")
    ax.legend(fontsize=8)

    names = outlier_result.sample_names
    for i in np.where(outlier_mask)[0]:
        ax.annotate(
            str(names[i]),
            (i, dmodx[i]),
            fontsize=7,
            color="red",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig
