"""
Random Forest visualizations:
- feature importance
- confusion matrix
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


_INVALID_LABELS = {"", "na", "nan", "none", "null"}


def _safe_feature_label(name, idx: int) -> str:
    if name is None:
        return f"Feature_{idx + 1}"
    text = str(name).strip()
    if text.lower() in _INVALID_LABELS:
        return f"Feature_{idx + 1}"
    return text


def plot_rf_importance(
    rf_result,
    fig: Figure = None,
    top_n: int = 25,
):
    """
    Plot top-N Random Forest feature importance.
    """
    imp_df = rf_result.feature_importance.head(top_n).copy()
    if "Importance" not in imp_df.columns or "Feature" not in imp_df.columns:
        raise ValueError("rf_result.feature_importance must contain 'Feature' and 'Importance' columns.")

    importances = imp_df["Importance"].fillna(0.0).to_numpy(dtype=float)
    max_importance = float(np.nanmax(importances)) if len(importances) else 0.0
    if not np.isfinite(max_importance) or max_importance <= 0:
        color_scale = np.zeros_like(importances)
    else:
        color_scale = importances / max_importance

    feature_labels = [
        _safe_feature_label(name, idx)
        for idx, name in enumerate(imp_df["Feature"].tolist())
    ]

    if fig is None:
        fig = plt.figure(figsize=(8, max(4, len(imp_df) * 0.35)))
    fig.clf()
    ax = fig.add_subplot(111)

    colors = plt.cm.YlOrRd(color_scale)
    ax.barh(range(len(imp_df)), importances, color=colors)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(feature_labels, fontsize=8)
    ax.set_xlabel("Mean Decrease Impurity (Gini)")
    ax.set_title(
        "Random Forest Feature Importance\n"
        f"OOB Acc={rf_result.oob_accuracy:.3f} | "
        f"CV Acc={rf_result.cv_accuracy:.3f} +/- {rf_result.cv_std:.3f}"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    rf_result,
    fig: Figure = None,
):
    """
    Plot Random Forest confusion matrix.
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    fig.clf()
    ax = fig.add_subplot(111)

    cm = rf_result.confusion_mat
    class_names = rf_result.class_names

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (CV)\nAccuracy {rf_result.cv_accuracy:.1%}")
    fig.tight_layout()
    return fig

