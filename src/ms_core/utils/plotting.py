"""
matplotlib 繪圖設定

包含共用的繪圖設定和色彩定義。
"""
import os
import sys

# Use a non-interactive backend by default to avoid Tk/Tcl dependency
# (common in CI / minimal Python installs). Users can override by setting
# the environment variable MPLBACKEND before running.
import matplotlib

if not os.environ.get('MPLBACKEND'):
    matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt

import numpy as np

# Import shared constants
from .constants import FONT_SIZES, COLORBLIND_COLORS, SAMPLE_TYPE_COLORS, SAMPLE_TYPE_MARKERS
from .sample_classification import normalize_sample_type


def plot_pca_comparison_qc_style(
    scores_left,
    scores_right,
    var_left,
    var_right,
    sample_names,
    sample_types,
    batch_labels=None,
    *,
    grouping='batch',
    suptitle=None,
    left_title=None,
    right_title=None,
    left_threshold_text=None,
    right_threshold_text=None,
    qc_outlier_names_left=None,
    qc_outlier_names_right=None,
    output_path=None,
    dpi=300,
):
    """Create a QC_LOWESS-style 2-panel PCA comparison plot.

    This is a shared plotting utility to keep PCA figure style consistent across
    subprograms (legend, figsize, layout).

    Parameters
    ----------
    scores_left, scores_right : array-like, shape (n_samples, 2)
        PCA scores (PC1/PC2) for left/right panels.
    var_left, var_right : array-like, shape (2,)
        Explained variance ratios for PC1/PC2.
    sample_names : list[str]
        Sample names aligned with rows of scores.
    sample_types : list[str]
        Each is one of: 'QC', 'Control', 'Exposure' (case-insensitive allowed).
    batch_labels : list[str] | None
        Batch label per sample; required when grouping='batch'.
    grouping : {'batch','sample_type'}
        Controls confidence ellipse mode.
    qc_outlier_names_left/right : set[str] | None
        Names of QC samples marked as outliers for each panel.
    output_path : str | os.PathLike | None
        If provided, saves the figure.

    Returns
    -------
    (fig, (ax_left, ax_right))
    """
    from .statistics import draw_hotelling_t2_ellipse

    scores_left = np.asarray(scores_left)
    scores_right = np.asarray(scores_right)
    var_left = np.asarray(var_left)
    var_right = np.asarray(var_right)

    if scores_left.shape[1] != 2 or scores_right.shape[1] != 2:
        raise ValueError('scores_left/scores_right must be (n_samples, 2)')
    if len(sample_names) != scores_left.shape[0] or len(sample_names) != scores_right.shape[0]:
        raise ValueError('sample_names length must match number of rows in scores')
    if len(sample_types) != len(sample_names):
        raise ValueError('sample_types length must match sample_names')

    if grouping not in ('batch', 'sample_type'):
        raise ValueError("grouping must be 'batch' or 'sample_type'")
    if grouping == 'batch':
        if batch_labels is None or len(batch_labels) != len(sample_names):
            raise ValueError('batch_labels is required for grouping=\'batch\' and must align with sample_names')

    qc_outlier_names_left = set(qc_outlier_names_left or [])
    qc_outlier_names_right = set(qc_outlier_names_right or [])

    sample_types_norm = [normalize_sample_type(t) for t in sample_types]

    # Dynamic color/marker maps based on actual types present
    color_map = dict(SAMPLE_TYPE_COLORS)  # copy defaults
    markers = dict(SAMPLE_TYPE_MARKERS)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.98, fontweight='bold')

    if grouping == 'batch':
        unique_batches = sorted(set(str(b) for b in batch_labels))
        batch_colors = COLORBLIND_COLORS * ((len(unique_batches) // len(COLORBLIND_COLORS)) + 1)
        batch_color_map = {batch: batch_colors[i] for i, batch in enumerate(unique_batches)}
    else:
        unique_batches = []
        batch_color_map = {}

    def scatter_panel(ax, scores, var, title_text, threshold_text, qc_outliers):
        for i, sample in enumerate(sample_names):
            s_type = sample_types_norm[i]
            color = color_map.get(s_type, SAMPLE_TYPE_COLORS.get('Unknown', '#808080'))
            marker = markers.get(s_type, SAMPLE_TYPE_MARKERS.get('Unknown', 'x'))
            is_outlier = (s_type == 'QC') and (sample in qc_outliers)

            if is_outlier:
                edgecolor = 'red'
                linewidth = 3
                size = 150
                alpha = 0.9
            else:
                edgecolor = 'black'
                linewidth = 1
                size = 100
                alpha = 0.7

            ax.scatter(
                scores[i, 0], scores[i, 1],
                c=[color], marker=marker,
                s=size, alpha=alpha,
                edgecolors=edgecolor, linewidths=linewidth,
            )

        all_bounds = []
        if grouping == 'batch':
            for batch in unique_batches:
                batch_indices = [i for i, b in enumerate(batch_labels) if str(b) == batch]
                if len(batch_indices) >= 3:
                    batch_scores = scores[batch_indices]
                    bounds = draw_hotelling_t2_ellipse(
                        ax,
                        batch_scores,
                        label=f'95% CI (Batch {batch})',
                        edgecolor=batch_color_map[batch],
                        linestyle='-',
                        linewidth=2.5,
                    )
                    if bounds is not None:
                        all_bounds.append(bounds)
        else:
            bounds_all = draw_hotelling_t2_ellipse(
                ax,
                scores,
                label='95% CI (All Samples)',
                edgecolor='gray',
                linestyle='--',
                linewidth=3,
            )
            if bounds_all is not None:
                all_bounds.append(bounds_all)

            qc_indices = [i for i, t in enumerate(sample_types_norm) if t == 'QC']
            if len(qc_indices) >= 3:
                qc_scores = scores[qc_indices]
                bounds_qc = draw_hotelling_t2_ellipse(
                    ax,
                    qc_scores,
                    label='95% CI (QC Only)',
                    edgecolor='#9370DB',
                    linestyle='-',
                    linewidth=3,
                )
                if bounds_qc is not None:
                    all_bounds.append(bounds_qc)

        if all_bounds:
            x_min = min(b[0] for b in all_bounds)
            x_max = max(b[1] for b in all_bounds)
            y_min = min(b[2] for b in all_bounds)
            y_max = max(b[3] for b in all_bounds)
        else:
            x_min, x_max = np.min(scores[:, 0]), np.max(scores[:, 0])
            y_min, y_max = np.min(scores[:, 1]), np.max(scores[:, 1])

        x_range = (x_max - x_min) or 1
        y_range = (y_max - y_min) or 1
        ax.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
        ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)

        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=12, fontweight='bold')

        title_lines = [title_text]
        if threshold_text:
            title_lines.append(threshold_text)
        ax.set_title('\n'.join(title_lines), fontsize=14, fontweight='bold', pad=15)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')

    scatter_panel(
        ax_left,
        scores_left,
        var_left,
        left_title or 'Left',
        left_threshold_text,
        qc_outlier_names_left,
    )
    scatter_panel(
        ax_right,
        scores_right,
        var_right,
        right_title or 'Right',
        right_threshold_text,
        qc_outlier_names_right,
    )

    # Build legend dynamically from the sample types actually present
    unique_types = sorted(set(sample_types_norm), key=lambda t: (t != 'QC', t != 'Control', t != 'Exposure', t))
    sample_legend_elements = []
    for stype in unique_types:
        sample_legend_elements.append(
            plt.Line2D([0], [0],
                       marker=markers.get(stype, 'x'), color='w',
                       markerfacecolor=color_map.get(stype, '#808080'),
                       markersize=10, label=stype,
                       markeredgecolor='black', markeredgewidth=1)
        )
    # Always add QC Outlier entry if QC is present
    if 'QC' in unique_types:
        sample_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9370DB',
                       markersize=10, label='QC Outlier', markeredgecolor='red', markeredgewidth=3),
        )

    if grouping == 'batch':
        ellipse_legend_elements = [
            plt.Line2D([0], [0], linestyle='-', color=batch_color_map[batch],
                       linewidth=2.5, label=f'95% CI (Batch {batch})')
            for batch in unique_batches
        ]
    else:
        ellipse_legend_elements = [
            plt.Line2D([0], [0], linestyle='-', color='#9370DB', linewidth=3, label='95% CI (QC Only)'),
            plt.Line2D([0], [0], linestyle='--', color='gray', linewidth=3, label='95% CI (All Samples)'),
        ]

    for ax in (ax_left, ax_right):
        legend1 = ax.legend(
            handles=sample_legend_elements,
            loc='upper left',
            fontsize=9,
            title='Sample Type',
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=ellipse_legend_elements,
            loc='upper right',
            fontsize=9,
            title='Confidence Ellipse',
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    return fig, (ax_left, ax_right)


# COLORBLIND_COLORS is imported from .constants


def setup_matplotlib():
    """
    設定 matplotlib 全域參數

    - 關閉 LaTeX 渲染
    - 根據作業系統設定適當的字體
    - 設定預設樣式

    Returns:
    --------
    dict : 包含設定資訊的字典
    """
    # 關閉 LaTeX
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    # 根據作業系統設定字體
    if sys.platform == 'darwin':
        plt.rcParams['font.family'] = 'Helvetica'
    else:
        plt.rcParams['font.family'] = 'Arial'

    return {
        'font_family': plt.rcParams['font.family'],
        'platform': sys.platform
    }
