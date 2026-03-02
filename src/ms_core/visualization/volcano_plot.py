"""
Volcano plot visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from adjustText import adjust_text

    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False


def plot_volcano(volcano_result, top_n: int = 5, fig=None):
    """
    Render a volcano plot.

    The y-axis and significance labels follow the analysis mode:
    - use_fdr=True  -> FDR-adjusted p-values
    - use_fdr=False -> raw p-values
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    rdf = volcano_result.result_df
    fc_thresh = volcano_result.fc_thresh
    p_thresh = volcano_result.p_thresh
    use_fdr = bool(getattr(volcano_result, "use_fdr", False))
    fdr_method = str(getattr(volcano_result, "fdr_method", "fdr_bh"))

    log2fc = rdf["log2FC"].to_numpy(dtype=float)
    neg_log10p = rdf["neg_log10p"].to_numpy(dtype=float)
    sig = rdf["significant"].to_numpy(dtype=bool)
    features = rdf["Feature"].astype(str).to_numpy()

    nonsig_label = "Not significant"
    sig_label = "Significant (FDR)" if use_fdr else "Significant"
    ax.scatter(log2fc[~sig], neg_log10p[~sig], c="grey", alpha=0.5, s=20, label=nonsig_label)
    ax.scatter(log2fc[sig], neg_log10p[sig], c="red", alpha=0.7, s=30, label=sig_label)

    ax.axhline(-np.log10(p_thresh), ls="--", c="blue", alpha=0.4, linewidth=0.8)
    ax.axvline(np.log2(fc_thresh), ls="--", c="blue", alpha=0.4, linewidth=0.8)
    ax.axvline(-np.log2(fc_thresh), ls="--", c="blue", alpha=0.4, linewidth=0.8)

    rank_col = "pvalue_adj" if use_fdr and "pvalue_adj" in rdf.columns else "pvalue"
    top_idx = np.argsort(rdf[rank_col].to_numpy(dtype=float))[:top_n]
    texts = []
    for idx in top_idx:
        name = features[idx]
        if len(name) > 20:
            name = f"{name[:18]}.."
        texts.append(ax.text(log2fc[idx], neg_log10p[idx], name, fontsize=7, ha="center"))
    if texts and HAS_ADJUSTTEXT:
        adjust_text(texts, ax=ax)

    ylabel = "-log10(FDR-adjusted p-value)" if use_fdr else "-log10(p-value)"
    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel(ylabel)

    mode_text = f"FDR={fdr_method}" if use_fdr else "raw p-value"
    ax.set_title(f"Volcano Plot ({volcano_result.group1} vs {volcano_result.group2}, {mode_text})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

