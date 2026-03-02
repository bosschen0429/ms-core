"""
互動式 3D PCA — 使用 Plotly

對應 MetaboAnalyst 的 3D PCA Score Plot
"""

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def plot_pca_3d(pca_result, pc_x=0, pc_y=1, pc_z=2):
    """
    繪製互動式 3D PCA Score Plot

    Parameters
    ----------
    pca_result : PCAResult
    pc_x, pc_y, pc_z : int
        三個主成分索引 (0-based)

    Returns
    -------
    plotly Figure object, or None if plotly not installed
    """
    if not HAS_PLOTLY:
        return None

    scores = pca_result.scores
    labels = pca_result.labels
    var_ratio = pca_result.explained_variance_ratio
    sample_names = pca_result.sample_names

    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    groups = sorted(set(labels_arr))
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    fig = go.Figure()

    for i, group in enumerate(groups):
        mask = labels_arr == group
        color = colors[i % len(colors)]

        # 取得樣本名稱
        names = [sample_names[j] for j in range(len(mask)) if mask[j]]

        fig.add_trace(go.Scatter3d(
            x=scores[mask, pc_x],
            y=scores[mask, pc_y],
            z=scores[mask, pc_z],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            name=str(group),
            text=names,
            hovertemplate=(
                f"<b>%{{text}}</b><br>"
                f"PC{pc_x+1}: %{{x:.3f}}<br>"
                f"PC{pc_y+1}: %{{y:.3f}}<br>"
                f"PC{pc_z+1}: %{{z:.3f}}<br>"
                f"<extra>{group}</extra>"
            ),
        ))

    fig.update_layout(
        title="3D PCA Score Plot",
        scene=dict(
            xaxis_title=f"PC{pc_x+1} ({var_ratio[pc_x]*100:.1f}%)",
            yaxis_title=f"PC{pc_y+1} ({var_ratio[pc_y]*100:.1f}%)",
            zaxis_title=f"PC{pc_z+1} ({var_ratio[pc_z]*100:.1f}%)",
        ),
        legend=dict(title="Group"),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def pca_3d_to_html(fig) -> str:
    """將 Plotly figure 轉為 HTML 字串"""
    if not HAS_PLOTLY or fig is None:
        return "<p>需要安裝 plotly: pip install plotly</p>"
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
