import matplotlib
import matplotlib.pyplot as plt
import platform
import seaborn as sns

# 設定中文字體（依平台 fallback）
if platform.system() == "Windows":
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
elif platform.system() == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC", "PingFang TC", "DejaVu Sans"
    ]
else:
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC", "WenQuanYi Micro Hei", "DejaVu Sans"
    ]
matplotlib.rcParams["axes.unicode_minus"] = False

# Colorblind-safe 預設色盤
COLORBLIND_PALETTE = sns.color_palette("colorblind")
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=COLORBLIND_PALETTE)

from ms_core.visualization.pca_plot import plot_pca_score, plot_pca_scree, plot_pca_loading
from ms_core.visualization.pca_3d import plot_pca_3d, pca_3d_to_html
from ms_core.visualization.boxplot import plot_group_boxplot, plot_sample_boxplot
from ms_core.visualization.density_plot import plot_density
from ms_core.visualization.volcano_plot import plot_volcano
from ms_core.visualization.heatmap import plot_heatmap
from ms_core.visualization.vip_plot import plot_vip
from ms_core.visualization.norm_preview import plot_norm_comparison
from ms_core.visualization.anova_plot import plot_anova_importance, plot_feature_boxplot
from ms_core.visualization.correlation_plot import plot_correlation_heatmap, plot_correlation_network
from ms_core.visualization.roc_plot import plot_roc_curves, plot_auc_ranking
from ms_core.visualization.outlier_plot import plot_outlier_score, plot_dmodx
from ms_core.visualization.rf_plot import plot_rf_importance, plot_confusion_matrix

try:
    from ms_core.visualization.oplsda_plot import plot_oplsda_score, plot_oplsda_splot
except ImportError:
    pass  # pyopls not installed
