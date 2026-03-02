from ms_core.analysis.pca import run_pca, PCAResult
from ms_core.analysis.plsda import run_plsda, PLSDAResult
from ms_core.analysis.univariate import volcano_analysis, VolcanoResult
from ms_core.analysis.anova import run_anova, ANOVAResult
from ms_core.analysis.clustering import compute_linkage, select_top_features
from ms_core.analysis.correlation import run_correlation, CorrelationResult
from ms_core.analysis.roc import run_roc_analysis, ROCResult
from ms_core.analysis.outlier import run_outlier_detection, OutlierResult
from ms_core.analysis.random_forest import run_random_forest, RFResult

try:
    from ms_core.analysis.oplsda import run_oplsda, OPLSDAResult
except ImportError:
    pass  # pyopls not installed
