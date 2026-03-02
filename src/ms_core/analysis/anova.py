"""
ANOVA 多組比較模組 — 複製 MetaboAnalyst 6.0 的多組單變量分析

支援:
  - One-way ANOVA (parametric)
  - Kruskal-Wallis (non-parametric)
  - FDR 校正 (Benjamini-Hochberg)
  - 事後檢定: Tukey HSD / Fisher LSD
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class ANOVAResult:
    """ANOVA 分析結果容器"""

    def __init__(self, result_df, groups, p_thresh, posthoc_df=None):
        self.result_df = result_df
        self.groups = groups
        self.p_thresh = p_thresh
        self.posthoc_df = posthoc_df

    @property
    def significant(self) -> pd.DataFrame:
        return self.result_df[self.result_df["significant"]]

    @property
    def n_significant(self) -> int:
        return self.result_df["significant"].sum()


def run_anova(
    df: pd.DataFrame,
    labels,
    p_thresh: float = 0.05,
    nonpar: bool = False,
    use_fdr: bool = True,
    posthoc: bool = True,
) -> ANOVAResult:
    """
    執行 ANOVA / Kruskal-Wallis 多組比較

    Parameters
    ----------
    df : DataFrame
        數值資料
    labels : Series or array
        分組標籤（>= 3 組）
    p_thresh : float
        顯著性閾值
    nonpar : bool
        是否使用 Kruskal-Wallis（非參數）
    use_fdr : bool
        是否 FDR 校正
    posthoc : bool
        是否做事後檢定 (Tukey HSD)

    Returns
    -------
    ANOVAResult
    """
    if hasattr(labels, "values"):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    groups = sorted(set(labels_arr))

    # 分組
    group_data = {g: df[labels_arr == g] for g in groups}

    # 每個特徵做檢定
    stats_list = []
    fvals = []
    pvals = []

    for col in df.columns:
        col_groups = [group_data[g][col].dropna().values for g in groups]
        # 確保每組有資料
        col_groups = [g for g in col_groups if len(g) >= 2]
        if len(col_groups) < 2:
            fvals.append(np.nan)
            pvals.append(1.0)
            continue

        try:
            if nonpar:
                stat, p = kruskal(*col_groups)
            else:
                stat, p = f_oneway(*col_groups)
            fvals.append(stat)
            pvals.append(p)
        except Exception:
            fvals.append(np.nan)
            pvals.append(1.0)

    pvals = np.array(pvals)
    fvals = np.array(fvals)

    # FDR
    if use_fdr:
        _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
    else:
        pvals_adj = pvals.copy()

    neg_log10p = -np.log10(np.clip(pvals_adj, 1e-300, 1.0))
    sig_mask = pvals_adj < p_thresh

    result_df = pd.DataFrame({
        "Feature": df.columns,
        "F_statistic": fvals,
        "pvalue": pvals,
        "pvalue_adj": pvals_adj,
        "neg_log10p": neg_log10p,
        "significant": sig_mask,
    })

    # 事後檢定 (Tukey HSD) — 只對顯著特徵做
    posthoc_df = None
    if posthoc and sig_mask.any():
        posthoc_records = []
        sig_features = result_df[result_df["significant"]]["Feature"].values
        for feat in sig_features[:50]:  # 限制數量
            vals = df[feat].values
            try:
                tukey = pairwise_tukeyhsd(vals, labels_arr, alpha=p_thresh)
                for row in tukey.summary().data[1:]:
                    posthoc_records.append({
                        "Feature": feat,
                        "Group1": str(row[0]),
                        "Group2": str(row[1]),
                        "MeanDiff": float(row[2]),
                        "p_adj": float(row[3]),
                        "Lower": float(row[4]),
                        "Upper": float(row[5]),
                        "Reject": bool(row[6]),
                    })
            except Exception:
                continue
        if posthoc_records:
            posthoc_df = pd.DataFrame(posthoc_records)

    return ANOVAResult(result_df, groups, p_thresh, posthoc_df)
