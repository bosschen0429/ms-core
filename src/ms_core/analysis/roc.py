"""
ROC analysis module.

Implements:
- Single-feature ROC (AUC based on out-of-fold probabilities)
- Multi-feature Logistic Regression ROC (cross-validated)
- Youden's J optimal cutoff
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


@dataclass
class SingleROCResult:
    """Single-feature ROC result."""

    feature: str
    fpr: np.ndarray
    tpr: np.ndarray
    auc_score: float
    optimal_cutoff: float
    sensitivity: float
    specificity: float


@dataclass
class ROCResult:
    """Complete ROC analysis result."""

    single_rocs: List[SingleROCResult]
    multi_fpr: Optional[np.ndarray] = None
    multi_tpr: Optional[np.ndarray] = None
    multi_auc: Optional[float] = None
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    single_cv_folds_used: int = 0
    multi_cv_folds_used: int = 0

    def get_top_biomarkers(self, n: int = 10) -> pd.DataFrame:
        return self.summary_df.head(n)


def _resolve_cv_folds(y_bin: np.ndarray, requested_folds: int) -> int:
    """Return valid StratifiedKFold split count for binary labels."""
    class_counts = np.bincount(y_bin)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    if min_class_count < 2:
        return 0
    return max(2, min(int(requested_folds), min_class_count))


def _build_binary_oof_probabilities(
    x: np.ndarray,
    y_bin: np.ndarray,
    cv_folds: int,
    random_state: int = 42,
) -> tuple[np.ndarray, int]:
    """Compute out-of-fold class-1 probabilities with logistic regression."""
    folds_used = _resolve_cv_folds(y_bin, cv_folds)
    if folds_used < 2:
        raise ValueError("At least 2 samples per class are required for cross-validated ROC.")

    cv = StratifiedKFold(n_splits=folds_used, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    y_prob = cross_val_predict(clf, x, y_bin, cv=cv, method="predict_proba")[:, 1]
    return y_prob, folds_used


def _compute_single_roc_cv(
    x: np.ndarray,
    y_bin: np.ndarray,
    feature_name: str,
    cv_folds: int,
) -> SingleROCResult:
    """Compute cross-validated ROC for a single feature."""
    x_2d = np.asarray(x, dtype=float).reshape(-1, 1)
    y_prob, _ = _build_binary_oof_probabilities(x_2d, y_bin, cv_folds=cv_folds)
    fpr, tpr, thresholds = roc_curve(y_bin, y_prob)
    auc_val = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_cutoff = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
    best_sens = float(tpr[best_idx])
    best_spec = float(1 - fpr[best_idx])

    return SingleROCResult(
        feature=feature_name,
        fpr=fpr,
        tpr=tpr,
        auc_score=float(auc_val),
        optimal_cutoff=optimal_cutoff,
        sensitivity=best_sens,
        specificity=best_spec,
    )


def run_roc_analysis(
    df: pd.DataFrame,
    labels: pd.Series,
    group1: str,
    group2: str,
    top_n: int = 10,
    multi_feature: bool = True,
    cv_folds: int = 5,
) -> ROCResult:
    """Run ROC analysis for group1 vs group2."""
    mask = labels.isin([group1, group2])
    df_sub = df[mask].copy()
    y = labels[mask].copy()

    le = LabelEncoder()
    y_bin = le.fit_transform(y)

    single_cv_folds_used = _resolve_cv_folds(y_bin, cv_folds)
    if single_cv_folds_used < 2:
        raise ValueError("At least 2 samples per class are required for cross-validated ROC.")

    single_rocs: list[SingleROCResult] = []
    for col in df_sub.columns:
        x = pd.to_numeric(df_sub[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(x).any():
            continue
        if np.std(x) == 0:
            continue
        try:
            result = _compute_single_roc_cv(x, y_bin, col, cv_folds=single_cv_folds_used)
            single_rocs.append(result)
        except Exception:
            continue

    single_rocs.sort(key=lambda result: max(result.auc_score, 1 - result.auc_score), reverse=True)
    single_rocs = single_rocs[: int(max(1, top_n))]

    summary = pd.DataFrame(
        [
            {
                "Feature": result.feature,
                "AUC": result.auc_score,
                "Optimal_Cutoff": result.optimal_cutoff,
                "Sensitivity": result.sensitivity,
                "Specificity": result.specificity,
            }
            for result in single_rocs
        ]
    )

    multi_fpr = multi_tpr = multi_auc_val = None
    multi_cv_folds_used = 0
    if multi_feature and len(df_sub.columns) >= 2:
        try:
            x_multi = df_sub.apply(pd.to_numeric, errors="coerce")
            finite_mask = np.isfinite(x_multi.values).all(axis=1)
            x_multi = x_multi.loc[finite_mask]
            y_multi = y_bin[finite_mask]
            if len(x_multi) >= 4:
                y_prob, multi_cv_folds_used = _build_binary_oof_probabilities(
                    x_multi.values, y_multi, cv_folds=cv_folds
                )
                multi_fpr, multi_tpr, _ = roc_curve(y_multi, y_prob)
                multi_auc_val = float(auc(multi_fpr, multi_tpr))
        except Exception:
            multi_fpr = multi_tpr = multi_auc_val = None
            multi_cv_folds_used = 0

    return ROCResult(
        single_rocs=single_rocs,
        multi_fpr=multi_fpr,
        multi_tpr=multi_tpr,
        multi_auc=multi_auc_val,
        summary_df=summary,
        single_cv_folds_used=single_cv_folds_used,
        multi_cv_folds_used=multi_cv_folds_used,
    )
