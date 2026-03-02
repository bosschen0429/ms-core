"""
Random Forest analysis:
- feature importance
- OOB accuracy
- CV accuracy
- confusion matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder


_INVALID_FEATURE_TOKENS = {"", "na", "nan", "none", "null"}


@dataclass
class RFResult:
    feature_importance: pd.DataFrame
    oob_accuracy: float
    cv_accuracy: float
    cv_std: float
    cv_folds_used: int
    confusion_mat: np.ndarray
    class_names: List[str]
    model: RandomForestClassifier
    dropped_unnamed_features: int = 0

    def get_top_features(self, n: int = 25) -> pd.DataFrame:
        return self.feature_importance.head(n)


def _get_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    """Choose valid StratifiedKFold split count from class distribution."""
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    if min_class_count < 2:
        return 0
    return max(2, min(requested_folds, min_class_count))


def _is_invalid_feature_name(name) -> bool:
    if pd.isna(name):
        return True
    text = str(name).strip().lower()
    return text in _INVALID_FEATURE_TOKENS


def _clean_feature_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop unnamed/invalid feature columns and normalize duplicate names."""
    kept_positions: list[int] = []
    clean_names: list[str] = []
    dropped = 0
    seen: dict[str, int] = {}

    for pos, col in enumerate(df.columns):
        if _is_invalid_feature_name(col):
            dropped += 1
            continue

        base = str(col).strip()
        count = seen.get(base, 0) + 1
        seen[base] = count
        clean = base if count == 1 else f"{base}_{count}"

        kept_positions.append(pos)
        clean_names.append(clean)

    out = df.iloc[:, kept_positions].copy()
    out.columns = clean_names
    return out, dropped


def run_random_forest(
    df: pd.DataFrame,
    labels: pd.Series,
    n_trees: int = 500,
    cv_folds: int = 5,
    top_n: int = 30,
    random_state: int = 42,
) -> RFResult:
    """
    Run Random Forest classification and return importance + performance summary.
    """
    clean_df, dropped_unnamed = _clean_feature_dataframe(df)
    if clean_df.shape[1] == 0:
        raise ValueError("No valid feature columns remain after dropping unnamed features.")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = clean_df.values
    class_names = list(le.classes_)

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    oob_acc = float(rf.oob_score_)

    used_folds = _get_cv_folds(y, cv_folds)
    if used_folds >= 2:
        cv = StratifiedKFold(
            n_splits=used_folds,
            shuffle=True,
            random_state=random_state,
        )
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        cv_acc = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        y_pred = cross_val_predict(rf, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)
    else:
        # Not enough samples per class for CV; fall back to in-sample prediction.
        cv_acc = float("nan")
        cv_std = float("nan")
        cm = confusion_matrix(y, rf.predict(X))

    top_n = int(max(1, min(top_n, clean_df.shape[1])))
    importance_df = (
        pd.DataFrame(
            {
                "Feature": clean_df.columns,
                "Importance": rf.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )

    return RFResult(
        feature_importance=importance_df,
        oob_accuracy=oob_acc,
        cv_accuracy=cv_acc,
        cv_std=cv_std,
        cv_folds_used=used_folds,
        confusion_mat=cm,
        class_names=class_names,
        model=rf,
        dropped_unnamed_features=dropped_unnamed,
    )
