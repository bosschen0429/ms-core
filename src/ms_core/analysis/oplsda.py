"""
OPLS-DA 分析 — 使用 pyopls 套件
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

try:
    from pyopls import OPLS
    HAS_PYOPLS = True
except ImportError:
    HAS_PYOPLS = False

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression


@dataclass
class OPLSDAResult:
    """OPLS-DA 分析結果"""
    scores_predictive: np.ndarray       # T predictive (n_samples, 1)
    scores_orthogonal: np.ndarray       # T orthogonal (n_samples, 1)
    labels: object                       # group labels
    r2x: float = 0.0                    # R²X
    r2y: float = 0.0                    # R²Y
    q2: float = 0.0                     # Q² (交叉驗證)
    feature_names: list = field(default_factory=list)
    loadings_predictive: np.ndarray = None  # P predictive
    vip_scores: np.ndarray = None       # VIP-like importance
    class_names: list = field(default_factory=list)

    def get_score_df(self) -> pd.DataFrame:
        """回傳 score DataFrame"""
        labels_arr = self.labels.values if hasattr(self.labels, 'values') else np.array(self.labels)
        return pd.DataFrame({
            'T_predictive': self.scores_predictive[:, 0],
            'T_orthogonal': self.scores_orthogonal[:, 0],
            'Group': labels_arr,
        })

    def get_importance_df(self) -> pd.DataFrame:
        """回傳特徵重要性 DataFrame"""
        if self.loadings_predictive is None:
            return pd.DataFrame()
        importance = np.abs(self.loadings_predictive[:, 0])
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Loading': self.loadings_predictive[:, 0],
            'Importance': importance,
        })
        return df.sort_values('Importance', ascending=False).reset_index(drop=True)


def run_oplsda(data, labels, n_components=1, cv_method="loo") -> OPLSDAResult:
    """
    執行 OPLS-DA 分析

    Parameters
    ----------
    data : DataFrame
        特徵矩陣 (samples x features)
    labels : Series/array
        分組標籤
    n_components : int
        正交成分數 (預設 1)
    cv_method : str
        "loo" 或 "kfold5"

    Returns
    -------
    OPLSDAResult
    """
    if not HAS_PYOPLS:
        raise ImportError(
            "OPLS-DA 需要 pyopls 套件。請執行: pip install pyopls"
        )

    X = data.values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    feature_names = list(data.columns)
    class_names = list(le.classes_)

    # OPLS 分解
    opls = OPLS(n_components=n_components)
    Z = opls.fit_transform(X, y)  # Z = X 去除正交成分後

    # 取得 scores
    scores_predictive = opls.T_ortho_  # 這是正交 score
    # 用去除正交後的 Z 做 PLS 得到 predictive score
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(Z, y)
    t_pred = pls.x_scores_

    # R² 和 Q²
    r2x = 1 - np.var(Z) / np.var(X) if np.var(X) > 0 else 0
    r2y = pls.score(Z, y)

    # 交叉驗證 Q²
    if cv_method == "loo":
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        cv_scores = cross_val_score(pls, Z, y, cv=cv, scoring='r2')
        q2 = np.mean(cv_scores)
    except Exception:
        q2 = 0.0

    # Loadings
    loadings_pred = pls.x_loadings_

    # VIP-like scores (simplified from PLS loadings)
    vip = np.abs(loadings_pred[:, 0])

    return OPLSDAResult(
        scores_predictive=t_pred,
        scores_orthogonal=opls.T_ortho_,
        labels=labels,
        r2x=r2x,
        r2y=r2y,
        q2=q2,
        feature_names=feature_names,
        loadings_predictive=loadings_pred,
        vip_scores=vip,
        class_names=class_names,
    )
