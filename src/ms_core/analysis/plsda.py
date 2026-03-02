"""
PLS-DA + VIP 分析模組 — 複製 MetaboAnalyst 6.0 的 PLSR.Anal()

對應 R: pls::plsr(method='oscorespls')
VIP 公式: VIP_j = sqrt(p * sum_h [w_jh^2 * SS_h] / sum_h SS_h)

VIP > 1 為重要特徵閾值
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold


class PLSDAResult:
    """PLS-DA 分析結果容器"""

    def __init__(self, scores, vips, feature_names, labels, pls_model,
                 explained_variance, q2=None):
        self.scores = scores          # n_samples x n_components
        self.vips = vips              # n_features
        self.feature_names = feature_names
        self.labels = labels
        self.model = pls_model
        self.explained_variance = explained_variance
        self.q2 = q2

    def get_vip_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "VIP": self.vips,
        })
        return df.sort_values("VIP", ascending=False).reset_index(drop=True)

    def get_important_features(self, threshold: float = 1.0) -> pd.DataFrame:
        vip_df = self.get_vip_df()
        return vip_df[vip_df["VIP"] >= threshold]


def _compute_vip(pls_model, X, y) -> np.ndarray:
    """
    計算 VIP scores

    VIP_j = sqrt(p * sum_h [w_jh^2 * SS_h] / sum_h SS_h)
    """
    t = pls_model.x_scores_        # n x h
    w = pls_model.x_weights_       # p x h
    q = pls_model.y_loadings_      # 1 x h
    p_feat, h = w.shape

    # SS per component: diag(T'T * Q'Q)
    ss = np.diag(t.T @ t @ q.T @ q).reshape(h,)
    total_ss = ss.sum()

    if total_ss == 0:
        return np.ones(p_feat)

    vips = np.zeros(p_feat)
    for i in range(p_feat):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)
        ])
        vips[i] = np.sqrt(p_feat * np.sum(ss * weight) / total_ss)

    return vips


def run_plsda(
    data: pd.DataFrame,
    labels,
    n_components: int = 3,
    cv_method: str = "loo",
) -> PLSDAResult:
    """
    執行 PLS-DA 分析

    Parameters
    ----------
    data : DataFrame
        已標準化的數值資料
    labels : Series or array
        分組標籤
    n_components : int
        成分數量，預設 3
    cv_method : str
        交叉驗證方法: "loo" (Leave-One-Out) 或 "kfold5"

    Returns
    -------
    PLSDAResult
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = data.values

    n_components = min(n_components, min(X.shape) - 1, len(np.unique(y)))

    # 建立 PLS 模型
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, y)

    # VIP
    vips = _compute_vip(pls, X, y)

    # Scores
    scores = pls.x_scores_

    # 計算 explained variance (X space)
    total_var = np.var(X, axis=0).sum()
    explained = []
    for i in range(n_components):
        t_i = scores[:, i:i+1]
        p_i = pls.x_loadings_[:, i:i+1]
        x_hat = t_i @ p_i.T
        explained.append(np.var(x_hat, axis=0).sum() / total_var)

    # 交叉驗證 Q²
    q2 = None
    try:
        if cv_method == "loo":
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(
            PLSRegression(n_components=n_components, scale=False),
            X, y, cv=cv,
        )
        ss_res = np.sum((y - y_pred.ravel()) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        q2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except Exception:
        pass

    return PLSDAResult(
        scores=scores,
        vips=vips,
        feature_names=list(data.columns),
        labels=labels,
        pls_model=pls,
        explained_variance=np.array(explained),
        q2=q2,
    )
