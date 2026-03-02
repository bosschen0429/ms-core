"""
統計分析函數

包含 Hotelling T² 相關的統計分析函數。
"""
import numpy as np
from scipy.stats import chi2, f as f_dist
from matplotlib.patches import Ellipse


def calculate_hotelling_t2_outliers(qc_scores, all_scores=None, alpha=0.05):
    """
    使用 Hotelling T² 檢測 QC 樣本是否偏離整體中心

    如果提供 all_scores，則使用「所有樣本」的協方差矩陣。
    如果只有 qc_scores，則使用 QC 樣本自身的統計量。

    Parameters:
    -----------
    qc_scores : np.ndarray
        QC 樣本的 PCA 分數 (n_qc x n_components)
    all_scores : np.ndarray, optional
        所有樣本的 PCA 分數 (n_all x n_components)
    alpha : float
        顯著性水準 (預設 0.05)

    Returns:
    --------
    tuple : (t2_values, threshold, outliers)
        - t2_values : 每個 QC 樣本的 T² 值
        - threshold : 異常值閾值
        - outliers : 布林陣列，標記異常值
    """
    n_qc, p = qc_scores.shape

    # 確定使用哪組樣本計算統計量
    if all_scores is not None:
        n_all = all_scores.shape[0]
        if n_qc < 1 or n_all < 3:
            return np.zeros(n_qc), 0, np.zeros(n_qc, dtype=bool)
        reference_scores = all_scores
        n_ref = n_all
    else:
        if n_qc < 3:
            return np.zeros(n_qc), 0, np.zeros(n_qc, dtype=bool)
        reference_scores = qc_scores
        n_ref = n_qc

    # 計算參考樣本的統計量
    mean_ref = np.mean(reference_scores, axis=0)
    cov_ref = np.cov(reference_scores, rowvar=False)

    # 正則化協方差矩陣
    cov_reg = cov_ref + np.eye(p) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    # 計算每個 QC 樣本的 Hotelling T² 值
    t2_values = np.zeros(n_qc)
    for i in range(n_qc):
        diff = qc_scores[i] - mean_ref
        t2_values[i] = np.dot(np.dot(diff, cov_inv), diff.T)

    # 計算閾值
    if all_scores is not None:
        # 使用卡方分布
        threshold = chi2.ppf(1 - alpha, p)
    else:
        # 使用 F 分布的正確公式
        if n_ref - p - 1 > 0:
            f_critical = f_dist.ppf(1 - alpha, p, n_ref - p - 1)
            threshold = (p * (n_ref + 1) * (n_ref - 1)) / (n_ref * (n_ref - p - 1)) * f_critical
        else:
            threshold = chi2.ppf(1 - alpha, p)

    outliers = t2_values > threshold

    return t2_values, threshold, outliers


def calculate_hotelling_t2_outliers_internal(scores, alpha=0.05):
    """
    使用 Hotelling T² 檢測樣本內部的異常值

    使用樣本自身的內部統計量。

    Parameters:
    -----------
    scores : np.ndarray
        樣本的 PCA 分數 (n x n_components)
    alpha : float
        顯著性水準 (預設 0.05)

    Returns:
    --------
    tuple : (t2_values, threshold, outliers)
    """
    n, p = scores.shape

    if n < 3:
        return np.zeros(n), 0, np.zeros(n, dtype=bool)

    # 使用內部統計量
    mean = np.mean(scores, axis=0)
    cov = np.cov(scores, rowvar=False)

    # 正則化協方差矩陣
    cov_reg = cov + np.eye(p) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    # 計算 Hotelling T² 值
    t2_values = np.zeros(n)
    for i in range(n):
        diff = scores[i] - mean
        t2_values[i] = np.dot(np.dot(diff, cov_inv), diff.T)

    # 使用正確的閾值公式
    if n - p - 1 > 0:
        f_critical = f_dist.ppf(1 - alpha, p, n - p - 1)
        threshold = (p * (n + 1) * (n - 1)) / (n * (n - p - 1)) * f_critical
    else:
        threshold = chi2.ppf(1 - alpha, p)

    outliers = t2_values > threshold

    return t2_values, threshold, outliers


def draw_hotelling_t2_ellipse(ax, scores, alpha=0.05, label=None,
                               edgecolor='red', linestyle='-', linewidth=2.5):
    """
    在 2D PCA 圖上繪製 95% Hotelling T² 橢圓

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        matplotlib 軸物件
    scores : np.ndarray
        樣本的 2D PCA 分數 (n x 2)
    alpha : float
        顯著性水準 (預設 0.05)
    label : str, optional
        橢圓的標籤
    edgecolor : str
        邊框顏色
    linestyle : str
        線條樣式
    linewidth : float
        線條寬度

    Returns:
    --------
    tuple or None : (x_min, x_max, y_min, y_max) 橢圓邊界，如果樣本不足則返回 None
    """
    n, p = scores.shape

    if n < 3:
        return None

    mean = np.mean(scores, axis=0)
    cov = np.cov(scores, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    f_critical = f_dist.ppf(1 - alpha, p, n - p)
    scale_factor = np.sqrt((p * (n - 1) * (n + 1)) / (n * (n - p)) * f_critical)

    width = 2 * scale_factor * np.sqrt(eigenvalues[0])
    height = 2 * scale_factor * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(mean, width, height, angle=angle,
                     facecolor='none', edgecolor=edgecolor,
                     linewidth=linewidth, linestyle=linestyle, label=label)
    ax.add_patch(ellipse)

    # 計算橢圓邊界
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_x = (width/2) * np.cos(t)
    ellipse_y = (height/2) * np.sin(t)

    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + mean[0]
    y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + mean[1]

    bounds = (np.min(x_rot), np.max(x_rot), np.min(y_rot), np.max(y_rot))

    return bounds
