"""
資料處理輔助函數

包含各模組共用的資料處理函數。
"""
import pandas as pd
import numpy as np


def get_valid_values(row, columns):
    """
    從 DataFrame 的一行中提取有效值（>0 且非 NaN）

    Parameters:
    -----------
    row : pd.Series
        DataFrame 的一行資料
    columns : list
        要提取的欄位名稱列表

    Returns:
    --------
    list : 有效的浮點數值列表（>0 且非 NaN）

    Examples:
    ---------
    >>> row = pd.Series({'A': 100, 'B': 200, 'C': 0, 'D': -5})
    >>> get_valid_values(row, ['A', 'B', 'C', 'D'])
    [100.0, 200.0]
    """
    values = []
    for col in columns:
        # 支援兩種檢查方式: dict-like 和 Series
        if hasattr(row, 'index'):
            if col not in row.index:
                continue
        elif col not in row:
            continue

        try:
            val = float(row[col])
            if not pd.isna(val) and val > 0:
                values.append(val)
        except (ValueError, TypeError):
            pass
    return values


def extract_sample_type_row(df, feature_col='Mz/RT'):
    """從 DataFrame 中提取 Sample_Type 資訊行（如果存在）。

    ISTD 輸出的資料 sheet 中，第一資料行可能是 Sample_Type 行
    （Mz/RT='Sample_Type'，各樣本欄位填入 QC/Control/Exposure 等）。
    此函式將其提取出來以避免干擾數值計算。

    移除字串行後，會將非 feature_col 的欄位轉回 numeric dtype，
    避免下游 np.isnan() 等操作因 object dtype 而失敗。

    Returns:
        (df_clean, type_row_dict) — 若無 Sample_Type 行則 type_row_dict 為 None
    """
    mask = df[feature_col].astype(str).str.strip().str.lower() == 'sample_type'
    if mask.any():
        type_row = df[mask].iloc[0].to_dict()
        df_clean = df[~mask].reset_index(drop=True)
        # 移除字串行後，將數值欄位的 dtype 從 object 轉回 numeric
        for col in df_clean.columns:
            if col != feature_col:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean, type_row
    return df, None


def insert_sample_type_row(df, type_row, feature_col='Mz/RT'):
    """將 Sample_Type 資訊行回插到 DataFrame 的第一行。

    只插入 df 中存在的欄位，缺少的欄位填空字串。

    Args:
        df: 目標 DataFrame
        type_row: 由 extract_sample_type_row 提取的字典，None 則不插入
        feature_col: 特徵 ID 欄位名稱
    """
    if type_row is None:
        return df
    row_data = {}
    for col in df.columns:
        if col == feature_col:
            row_data[col] = 'Sample_Type'
        else:
            row_data[col] = type_row.get(col, '')
    type_df = pd.DataFrame([row_data], columns=df.columns)
    return pd.concat([type_df, df], ignore_index=True)
