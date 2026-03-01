# 質譜數據工具箱 — 三專案整合計畫

> **日期：** 2026-03-02
> **範圍：** MetaboAnalyst Clone、ms-preprocessing-toolkit、Data_Normalization_project_v2
> **目標：** 建立共享 core library，消除重複實作，形成端到端 pipeline

---

## 1. 現況分析

### 1.1 三專案定位

| 專案 | 定位 | GUI 框架 | 核心能力 |
|------|------|----------|---------|
| **ms-preprocessing-toolkit** (Toolkit) | LC-MS 原始資料前處理 | customtkinter | 資料整理、ISTD 標記、重複移除、品質篩選 |
| **Data_Normalization_project_v2** (DNP) | 儀器校正與正規化 | Tkinter | ISTD 校正、QC-LOWESS、ComBat、PQN |
| **Metaboanalyst_clone** (MA) | 統計分析與視覺化 | PySide6 | 缺值處理、filtering、normalization、transformation、scaling、10+ 統計方法 |

### 1.2 功能重疊地圖

```
                    Toolkit         DNP         MetaboAnalyst
                    ───────         ───         ─────────────
資料整理 (m/z+RT)      ✅            —              —
ISTD 標記              ✅            —              —
ISTD 校正              —            ✅              —
重複訊號移除            ✅            —              —
MS 品質篩選 (ratio)    ✅            —              —
QC-LOWESS              —            ✅              —
Batch Effect           —            ✅              —
Missing Value          ✅ min/2      —             ✅ 8 種方法     ← 重疊
Feature Filtering      ✅ 3-criteria  —             ✅ IQR/RSD     ← 重疊(互補)
PQN 正規化             —            ✅ smart QC     ✅ 基礎版       ← 重疊
其他正規化              —             —             ✅ 6 種方法
Creatinine 校正        —            ✅              —
glog 轉換              —             —             ✅
Scaling                —             —             ✅ 4 種方法
PCA                    —            ✅ 驗證用       ✅ 完整分析     ← 重疊
統計分析 (10+種)        —             —             ✅
視覺化 (14 種圖)        —             —             ✅
QC 樣本處理             ✅            ✅             ✅             ← 三者重疊
Sample 分類             ✅            ✅             ✅             ← 三者重疊
```

---

## 2. 整合決策

### 2.1 整合策略：共享 Core Library

```
┌─────────────────────────────────────────────────────────┐
│                    ms-core (共享核心)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ 資料整理  │ │ ISTD     │ │ 校正模組  │ │ 統計分析    │  │
│  │ 重複移除  │ │ 標記+校正│ │ LOWESS   │ │ PCA/PLSDA  │  │
│  │ 品質篩選  │ │          │ │ ComBat   │ │ Volcano    │  │
│  │ 缺值處理  │ │          │ │ PQN      │ │ ROC/RF/... │  │
│  │ 統計篩選  │ │          │ │          │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  │ 轉換/縮放│ │ 視覺化    │ │ 共用工具  │                  │
│  │ glog     │ │ 14 種圖  │ │ QC 處理   │                  │
│  │ scaling  │ │          │ │ Sample分類│                  │
│  └──────────┘ └──────────┘ └──────────┘                  │
└─────────────────────────────────────────────────────────┘
        ↑               ↑               ↑
   Toolkit GUI     DNP GUI        MetaboAnalyst GUI
  (customtkinter)  (Tkinter)      (PySide6)
```

**原則：**
- 三個 GUI 各自保留，各自引用 `ms-core` 作為依賴
- `ms-core` 為純函數模組，無 GUI import，DataFrame in → DataFrame out
- 消除所有 adapter 層 — 統一資料格式後不再需要轉換
- 每個專案的 GUI 只暴露與其定位相關的步驟

### 2.2 重疊功能的處理決策

| 重疊功能 | 決策 | 原因 |
|---------|------|------|
| **Missing Value** | 採用 MA 版（8 種方法） | MA 版更完整，涵蓋 min/LoD、mean、median、KNN、PPCA、BPCA、SVD |
| **Feature Filtering** | **兩者都保留，分別用於不同時機** | Toolkit 的 ratio filter 用於早期 MS 品質篩選；MA 的 IQR/RSD filter 用於晚期統計篩選 |
| **PQN 正規化** | 採用 DNP 版 | DNP 的 PQN 有 smart QC assessment + creatinine 雙階段，更完整 |
| **PCA** | 採用 MA 版（完整分析） | DNP 的 PCA 僅用於驗證，MA 版有 scores/loadings/scree + 3D |
| **QC 處理** | 統一成一個模組 | 三個專案各有 QC 偵測邏輯，應合併為 `ms-core/qc.py` |
| **Sample 分類** | 統一成一個模組 | 合併為 `ms-core/sample_classification.py` |

---

## 3. 統一 Pipeline 順序

```
 階段              步驟                          來源            說明
═══════════════════════════════════════════════════════════════════════════
 MS 前處理    1. 資料整理                        Toolkit     m/z+RT 合併、欄位重命名
              2. 重複訊號移除                    Toolkit     ppm+RT tolerance 去重
              3. MS 品質篩選 (ratio filter)      Toolkit     3-criteria 組別導向篩選
─────────────────────────────────────────────────────────────────────────
 缺值+校正    4. Missing Value 填補              MA          8 種方法 (min/LoD 為預設)
              5. ISTD 標記+校正                  Toolkit+DNP 標記→加權匹配→校正
─────────────────────────────────────────────────────────────────────────
 趨勢+批次    6. QC-LOWESS 趨勢校正              DNP         動態 frac、smart 判斷
              7. Batch Effect ComBat             DNP         PERMANOVA 驗證→條件式套用
─────────────────────────────────────────────────────────────────────────
 統計前處理   8. Feature Filtering (IQR/RSD)     MA          整體變異度篩選 (batch 校正後更準)
              9. 正規化 — PQN                    DNP         smart QC assessment, creatinine
             10. glog 轉換                       MA          Generalized log (非 log2(x+1))
             11. Scaling                         MA          MeanCenter/Auto/Pareto/Range
─────────────────────────────────────────────────────────────────────────
 分析+視覺   12. 統計分析 + 視覺化               MA          PCA, PLS-DA, Volcano, ANOVA,
                                                             ROC, RF, OPLS-DA, Heatmap...
```

### 3.1 Pipeline 順序設計理由

| 順序決策 | 理由 |
|---------|------|
| **Step 3 → Step 4**（品質篩選先於缺值填補） | 先移除偵測率差的 feature，避免對大量缺失的 feature 做無意義的填補 |
| **Step 4 → Step 5**（缺值填補先於 ISTD 校正） | ISTD 校正需要完整的數據矩陣，缺值會導致校正比率計算不正確 |
| **Step 5 → Step 6**（ISTD 先於 QC-LOWESS） | ISTD 校正消除基質效應和離子抑制；QC-LOWESS 校正的是校正後的殘餘趨勢漂移 |
| **Step 7 → Step 8**（Batch Effect 先於 IQR filter） | Batch effect 會人為膨脹或抵消 feature 變異度，校正後再算 IQR/RSD 更能反映真實生物變異 |
| **Step 8 → Step 9**（IQR filter 先於 PQN） | 移除低變異 feature 後，PQN 的 reference spectrum 更穩健（不被雜訊 feature 拉偏） |
| **Step 9 有 PQN 就不做 SpecNorm** | PQN 和 SpecNorm 都是 row-wise normalization，重複套用會扭曲數據 |

### 3.2 兩階段 Filtering 設計

```
Step 3: MS 品質篩選 (Toolkit ratio filter)
├── 目的：移除 MS 儀器偵測不可靠的 features
├── 方法：組別導向，看各組的訊號偵測率
├── 三標準：
│   ├── 穩定性：≥2 組 signal ratio ≥ 0.33
│   ├── 偏態性：任一組 ratio ≥ 0.66
│   └── 差異性：組間 ratio 差 ≥ 0.30
├── 時機：早期（原始資料品質把關）
└── 保留條件：三者滿足其一即保留

Step 8: Feature Filtering (MetaboAnalyst IQR/RSD)
├── 目的：移除統計上無法貢獻的低變異 features
├── 方法：整體導向，看數值分佈的變異度
├── 可選方法：IQR / SD / MAD / RSD / NRSD
├── 自動切割點：依 feature 數量調整 (5%~40%)
├── 時機：晚期（batch 校正後，正規化前）
└── 可選：QC-RSD 前置篩選
```

---

## 4. 共享 Core Library 架構

### 4.1 模組結構

```
ms-core/
├── __init__.py
├── py.typed
│
├── preprocessing/                    # ← 來自 Toolkit
│   ├── data_organizer.py            # m/z+RT 合併、欄位重命名
│   ├── istd_marker.py               # ISTD 標記 (ppm matching)
│   ├── duplicate_remover.py         # 重複訊號移除
│   └── ms_quality_filter.py         # 3-criteria ratio filter
│
├── calibration/                      # ← 來自 DNP
│   ├── istd_correction.py           # ISTD 加權校正演算法
│   ├── qc_lowess.py                 # QC-LOWESS 趨勢校正
│   ├── batch_effect.py              # ComBat batch correction
│   └── pqn.py                       # PQN 正規化 (smart QC assessment)
│
├── processing/                       # ← 來自 MetaboAnalyst
│   ├── missing_values.py            # 8 種缺值填補方法
│   ├── feature_filter.py            # IQR/SD/MAD/RSD/NRSD filter
│   ├── normalization.py             # SumNorm, MedianNorm, QuantileNorm 等
│   ├── transformation.py            # glog, log10, sqrt, cbrt
│   └── scaling.py                   # MeanCenter, Auto, Pareto, Range
│
├── analysis/                         # ← 來自 MetaboAnalyst
│   ├── pca.py
│   ├── plsda.py
│   ├── oplsda.py
│   ├── univariate.py                # t-test, volcano
│   ├── anova.py
│   ├── roc.py
│   ├── random_forest.py
│   ├── clustering.py
│   ├── correlation.py
│   └── outlier.py
│
├── visualization/                    # ← 來自 MetaboAnalyst
│   ├── pca_plot.py
│   ├── pca_3d.py
│   ├── volcano_plot.py
│   ├── vip_plot.py
│   ├── heatmap.py
│   ├── boxplot.py
│   ├── density_plot.py
│   ├── roc_plot.py
│   ├── rf_plot.py
│   ├── correlation_plot.py
│   ├── outlier_plot.py
│   ├── oplsda_plot.py
│   ├── anova_plot.py
│   └── norm_preview.py
│
├── utils/                            # ← 三者合併
│   ├── qc.py                        # 統一 QC 偵測與處理
│   ├── sample_classification.py     # 統一 Sample 分類邏輯
│   ├── file_handler.py              # Excel/CSV I/O (合併三者)
│   ├── statistics.py                # 共用統計工具 (Hotelling T², etc.)
│   ├── safe_math.py                 # 安全數值運算
│   └── validators.py                # 資料驗證
│
└── pipeline.py                       # 統一 12 步 pipeline orchestrator
```

### 4.2 統一資料格式

整合後採用統一的 DataFrame 格式，消除所有 adapter：

**核心資料矩陣：**
```
              Feature_1  Feature_2  Feature_3  ...
Sample_001       5000       6000       7000
Sample_002       5500       6500       7500
QC_001           5200       6200       7200
...
```
- **Index:** Sample ID（字串）
- **Columns:** Feature ID（格式 `m/z_value/RT_value` 或自由命名）
- **Values:** 數值強度（float64）

**附帶 metadata：**
```python
@dataclass
class MSDataset:
    matrix: pd.DataFrame          # 核心數據矩陣 (samples × features)
    labels: pd.Series             # Group labels (與 matrix.index 對齊)
    sample_info: pd.DataFrame     # SampleInfo (Name, Type, Order, Volume, Batch, ...)
    feature_info: pd.DataFrame    # FeatureInfo (m/z, RT, ISTD flag, ...)
    processing_log: list[str]     # 處理記錄
```

### 4.3 各 GUI 使用的步驟

```
Toolkit GUI (customtkinter):
└── Steps 1-3: 資料整理 → 重複移除 → MS 品質篩選

DNP GUI (Tkinter):
└── Steps 5-7, 9: ISTD 校正 → QC-LOWESS → Batch Effect → PQN

MetaboAnalyst GUI (PySide6):
└── Steps 4, 8, 10-12: 缺值 → IQR filter → glog → Scaling → 統計分析
└── 也可直接跑 Step 1-12（完整 pipeline 模式）
```

---

## 5. 遷移計畫

### Phase 1：建立 ms-core 骨架

- [ ] 建立 `ms-core/` repo 和 package 結構
- [ ] 定義 `MSDataset` 統一資料格式
- [ ] 設定 `pyproject.toml`（可用 `pip install -e .` 開發模式安裝）
- [ ] 建立共用 utils（qc.py, sample_classification.py, file_handler.py）

### Phase 2：遷移 Processing 模組（來自 MetaboAnalyst）

- [ ] 搬移 `missing_values.py` → `ms-core/processing/`
- [ ] 搬移 `feature_filter.py` (IQR/RSD) → `ms-core/processing/`
- [ ] 搬移 `normalization.py` → `ms-core/processing/`
- [ ] 搬移 `transformation.py` → `ms-core/processing/`
- [ ] 搬移 `scaling.py` → `ms-core/processing/`
- [ ] 搬移所有 `analysis/` 和 `visualization/` → `ms-core/`
- [ ] 更新 MetaboAnalyst GUI import 路徑

### Phase 3：遷移 Preprocessing 模組（來自 Toolkit）

- [ ] 搬移 `data_organizer.py` → `ms-core/preprocessing/`
- [ ] 搬移 `istd_marker.py` → `ms-core/preprocessing/`
- [ ] 搬移 `duplicate_remover.py` → `ms-core/preprocessing/`
- [ ] 搬移 `feature_filter.py` (ratio) → `ms-core/preprocessing/ms_quality_filter.py`
- [ ] 更新 Toolkit GUI import 路徑

### Phase 4：遷移 Calibration 模組（來自 DNP）

- [ ] 搬移 `istd.py` → `ms-core/calibration/istd_correction.py`
- [ ] 搬移 `qc_lowess.py` → `ms-core/calibration/`
- [ ] 搬移 `batch_effect.py` → `ms-core/calibration/`
- [ ] 搬移 `normalization.py` (PQN) → `ms-core/calibration/pqn.py`
- [ ] 移除 `adapters/` 目錄（不再需要）
- [ ] 更新 DNP GUI import 路徑

### Phase 5：統一 Pipeline Orchestrator

- [ ] 建立 `ms-core/pipeline.py` 實作 12 步 pipeline
- [ ] 整合 undo/redo（snapshot 機制）
- [ ] 整合 processing log
- [ ] 為 MetaboAnalyst GUI 新增「完整 pipeline」模式（可從 Step 1 開始）

### Phase 6：清理與驗證

- [ ] 搬移所有 tests 到 `ms-core/tests/`
- [ ] 跑完整測試套件確保無迴歸
- [ ] 移除三個專案中已搬到 ms-core 的重複程式碼
- [ ] 移除所有 adapter 層
- [ ] 更新三個專案的 `requirements.txt` 加入 `ms-core` 依賴
- [ ] 更新各專案的 CLAUDE.md 和文件

---

## 6. 風險與注意事項

| 風險 | 緩解措施 |
|------|---------|
| **搬移後 import 路徑全部壞掉** | 使用 `pip install -e .` 開發模式，一次只搬一個模組並立即測試 |
| **三個專案的 QC 邏輯不完全一致** | 以 DNP 的為基礎（最完整），合併 Toolkit 和 MA 的特殊 case |
| **Toolkit 用 openpyxl 紅字/藍字標記** | `ms-core` 保持純數據處理，Excel 格式化由各 GUI 的 file_handler 負責 |
| **DNP 的 PQN 有 creatinine 雙階段** | creatinine 校正作為 PQN 的可選子步驟保留，預設不啟用 |
| **MetaboAnalyst 的 SpecNorm 被移除** | 不移除模組，但 pipeline 中若選了 PQN 就自動跳過 SpecNorm |
| **三個 GUI 框架不同** | 不統一 GUI 框架，各自保留。ms-core 為純 Python，無 GUI 依賴 |

---

## 7. 目錄結構（整合後）

```
質譜數據工具箱/
├── ms-core/                         # 共享核心 library (新建)
│   ├── pyproject.toml
│   ├── src/ms_core/
│   │   ├── preprocessing/
│   │   ├── calibration/
│   │   ├── processing/
│   │   ├── analysis/
│   │   ├── visualization/
│   │   ├── utils/
│   │   └── pipeline.py
│   └── tests/
│
├── ms-preprocessing-toolkit/        # Toolkit GUI (瘦身後)
│   ├── src/ms_preprocessing/
│   │   ├── gui/                     # 保留 GUI 層
│   │   └── config/                  # 保留設定
│   └── requirements.txt             # 加入 ms-core 依賴
│
├── Data_Normalization_project_v2/   # DNP GUI (瘦身後)
│   ├── src/metabolomics/
│   │   ├── gui/                     # 保留 GUI 層
│   │   └── utils/console.py         # 保留 GUI 工具
│   └── requirements.txt             # 加入 ms-core 依賴
│
└── Metaboanalyst_clone/             # MA GUI (瘦身後)
    ├── gui/                         # 保留 GUI 層
    ├── translations/                # 保留 i18n
    └── requirements.txt             # 加入 ms-core 依賴
```

---

## 附錄 A：各模組來源對照表

| ms-core 模組 | 來源專案 | 原始檔案 |
|-------------|---------|---------|
| `preprocessing/data_organizer.py` | Toolkit | `src/ms_preprocessing/core/data_organizer.py` |
| `preprocessing/istd_marker.py` | Toolkit | `src/ms_preprocessing/core/istd_marker.py` |
| `preprocessing/duplicate_remover.py` | Toolkit | `src/ms_preprocessing/core/duplicate_remover.py` |
| `preprocessing/ms_quality_filter.py` | Toolkit | `src/ms_preprocessing/core/feature_filter.py` |
| `calibration/istd_correction.py` | DNP | `src/metabolomics/processors/istd.py` |
| `calibration/qc_lowess.py` | DNP | `src/metabolomics/processors/qc_lowess.py` |
| `calibration/batch_effect.py` | DNP | `src/metabolomics/processors/batch_effect.py` |
| `calibration/pqn.py` | DNP | `src/metabolomics/processors/normalization.py` |
| `processing/missing_values.py` | MA | `core/missing_values.py` |
| `processing/feature_filter.py` | MA | `core/filtering.py` |
| `processing/normalization.py` | MA | `core/normalization.py` |
| `processing/transformation.py` | MA | `core/transformation.py` |
| `processing/scaling.py` | MA | `core/scaling.py` |
| `analysis/*` | MA | `analysis/*` |
| `visualization/*` | MA | `visualization/*` |
| `utils/qc.py` | 三者合併 | Toolkit validators + DNP sample_classification + MA qc.py |
| `utils/sample_classification.py` | 三者合併 | Toolkit validators + DNP sample_classification + MA qc.py |
| `utils/file_handler.py` | 三者合併 | Toolkit file_handler + DNP file_io + MA data_import |
| `pipeline.py` | MA 為基礎擴展 | MA `core/pipeline.py` + 新增 12 步 |
