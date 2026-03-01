"""Unified 12-step pipeline orchestrator.

Pipeline order (NEVER reorder):
    1.  Data organization         (preprocessing)
    2.  Duplicate signal removal  (preprocessing)
    3.  MS quality filter         (preprocessing)  — ratio-based, per-group
    4.  Missing value imputation  (processing)     — 8 methods
    5.  ISTD marking + correction (preprocessing + calibration)
    6.  QC-LOWESS trend correction(calibration)
    7.  Batch effect ComBat       (calibration)
    8.  Feature filtering IQR/RSD (processing)     — overall variance
    9.  PQN normalization         (calibration)     — smart QC assessment
    10. Generalized log transform (processing)     — glog, NOT log2(x+1)
    11. Scaling                   (processing)     — MeanCenter/Auto/Pareto/Range
    12. Statistical analysis      (analysis + visualization)
"""

from __future__ import annotations

# Placeholder — will be implemented in Phase 5 of the migration plan.
# Each step will call the corresponding module from preprocessing/,
# calibration/, processing/, or analysis/.
