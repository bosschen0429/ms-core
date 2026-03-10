"""Unified parquet intermediate storage contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


class IntermediateStore:
    """Persist dataframe + metadata sidecar for step-to-step handoff."""

    @staticmethod
    def _meta_path(parquet_path: Path) -> Path:
        return parquet_path.with_suffix(parquet_path.suffix + ".meta.json")

    @staticmethod
    def save(
        df: pd.DataFrame,
        parquet_path: str | Path,
        metadata: dict[str, Any] | None = None,
        index: bool = False,
    ) -> Path:
        path = Path(parquet_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=index)
        except Exception:
            normalized_df = IntermediateStore._normalize_dataframe_for_parquet(df)
            normalized_df.to_parquet(path, index=index)

        payload = IntermediateStore._normalize_metadata(metadata or {})
        IntermediateStore._meta_path(path).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def load(parquet_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        path = Path(parquet_path)
        df = pd.read_parquet(path)
        meta_path = IntermediateStore._meta_path(path)
        if not meta_path.exists():
            return df, {}

        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return df, {}
        return df, raw

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {str(key): IntermediateStore._normalize_value(value) for key, value in metadata.items()}

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): IntermediateStore._normalize_value(v) for k, v in value.items()}
        if isinstance(value, set):
            return [IntermediateStore._normalize_value(item) for item in sorted(value)]
        if isinstance(value, (list, tuple)):
            return [IntermediateStore._normalize_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _normalize_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for idx, dtype in enumerate(normalized.dtypes):
            if dtype != "object":
                continue

            series = normalized.iloc[:, idx]
            non_null = series[series.notna()]
            if non_null.empty:
                continue

            if non_null.map(lambda v: isinstance(v, str)).all():
                continue

            if non_null.map(lambda v: isinstance(v, (bytes, bytearray))).all():
                col_name = normalized.columns[idx]
                normalized[col_name] = series.map(IntermediateStore._decode_bytes).to_list()
                continue

            if non_null.map(lambda v: isinstance(v, (int, float, bool, np.number, np.bool_))).all():
                col_name = normalized.columns[idx]
                normalized[col_name] = pd.to_numeric(series, errors="coerce").to_list()
                continue

            col_name = normalized.columns[idx]
            normalized[col_name] = series.map(IntermediateStore._stringify_mixed_value).to_list()

        return normalized

    @staticmethod
    def _decode_bytes(value: Any) -> Any:
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except Exception:
            pass
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _stringify_mixed_value(value: Any) -> Any:
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except Exception:
            pass
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", errors="replace")
        return str(value)
