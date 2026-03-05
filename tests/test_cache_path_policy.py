"""Tests for parquet cache path policy and fail-open behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def test_excel_cache_path_not_co_located_in_output_directory(monkeypatch) -> None:
    from ms_core.utils.file_handler import FileHandler

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        base = Path(temp_dir)
        cache_root = base / "internal-cache"
        monkeypatch.setenv("MSPTK_PARQUET_CACHE_ROOT", str(cache_root))

        output_dir = base / "OUTPUT"
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = output_dir / "result.xlsx"

        df = pd.DataFrame({"Mz/RT": ["Sample_Type", "100.1/1.0"], "S1": ["case", 1234]})
        handler = FileHandler()
        handler.save_data(df, excel_path, sheet_name="RawIntensity", save_parquet_cache=True)

        cache_path = handler._resolve_parquet_cache(excel_path)
        assert cache_path is not None
        assert output_dir not in cache_path.parents
        assert cache_root in cache_path.parents


def test_cache_failures_do_not_block_excel_save(monkeypatch) -> None:
    from ms_core.utils import intermediate_store as store_module
    from ms_core.utils.file_handler import FileHandler

    def _always_fail_store(*_args, **_kwargs):
        raise RuntimeError("store write failed")

    def _always_fail_parquet(*_args, **_kwargs):
        raise RuntimeError("parquet write failed")

    monkeypatch.setattr(store_module.IntermediateStore, "save", _always_fail_store)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _always_fail_parquet)

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        excel_path = Path(temp_dir) / "result.xlsx"
        df = pd.DataFrame({"Mz/RT": ["Sample_Type", "100.1/1.0"], "S1": ["case", 1234]})
        handler = FileHandler()
        saved = handler.save_data(df, excel_path, sheet_name="RawIntensity", save_parquet_cache=True)

        assert saved == excel_path
        assert excel_path.exists()
