"""Tests for unified parquet intermediate store contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_intermediate_store_roundtrip_preserves_metadata(project_temp_dir) -> None:
    from ms_core.utils.intermediate_store import IntermediateStore

    df = pd.DataFrame(
        {
            "Mz/RT": ["100.1/1.0", "200.2/2.0"],
            "Sample1": [10.0, 20.0],
        }
    )
    metadata = {
        "red_font_rows": [1],
        "blue_font_cells": [[1, 1]],
        "protected_rows": [0],
        "sample_info_ref": "sample_info.xlsx",
        "deleted_feature_ref": "deleted_feature.xlsx",
    }
    with project_temp_dir() as temp_dir:
        parquet_path = temp_dir / "step1.parquet"
        IntermediateStore.save(df=df, parquet_path=parquet_path, metadata=metadata)
        loaded_df, loaded_metadata = IntermediateStore.load(parquet_path=parquet_path)

    pd.testing.assert_frame_equal(loaded_df, df)
    for key, expected in metadata.items():
        assert loaded_metadata.get(key) == expected


def test_save_parquet_cache_default_enabled() -> None:
    from ms_core.preprocessing.settings import Settings

    assert Settings.SAVE_PARQUET_CACHE is True


def test_intermediate_store_save_handles_mixed_object_columns(project_temp_dir) -> None:
    from ms_core.utils.intermediate_store import IntermediateStore

    df = pd.DataFrame(
        {
            "MZ": ["Sample_Type", 123.456],
            "RT": ["Tolerance", 1.5],
        }
    )

    with project_temp_dir() as temp_dir:
        parquet_path = temp_dir / "mixed.parquet"
        IntermediateStore.save(df=df, parquet_path=parquet_path, metadata={})
        loaded_df, _ = IntermediateStore.load(parquet_path=parquet_path)

    assert loaded_df.shape == df.shape
