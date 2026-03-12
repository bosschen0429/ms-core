from __future__ import annotations

from pathlib import Path


def test_ms_core_project_temp_root_stays_under_dot_tmp(project_temp_root: Path) -> None:
    assert project_temp_root.parts[-2:] == (".tmp", "tests")


def test_ms_core_project_temp_dir_creates_path_under_dot_tmp(project_temp_dir) -> None:
    with project_temp_dir() as temp_dir:
        assert temp_dir.parts[-3] == ".tmp"
