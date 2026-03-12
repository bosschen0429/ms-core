from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".tmp"
PROJECT_TEST_TEMP_ROOT = TMP_ROOT / "tests"


@pytest.fixture(scope="session")
def project_temp_root() -> Path:
    PROJECT_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECT_TEST_TEMP_ROOT


@pytest.fixture
def project_temp_dir(project_temp_root: Path):
    @contextmanager
    def _factory(prefix: str = "case-"):
        temp_dir = project_temp_root / f"{prefix}{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return _factory
