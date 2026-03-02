"""
Excel file I/O utilities for metabolomics data processing.

Provides output path generation, directory management, and column validation helpers.
"""
import pandas as pd
from openpyxl import load_workbook
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .constants import VALIDATION_THRESHOLDS, DATETIME_FORMAT_FULL

def get_project_root() -> Path:
    """
    Resolve the project root based on the package location.

    Falls back to the current working directory if the expected layout
    is not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


def get_output_root() -> Path:
    """
    Return the project-level output directory and ensure it exists.
    """
    output_root = get_project_root() / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def validate_required_columns(
    df: pd.DataFrame,
    required: List[str],
    sheet_name: str = "DataFrame"
) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist in a DataFrame.

    Args:
        df: DataFrame to validate
        required: List of required column names
        sheet_name: Name of the sheet (for error messages)

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, [f"'{sheet_name}' 缺少必要欄位: {', '.join(missing)}"]
    return True, []


def generate_output_filename(
    prefix: str,
    input_file: str = None,
    timestamp: str = None,
    extension: str = ".xlsx"
) -> str:
    """
    Generate a timestamped output filename.

    Args:
        prefix: Prefix for the filename (e.g., "ISTD_Results")
        input_file: Optional input file to base the name on
        timestamp: Optional timestamp string (uses current time if not provided)
        extension: File extension (default: .xlsx)

    Returns:
        Generated filename string
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().strftime(DATETIME_FORMAT_FULL)

    if input_file:
        # Append to existing filename pattern
        input_base = Path(input_file).stem
        return f"{input_base}_{prefix}_{timestamp}{extension}"
    else:
        return f"{prefix}_{timestamp}{extension}"


def get_output_directory(
    input_file: str = None,
    subdir: str = None
) -> Path:
    """
    Get the output directory for a given input file.

    Creates the directory if it doesn't exist.

    Args:
        input_file: Input file path (unused; retained for compatibility)
        subdir: Optional subdirectory name (e.g., "plots")

    Returns:
        Path object for the output directory
    """
    output_dir = get_output_root()
    if subdir:
        output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_path(
    prefix: str,
    timestamp: str = None,
    extension: str = ".xlsx"
) -> Path:
    """
    Build a full output file path under the project output directory.
    """
    output_dir = get_output_root()
    filename = generate_output_filename(prefix, timestamp=timestamp, extension=extension)
    return output_dir / filename


def build_plots_dir(
    subdir: str,
    timestamp: str = None,
    session_prefix: str = None
) -> Path:
    """
    Build a plots directory under the project output directory.
    """
    output_dir = get_output_root()
    plots_root = output_dir / subdir
    if session_prefix:
        session_name = f"{session_prefix}_{timestamp}" if timestamp else session_prefix
        plots_dir = plots_root / session_name
    else:
        plots_dir = plots_root
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir
