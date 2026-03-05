"""
Configuration settings for MS Preprocessing Toolkit.

This module defines all configurable parameters and default values
used throughout the preprocessing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import os
import tempfile


@dataclass
class ISTDConfig:
    """Configuration for ISTD Marker module."""

    # Default tolerances for duplicate detection
    default_ppm_tolerance: float = 20.0
    default_rt_tolerance: float = 1.0

    # Column names
    feature_id_col: str = "Mz/RT"
    tolerance_col: str = "m/z Tolerance( ppm)/RT Tolerance"
    sample_type_col: str = "Sample_Type"

    # Default ISTD m/z list
    default_istd_mz: list = field(default_factory=lambda: [
        261.1273,
        245.1324,
        289.0841,
        300.1605,
        269.1436,
        482.2087,
        303.0913,
    ])


@dataclass
class DuplicateRemovalConfig:
    """Configuration for Duplicate Remover module."""

    # Tolerance settings
    mz_tolerance_ppm: float = 20.0
    rt_tolerance: float = 1.0

    # Processing options
    preserve_red_font: bool = True
    top_n_results: Optional[int] = None

    # Column detection keywords
    rt_keywords: list = field(default_factory=lambda: ["rt", "retention", "time", "min"])
    mz_keywords: list = field(default_factory=lambda: ["mz", "m/z", "mass"])
    intensity_keywords: list = field(default_factory=lambda: ["intensity", "area", "height", "abundance"])


@dataclass
class FeatureFilterConfig:
    """Configuration for Feature Filter module."""

    # Signal threshold
    signal_threshold: float = 5000.0

    # QC warning threshold
    qc_warning_threshold: float = 0.5

    # Default filter thresholds
    default_background_threshold: float = 0.33
    default_skew_threshold: float = 0.66
    default_diff_threshold: float = 0.30
    default_qc_ratio_threshold: float = 0.0

    # Excluded sample types (not included in analysis)
    excluded_types: list = field(default_factory=lambda: ["blank", "standard", "sdolek", "qc"])


@dataclass
class DataOrganizerConfig:
    """Configuration for Data Organizer module."""

    # Expected column structure
    required_columns: list = field(default_factory=lambda: ["Mz/RT", "Sample_Type"])

    # Data validation settings
    min_samples: int = 1
    min_features: int = 1


@dataclass
class ProcessingConfig:
    """Combined configuration for the entire processing pipeline."""

    istd: ISTDConfig = field(default_factory=ISTDConfig)
    duplicate: DuplicateRemovalConfig = field(default_factory=DuplicateRemovalConfig)
    filter: FeatureFilterConfig = field(default_factory=FeatureFilterConfig)
    organizer: DataOrganizerConfig = field(default_factory=DataOrganizerConfig)


class Settings:
    """
    Application-wide settings manager.

    Handles loading, saving, and accessing configuration settings
    for the MS Preprocessing Toolkit.
    """

    # Supported file formats (canonical source ??used by FileHandler too)
    SUPPORTED_FORMATS = frozenset({".xlsx", ".xls", ".csv", ".tsv", ".txt", ".parquet"})
    # Enable parquet intermediates by default for Step1-4 chaining performance.
    SAVE_PARQUET_CACHE = True
    PARQUET_CACHE_ROOT_ENV = "MSPTK_PARQUET_CACHE_ROOT"
    PARQUET_CACHE_DIRNAME = "ms-preprocessing-toolkit/cache"

    # GUI Settings
    WINDOW_TITLE = "MS Preprocessing Toolkit"
    WINDOW_SIZE = (1100, 720)
    THEME = "dark-blue"  # customtkinter theme

    # Workflow steps
    WORKFLOW_STEPS = [
        ("data_organizer", "鞈??渡?", "Data Organization"),
        ("istd_marker", "ISTD 璅?", "ISTD Marking"),
        ("duplicate_remover", "??閮??芷", "Duplicate Removal"),
        ("feature_filter", "Feature Filtering", "Feature Filtering"),
    ]

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize settings with optional custom configuration."""
        self.config = config or ProcessingConfig()
        self._config_path: Optional[Path] = None

    @classmethod
    def load_from_file(cls, path: Path) -> "Settings":
        """Load settings from a configuration file."""
        # TODO: Implement JSON/YAML config file loading
        return cls()

    def save_to_file(self, path: Path) -> None:
        """Save current settings to a configuration file."""
        # TODO: Implement JSON/YAML config file saving
        self._config_path = path

    def get_processing_config(self) -> ProcessingConfig:
        """Get the current processing configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    @classmethod
    def get_parquet_cache_root(cls) -> Path:
        """Resolve machine-local parquet cache root."""
        override = os.getenv(cls.PARQUET_CACHE_ROOT_ENV)
        if override:
            return Path(override)

        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / cls.PARQUET_CACHE_DIRNAME

        return Path(tempfile.gettempdir()) / cls.PARQUET_CACHE_DIRNAME


