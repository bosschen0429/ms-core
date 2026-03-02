"""
Base classes for MS Preprocessing processors.

This module defines the abstract base class and common interfaces
for all processing modules in the toolkit.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

import pandas as pd


@dataclass
class ProcessingResult:
    """
    Container for processing results.

    Attributes:
        success: Whether the processing completed successfully
        data: The processed DataFrame
        message: Human-readable status message
        statistics: Dictionary of processing statistics
        warnings: List of warning messages
        errors: List of error messages
        metadata: Additional metadata about the processing
    """

    success: bool
    data: Optional[pd.DataFrame] = None
    message: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Add timestamp to metadata."""
        self.metadata["processed_at"] = datetime.now().isoformat()


class BaseProcessor(ABC):
    """
    Abstract base class for all processing modules.

    Provides common interface and utility methods for
    data processing operations.
    """

    def __init__(self, name: str):
        """
        Initialize the processor.

        Args:
            name: Human-readable name of the processor
        """
        self.name = name
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._cancelled = False

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """
        Set a callback function for progress updates.

        Args:
            callback: Function that takes (progress_percent, status_message)
        """
        self._progress_callback = callback

    def update_progress(self, percent: float, message: str = "") -> None:
        """
        Update processing progress.

        Args:
            percent: Progress percentage (0-100)
            message: Status message
        """
        if self._progress_callback:
            self._progress_callback(percent, message)

    def cancel(self) -> None:
        """Request cancellation of the current processing operation."""
        self._cancelled = True

    def reset(self) -> None:
        """Reset the processor state for a new operation."""
        self._cancelled = False

    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> ProcessingResult:
        """
        Process the input data.

        Args:
            df: Input DataFrame to process
            **kwargs: Additional processing parameters

        Returns:
            ProcessingResult containing the processed data and statistics
        """
        pass

    @abstractmethod
    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data before processing.

        Args:
            df: Input DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this processor.

        Returns:
            Dictionary containing processor metadata
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
        }
