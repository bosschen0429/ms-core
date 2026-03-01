"""Unified data container for the entire MS processing pipeline.

All core modules accept and return MSDataset instances, providing a single
interface that eliminates the need for adapter layers between projects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MSDataset:
    """Immutable-ish container that travels through the 12-step pipeline.

    Attributes:
        matrix: Intensity matrix (samples × features). Index = sample IDs,
            columns = feature IDs (e.g. ``"252.1098/18.45"``).
        labels: Group labels aligned with ``matrix.index``.
        sample_info: Per-sample metadata — columns may include
            ``Sample_Type``, ``Injection_Order``, ``Injection_Volume``,
            ``Batch``, etc.  Index = sample IDs.
        feature_info: Per-feature metadata — columns may include
            ``mz``, ``rt``, ``is_istd``, ``istd_mz``, etc.
            Index = feature IDs (matching ``matrix.columns``).
        processing_log: Ordered list of human-readable log entries
            appended by each pipeline step.
    """

    matrix: pd.DataFrame
    labels: pd.Series
    sample_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    processing_log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    @property
    def sample_ids(self) -> pd.Index:
        return self.matrix.index

    @property
    def feature_ids(self) -> pd.Index:
        return self.matrix.columns

    @property
    def groups(self) -> list[str]:
        """Sorted list of unique group labels."""
        return sorted(self.labels.unique().astype(str))

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Append a processing log entry."""
        self.processing_log.append(message)

    def replace_matrix(self, new_matrix: pd.DataFrame, msg: str | None = None) -> MSDataset:
        """Return a new MSDataset with an updated matrix (and optional log).

        Labels and metadata are re-aligned to the new matrix's index/columns.
        """
        new_labels = self.labels.reindex(new_matrix.index)

        new_sample_info = self.sample_info
        if not self.sample_info.empty:
            common_idx = new_matrix.index.intersection(self.sample_info.index)
            new_sample_info = self.sample_info.loc[common_idx]

        new_feature_info = self.feature_info
        if not self.feature_info.empty:
            common_cols = new_matrix.columns.intersection(self.feature_info.index)
            new_feature_info = self.feature_info.loc[common_cols]

        new_log = list(self.processing_log)
        if msg:
            new_log.append(msg)

        return MSDataset(
            matrix=new_matrix,
            labels=new_labels,
            sample_info=new_sample_info,
            feature_info=new_feature_info,
            processing_log=new_log,
        )

    def copy(self) -> MSDataset:
        """Deep copy for snapshot/undo purposes."""
        return MSDataset(
            matrix=self.matrix.copy(),
            labels=self.labels.copy(),
            sample_info=self.sample_info.copy(),
            feature_info=self.feature_info.copy(),
            processing_log=list(self.processing_log),
        )
