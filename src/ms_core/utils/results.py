from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProcessingResult:
    """
    Standard return type for processor main() functions.
    """
    file_path: str
    output_path: str
    metabolites: int
    samples: int
    plots_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dict for UI/update pipelines.
        """
        data = {
            "file_path": self.file_path,
            "output_path": self.output_path,
            "metabolites": self.metabolites,
            "samples": self.samples,
        }
        if self.plots_dir:
            data["plots_dir"] = self.plots_dir
        if self.extra:
            data.update(self.extra)
        return data
