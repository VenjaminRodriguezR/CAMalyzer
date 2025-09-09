"""Parameter dataclasses for CAMalyzer."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class InferenceParams:
    """Parameters controlling inference."""

    roi_size: Tuple[int, int, int] = (64, 64, 64)
    threshold: float = 0.5

    def validate(self) -> None:
        if len(self.roi_size) != 3:
            raise ValueError("roi_size must be a 3-tuple")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
