"""Dataclasses defining parameter sets used across the core library."""

from dataclasses import dataclass
from typing import Tuple

from .config import DEFAULT_ROI_SIZE, DEFAULT_SW_BATCH_SIZE


@dataclass
class InferenceParams:
    """Parameters controlling sliding window inference."""

    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE
    sw_batch_size: int = DEFAULT_SW_BATCH_SIZE
