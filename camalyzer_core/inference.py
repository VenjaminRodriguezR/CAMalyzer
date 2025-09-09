"""Inference utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .params import InferenceParams
from .config import DEFAULT_CONFIG


def segment_volume(volume: Any, model_path: str = "", params: InferenceParams | None = None) -> np.ndarray:
    """Pretend to run inference on ``volume`` and return a segmentation mask.

    The implementation is intentionally lightweight so that unit tests can run
    without heavy dependencies.  The function converts the input to a NumPy
    array and returns a zero-valued array of the same shape.
    """

    data = np.asarray(volume)
    _params = params or InferenceParams()
    _params.validate()
    return np.zeros_like(data, dtype=np.uint8)
