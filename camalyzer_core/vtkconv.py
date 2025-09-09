"""Utilities for converting between NumPy arrays and VTK-style lists."""

from typing import Any

import numpy as np


def numpy_to_list(arr: np.ndarray) -> list:
    """Convert a NumPy array to a nested Python list."""
    return arr.tolist()


def list_to_numpy(data: Any) -> np.ndarray:
    """Convert a Python sequence to ``np.ndarray``."""
    return np.asarray(data)
