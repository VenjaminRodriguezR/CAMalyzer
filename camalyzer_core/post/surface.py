"""Surface extraction helpers."""

import numpy as np


def extract_surface(volume: np.ndarray) -> np.ndarray:
    """Return the set of non-zero voxel indices as a stand-in surface."""
    volume = np.asarray(volume)
    return np.argwhere(volume > 0)
