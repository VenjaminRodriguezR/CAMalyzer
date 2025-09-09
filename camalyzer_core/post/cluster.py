"""Clustering helpers."""

from typing import Iterable

import numpy as np


def select_largest(clusters: Iterable[np.ndarray]) -> np.ndarray:
    """Return the cluster with the most elements."""
    clusters = list(clusters)
    if not clusters:
        raise ValueError("no clusters provided")
    return max(clusters, key=lambda c: len(c))
