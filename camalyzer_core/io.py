"""I/O helpers for models and volumes."""

from typing import Any

import numpy as np


def load_model(path: str) -> dict:
    """Dummy model loader returning path info."""
    return {"model_path": path}


def load_volume(data: Any) -> np.ndarray:
    """Return numpy representation of the volume."""
    return np.asarray(data)


def get_device(prefer_gpu: bool = True) -> str:
    """Return ``"cuda"`` if available and requested, else ``"cpu"``."""
    try:
        import torch

        if prefer_gpu and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
