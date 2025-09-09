"""Input/output helpers working on ``numpy`` arrays."""

from __future__ import annotations

import numpy as np
import torch

try:  # monai is optional during testing
    from monai.transforms import ScaleIntensity
except Exception:  # pragma: no cover
    class ScaleIntensity:  # type: ignore
        def __call__(self, x):
            return x


def volume_to_tensor(volume: np.ndarray) -> torch.Tensor:
    """Convert a numpy volume to a torch tensor and normalize it."""
    tensor = torch.from_numpy(volume).float().permute(2, 1, 0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return ScaleIntensity()(tensor)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a single channel tensor back to a numpy array."""
    array = tensor.squeeze().permute(2, 1, 0).cpu().numpy()
    return array
