"""Core functionality for CAMalyzer module.

This package is independent from the Slicer module GUI and
implements reusable pieces such as configuration, inference and
post-processing utilities.
"""

from .config import DEVICE, DEFAULT_ROI_SIZE, DEFAULT_SW_BATCH_SIZE
from .params import InferenceParams

__all__ = [
    "DEVICE",
    "DEFAULT_ROI_SIZE",
    "DEFAULT_SW_BATCH_SIZE",
    "InferenceParams",
]

try:  # skimage/vtk may not be present during lightweight testing
    from .vtkconv import array_to_polydata

    __all__ += ["array_to_polydata"]
except Exception:  # pragma: no cover
    array_to_polydata = None  # type: ignore

try:  # Optional dependencies (monai) may not be available during testing
    from .inference import load_model, run_inference

    __all__ += ["load_model", "run_inference"]
except Exception:  # pragma: no cover - used only when monai missing
    load_model = None  # type: ignore
    run_inference = None  # type: ignore
