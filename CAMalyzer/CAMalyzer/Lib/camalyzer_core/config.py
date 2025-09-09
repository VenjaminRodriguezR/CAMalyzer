"""Configuration values used by the core library."""

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover - torch may be missing during docs/tests
    DEVICE = "cpu"

# Default sliding window parameters
DEFAULT_ROI_SIZE = (96, 96, 96)
DEFAULT_SW_BATCH_SIZE = 4
