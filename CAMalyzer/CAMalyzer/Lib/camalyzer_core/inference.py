"""High level inference routines used by the CAMalyzer module."""

from __future__ import annotations

import os
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch

from .config import DEVICE
from .io import volume_to_tensor, tensor_to_numpy
from .params import InferenceParams


def load_model(model_path: str) -> torch.nn.Module:
    """Load a MONAI UNet model from ``model_path``."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    import monai

    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.125,
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_inference(volume: torch.Tensor, model: torch.nn.Module, params: InferenceParams) -> torch.Tensor:
    """Run sliding window inference and return the segmentation tensor."""
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        prediction = sliding_window_inference(volume, params.roi_size, params.sw_batch_size, model)
        prediction = [post_trans(p) for p in decollate_batch(prediction)]
    return prediction[0]


def segment_numpy(volume_array, model_path: str, params: InferenceParams | None = None) -> np.ndarray:
    """Convenience wrapper to perform full inference on a numpy array."""
    if params is None:
        params = InferenceParams()
    model = load_model(model_path)
    tensor = volume_to_tensor(volume_array).to(DEVICE)
    seg = run_inference(tensor, model, params)
    return tensor_to_numpy(seg)
