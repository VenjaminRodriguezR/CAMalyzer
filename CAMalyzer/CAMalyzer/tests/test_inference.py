import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "Lib"))

from camalyzer_core.io import volume_to_tensor, tensor_to_numpy


def test_round_trip():
    data = np.random.rand(5, 5, 5).astype("float32")
    tensor = volume_to_tensor(data)
    back = tensor_to_numpy(tensor)
    assert back.shape == data.shape
