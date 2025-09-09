import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from camalyzer_core.inference import segment_volume


def test_segment_volume_shape():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    mask = segment_volume(volume)
    assert mask.shape == volume.shape
    assert mask.dtype == np.uint8
