import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from camalyzer_core.post import extract_surface, select_largest, poisson_reconstruct


def test_post_pipeline():
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    volume[0, 0, 0] = 1
    surface = extract_surface(volume)
    cluster = select_largest([surface])
    mesh = poisson_reconstruct(cluster)
    assert mesh.shape[1] == 3  # each point is 3D
