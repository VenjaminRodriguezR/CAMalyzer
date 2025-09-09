import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "Lib"))

vtk = pytest.importorskip("vtk")
from camalyzer_core.post import smooth_polydata


def test_smooth_polydata():
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    poly = sphere.GetOutput()
    smoothed = smooth_polydata(poly, iterations=5)
    assert smoothed.GetNumberOfPoints() == poly.GetNumberOfPoints()
