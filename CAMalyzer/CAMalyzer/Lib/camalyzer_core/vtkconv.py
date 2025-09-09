"""Conversion helpers producing VTK objects from numpy arrays."""

from __future__ import annotations

import numpy as np
import vtk
from skimage import measure


def array_to_polydata(segmented: np.ndarray) -> vtk.vtkPolyData:
    """Generate a ``vtkPolyData`` surface from a binary segmentation array."""
    verts, faces, _, _ = measure.marching_cubes(segmented, level=0.5)
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(*v)
    triangles = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, f[0])
        triangle.GetPointIds().SetId(1, f[1])
        triangle.GetPointIds().SetId(2, f[2])
        triangles.InsertNextCell(triangle)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    return polydata
