"""Surface related post-processing functions."""

from __future__ import annotations

import vtk


def smooth_polydata(polydata: vtk.vtkPolyData, iterations: int = 30) -> vtk.vtkPolyData:
    """Return a smoothed copy of *polydata*."""
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(smoother.GetOutput())
    return result
