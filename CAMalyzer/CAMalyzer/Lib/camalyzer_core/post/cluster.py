"""Utilities for working with connected components."""

from __future__ import annotations

import vtk


def largest_cluster(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Extract the largest connected surface from *polydata*."""
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(connectivity.GetOutput())
    return result
