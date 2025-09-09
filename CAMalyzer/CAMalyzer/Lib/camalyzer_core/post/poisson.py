"""Poisson surface reconstruction helpers."""

from __future__ import annotations

import open3d as o3d
import numpy as np
import vtk


def poisson_reconstruction(polydata: vtk.vtkPolyData, depth: int = 8) -> vtk.vtkPolyData:
    """Run Poisson surface reconstruction using :mod:`open3d`."""
    points = o3d.utility.Vector3dVector([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    pcd = o3d.geometry.PointCloud(points)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    o3d_poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    for pt in np.asarray(mesh.vertices):
        vtk_points.InsertNextPoint(*pt)
    o3d_poly.SetPoints(vtk_points)
    return o3d_poly
