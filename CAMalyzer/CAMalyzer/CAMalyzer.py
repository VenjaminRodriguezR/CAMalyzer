import logging
import os
from typing import Annotated, Optional
from qt import QFileDialog  
import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import time
import threading
import statistics
import gc
import psutil
import math
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLLabelMapVolumeNode, vtkMRMLModelNode

import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensity, Activations, AsDiscrete
from monai.data import decollate_batch
from skimage import measure
import open3d as o3d
from vtk.util.numpy_support import vtk_to_numpy
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest

#
# CAMalyzer
#

class CAMalyzer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CAMalyzer"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#CAMalyzer">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # CAMalyzer1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CAMalyzer',
        sampleName='CAMalyzer1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'CAMalyzer1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='CAMalyzer1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='CAMalyzer1'
    )

    # CAMalyzer2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CAMalyzer',
        sampleName='CAMalyzer2',
        thumbnailFileName=os.path.join(iconsPath, 'CAMalyzer2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='CAMalyzer2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='CAMalyzer2'
    )


#
# CAMalyzerParameterNode
#

@parameterNodeWrapper
class CAMalyzerParameterNode:
    """
    Parameters for the CAMalyzer module.

    inputVolume - The input volume for segmentation.
    modelForPrediction - File path to the .pth model to be used for prediction.
    outputLabelMap - The output label map resulting from the segmentation.
    modelOutput - The output 3D model generated by the module.
    """
    inputVolume: vtkMRMLScalarVolumeNode  # Input volume to process
    modelForPrediction: str = ""  # Path to the model file (.pth)
    outputLabelMap: vtkMRMLLabelMapVolumeNode  # Output label map
    modelOutput: vtkMRMLModelNode  # Output 3D model


#
# CAMalyzerWidget
#
class CAMalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class."""

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CAMalyzer.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = CAMalyzerLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.BrowseModelButton.connect('clicked(bool)', self.onModelPathButtonClicked)

        # Initialize parameter node
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

        # Ensure inputVolume is initialized
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
            else:
                logging.warning("No ScalarVolumeNode found in the scene for inputVolume.")

        # Ensure outputLabelMap is initialized
        if not self._parameterNode.outputLabelMap:
            self._parameterNode.outputLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            logging.info("Created a new LabelMapVolumeNode for outputLabelMap.")

        # Ensure modelOutput is initialized
        if not self._parameterNode.modelOutput:
            self._parameterNode.modelOutput = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            logging.info("Created a new ModelNode for modelOutput.")

    def setParameterNode(self, inputParameterNode: Optional[CAMalyzerParameterNode]) -> None:
        """
        Set and observe parameter node.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """
        Enable or disable the Apply button based on parameter validity.
        """
        missing = []
        if not self._parameterNode.inputVolume:
            missing.append("Input Volume")
        if not self._parameterNode.modelForPrediction:
            missing.append("Model for Prediction")
        if not self._parameterNode.outputLabelMap:
            missing.append("Output Label Map")
        if not self._parameterNode.modelOutput:
            missing.append("Model Output")

        if not missing:
            self.ui.applyButton.toolTip = "Run segmentation and model generation"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = f"Missing parameters: {', '.join(missing)}"
            self.ui.applyButton.enabled = False


    
    
    def onModelPathButtonClicked(self) -> None:
        """
        Open a file dialog to select a model file and update the QLineEdit widget.
        """
        try:
            filePath = QFileDialog.getOpenFileName(
                self.parent,
                "Select Model File",  # Title of the dialog
                "",                   # Initial directory (empty means default)
                "Model Files (*.pth)" # Filter for file types
            )
            if filePath:  # Check if a file was selected
                self.ui.modelForPrediction.setText(filePath)  # Update QLineEdit widget
                if self._parameterNode:
                    self._parameterNode.modelForPrediction = filePath  # Update parameter node
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to select model file: {str(e)}")

    def onApplyButton(self) -> None:
        """
        Run the processing when the Apply button is clicked.
        """
        with slicer.util.tryWithErrorDisplay("Failed to run segmentation and model generation.", waitCursor=True):
            self.logic.process(
                inputVolume=self._parameterNode.inputVolume,
                modelForPrediction=self._parameterNode.modelForPrediction,
                outputLabelMap=self._parameterNode.outputLabelMap,
                modelOutput=self._parameterNode.modelOutput,roi_size=(96, 96, 96),
                sw_batch_size=4
            )

#
# CAMalyzerLogic
#
import logging
import slicer
import os
import sys
import importlib.util
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic


class CAMalyzerLogic(ScriptedLoadableModuleLogic):
    """
    CAMalyzerLogic handles automatic segmentation, dependency management, and 3D model generation.
    """

    def __init__(self) -> None:
        """
        Initialize logic and ensure dependencies are installed.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device set to: {self.device}")
        self.check_and_install_dependencies()

    def check_and_install_dependencies(self):
                required_packages = [
                    "torch",
                    "monai",
                    "numpy",
                    "scikit-image",
                    "open3d",
                    "psutil",
                    "pynvml",
                    "scipy"
                ]
                for package in required_packages:
                    self.install_package_if_missing(package)

    def install_package_if_missing(self, package_name):
        """
        Check if a package is installed, and if not, install it using pip.
        """
        try:
            # Use importlib.util.find_spec to check if the package is installed
            if importlib.util.find_spec(package_name) is None:
                logging.info(f"Installing missing package: {package_name}")
                slicer.util.showStatusMessage(f"Installing {package_name}...", 2000)
                slicer.util.pip_install(package_name)
                logging.info(f"Successfully installed package: {package_name}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to install {package_name}: {str(e)}")
            raise

    def getParameterNode(self):
        """
        Get the parameter node wrapped with CAMalyzerParameterNode.
        """
        return CAMalyzerParameterNode(super().getParameterNode())

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a pre-trained PyTorch model.
        """
        if not os.path.isfile(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        import monai
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.125
        ).to(self.device)

        import torch
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info(f"Model loaded from {model_path}")
        return model

    def process(self,
            inputVolume: vtkMRMLScalarVolumeNode,
            modelForPrediction: str,
            outputLabelMap: vtkMRMLLabelMapVolumeNode,
            modelOutput: vtkMRMLModelNode,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            showResult: bool = True) -> None:

        if not inputVolume or not outputLabelMap or not modelOutput:
            raise ValueError("Input volume, output label map, or model output is invalid")
        if not os.path.isfile(modelForPrediction):
            raise ValueError(f"Model file not found: {modelForPrediction}")

        # ---- Names derived from input (consistent, human-readable)
        base = inputVolume.GetName() or "Input"
        name_seg_raw     = f"{base}_SegRaw"
        name_seg_opened  = f"{base}_SegOpened"
        name_model_final = f"{base}_ModelPoisson"
        name_seg_frommdl = f"{base}_SegFromModel"          # editable segmentation
        name_lbl_frommdl = f"{base}_SegFromModel_Label"    # exported labelmap

        # Ensure OpenCV is available
        try:
            import cv2  # noqa
        except Exception:
            self.install_package_if_missing("opencv-python-headless")
            import cv2  # noqa

        logging.info(f"Processing started with model: {modelForPrediction}")

        # ---- Load model
        model = self.load_model(modelForPrediction)

        # ---- Preprocess + inference
        import numpy as np
        import torch
        from monai.inferers import sliding_window_inference
        from monai.transforms import Compose, ScaleIntensity, Activations, AsDiscrete
        from monai.data import decollate_batch
        from skimage import measure
        import open3d as o3d
        from vtk.util.numpy_support import vtk_to_numpy

        volume_array = slicer.util.arrayFromVolume(inputVolume)  # (z,y,x)
        volume_tensor = torch.from_numpy(volume_array).float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        volume_tensor = ScaleIntensity()(volume_tensor).to(self.device)

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        with torch.no_grad():
            prediction = sliding_window_inference(volume_tensor, roi_size, sw_batch_size, model)
            prediction = [post_trans(p) for p in decollate_batch(prediction)]
            segmented_array = prediction[0].cpu().squeeze().permute(2, 1, 0).numpy().astype(np.uint8)

        logging.info(f"Segmentation completed. Shape: {segmented_array.shape}")

        # ---- Raw segmentation → outputLabelMap (named, aligned)
        outputLabelMap.SetName(name_seg_raw)
        slicer.util.updateVolumeFromArray(outputLabelMap, segmented_array)
        ijk_to_ras_matrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijk_to_ras_matrix)
        outputLabelMap.SetIJKToRASMatrix(ijk_to_ras_matrix)

        # ---- OpenCV 2D opening (kernel 7, repeat 10×) per axial slice
        opened_array = np.zeros_like(segmented_array, dtype=np.uint8)
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for z in range(segmented_array.shape[0]):
            slice_img = (segmented_array[z] * 255).astype(np.uint8)
            slice_opened = slice_img
            for _ in range(10):
                slice_opened = cv2.morphologyEx(slice_opened, cv2.MORPH_OPEN, kernel)
            opened_array[z] = (slice_opened > 0).astype(np.uint8)

        # ---- Intermediate opened labelmap (new node, named, aligned)
        openedLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name_seg_opened)
        slicer.util.updateVolumeFromArray(openedLabelMap, opened_array)
        openedLabelMap.SetIJKToRASMatrix(ijk_to_ras_matrix)

        # ---- Surface from opened mask → smooth → DBSCAN → Poisson
        verts, faces, _, _ = measure.marching_cubes(opened_array, level=0.5)

        points = vtk.vtkPoints()
        for v in verts:
            points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        triangles = vtk.vtkCellArray()
        for f in faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(f[0]))
            tri.GetPointIds().SetId(1, int(f[1]))
            tri.GetPointIds().SetId(2, int(f[2]))
            triangles.InsertNextCell(tri)
        polyData.SetPolys(triangles)

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(polyData)
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

        smoothedPolyData = smoother.GetOutput()
        verts_sm = vtk_to_numpy(smoothedPolyData.GetPoints().GetData())
        faces_sm = vtk_to_numpy(smoothedPolyData.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

        mesh_sm = o3d.geometry.TriangleMesh()
        mesh_sm.vertices = o3d.utility.Vector3dVector(verts_sm)
        mesh_sm.triangles = o3d.utility.Vector3iVector(faces_sm)
        mesh_sm.compute_vertex_normals()

        point_cloud = mesh_sm.sample_points_uniformly(number_of_points=100_000)
        labels = np.array(point_cloud.cluster_dbscan(eps=10, min_points=150, print_progress=True))
        valid = labels >= 0
        if not np.any(valid):
            raise RuntimeError("DBSCAN produced no valid clusters; adjust eps/min_points.")
        max_cluster = np.bincount(labels[valid]).argmax()
        pcd_filtered = point_cloud.select_by_index(np.where(labels == max_cluster)[0])

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh_filtered, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=9)
        mesh_filtered.compute_vertex_normals()

        # ---- Poisson mesh → VTK → smooth again → modelOutput (named)
        polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_tris = vtk.vtkCellArray()
        verts_np = np.asarray(mesh_filtered.vertices)
        tris_np = np.asarray(mesh_filtered.triangles)
        for v in verts_np:
            vtk_points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
        for t in tris_np:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(t[0]))
            tri.GetPointIds().SetId(1, int(t[1]))
            tri.GetPointIds().SetId(2, int(t[2]))
            vtk_tris.InsertNextCell(tri)
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_tris)

        sm2 = vtk.vtkSmoothPolyDataFilter()
        sm2.SetInputData(polydata)
        sm2.SetNumberOfIterations(100)
        sm2.Update()

        modelOutput.SetName(name_model_final)
        smoothed_polydata = sm2.GetOutput()
        modelOutput.SetAndObservePolyData(smoothed_polydata)
        modelOutput.CreateDefaultDisplayNodes()
        modelOutput.GetDisplayNode().SetColor(0.5, 0.8, 1.0)

        logging.info("3D model generation completed (Poisson + normals).")

        # =====================================================================
        # Convert final model → editable Segmentation (robust, version-safe)
        # =====================================================================
        logicSeg = slicer.modules.segmentations.logic()

        # 1) Segmentation node (geometry will be seeded from labelmap)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name_seg_frommdl)
        segNode.CreateDefaultDisplayNodes()

        # 2) Seed segmentation geometry by importing the OPENED labelmap
        #    This guarantees a Binary labelmap master with correct grid.
        logicSeg.ImportLabelmapToSegmentationNode(openedLabelMap, segNode)

        # 3) Hide the 'opened' seed segment so only the model will be exported later
        seg = segNode.GetSegmentation()
        ids = vtk.vtkStringArray()
        seg.GetSegmentIDs(ids)
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            segNode.GetDisplayNode().SetSegmentVisibility(sid, False)

        # 4) Import the Poisson model as a new segment; make it visible
        before = seg.GetNumberOfSegments()
        logicSeg.ImportModelToSegmentationNode(modelOutput, segNode)
        after = seg.GetNumberOfSegments()
        if after <= before:
            raise RuntimeError("ImportModelToSegmentationNode failed: no new segment added from model.")

        # Rename the last-added segment to "Poisson" and make it visible
        last_id = seg.GetNthSegmentID(after - 1)
        seg.GetSegment(last_id).SetName("Poisson")
        segNode.GetDisplayNode().SetSegmentVisibility(last_id, True)

        # 5) Create the TARGET LabelMap and initialize its geometry/data to match input
        segLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name_lbl_frommdl)
        zeros = np.zeros_like(opened_array, dtype=np.uint8)  # correct shape
        slicer.util.updateVolumeFromArray(segLabel, zeros)
        segLabel.SetIJKToRASMatrix(ijk_to_ras_matrix)

        # 6) Export *visible* segments (only "Poisson") to the labelmap, aligned to input
        ok = logicSeg.ExportVisibleSegmentsToLabelmapNode(segNode, segLabel, inputVolume)
        if not ok:
            raise RuntimeError("ExportVisibleSegmentsToLabelmapNode failed; check segment visibility and geometry.")

        if showResult:
            # Show the exported labelmap to confirm voxel-space export
            slicer.util.setSliceViewerLayers(background=inputVolume, label=segLabel)
            slicer.util.showStatusMessage(
                "Opened labelmap, Poisson model, editable segmentation, and exported labelmap created.", 2000
            )

        logging.info("Processing finished.")



#
# CAMalyzerTest
#

class CAMalyzerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for the CAMalyzer module.
    """

    def setUp(self):
        """
        Reset the state by clearing the scene.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """
        Run the test case.
        """
        self.setUp()
        self.test_segmentation_with_test_data()

    def test_segmentation_with_test_data(self):
        """
        Test the segmentation logic using test data.
        """
        logging.info("Starting CAMalyzer test with test data.")

        # Define the path to the test files
        test_data_dir = "/home/notvenja24/Escritorio/CAMalyzer/CAMalyzer/CAMalyzer/Testing/Test_files"
        volume_path = os.path.join(test_data_dir, "RU019FOVB.nii.gz")
        model_path = os.path.join(test_data_dir, "Best_Model_2480.pth")

        # Check that the test files exist
        self.assertTrue(os.path.exists(volume_path), f"Test volume not found at {volume_path}")
        self.assertTrue(os.path.exists(model_path), f"Test model not found at {model_path}")

        # Load the test volume
        logging.info(f"Loading test volume from {volume_path}")
        volume_node = slicer.util.loadVolume(volume_path)
        self.assertIsNotNone(volume_node, "Failed to load test volume.")

        # Create output nodes
        logging.info("Creating output nodes for label map and 3D model.")
        output_label_map = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TestOutputLabelMap")
        model_output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestModelOutput")

        # Run the CAMalyzer logic
        logging.info("Running CAMalyzer logic.")
        logic = CAMalyzerLogic()
        logic.process(
            inputVolume=volume_node,
            modelForPrediction=model_path,
            outputLabelMap=output_label_map,
            modelOutput=model_output,
            showResult=False  # Disable GUI updates for testing
        )

        # Validate the outputs
        logging.info("Validating the outputs.")
        self.assertIsNotNone(output_label_map.GetImageData(), "Label map was not generated.")
        self.assertIsNotNone(model_output.GetPolyData(), "3D model was not generated.")

        logging.info("CAMalyzer test completed successfully.")
