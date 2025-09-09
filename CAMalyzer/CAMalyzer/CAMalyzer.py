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
                outputLabelMap: vtkMRMLScalarVolumeNode,
                modelOutput: vtkMRMLModelNode,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                showResult: bool = True) -> None:
        """
        Process the input volume using the specified model and generate outputs.
        """
        if not inputVolume or not outputLabelMap or not modelOutput:
            raise ValueError("Input volume, output label map, or model output is invalid")
        if not os.path.isfile(modelForPrediction):
            raise ValueError(f"Model file not found: {modelForPrediction}")

        logging.info(f"Processing started with model: {modelForPrediction}")

        # Load the model
        model = self.load_model(modelForPrediction)

        # Preprocess and segment
        import numpy as np
        import torch
        from monai.inferers import sliding_window_inference
        from monai.transforms import Compose, ScaleIntensity, Activations, AsDiscrete
        from monai.data import decollate_batch
        from skimage import measure
        import open3d as o3d

        volume_array = slicer.util.arrayFromVolume(inputVolume)
        volume_tensor = torch.from_numpy(volume_array).float().permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        volume_tensor = ScaleIntensity()(volume_tensor).to(self.device)

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        with torch.no_grad():
            prediction = sliding_window_inference(volume_tensor, roi_size, sw_batch_size, model)
            prediction = [post_trans(p) for p in decollate_batch(prediction)]
            segmented_array = prediction[0].cpu().squeeze().permute(2, 1, 0).numpy()

        logging.info(f"Segmentation completed. Shape: {segmented_array.shape}")

        # Update the label map
        slicer.util.updateVolumeFromArray(outputLabelMap, segmented_array)
        ijk_to_ras_matrix = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijk_to_ras_matrix)
        outputLabelMap.SetIJKToRASMatrix(ijk_to_ras_matrix)

        # Generate and clean the 3D model
        verts, faces, _, _ = measure.marching_cubes(segmented_array, level=0.5)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        point_cloud = mesh.sample_points_uniformly(number_of_points=100000)
        labels = np.array(point_cloud.cluster_dbscan(eps=10, min_points=150, print_progress=True))
        max_cluster_index = np.bincount(labels[labels >= 0]).argmax()
        filtered_points = point_cloud.select_by_index(np.where(labels == max_cluster_index)[0])

        mesh_filtered, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_points, depth=9)

        # Smooth the mesh using VTK
        polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_triangles = vtk.vtkCellArray()
        for vert in np.asarray(mesh_filtered.vertices):
            vtk_points.InsertNextPoint(vert)
        for face in np.asarray(mesh_filtered.triangles):
            triangle = vtk.vtkTriangle()
            for i in range(3):
                triangle.GetPointIds().SetId(i, face[i])
            vtk_triangles.InsertNextCell(triangle)
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_triangles)

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(polydata)
        smoother.SetNumberOfIterations(50)
        smoother.Update()

        smoothed_polydata = smoother.GetOutput()
        modelOutput.SetAndObservePolyData(smoothed_polydata)
        modelOutput.CreateDefaultDisplayNodes()
        modelOutput.GetDisplayNode().SetColor(0.5, 0.8, 1.0)

        logging.info("3D model generation completed.")

        if showResult:
            slicer.util.setSliceViewerLayers(background=inputVolume, label=outputLabelMap)
            slicer.util.showStatusMessage("Segmentation and model generation completed.", 2000)

        logging.info("Processing finished.")


"""
class CAMalyzerLogic(ScriptedLoadableModuleLogic):
    #This class should implement all the actual
    #computation done by your module.  The interface
    #should be such that other python code can import
    #this class and make use of the functionality without
    #requiring an instance of the Widget.
    #Uses ScriptedLoadableModuleLogic base class, available at:
    #https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    

    def __init__(self) -> None:
    
        #Called when the logic class is instantiated. Can be used for initializing member variables.
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return CAMalyzerParameterNode(super().getParameterNode())

    def process(self,
            inputVolume: vtkMRMLScalarVolumeNode,
            modelForPrediction: str,
            outputLabelMap: vtkMRMLScalarVolumeNode,
            modelOutput: str = "",
            showResult: bool = True) -> None:
        
        #Process the input volume using the specified model and generate outputs.
        
        if not inputVolume or not outputLabelMap:
            raise ValueError("Input volume or output label map is invalid")
        if not os.path.isfile(modelForPrediction):
            raise ValueError(f"Model file not found: {modelForPrediction}")

        logging.info(f"Processing started with model: {modelForPrediction}")

        # Example logic for loading and applying the model
        # Replace this with actual deep learning inference logic
        result = "Prediction completed successfully."  # Placeholder for model result
        modelOutput = result

        if showResult:
            slicer.util.showStatusMessage(f"Processing completed: {modelOutput}", 2000)

        logging.info("Processing finished.")
"""

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
