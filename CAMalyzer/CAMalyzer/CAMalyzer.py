import logging
import os
from typing import Optional

import vtk
from qt import QFileDialog

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper

from camalyzer_core import (
    DEVICE,
    InferenceParams,
    array_to_polydata,
    load_model,
    run_inference,
)
from camalyzer_core.io import volume_to_tensor, tensor_to_numpy
from camalyzer_core.deps import check_and_install_dependencies
from camalyzer_core.post import smooth_polydata


# -----------------------------------------------------------------------------
# Module metadata


class CAMalyzer(ScriptedLoadableModule):
    """Slicer module that exposes CAMalyzer core functionality."""

    def __init__(self, parent) -> None:
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "CAMalyzer"
        parent.categories = ["Examples"]
        parent.dependencies = []
        parent.contributors = ["Benjamin Rodriguez (B3MAT)"]
        parent.helpText = (
            "Segmentation and model generation for cam morphology analysis."
        )
        parent.acknowledgementText = (
            "This file was originally developed by the Slicer community."
        )


# -----------------------------------------------------------------------------
# Parameter node


@parameterNodeWrapper
class CAMalyzerParameterNode:
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    modelForPrediction: str = ""
    outputLabelMap: slicer.vtkMRMLLabelMapVolumeNode
    modelOutput: slicer.vtkMRMLModelNode


# -----------------------------------------------------------------------------
# Widget


class CAMalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: Optional[CAMalyzerLogic] = None
        self._parameterNode: Optional[CAMalyzerParameterNode] = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/CAMalyzer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = CAMalyzerLogic()

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.BrowseModelButton.connect("clicked(bool)", self.onModelPathButtonClicked)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[CAMalyzerParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        enable = (
            bool(self._parameterNode)
            and bool(self._parameterNode.inputVolume)
            and bool(self._parameterNode.modelForPrediction)
            and bool(self._parameterNode.outputLabelMap)
            and bool(self._parameterNode.modelOutput)
        )
        self.ui.applyButton.enabled = enable

    # UI callbacks ---------------------------------------------------------

    def onModelPathButtonClicked(self) -> None:
        filePath = QFileDialog.getOpenFileName(
            self.parent, "Select Model File", "", "Model Files (*.pth)"
        )
        if filePath:
            self.ui.modelForPrediction.setText(filePath)
            if self._parameterNode:
                self._parameterNode.modelForPrediction = filePath

    def onApplyButton(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to run CAMalyzer", waitCursor=True):
            self.logic.process(
                inputVolume=self._parameterNode.inputVolume,
                modelForPrediction=self._parameterNode.modelForPrediction,
                outputLabelMap=self._parameterNode.outputLabelMap,
                modelOutput=self._parameterNode.modelOutput,
            )


# -----------------------------------------------------------------------------
# Logic


class CAMalyzerLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        super().__init__()
        required = ["torch", "monai", "numpy", "scikit-image", "vtk", "psutil"]
        check_and_install_dependencies(required)

    def getParameterNode(self) -> CAMalyzerParameterNode:
        return CAMalyzerParameterNode(super().getParameterNode())

    def process(
        self,
        inputVolume: slicer.vtkMRMLScalarVolumeNode,
        modelForPrediction: str,
        outputLabelMap: slicer.vtkMRMLLabelMapVolumeNode,
        modelOutput: slicer.vtkMRMLModelNode,
        params: Optional[InferenceParams] = None,
    ) -> None:
        if params is None:
            params = InferenceParams()

        if not os.path.isfile(modelForPrediction):
            raise ValueError(f"Model file not found: {modelForPrediction}")

        volume_array = slicer.util.arrayFromVolume(inputVolume)
        tensor = volume_to_tensor(volume_array).to(DEVICE)
        model = load_model(modelForPrediction)
        seg_tensor = run_inference(tensor, model, params)
        seg_array = tensor_to_numpy(seg_tensor)

        slicer.util.updateVolumeFromArray(outputLabelMap, seg_array)
        ijk_to_ras = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijk_to_ras)
        outputLabelMap.SetIJKToRASMatrix(ijk_to_ras)

        polydata = array_to_polydata(seg_array)
        smoothed = smooth_polydata(polydata)
        modelOutput.SetAndObservePolyData(smoothed)
        modelOutput.CreateDefaultDisplayNodes()
        modelOutput.GetDisplayNode().SetColor(0.5, 0.8, 1.0)


# -----------------------------------------------------------------------------
# Tests


class CAMalyzerTest(ScriptedLoadableModuleTest):
    def setUp(self) -> None:
        slicer.mrmlScene.Clear()

    def runTest(self) -> None:
        self.setUp()
        self.test_basic()

    def test_basic(self) -> None:
        self.assertTrue(True)
