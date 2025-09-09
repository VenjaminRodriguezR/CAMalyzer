import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


class CAMalyzer(ScriptedLoadableModule):
    """Thin Slicer front-end for ``camalyzer_core``."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        parent.title = "CAMalyzer"
        parent.categories = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["Auto Generated"]
        parent.helpText = (
            "Front-end that delegates heavy lifting to the ``camalyzer_core``"
            " package."
        )
        parent.acknowledgementText = """Generated for modularization demo."""


class CAMalyzerWidget(ScriptedLoadableModuleWidget):
    """Minimal user interface built from ``UI/CAMalyzer.ui``."""

    def setup(self) -> None:
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/CAMalyzer.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self.logic = CAMalyzerLogic()
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.connect("clicked(bool)", self.onApply)

    def onApply(self) -> None:
        volume = getattr(self.ui, "inputVolume", None)
        model_path = getattr(self.ui, "modelPath", None)
        self.logic.run(
            volume.currentNode() if volume else None,
            model_path.currentPath if model_path else "",
        )


class CAMalyzerLogic(ScriptedLoadableModuleLogic):
    """Lightweight wrapper that calls ``camalyzer_core``."""

    def run(self, volume_node, model_path: str = "") -> bool:
        from camalyzer_core.inference import segment_volume

        if volume_node is None:
            raise ValueError("No input volume provided")
        segment_volume(volume_node, model_path)
        return True
