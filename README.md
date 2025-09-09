# CAMalyzer

CAMalyzer is a 3D Slicer extension for automated image segmentation and
3D model generation using deep‐learning models. The module wraps a
PyTorch inference pipeline and provides a simple user interface for
loading a volume, applying a trained network and visualising the
resulting label map and surface model.

## Repository structure

```
CAMalyzer/
├── CAMalyzer/              # Python module and resources
│   ├── CAMalyzer.py        # Module entry point and logic
│   ├── Resources/
│   │   ├── Icons/          # Module icons and sample thumbnails
│   │   └── UI/CAMalyzer.ui # Qt designer file for the widget
│   └── Testing/            # Test scaffolding
└── CAMalyzer.png           # Extension icon
```

## Installation

1. Clone this repository in your Slicer extensions directory or use the
   Extension Wizard to build it with CMake.
2. Launch 3D Slicer and add the built extension to your Slicer
   installation.
3. After installation the module appears as **CAMalyzer** in the module
   list.

## Usage

1. Load or download a volume. Two sample data sets are available via the
   *Sample Data* module (look for **CAMalyzer1** and **CAMalyzer2**).
2. Open the **CAMalyzer** module.
3. Select the **Input Volume** to be segmented.
4. Press **Browse** to choose the path to a trained `.pth` model
   (required).
5. Optionally select output nodes for the **Label Map** and **Model**.
6. Click **Apply** to run the pipeline. The module runs the model,
   creates a label map and converts it to a smoothed 3D surface.

## Classes and components

### `CAMalyzer`

Module entry point that sets metadata and registers example data during
startup【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L32-L55】.

### `CAMalyzerParameterNode`

Typed storage for user inputs including the input volume, model path,
output label map and generated 3D model【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L112-L125】.

### `CAMalyzerWidget`

Loads the Qt `.ui` file, connects GUI elements and handles events such as
selecting the model path and running the process when **Apply** is
pressed【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L131-L172】【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L262-L293】.

### `CAMalyzerLogic`

Core processing layer. It checks/installs Python dependencies, loads a
UNet model, applies sliding‑window inference and builds a smoothed 3D
mesh from the segmentation【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L306-L475】.

### `CAMalyzerTest`

Smoke tests demonstrating how to run the logic on sample data and verify
results【F:CAMalyzer/CAMalyzer/CAMalyzer.py†L528-L590】.

## Resources

- UI definition: `CAMalyzer/CAMalyzer/Resources/UI/CAMalyzer.ui`
- Icons: `CAMalyzer/CAMalyzer/Resources/Icons`
- Test scaffolding: `CAMalyzer/CAMalyzer/Testing`

## License

This project inherits the licensing terms of 3D Slicer and the included
dependencies. See individual files for details.

