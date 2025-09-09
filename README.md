# CAMalyzer

CAMalyzer is a sample 3D Slicer extension that separates a thin Qt-based front-end from a pure-Python processing library. The project demonstrates how a Slicer module can delegate heavy work to a standalone package while keeping the GUI lightweight.

## Directory structure

- `CAMalyzer.py` – Slicer module wrapper that exposes the user interface and delegates work to the core package.
- `UI/` – Qt `.ui` files describing the module interface.
- `Resources/` – Icon assets used by the module.
- `camalyzer_core/` – Pure Python package with configuration, inference, I/O and post-processing utilities.
- `tests/` – Pytest test suite for the core functionality.

## Classes and components

### Slicer front-end (`CAMalyzer.py`)

- **`CAMalyzer`** – Registers the module with Slicer and defines metadata.
- **`CAMalyzerWidget`** – Loads the Qt interface, wires up the Apply button and gathers user input.
- **`CAMalyzerLogic`** – Lightweight wrapper that forwards the selected volume and model path to the core package.

### Core package (`camalyzer_core`)

- **`DEFAULT_CONFIG`** – Default hyper-parameters for the pipeline (`config.py`).
- **`InferenceParams`** – Dataclass storing ROI size and threshold with validation (`params.py`).
- **`segment_volume`** – Stub inference routine returning a zero-valued mask (`inference.py`).
- **`check_optional_deps`** – Helper reporting missing optional libraries (`deps.py`).
- **`load_model`, `load_volume`, `get_device`** – Simplified I/O utilities (`io.py`).
- **Post-processing helpers** – `extract_surface`, `select_largest`, `poisson_reconstruct` (`post/`).
- **Utility modules** – `vtkconv.py` for NumPy/Vtk conversions and `profiling.py` for a timing context manager.

## Usage

### Core library

```python
import numpy as np
from camalyzer_core import segment_volume

volume = np.zeros((64, 64, 64))
mask = segment_volume(volume)
```

### As a Slicer module

1. Copy this repository into a Slicer extensions directory or build it with the Extension Wizard.
2. Launch 3D Slicer and load the **CAMalyzer** module.
3. Select an input volume, choose a model file and press **Apply** to generate a label map.

### Running tests

Install the test dependencies and run the suite:

```bash
pip install numpy pytest
pytest
```

## License

This project inherits the licensing terms of 3D Slicer and its dependencies.

