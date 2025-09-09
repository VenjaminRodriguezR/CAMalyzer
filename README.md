# CAMalyzer

CAMalyzer is a [3D Slicer](https://www.slicer.org/) extension that automatically segments 3 T hip MRI scans and generates a 3D model of the femoral head to assist in cam morphology analysis.

## Features
- Loads a pre‑trained PyTorch model and performs sliding‑window inference on the input volume
- Cleans the segmentation with morphological opening and generates an editable segmentation node
- Produces a smoothed 3D surface using DBSCAN filtering and Poisson reconstruction
- Installs required Python dependencies at runtime if they are missing

## Requirements
- 3D Slicer 5.0 or later with the CAMalyzer extension installed
- A pre‑trained 3D U‑Net model stored as a `.pth` file. The network architecture used during training is:
  - Spatial dimensions: 3
  - In/out channels: 1
  - Kernel size: 3
  - Channels: (16, 32, 64, 128, 256)
  - Strides: (2, 2, 2, 2)
  - Residual units: 2
  - Dropout: 0.125
- Optional: GPU with CUDA for faster inference

## Inputs
- **Input Volume**: Hip MRI that can be read by Slicer (e.g., NIfTI `.nii.gz`).
- **Model for Prediction**: Path to the trained `.pth` file matching the architecture above.

## Outputs
Running CAMalyzer creates the following nodes in the Slicer scene:
- **Output Label Map**: Binary segmentation of the region of interest.
- **Model Output**: Poisson‑reconstructed and smoothed 3D mesh derived from the segmentation.
- Additional intermediate nodes: raw segmentation, opened segmentation, and a label map exported from the 3D model for further editing.

## Usage
1. Launch 3D Slicer and load your MRI volume.
2. Open the **CAMalyzer** module.
3. Select the input volume and choose the `.pth` model file on disk.
4. Optionally specify output nodes for the label map and 3D model.
5. Click **Apply**. The module performs preprocessing, inference, cleanup, and mesh generation.
6. Inspect the generated label map in slice views and the 3D model in the 3D view. Edit or export the results as needed.

## Testing
The repository includes a basic test that verifies segmentation using sample data.
Run the tests (inside a Slicer Python environment) with:

```bash
pytest
```

## License
Released under the terms of the MIT license.
