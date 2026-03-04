# Inference Scripts

This directory contains standalone scripts for running inference and generating visualizations using the MoME+ model.

## Available Scripts

### `run_inference_and_visualize.py`
End-to-end script that loads the trained MoME+ model (4 modality experts + fusion network), runs inference on a specific BraTS case, and generates 2D slice visualization images.

**Usage:**
```bash
python scripts/run_inference_and_visualize.py --case_dir <path_to_brats_case_folder> [--with_gt] [--display_modality T1ce]
```

**Arguments:**
- `--case_dir`: Required. Path to the folder containing the 4 NIfTI MRI modalities (T1, T1c, T2w, T2f).
- `--with_gt`: Optional flag. If provided, the script will also look for a `<case>-seg.nii.gz` file (ground truth) and generate a comparison visualization.
- `--display_modality`: Optional. Which MRI modality to use as the background for the segmentation overlay (default is T1ce).
- `--output`: Optional. Directory to save the generated PNGs (default: `output/slice_visualizations`).

**Output:**
The script saves several `.png` files to the output directory, showing:
- **Prediction Overlay:** The model's predicted Whole Tumor (WT - green), Tumor Core (TC - yellow), and Enhancing Tumor (ET - red) overlaid on the MRI slice with the maximum predicted tumor area.
- **Prediction Standalone:** Just the masks on a black background.
- **Ground Truth Overlay:** (If `--with_gt` used) The ground truth masks overlaid on the MRI slice with the maximum ground truth tumor area.

### `inspect_checkpoint.py`
A small utility script to inspect the contents and keys of saved PyTorch `.pth` checkpoint files, useful for debugging model loading issues.

## Recent Fixes
The underlying `InferenceEngine` (`src/inference/inference_engine.py`) was recently fixed to correctly align with the backend's proven preprocessing pipeline:
1. **Model Outputs:** Interprets logits using `Sigmoid` (independent multi-label) instead of `Softmax` (mutually exclusive classes).
2. **Axis Routing:** Permutes NIfTI native `(H,W,D)` spacing into the `(D,H,W)` PyTorch tensor shape expected by the model.
3. **Z-Score Normalization:** Correctly computes the mean and standard deviation from the brain tissue ONLY, and normalizes only the brain tissue, ensuring the background zeroes remain exactly 0.
