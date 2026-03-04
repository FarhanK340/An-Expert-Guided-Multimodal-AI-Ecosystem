"""
Test script for SliceVisualizer.

Creates synthetic 3D MRI volume + BraTS mask data and generates
2D slice visualizations to verify the visualizer works correctly.

Usage:
    python scripts/test_slice_visualizer.py

    # Or with real NIfTI data:
    python scripts/test_slice_visualizer.py --mri path/to/t1c.nii.gz --mask path/to/mask.nii.gz
"""

import sys
import numpy as np
from pathlib import Path

# Ensure repo root is on sys.path so we can import src.*
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.slice_visualizer import SliceVisualizer


def create_synthetic_data(
    shape: tuple = (128, 128, 128),
) -> tuple:
    """
    Create a synthetic MRI volume and BraTS-style segmentation mask
    with a realistic-ish nested tumor structure.

    Returns:
        (mri_volume, brats_mask)
    """
    D, H, W = shape
    rng = np.random.default_rng(42)

    # --- Synthetic MRI ---
    # Start with smooth background
    mri = rng.normal(0.3, 0.1, shape).astype(np.float32)

    # Add a brain-like ellipsoid
    z, y, x = np.mgrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2
    brain_mask = (
        ((z - cz) / (D * 0.35)) ** 2 +
        ((y - cy) / (H * 0.40)) ** 2 +
        ((x - cx) / (W * 0.38)) ** 2
    ) < 1.0
    mri[brain_mask] += 0.5
    mri += rng.normal(0, 0.05, shape).astype(np.float32)

    # --- Synthetic BraTS mask ---
    mask = np.zeros(shape, dtype=np.uint8)

    # Whole tumor (edema) — large blob
    wt_mask = (
        ((z - cz + 5) / 18) ** 2 +
        ((y - cy - 3) / 22) ** 2 +
        ((x - cx + 4) / 20) ** 2
    ) < 1.0
    mask[wt_mask] = 2  # edema

    # Tumor core (necrotic) — medium blob inside WT
    tc_mask = (
        ((z - cz + 5) / 10) ** 2 +
        ((y - cy - 3) / 12) ** 2 +
        ((x - cx + 4) / 11) ** 2
    ) < 1.0
    mask[tc_mask] = 1  # necrotic

    # Enhancing tumor — small blob inside TC
    et_mask = (
        ((z - cz + 5) / 5) ** 2 +
        ((y - cy - 3) / 6) ** 2 +
        ((x - cx + 4) / 5) ** 2
    ) < 1.0
    mask[et_mask] = 4  # enhancing

    # Add intensity variation in tumor region
    mri[wt_mask] += 0.3
    mri[tc_mask] -= 0.1
    mri[et_mask] += 0.5

    return mri, mask


def run_test(mri_path=None, mask_path=None):
    """Run the visualization test."""

    output_dir = Path('output/slice_visualizations')
    viz = SliceVisualizer()

    if mri_path and mask_path:
        # ----------------------------------------------------------
        # Test with real NIfTI files
        # ----------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"Testing with real NIfTI data")
        print(f"  MRI:  {mri_path}")
        print(f"  Mask: {mask_path}")
        print(f"{'='*60}\n")

        # Overlay mode
        results = viz.generate_from_nifti(
            mri_path, mask_path, output_dir,
            prefix='real_overlay', overlay_mode=True,
        )
        print("Overlay mode results:")
        for name, path in results.items():
            print(f"  {name}: {path}")

        # Standalone mode
        results = viz.generate_from_nifti(
            mri_path, mask_path, output_dir,
            prefix='real_standalone', overlay_mode=False,
        )
        print("\nStandalone mode results:")
        for name, path in results.items():
            print(f"  {name}: {path}")

    else:
        # ----------------------------------------------------------
        # Test with synthetic data
        # ----------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"Testing with synthetic data (128x128x128)")
        print(f"{'='*60}\n")

        mri, mask = create_synthetic_data()

        print(f"MRI volume shape: {mri.shape}")
        print(f"Mask shape:       {mask.shape}")
        print(f"Mask labels present: {np.unique(mask)}")

        sub = viz.decompose_brats_mask(mask)
        print(f"WT voxels: {sub['WT'].sum():,}")
        print(f"TC voxels: {sub['TC'].sum():,}")
        print(f"ET voxels: {sub['ET'].sum():,}")

        # --- Test 1: Overlay mode (axial, best slice) ---
        print(f"\n--- Test 1: Overlay mode (axial, auto-slice) ---")
        best = viz.find_best_slice(mask, 'axial')
        print(f"Best axial slice: {best}")

        results = viz.generate_from_arrays(
            mri, mask, output_dir,
            prefix='synth_overlay', overlay_mode=True,
        )
        print("Saved:")
        for name, path in results.items():
            print(f"  {name}: {path}")

        # --- Test 2: Standalone mode ---
        print(f"\n--- Test 2: Standalone mode (axial, auto-slice) ---")
        results = viz.generate_from_arrays(
            mri, mask, output_dir,
            prefix='synth_standalone', overlay_mode=False,
        )
        print("Saved:")
        for name, path in results.items():
            print(f"  {name}: {path}")

        # --- Test 3: All three planes ---
        print(f"\n--- Test 3: All planes (overlay mode) ---")
        results = viz.generate_all_planes(
            mri, mask, output_dir,
            prefix='synth_allplanes', overlay_mode=True,
        )
        for plane, files in results.items():
            print(f"  {plane}:")
            for name, path in files.items():
                print(f"    {name}: {path}")

        # --- Test 4: Explicit slice index ---
        print(f"\n--- Test 4: Explicit slice index (axial slice 64) ---")
        results = viz.generate_from_arrays(
            mri, mask, output_dir,
            prefix='synth_explicit', overlay_mode=True,
            slice_idx=64,
        )
        print("Saved:")
        for name, path in results.items():
            print(f"  {name}: {path}")

    print(f"\n✅ All tests passed! Output dir: {output_dir.resolve()}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test SliceVisualizer')
    parser.add_argument('--mri', type=str, default=None,
                        help='Path to MRI NIfTI (optional, uses synthetic if not given)')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to BraTS mask NIfTI (optional)')
    args = parser.parse_args()

    run_test(args.mri, args.mask)
