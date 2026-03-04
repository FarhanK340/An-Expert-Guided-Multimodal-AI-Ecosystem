"""
2D Slice Visualizer for MRI Volumes and Segmentation Masks.

Extracts 2D slices from 3D NIfTI volumes and generates visualization
images showing the input MRI alongside WT, TC, and ET segmentation
masks — either as separate images or a composite panel.

Usage:
    from src.inference.slice_visualizer import SliceVisualizer

    viz = SliceVisualizer()

    # From NIfTI files (after inference)
    results = viz.generate_from_nifti(
        mri_path="path/to/t1ce.nii.gz",
        mask_path="path/to/full_segmentation.nii.gz",
        output_dir="output/slices",
    )

    # From numpy arrays (inline with inference pipeline)
    results = viz.generate_from_arrays(
        mri_volume=mri_np,          # (D, H, W) float32
        brats_mask=brats_mask_np,   # (D, H, W) uint8, BraTS labels
        output_dir="output/slices",
    )
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List


class SliceVisualizer:
    """
    Extract 2D slices from 3D MRI volumes and segmentation masks,
    and generate visualisation images (PNG).
    """

    # BraTS label convention used by the backend inference_utils
    BRATS_LABELS = {
        'background': 0,
        'necrotic':   1,   # NCR / Necrotic — part of TC
        'edema':      2,   # ED  / Edema    — part of WT only
        'enhancing':  4,   # ET  / Enhancing — part of TC and ET
    }

    # Overlay colours (R, G, B) in 0-1 range
    COLORS = {
        'WT': (0.2, 0.8, 0.2),    # green
        'TC': (1.0, 0.85, 0.0),   # yellow
        'ET': (1.0, 0.2, 0.2),    # red
    }

    PLANES = ('axial', 'coronal', 'sagittal')

    # ------------------------------------------------------------------
    # Core slice helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_slice(
        volume: np.ndarray,
        slice_idx: int,
        plane: str = 'axial',
    ) -> np.ndarray:
        """
        Extract a 2D slice from a 3D volume.

        Args:
            volume:    3D array (D, H, W)
            slice_idx: Index along the slicing axis
            plane:     'axial' (D axis), 'coronal' (H axis),
                       or 'sagittal' (W axis)

        Returns:
            2D numpy array
        """
        if plane == 'axial':
            return volume[slice_idx, :, :]
        elif plane == 'coronal':
            return volume[:, slice_idx, :]
        elif plane == 'sagittal':
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Unknown plane: {plane}. Use axial/coronal/sagittal.")

    @staticmethod
    def find_best_slice(
        mask: np.ndarray,
        plane: str = 'axial',
    ) -> int:
        """
        Find the slice index with the largest tumor area.

        Args:
            mask:  3D binary or labelled mask (D, H, W)
            plane: slicing plane

        Returns:
            Slice index with most non-zero voxels.
            Falls back to the volume centre if mask is empty.
        """
        binary = (mask > 0).astype(np.float32)

        if plane == 'axial':
            area_per_slice = binary.sum(axis=(1, 2))    # (D,)
        elif plane == 'coronal':
            area_per_slice = binary.sum(axis=(0, 2))    # (H,)
        elif plane == 'sagittal':
            area_per_slice = binary.sum(axis=(0, 1))    # (W,)
        else:
            raise ValueError(f"Unknown plane: {plane}")

        if area_per_slice.max() == 0:
            # No tumor — return centre slice
            return int(area_per_slice.shape[0] // 2)

        return int(np.argmax(area_per_slice))

    # ------------------------------------------------------------------
    # BraTS mask → binary sub-region masks
    # ------------------------------------------------------------------

    @staticmethod
    def decompose_brats_mask(
        brats_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Decompose a BraTS-labelled mask (0/1/2/4) into binary masks
        for Whole Tumor, Tumor Core, and Enhancing Tumor.

        Returns:
            Dict with keys 'WT', 'TC', 'ET', each a bool array.
        """
        wt = (brats_mask > 0)                                         # all tumour
        tc = ((brats_mask == 1) | (brats_mask == 4))                  # necrotic + enhancing
        et = (brats_mask == 4)                                        # enhancing only
        return {'WT': wt, 'TC': tc, 'ET': et}

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_mri_slice(mri_slice: np.ndarray) -> np.ndarray:
        """Normalize a 2D MRI slice to [0, 1] for display."""
        s = mri_slice.astype(np.float64)
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-8:
            return np.zeros_like(s)
        return (s - mn) / (mx - mn)

    @classmethod
    def render_mask_overlay(
        cls,
        mri_slice: np.ndarray,
        mask_slice: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay a binary mask on a grayscale MRI slice.

        Args:
            mri_slice:  2D array, already normalised to [0, 1]
            mask_slice: 2D bool/uint8 array (same spatial shape)
            color:      (R, G, B) in [0, 1]
            alpha:      overlay opacity

        Returns:
            (H, W, 3) float32 RGB image in [0, 1]
        """
        # Grayscale → RGB
        rgb = np.stack([mri_slice] * 3, axis=-1).astype(np.float32)  # (H, W, 3)

        mask_bool = mask_slice.astype(bool)
        if mask_bool.any():
            overlay = np.array(color, dtype=np.float32)
            rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * overlay

        return np.clip(rgb, 0.0, 1.0)

    @classmethod
    def render_mask_standalone(
        cls,
        mask_slice: np.ndarray,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        bg_intensity: float = 0.95,
    ) -> np.ndarray:
        """
        Render a binary mask as a standalone coloured image (no MRI background).
        Similar to the reference screenshots the user shared.

        Returns:
            (H, W, 3) float32 RGB image in [0, 1]
        """
        h, w = mask_slice.shape[:2]
        rgb = np.full((h, w, 3), bg_intensity, dtype=np.float32)

        mask_bool = mask_slice.astype(bool)
        if mask_bool.any():
            rgb[mask_bool] = np.array(color, dtype=np.float32) * 0.65

        return rgb

    # ------------------------------------------------------------------
    # High-level generation from numpy arrays
    # ------------------------------------------------------------------

    def generate_from_arrays(
        self,
        mri_volume: np.ndarray,
        brats_mask: np.ndarray,
        output_dir: Union[str, Path] = 'output/slice_visualizations',
        plane: str = 'axial',
        slice_idx: Optional[int] = None,
        prefix: str = '',
        save_individual: bool = True,
        save_composite: bool = True,
        overlay_mode: bool = True,
        dpi: int = 150,
    ) -> Dict[str, str]:
        """
        Generate 2D slice visualizations from 3D numpy arrays.

        Args:
            mri_volume:     3D MRI volume (D, H, W), any float range
            brats_mask:     3D BraTS mask (D, H, W), labels 0/1/2/4
            output_dir:     directory for output PNGs
            plane:          'axial', 'coronal', or 'sagittal'
            slice_idx:      explicit slice index (None = auto-pick best)
            prefix:         filename prefix (e.g. case ID)
            save_individual: save separate WT/TC/ET images
            save_composite:  save 4-panel composite image
            overlay_mode:   True = overlay on MRI; False = standalone masks
            dpi:            output image DPI

        Returns:
            Dict mapping image names to saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-select slice
        if slice_idx is None:
            slice_idx = self.find_best_slice(brats_mask, plane)

        # Extract 2D slices
        mri_2d = self.normalize_mri_slice(
            self.extract_slice(mri_volume, slice_idx, plane)
        )
        sub_masks = self.decompose_brats_mask(brats_mask)
        mask_slices = {
            region: self.extract_slice(m.astype(np.uint8), slice_idx, plane)
            for region, m in sub_masks.items()
        }

        # Render images
        rendered: Dict[str, np.ndarray] = {}

        # Input MRI (grayscale → RGB for consistency)
        rendered['input'] = np.stack([mri_2d] * 3, axis=-1).astype(np.float32)

        for region in ('WT', 'TC', 'ET'):
            if overlay_mode:
                rendered[region] = self.render_mask_overlay(
                    mri_2d, mask_slices[region], self.COLORS[region]
                )
            else:
                rendered[region] = self.render_mask_standalone(
                    mask_slices[region], self.COLORS[region]
                )

        saved: Dict[str, str] = {}
        pfx = f"{prefix}_" if prefix else ''

        # Save individual images
        if save_individual:
            for name, img in rendered.items():
                fname = f"{pfx}{name}_{plane}_slice{slice_idx}.png"
                fpath = output_dir / fname
                plt.imsave(str(fpath), img, dpi=dpi)
                saved[name] = str(fpath)

        # Save composite 4-panel image
        if save_composite:
            composite = self._make_composite(
                rendered['input'], rendered['WT'],
                rendered['TC'], rendered['ET'],
                titles=['Input MRI', 'Whole Tumor (WT)',
                        'Tumor Core (TC)', 'Enhancing Tumor (ET)'],
                plane=plane,
                slice_idx=slice_idx,
            )
            fname = f"{pfx}composite_{plane}_slice{slice_idx}.png"
            fpath = output_dir / fname
            composite.savefig(str(fpath), dpi=dpi, bbox_inches='tight',
                              facecolor='black')
            plt.close(composite)
            saved['composite'] = str(fpath)

        return saved

    # ------------------------------------------------------------------
    # High-level generation from NIfTI files
    # ------------------------------------------------------------------

    def generate_from_nifti(
        self,
        mri_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_dir: Union[str, Path] = 'output/slice_visualizations',
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generate visualizations from NIfTI file paths.

        Args:
            mri_path:   Path to MRI NIfTI (.nii / .nii.gz)
            mask_path:  Path to BraTS segmentation mask NIfTI
            output_dir: Output directory
            **kwargs:   forwarded to generate_from_arrays

        Returns:
            Dict mapping image names to saved file paths
        """
        mri_nii = nib.load(str(mri_path))
        mri_vol = mri_nii.get_fdata().astype(np.float32)

        mask_nii = nib.load(str(mask_path))
        mask_vol = mask_nii.get_fdata().astype(np.uint8)

        # NIfTI is typically (H, W, D) — permute to (D, H, W) for consistency
        # with the inference engine convention
        mri_vol = np.transpose(mri_vol, (2, 0, 1))
        mask_vol = np.transpose(mask_vol, (2, 0, 1))

        return self.generate_from_arrays(
            mri_volume=mri_vol,
            brats_mask=mask_vol,
            output_dir=output_dir,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Multi-plane convenience
    # ------------------------------------------------------------------

    def generate_all_planes(
        self,
        mri_volume: np.ndarray,
        brats_mask: np.ndarray,
        output_dir: Union[str, Path] = 'output/slice_visualizations',
        prefix: str = '',
        **kwargs,
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate visualizations for all three planes (axial, coronal, sagittal).

        Returns:
            Dict[plane] → Dict[image_name → path]
        """
        results = {}
        for plane in self.PLANES:
            results[plane] = self.generate_from_arrays(
                mri_volume=mri_volume,
                brats_mask=brats_mask,
                output_dir=output_dir,
                plane=plane,
                prefix=prefix,
                **kwargs,
            )
        return results

    # ------------------------------------------------------------------
    # Composite panel
    # ------------------------------------------------------------------

    @staticmethod
    def _make_composite(
        input_img: np.ndarray,
        wt_img: np.ndarray,
        tc_img: np.ndarray,
        et_img: np.ndarray,
        titles: Optional[List[str]] = None,
        plane: str = 'axial',
        slice_idx: int = 0,
    ) -> plt.Figure:
        """
        Create a 4-panel figure: Input | WT | TC | ET.

        Returns:
            matplotlib Figure object (caller should save/close).
        """
        if titles is None:
            titles = ['Input MRI', 'Whole Tumor', 'Tumor Core', 'Enhancing Tumor']

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
        fig.suptitle(
            f'{plane.capitalize()} Plane — Slice {slice_idx}',
            color='white', fontsize=14, fontweight='bold', y=0.98,
        )

        images = [input_img, wt_img, tc_img, et_img]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, origin='lower')
            ax.set_title(title, color='white', fontsize=11, pad=8)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.05, top=0.88)
        return fig


# ======================================================================
# CLI entrypoint
# ======================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate 2D slice visualizations from 3D MRI + mask'
    )
    parser.add_argument('--mri', type=str, required=True,
                        help='Path to MRI NIfTI file')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to BraTS segmentation mask NIfTI')
    parser.add_argument('--output', type=str,
                        default='output/slice_visualizations',
                        help='Output directory')
    parser.add_argument('--plane', type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal', 'all'])
    parser.add_argument('--slice', type=int, default=None,
                        help='Explicit slice index (default: auto-pick)')
    parser.add_argument('--overlay', action='store_true', default=True,
                        help='Overlay masks on MRI (default)')
    parser.add_argument('--standalone', action='store_true',
                        help='Render masks without MRI background')
    parser.add_argument('--prefix', type=str, default='',
                        help='Filename prefix')

    args = parser.parse_args()

    viz = SliceVisualizer()
    overlay_mode = not args.standalone

    if args.plane == 'all':
        mri_nii = nib.load(args.mri)
        mri_vol = np.transpose(mri_nii.get_fdata().astype(np.float32), (2, 0, 1))
        mask_nii = nib.load(args.mask)
        mask_vol = np.transpose(mask_nii.get_fdata().astype(np.uint8), (2, 0, 1))

        results = viz.generate_all_planes(
            mri_vol, mask_vol, args.output,
            prefix=args.prefix, overlay_mode=overlay_mode,
        )
        for plane, files in results.items():
            print(f"\n  {plane}:")
            for name, path in files.items():
                print(f"    {name}: {path}")
    else:
        results = viz.generate_from_nifti(
            args.mri, args.mask, args.output,
            plane=args.plane, slice_idx=args.slice,
            prefix=args.prefix, overlay_mode=overlay_mode,
        )
        print("\nSaved visualizations:")
        for name, path in results.items():
            print(f"  {name}: {path}")
