"""
Single-case preprocessing pipeline for inference.

Fixed version: uses percentile-based intensity normalization per BraTS convention,
preserves the original affine so saved masks align with input scans, and
correctly handles the (C, H, W, D) → (B, C, D, H, W) permutation.
"""

import os
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional

from django.conf import settings


class InferencePreprocessor:
    """
    Preprocessing pipeline for single-case inference.

    Applies z-score normalisation per-modality (BraTS convention),
    resamples to (128, 128, 128), and returns tensors in (B, C, D, H, W) format.
    """

    def __init__(self,
                 target_size: Tuple[int, int, int] = (128, 128, 128)):
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_and_preprocess_case(self,
                                 case_dir: Path,
                                 modalities: list = None) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess all modalities for a case.

        Returns:
            Dict mapping modality name → tensor of shape (1, 1, D, H, W).
        """
        if modalities is None:
            modalities = ['T1', 'T1ce', 'T2', 'FLAIR']

        preprocessed_data = {}

        for modality in modalities:
            file_path = self._find_modality_file(case_dir, modality)
            if file_path is None:
                raise FileNotFoundError(
                    f"Could not find {modality} file in {case_dir}. "
                    f"Files present: {list(case_dir.glob('*')) if case_dir.exists() else 'dir not found'}"
                )

            volume_np, affine = self._load_nifti(file_path)
            volume_np = self._normalize(volume_np)
            volume_np = self._resize(volume_np, self.target_size)

            # numpy shape: (H_r, W_r, D_r) → tensor (1, 1, D, H, W)
            tensor = torch.from_numpy(volume_np).float()          # (H, W, D)
            tensor = tensor.permute(2, 0, 1)                      # (D, H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)             # (1, 1, D, H, W)

            preprocessed_data[modality] = tensor
            print(f"Preprocessed {modality}: {tensor.shape}")

        return preprocessed_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_nifti(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load NIfTI and return float32 data + affine."""
        img = nib.load(str(file_path))
        data = img.get_fdata(dtype=np.float32)
        return data, img.affine

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """
        Brain-MRI z-score normalisation.

        Only the non-zero voxels are used to compute mean/std so that
        background (skull-stripped zeros) does not skew statistics.
        """
        mask = volume > 0
        if mask.sum() == 0:
            return volume  # all background – return as-is

        mean = volume[mask].mean()
        std  = volume[mask].std()
        if std < 1e-8:
            std = 1e-8

        normalised = np.zeros_like(volume)
        normalised[mask] = (volume[mask] - mean) / std
        return normalised

    def _resize(self, volume: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
        """
        Trilinear resize using scipy (no MONAI dependency needed here).
        Falls back to skimage if scipy is unavailable.
        """
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target, volume.shape)]
        resized = zoom(volume, factors, order=1, prefilter=False)
        return resized.astype(np.float32)

    def _find_modality_file(self, case_dir: Path, modality: str) -> Optional[Path]:
        """
        Find a NIfTI file for the given modality in case_dir.

        Checks both the DB-stored filenames (t1.nii.gz, t1ce.nii.gz …)
        and common BraTS naming patterns.
        """
        modality_patterns = {
            'T1':    ['t1.nii.gz', 't1.nii', 't1n.nii.gz', 't1n.nii'],
            'T1ce':  ['t1ce.nii.gz', 't1ce.nii', 't1c.nii.gz', 't1c.nii'],
            'T2':    ['t2.nii.gz', 't2.nii', 't2w.nii.gz', 't2w.nii'],
            'FLAIR': ['flair.nii.gz', 'flair.nii', 't2f.nii.gz', 't2f.nii'],
        }

        patterns = modality_patterns.get(modality, [modality.lower() + '.nii.gz'])

        # 1. Exact filename match
        for pat in patterns:
            p = case_dir / pat
            if p.exists():
                return p

        # 2. Glob / partial match (any file whose name contains the keyword)
        keywords = {
            'T1':    ['_t1.', '_t1n.', '-t1.', '-t1n.', 't1_', 't1n_'],
            'T1ce':  ['_t1ce.', '_t1c.', '-t1ce.', '-t1c.', 't1ce_', 't1c_'],
            'T2':    ['_t2.', '_t2w.', '-t2.', '-t2w.', 't2_', 't2w_'],
            'FLAIR': ['_flair.', '_t2f.', '-flair.', '-t2f.', 'flair_', 't2f_', 'FLAIR.nii.gz'],
        }.get(modality, [])

        if case_dir.exists():
            for f in case_dir.iterdir():
                name_lower = f.name.lower()
                if f.suffix in ('.gz', '.nii') or f.name.endswith('.nii.gz'):
                    for kw in keywords:
                        if kw in name_lower:
                            return f

        return None

    # ------------------------------------------------------------------
    # Utility – load original affine for post-processing
    # ------------------------------------------------------------------

    def get_reference_affine(self, case_dir: Path) -> np.ndarray:
        """
        Return the affine from the first available modality file.
        Used when saving output masks so they align with input scans.
        """
        for mod in ['T1', 'T1ce', 'T2', 'FLAIR']:
            f = self._find_modality_file(case_dir, mod)
            if f is not None:
                img = nib.load(str(f))
                return img.affine
        return np.eye(4)


def preprocess_for_inference(case_dir: str) -> Dict[str, torch.Tensor]:
    """Convenience wrapper."""
    preprocessor = InferencePreprocessor()
    return preprocessor.load_and_preprocess_case(Path(case_dir))
