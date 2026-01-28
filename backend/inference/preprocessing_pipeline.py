"""
Single-case preprocessing pipeline for inference.

Adapted from src/preprocessing/data_preprocessing.py for inference mode.
Loads and preprocesses MRI volumes from case directory.
"""

import os
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation,
    ScaleIntensityRange, CropForeground, Resize, ToTensor
)

from django.conf import settings


class InferencePreprocessor:
    """
    Preprocessing pipeline for single-case inference.
    
    Loads MRI volumes, applies normalization, resampling, and orientation
    to prepare inputs for the segmentation model.
    """
    
    def __init__(self, 
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize inference preprocessor.
        
        Args:
            target_spacing: Target voxel spacing in mm
            target_size: Target volume size
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        
        # Define transforms for each modality
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=target_spacing, mode="bilinear"),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
            CropForeground(),
            Resize(spatial_size=target_size, mode="trilinear"),
            ToTensor()
        ])
    
    def load_and_preprocess_case(self, case_dir: Path, modalities: list = None) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess all modalities for a case.
        
        Args:
            case_dir: Path to case directory containing MRI files
            modalities: List of modality names to load (default: ['T1', 'T1ce', 'T2', 'FLAIR'])
            
        Returns:
            Dictionary mapping modality names to preprocessed tensors
        """
        if modalities is None:
            modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
        
        preprocessed_data = {}
        
        for modality in modalities:
            # Find the file for this modality
            file_path = self._find_modality_file(case_dir, modality)
            
            if file_path is None:
                raise FileNotFoundError(f"Could not find {modality} file in {case_dir}")
            
            # Load and preprocess
            volume = self.transforms(str(file_path))
            
            # MONAI outputs shape: (C, H, W, D) for 3D volumes
            # Model expects shape: (B, C, D, H, W)
            # We need to permute from (C, H, W, D) -> (C, D, H, W), then add batch dim
            
            if volume.ndim == 4 and volume.shape[0] == 1:
                # Shape is (1, H, W, D), permute to (1, D, H, W) then add batch
                volume = volume.permute(0, 3, 1, 2)  # (1, H, W, D) -> (1, D, H, W)
                preprocessed_data[modality] = volume.unsqueeze(0)  # (1, 1, D, H, W)
            elif volume.ndim == 3:
                # Shape is (H, W, D), permute to (D, H, W), then add channel and batch
                volume = volume.permute(2, 0, 1)  # (H, W, D) -> (D, H, W)
                preprocessed_data[modality] = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            elif volume.ndim == 4:
                # Shape likely (C, H, W, D), permute to (C, D, H, W), then add batch
                volume = volume.permute(0, 3, 1, 2)  # (C, H, W, D) -> (C, D, H, W)
                preprocessed_data[modality] = volume.unsqueeze(0)  # (1, C, D, H, W)
            else:
                raise ValueError(f"Unexpected volume shape for {modality}: {volume.shape}")
            
            # Debug: print final shape
            print(f"Preprocessed {modality} shape: {preprocessed_data[modality].shape}")
        
        return preprocessed_data
    
    def _find_modality_file(self, case_dir: Path, modality: str) -> Optional[Path]:
        """
        Find the file path for a specific modality in the case directory.
        
        Args:
            case_dir: Path to case directory
            modality: Modality name (T1, T1ce, T2, FLAIR)
            
        Returns:
            Path to the modality file or None if not found
        """
        # Normalize modality name to match MRI file naming
        modality_map = {
            'T1': ['t1', 't1n'],
            'T1ce': ['t1ce', 't1c'],
            'T2': ['t2', 't2w'],
            'FLAIR': ['flair', 't2f']
        }
        
        possible_names = modality_map.get(modality, [modality.lower()])
        
        # Search for .nii or .nii.gz files
        for name in possible_names:
            # Try exact match
            for ext in ['.nii.gz', '.nii']:
                file_path = case_dir / f"{name}{ext}"
                if file_path.exists():
                    return file_path
                
                # Try with case-insensitive search
                for file in case_dir.glob(f"*{name}*{ext}"):
                    return file
        
        return None
    
    def load_nifti_as_numpy(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a NIfTI file as numpy array.
        
        Args:
            file_path: Path to NIfTI file
            
        Returns:
            Tuple of (data, affine)
        """
        nifti_img = nib.load(str(file_path))
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        return data, affine


def preprocess_for_inference(case_dir: str) -> Dict[str, torch.Tensor]:
    """
    Convenience function to preprocess a case for inference.
    
    Args:
        case_dir: Path to case directory
        
    Returns:
        Dictionary of preprocessed modality tensors
    """
    preprocessor = InferencePreprocessor()
    return preprocessor.load_and_preprocess_case(Path(case_dir))
