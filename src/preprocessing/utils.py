"""
Utility functions for medical imaging data processing.

Provides I/O functions for NIfTI files, normalization, and other
preprocessing utilities.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import zscore

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_nifti(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return data and affine matrix.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Tuple of (data_array, affine_matrix)
    """
    try:
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        affine = nii_img.affine
        
        # Convert to float32 for consistency
        data = data.astype(np.float32)
        
        logger.debug(f"Loaded NIfTI file: {file_path}, shape: {data.shape}")
        return data, affine
        
    except Exception as e:
        logger.error(f"Error loading NIfTI file {file_path}: {e}")
        raise


def save_nifti(data: np.ndarray, 
               affine: np.ndarray, 
               file_path: str,
               header: Optional[nib.Nifti1Header] = None) -> None:
    """
    Save data as a NIfTI file.
    
    Args:
        data: Data array to save
        affine: Affine transformation matrix
        file_path: Output file path
        header: Optional NIfTI header
    """
    try:
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Create NIfTI image
        if header is None:
            nii_img = nib.Nifti1Image(data, affine)
        else:
            nii_img = nib.Nifti1Image(data, affine, header)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        nib.save(nii_img, file_path)
        logger.debug(f"Saved NIfTI file: {file_path}, shape: {data.shape}")
        
    except Exception as e:
        logger.error(f"Error saving NIfTI file {file_path}: {e}")
        raise


def normalize_volume(data: np.ndarray, 
                    modality: str,
                    method: str = "zscore") -> np.ndarray:
    """
    Normalize a volume based on modality and method.
    
    Args:
        data: Input volume data
        modality: Modality name (T1, T1ce, T2, FLAIR)
        method: Normalization method ("zscore", "minmax", "percentile")
        
    Returns:
        Normalized volume
    """
    if method == "zscore":
        # Z-score normalization
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            normalized = (data - mean) / std
        else:
            normalized = data - mean
            
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(data)
            
    elif method == "percentile":
        # Percentile-based normalization
        p1, p99 = np.percentile(data, [1, 99])
        if p99 > p1:
            normalized = np.clip((data - p1) / (p99 - p1), 0, 1)
        else:
            normalized = np.zeros_like(data)
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def resample_volume(data: np.ndarray,
                   original_spacing: Tuple[float, float, float],
                   target_spacing: Tuple[float, float, float],
                   order: int = 1) -> np.ndarray:
    """
    Resample a volume to target spacing.
    
    Args:
        data: Input volume data
        original_spacing: Original voxel spacing (x, y, z)
        target_spacing: Target voxel spacing (x, y, z)
        order: Interpolation order (0=nearest, 1=linear, 2=quadratic)
        
    Returns:
        Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    
    # Resample using scipy
    resampled = ndimage.zoom(data, zoom_factors, order=order)
    
    return resampled.astype(np.float32)


def crop_volume(data: np.ndarray, 
                crop_size: Tuple[int, int, int],
                center: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Crop a volume to specified size.
    
    Args:
        data: Input volume data
        crop_size: Target crop size (x, y, z)
        center: Center point for cropping (if None, uses volume center)
        
    Returns:
        Cropped volume
    """
    if center is None:
        center = [s // 2 for s in data.shape]
    
    # Calculate crop boundaries
    start = [max(0, c - s // 2) for c, s in zip(center, crop_size)]
    end = [min(data.shape[i], start[i] + crop_size[i]) for i in range(3)]
    
    # Adjust start if crop would go out of bounds
    for i in range(3):
        if end[i] - start[i] < crop_size[i]:
            start[i] = max(0, end[i] - crop_size[i])
    
    # Crop the volume
    cropped = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Pad if necessary
    if cropped.shape != crop_size:
        padded = np.zeros(crop_size, dtype=data.dtype)
        pad_start = [(crop_size[i] - cropped.shape[i]) // 2 for i in range(3)]
        padded[pad_start[0]:pad_start[0]+cropped.shape[0],
               pad_start[1]:pad_start[1]+cropped.shape[1],
               pad_start[2]:pad_start[2]+cropped.shape[2]] = cropped
        return padded
    
    return cropped


def get_volume_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Get statistical information about a volume.
    
    Args:
        data: Input volume data
        
    Returns:
        Dictionary with volume statistics
    """
    stats = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "non_zero_voxels": int(np.count_nonzero(data)),
        "total_voxels": int(data.size)
    }
    
    return stats


def create_brain_mask(data: np.ndarray, 
                     threshold: float = 0.1) -> np.ndarray:
    """
    Create a simple brain mask by thresholding.
    
    Args:
        data: Input volume data
        threshold: Threshold value for masking
        
    Returns:
        Binary brain mask
    """
    # Simple thresholding
    mask = data > threshold
    
    # Remove small connected components
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
    
    return mask.astype(np.uint8)


def apply_brain_mask(data: np.ndarray, 
                    mask: np.ndarray) -> np.ndarray:
    """
    Apply a brain mask to volume data.
    
    Args:
        data: Input volume data
        mask: Binary brain mask
        
    Returns:
        Masked volume data
    """
    return data * mask


def compute_dice_coefficient(mask1: np.ndarray, 
                           mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dice coefficient
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union


def compute_hausdorff_distance(mask1: np.ndarray, 
                              mask2: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Hausdorff distance
    """
    from scipy.spatial.distance import directed_hausdorff
    
    # Get coordinates of non-zero voxels
    coords1 = np.argwhere(mask1 > 0)
    coords2 = np.argwhere(mask2 > 0)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(coords1, coords2)[0]
    d2 = directed_hausdorff(coords2, coords1)[0]
    
    return max(d1, d2)


def validate_nifti_file(file_path: str) -> bool:
    """
    Validate a NIfTI file.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        
        # Check for basic validity
        if data.size == 0:
            return False
            
        if not np.isfinite(data).all():
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating NIfTI file {file_path}: {e}")
        return False

