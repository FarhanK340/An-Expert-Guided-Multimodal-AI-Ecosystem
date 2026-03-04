"""
Registration utilities for aligning segmentation masks to atlas space.

Provides both affine (fast) and non-linear (ANTs - accurate) registration methods.
"""

import os
import subprocess
import nibabel as nib
from nilearn.image import resample_to_img
from pathlib import Path


def register_to_atlas_affine(segmentation_path_or_img, atlas_path_or_img):
    """
    Resample segmentation mask to atlas space using affine transformation.
    
    This is faster but less accurate than non-linear registration.
    Good for quick prototyping or when ANTs is not available.
    
    Args:
        segmentation_path_or_img: Path to segmentation mask (.nii.gz) OR nibabel image
        atlas_path_or_img: Path to brain atlas (.nii.gz) OR nibabel image object
    
    Returns:
        Nibabel image object in atlas space
    """
    # Handle both path strings and nibabel image objects for segmentation
    if isinstance(segmentation_path_or_img, (str, Path)):
        seg_img = nib.load(segmentation_path_or_img)
    else:
        # Assume it's already a nibabel image object
        seg_img = segmentation_path_or_img
    
    # Handle both path strings and nibabel image objects for atlas
    if isinstance(atlas_path_or_img, (str, Path)):
        atlas_img = nib.load(atlas_path_or_img)
    else:
        # Assume it's already a nibabel image object
        atlas_img = atlas_path_or_img
    
    # Resample to atlas space using nearest neighbor interpolation
    # to preserve discrete label values
    seg_resampled = resample_to_img(
        seg_img, 
        atlas_img, 
        interpolation='nearest'
    )
    
    return seg_resampled


def register_to_atlas_ants(
    segmentation_path, 
    t1_reference_path,
    atlas_template_path,
    output_dir
):
    """
    Use ANTs for precise non-linear registration.
    
    This requires ANTs to be installed on the system.
    Installation:
    - conda install -c conda-forge ants
    - or compile from source: https://github.com/ANTsX/ANTs
    
    Args:
        segmentation_path: Path to segmentation mask
        t1_reference_path: Path to patient's T1 scan
        atlas_template_path: Path to MNI152 template
        output_dir: Directory to save transformation files
    
    Returns:
        Nibabel image of segmentation in atlas space
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Register T1 scan to MNI152 template
    transform_prefix = os.path.join(output_dir, 'transform_')
    
    cmd = [
        'antsRegistrationSyNQuick.sh',
        '-d', '3',
        '-f', atlas_template_path,  # Fixed: MNI152 template
        '-m', t1_reference_path,     # Moving: Patient T1
        '-o', transform_prefix,
        '-t', 's'  # Use SyN (symmetric normalization)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ANTs not found. Please install ANTs:\n"
            "  conda install -c conda-forge ants\n"
            "or use affine registration instead."
        )
    
    # Step 2: Apply transformation to segmentation mask
    output_seg = os.path.join(output_dir, 'seg_in_atlas_space.nii.gz')
    
    cmd_transform = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', segmentation_path,
        '-r', atlas_template_path,
        '-o', output_seg,
        '-n', 'NearestNeighbor',  # Preserve discrete labels
        '-t', f'{transform_prefix}1Warp.nii.gz',
        '-t', f'{transform_prefix}0GenericAffine.mat'
    ]
    
    subprocess.run(cmd_transform, check=True, capture_output=True)
    
    return nib.load(output_seg)


def check_ants_available():
    """
    Check if ANTs is available on the system.
    
    Returns:
        bool: True if ANTs is available, False otherwise
    """
    try:
        subprocess.run(
            ['antsRegistrationSyNQuick.sh', '--help'],
            capture_output=True,
            check=False
        )
        return True
    except FileNotFoundError:
        return False
