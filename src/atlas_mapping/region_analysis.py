"""
Region analysis utilities for computing overlap between tumors and atlas regions.
"""

import numpy as np
from collections import defaultdict
from scipy.ndimage import binary_opening, binary_closing
from skimage.measure import label
from skimage.morphology import remove_small_objects


def compute_region_overlap(seg_atlas_space, atlas_labels):
    """
    Calculate voxel-wise overlap between segmentation and atlas regions.
    
    Args:
        seg_atlas_space: Nibabel image (registered segmentation)
        atlas_labels: Nibabel image (brain atlas)
    
    Returns:
        Dictionary mapping region IDs to voxel counts
    """
    seg_data = seg_atlas_space.get_fdata()
    atlas_data = atlas_labels.get_fdata().astype(int)
    
    # Binary mask: tumor present (non-zero)
    tumor_mask = seg_data > 0
    
    # Count voxels per region
    region_counts = defaultdict(int)
    unique_regions = np.unique(atlas_data[tumor_mask])
    
    for region_id in unique_regions:
        if region_id == 0:  # Skip background
            continue
        
        # Count tumor voxels in this region
        overlap_mask = (atlas_data == region_id) & tumor_mask
        region_counts[region_id] = np.sum(overlap_mask)
    
    return dict(region_counts)


def calculate_percentage_involvement(seg_atlas_space, atlas_labels, region_names):
    """
    Compute percentage of each brain region affected by tumor.
    
    Args:
        seg_atlas_space: Nibabel image in atlas space
        atlas_labels: Atlas label image
        region_names: Dictionary mapping region IDs to names
    
    Returns:
        List of dicts with region info sorted by involvement percentage
    """
    seg_data = seg_atlas_space.get_fdata()
    atlas_data = atlas_labels.get_fdata().astype(int)
    
    tumor_mask = seg_data > 0
    region_overlap = compute_region_overlap(seg_atlas_space, atlas_labels)
    
    results = []
    
    for region_id, tumor_voxels in region_overlap.items():
        # Total voxels in this region
        region_mask = atlas_data == region_id
        total_voxels = np.sum(region_mask)
        
        if total_voxels == 0:
            continue
        
        # Percentage involvement
        percentage = (tumor_voxels / total_voxels) * 100
        
        results.append({
            'region_id': int(region_id),
            'region_name': region_names.get(region_id, f'Unknown_{region_id}'),
            'tumor_voxels': int(tumor_voxels),
            'total_voxels': int(total_voxels),
            'percentage_involvement': round(percentage, 2),
            'tumor_volume_in_region_mm3': int(tumor_voxels * np.prod(seg_atlas_space.header.get_zooms()))
        })
    
    # Sort by percentage involvement (descending)
    results.sort(key=lambda x: x['percentage_involvement'], reverse=True)
    
    return results


def analyze_tumor_subregions(seg_atlas_space, atlas_labels, region_names):
    """
    Separate analysis for each tumor sub-component.
    
    For BraTS segmentations:
    - Label 1: Necrotic and non-enhancing tumor core
    - Label 2: Peritumoral edema
    - Label 4: Enhancing tumor
    
    Args:
        seg_atlas_space: Segmentation in atlas space
        atlas_labels: Atlas label image
        region_names: Region name mapping
    
    Returns:
        Dictionary with analysis for each tumor label
    """
    seg_data = seg_atlas_space.get_fdata()
    atlas_data = atlas_labels.get_fdata().astype(int)
    
    # Ensure both arrays have the same shape
    if seg_data.shape != atlas_data.shape:
        raise ValueError(
            f"Shape mismatch: segmentation {seg_data.shape} != atlas {atlas_data.shape}. "
            f"Registration may have failed. Segmentation should be resampled to atlas space."
        )
    
    # BraTS label mapping
    label_names = {
        1: 'necrotic_and_non_enhancing_tumor_core',
        2: 'peritumoral_edema',
        4: 'enhancing_tumor'
    }
    
    results = {}
    
    for label_id, label_name in label_names.items():
        label_mask = seg_data == label_id
        
        if not np.any(label_mask):
            results[label_name] = []
            continue
        
        # Calculate region overlap for this specific label
        region_counts = defaultdict(int)
        unique_regions = np.unique(atlas_data[label_mask])
        
        label_results = []
        
        for region_id in unique_regions:
            if region_id == 0:
                continue
            
            overlap_mask = (atlas_data == region_id) & label_mask
            label_voxels = np.sum(overlap_mask)
            
            region_mask = atlas_data == region_id
            total_voxels = np.sum(region_mask)
            
            if total_voxels > 0:
                percentage = (label_voxels / total_voxels) * 100
                
                label_results.append({
                    'region_id': int(region_id),
                    'region_name': region_names.get(region_id, f'Unknown_{region_id}'),
                    'voxels': int(label_voxels),
                    'percentage': round(percentage, 2),
                    'volume_mm3': int(label_voxels * np.prod(seg_atlas_space.header.get_zooms()))
                })
        
        label_results.sort(key=lambda x: x['percentage'], reverse=True)
        results[label_name] = label_results[:10]  # Top 10 regions
    
    return results


def apply_confidence_threshold(seg_data, threshold=0.5):
    """
    Only consider voxels with high confidence.
    Assumes probabilistic segmentation output.
    
    Args:
        seg_data: Segmentation data (probabilistic or discrete)
        threshold: Confidence threshold
    
    Returns:
        Binary mask of high-confidence voxels
    """
    return seg_data > threshold


def remove_small_components(mask, min_size=50):
    """
    Remove isolated voxel clusters (noise filtering).
    
    Args:
        mask: Binary mask
        min_size: Minimum cluster size in voxels
    
    Returns:
        Cleaned binary mask
    """
    labeled = label(mask)
    cleaned = remove_small_objects(labeled, min_size=min_size)
    return cleaned > 0


def morphological_cleanup(mask, operation='close'):
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask
        operation: 'close', 'open', or 'both'
    
    Returns:
        Cleaned mask
    """
    if operation in ['close', 'both']:
        mask = binary_closing(mask)
    
    if operation in ['open', 'both']:
        mask = binary_opening(mask)
    
    return mask
