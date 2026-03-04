"""
Main BrainAtlasMapper class for production use.
"""

import os
import numpy as np
import nibabel as nib
from .registration import register_to_atlas_affine, register_to_atlas_ants, check_ants_available
from .region_analysis import calculate_percentage_involvement, analyze_tumor_subregions
from .atlas_data import get_region_names


class BrainAtlasMapper:
    """
    Production class for brain atlas mapping pipeline.
    
    Handles registration, region overlap calculation, and anatomical analysis
    of brain tumor segmentations.
    
    Example:
        >>> mapper = BrainAtlasMapper(
        ...     atlas_path='atlases/MNI152_T1_1mm.nii.gz',
        ...     atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',
        ...     use_ants=True
        ... )
        >>> results = mapper.process_segmentation(
        ...     seg_path='segmentation.nii.gz',
        ...     t1_reference_path='t1.nii.gz'
        ... )
    """
    
    def __init__(
        self, 
        atlas_path=None, 
        atlas_labels_path=None, 
        atlas_name='harvard_oxford',
        use_ants=False
    ):
        """
        Initialize the atlas mapper.
        
        Args:
            atlas_path: Path to atlas template (e.g., MNI152). If None, will download.
            atlas_labels_path: Path to atlas labels. If None, will download.
            atlas_name: Atlas identifier ('harvard_oxford', 'AAL3', etc.)
            use_ants: Whether to use ANTs for non-linear registration
        """
        self.atlas_name = atlas_name
        self.use_ants = use_ants
        
        # Load or download atlas
        if atlas_path and atlas_labels_path:
            self.atlas_template_path = atlas_path
            self.atlas_labels_path = atlas_labels_path
            self.atlas_img = nib.load(atlas_path)
            self.atlas_labels = nib.load(atlas_labels_path)
        else:
            self._download_atlas()
        
        # Load region names
        self.region_names = get_region_names(atlas_name)
        
        # Check ANTs availability
        if use_ants and not check_ants_available():
            print("WARNING: ANTs not found. Falling back to affine registration.")
            self.use_ants = False
    
    def _download_atlas(self):
        """Download atlas data if not provided."""
        from .atlas_data import download_harvard_oxford_atlas, get_mni152_template
        
        print("Downloading atlas data...")
        atlas_data = download_harvard_oxford_atlas()
        
        # Use cortical atlas as primary
        self.atlas_labels_path = atlas_data['cortical_atlas']
        self.atlas_labels = nib.load(self.atlas_labels_path)
        
        # Download MNI152 template for registration
        self.atlas_template_path = get_mni152_template()
        self.atlas_img = nib.load(self.atlas_template_path)
        
        print("Atlas download complete.")
    
    def process_segmentation(
        self, 
        seg_path, 
        t1_reference_path=None,
        output_dir='./atlas_mapping_output',
        clean_segmentation=False
    ):
        """
        Complete atlas mapping workflow.
        
        Args:
            seg_path: Path to segmentation file (.nii.gz)
            t1_reference_path: Path to T1 reference scan (required for ANTs)
            output_dir: Directory to save intermediate results
            clean_segmentation: Whether to apply morphological cleanup
        
        Returns:
            Dictionary with anatomical analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Registration
        print("Step 1: Registering segmentation to atlas space...")
        if self.use_ants and t1_reference_path:
            # Use stored template path for ANTs
            template_path = getattr(self, 'atlas_template_path', None)
            if not template_path:
                raise ValueError("Atlas template path not available. Cannot use ANTs registration.")
            
            seg_atlas = register_to_atlas_ants(
                seg_path, 
                t1_reference_path,
                template_path,
                output_dir
            )
            # After ANTs registration, we still need to resample to atlas_labels space
            # since MNI152 and Harvard-Oxford labels may have different dimensions
            seg_atlas = register_to_atlas_affine(seg_atlas, self.atlas_labels)
        else:
            # Resample directly to atlas_labels space (not the template)
            seg_atlas = register_to_atlas_affine(seg_path, self.atlas_labels)
        
        # Optional: Clean segmentation
        if clean_segmentation:
            from .region_analysis import remove_small_components
            seg_data = seg_atlas.get_fdata()
            cleaned = remove_small_components(seg_data > 0, min_size=50)
            seg_data[~cleaned] = 0
            seg_atlas = nib.Nifti1Image(seg_data, seg_atlas.affine, seg_atlas.header)
        
        # Step 2: Multi-label analysis (for BraTS-style segmentations)
        print("Step 2: Analyzing tumor subregions...")
        subregion_analysis = analyze_tumor_subregions(
            seg_atlas,
            self.atlas_labels,
            self.region_names
        )
        
        # Step 3: Overall tumor analysis
        print("Step 3: Computing regional involvement...")
        overall_analysis = calculate_percentage_involvement(
            seg_atlas,
            self.atlas_labels,
            self.region_names
        )
        
        # Calculate total tumor volume
        seg_data = seg_atlas.get_fdata()
        total_tumor_voxels = np.sum(seg_data > 0)
        voxel_volume = np.prod(seg_atlas.header.get_zooms())
        total_tumor_volume_mm3 = int(total_tumor_voxels * voxel_volume)
        
        # Determine hemisphere
        hemisphere = self._determine_hemisphere(seg_data)
        crossing_midline = self._check_midline_crossing(seg_data)
        
        results = {
            'subregion_analysis': subregion_analysis,
            'overall_affected_regions': overall_analysis[:15],  # Top 15 regions
            'metadata': {
                'atlas': self.atlas_name,
                'registration': 'ANTs_SyN' if self.use_ants else 'affine',
                'total_tumor_volume_mm3': total_tumor_volume_mm3,
                'hemisphere': hemisphere,
                'crossing_midline': crossing_midline
            }
        }
        
        print(f"Atlas mapping complete. Found {len(overall_analysis)} affected regions.")
        
        return results
    
    def _determine_hemisphere(self, seg_data):
        """
        Determine primary hemisphere affected.
        
        Args:
            seg_data: Segmentation data array
        
        Returns:
            'left', 'right', or 'bilateral'
        """
        # Assumes standard orientation (RAS)
        midline_x = seg_data.shape[0] // 2
        
        left_voxels = np.sum(seg_data[:midline_x, :, :] > 0)
        right_voxels = np.sum(seg_data[midline_x:, :, :] > 0)
        
        if left_voxels > 2 * right_voxels:
            return "left"
        elif right_voxels > 2 * left_voxels:
            return "right"
        else:
            return "bilateral"
    
    def _check_midline_crossing(self, seg_data):
        """
        Check if tumor crosses the midline.
        
        Args:
            seg_data: Segmentation data array
        
        Returns:
            bool: True if tumor crosses midline
        """
        midline_x = seg_data.shape[0] // 2
        
        # Check if tumor is present on both sides (with a small margin)
        left_present = np.any(seg_data[:midline_x-2, :, :] > 0)
        right_present = np.any(seg_data[midline_x+2:, :, :] > 0)
        
        return bool(left_present and right_present)
