"""
Brain Atlas Mapping Module

Transforms segmentation masks into anatomically meaningful region labels
using standard brain atlases (Harvard-Oxford, AAL3, MNI152).
"""

import os
import numpy as np
import nibabel as nib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainAtlasMapper:
    """
    Production class for brain atlas mapping pipeline.
    Converts segmentation masks to anatomical region descriptors.
    """
    
    def __init__(
        self,
        atlas_path: str,
        atlas_labels_path: str,
        atlas_name: str = 'harvard_oxford',
        use_ants: bool = False
    ):
        """
        Initialize atlas mapper.
        
        Args:
            atlas_path: Path to brain atlas template (e.g., MNI152_T1_1mm.nii.gz)
            atlas_labels_path: Path to atlas labels (e.g., HarvardOxford-cort-maxprob.nii.gz)
            atlas_name: Name of atlas ('harvard_oxford', 'AAL3', 'MNI152')
            use_ants: Whether to use ANTs for non-linear registration
        """
        self.atlas_img = nib.load(atlas_path)
        self.atlas_labels = nib.load(atlas_labels_path)
        self.atlas_name = atlas_name
        self.use_ants = use_ants
        
        # Load region names
        self.region_names = self._load_region_names(atlas_name)
        
        logger.info(f"Initialized BrainAtlasMapper with {atlas_name} atlas")
    
    def _load_region_names(self, atlas_name: str) -> Dict[int, str]:
        """
        Load region ID to name mapping for the specified atlas.
        
        Returns:
            Dictionary mapping region IDs to anatomical names
        """
        if atlas_name == 'harvard_oxford':
            # Harvard-Oxford cortical atlas (48 regions)
            return {
                0: 'Background',
                1: 'Frontal Pole',
                2: 'Insular Cortex',
                3: 'Superior Frontal Gyrus',
                4: 'Middle Frontal Gyrus',
                5: 'Inferior Frontal Gyrus, pars triangularis',
                6: 'Inferior Frontal Gyrus, pars opercularis',
                7: 'Precentral Gyrus',
                8: 'Temporal Pole',
                9: 'Superior Temporal Gyrus, anterior',
                10: 'Superior Temporal Gyrus, posterior',
                11: 'Middle Temporal Gyrus, anterior',
                12: 'Middle Temporal Gyrus, posterior',
                13: 'Middle Temporal Gyrus, temporooccipital',
                14: 'Inferior Temporal Gyrus, anterior',
                15: 'Inferior Temporal Gyrus, posterior',
                16: 'Inferior Temporal Gyrus, temporooccipital',
                17: 'Postcentral Gyrus',
                18: 'Superior Parietal Lobule',
                19: 'Supramarginal Gyrus, anterior',
                20: 'Supramarginal Gyrus, posterior',
                21: 'Angular Gyrus',
                22: 'Lateral Occipital Cortex, superior',
                23: 'Lateral Occipital Cortex, inferior',
                24: 'Intracalcarine Cortex',
                25: 'Frontal Medial Cortex',
                26: 'Juxtapositional Lobule Cortex',
                27: 'Subcallosal Cortex',
                28: 'Paracingulate Gyrus',
                29: 'Cingulate Gyrus, anterior',
                30: 'Cingulate Gyrus, posterior',
                31: 'Precuneous Cortex',
                32: 'Cuneal Cortex',
                33: 'Frontal Orbital Cortex',
                34: 'Parahippocampal Gyrus, anterior',
                35: 'Parahippocampal Gyrus, posterior',
                36: 'Lingual Gyrus',
                37: 'Temporal Fusiform Cortex, anterior',
                38: 'Temporal Fusiform Cortex, posterior',
                39: 'Temporal Occipital Fusiform Cortex',
                40: 'Occipital Fusiform Gyrus',
                41: 'Frontal Operculum Cortex',
                42: 'Central Opercular Cortex',
                43: 'Parietal Operculum Cortex',
                44: 'Planum Polare',
                45: 'Heschl\'s Gyrus',
                46: 'Planum Temporale',
                47: 'Supracalcarine Cortex',
                48: 'Occipital Pole',
            }
        else:
            logger.warning(f"Atlas {atlas_name} not implemented, using generic names")
            return {i: f'Region_{i}' for i in range(200)}
    
    def register_to_atlas_affine(
        self,
        seg_path: str
    ) -> nib.Nifti1Image:
        """
        Resample segmentation mask to atlas space using affine transformation.
        
        Args:
            seg_path: Path to segmentation mask (.nii.gz)
        
        Returns:
            Nibabel image object in atlas space
        """
        from nilearn.image import resample_to_img
        
        seg_img = nib.load(seg_path)
        
        # Resample to atlas space
        seg_resampled = resample_to_img(
            seg_img,
            self.atlas_img,
            interpolation='nearest'  # Preserve integer labels
        )
        
        logger.info(f"Registered segmentation to atlas space (affine)")
        return seg_resampled
    
    def register_to_atlas_ants(
        self,
        seg_path: str,
        t1_reference_path: str,
        output_dir: str
    ) -> nib.Nifti1Image:
        """
        Use ANTs for precise non-linear registration.
        
        Args:
            seg_path: Path to segmentation mask
            t1_reference_path: Path to patient's T1 scan
            output_dir: Directory for intermediate files
        
        Returns:
            Nibabel image in atlas space
        """
        import subprocess
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Register T1 to MNI152 template
        atlas_template_path = str(self.atlas_img.get_filename())
        transform_prefix = os.path.join(output_dir, 'transform_')
        
        cmd_register = [
            'antsRegistrationSyNQuick.sh',
            '-d', '3',
            '-f', atlas_template_path,
            '-m', t1_reference_path,
            '-o', transform_prefix,
            '-t', 's'
        ]
        
        try:
            subprocess.run(cmd_register, check=True, capture_output=True)
            logger.info("ANTs registration completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"ANTs registration failed: {e}")
            raise
        
        # Step 2: Apply transformation to segmentation
        output_seg = os.path.join(output_dir, 'seg_in_atlas_space.nii.gz')
        
        cmd_transform = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', seg_path,
            '-r', atlas_template_path,
            '-o', output_seg,
            '-n', 'NearestNeighbor',
            '-t', f'{transform_prefix}1Warp.nii.gz',
            '-t', f'{transform_prefix}0GenericAffine.mat'
        ]
        
        try:
            subprocess.run(cmd_transform, check=True, capture_output=True)
            logger.info("Applied transformation to segmentation")
        except subprocess.CalledProcessError as e:
            logger.error(f"Transform application failed: {e}")
            raise
        
        return nib.load(output_seg)
    
    def compute_region_overlap(
        self,
        seg_atlas_space: nib.Nifti1Image
    ) -> Dict[int, int]:
        """
        Calculate voxel-wise overlap between segmentation and atlas regions.
        
        Args:
            seg_atlas_space: Segmentation in atlas space
        
        Returns:
            Dictionary mapping region IDs to tumor voxel counts
        """
        seg_data = seg_atlas_space.get_fdata()
        atlas_data = self.atlas_labels.get_fdata().astype(int)
        
        # Binary tumor mask
        tumor_mask = seg_data > 0
        
        # Count voxels per region
        region_counts = defaultdict(int)
        unique_regions = np.unique(atlas_data[tumor_mask])
        
        for region_id in unique_regions:
            if region_id == 0:  # Skip background
                continue
            
            overlap_mask = (atlas_data == region_id) & tumor_mask
            region_counts[region_id] = int(np.sum(overlap_mask))
        
        return dict(region_counts)
    
    def calculate_percentage_involvement(
        self,
        seg_atlas_space: nib.Nifti1Image
    ) -> List[Dict]:
        """
        Compute percentage of each brain region affected by tumor.
        
        Args:
            seg_atlas_space: Segmentation in atlas space
        
        Returns:
            List of dicts with region information, sorted by involvement
        """
        seg_data = seg_atlas_space.get_fdata()
        atlas_data = self.atlas_labels.get_fdata().astype(int)
        
        region_overlap = self.compute_region_overlap(seg_atlas_space)
        voxel_vol = np.prod(seg_atlas_space.header.get_zooms())
        
        results = []
        
        for region_id, tumor_voxels in region_overlap.items():
            # Total voxels in this region
            region_mask = atlas_data == region_id
            total_voxels = int(np.sum(region_mask))
            
            if total_voxels == 0:
                continue
            
            # Percentage involvement
            percentage = (tumor_voxels / total_voxels) * 100
            
            results.append({
                'region_id': int(region_id),
                'region_name': self.region_names.get(region_id, f'Unknown_{region_id}'),
                'tumor_voxels': int(tumor_voxels),
                'total_voxels': total_voxels,
                'percentage_involvement': round(percentage, 2),
                'tumor_volume_in_region_mm3': round(tumor_voxels * voxel_vol, 1),
                'total_region_volume_mm3': round(total_voxels * voxel_vol, 1)
            })
        
        # Sort by percentage involvement (descending)
        results.sort(key=lambda x: x['percentage_involvement'], reverse=True)
        
        return results
    
    def analyze_tumor_subregions(
        self,
        seg_atlas_space: nib.Nifti1Image
    ) -> Dict[str, List[Dict]]:
        """
        Separate anatomical analysis for each tumor sub-component.
        
        BraTS labels:
        - Label 1: Necrotic and non-enhancing tumor core (NCR/NET)
        - Label 2: Peritumoral edema (ED)
        - Label 4: Enhancing tumor (ET)
        
        Args:
            seg_atlas_space: Segmentation in atlas space
        
        Returns:
            Dictionary with region analysis per tumor component
        """
        seg_data = seg_atlas_space.get_fdata()
        atlas_data = self.atlas_labels.get_fdata().astype(int)
        voxel_vol = np.prod(seg_atlas_space.header.get_zooms())
        
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
            
            # Calculate region overlap for this label
            unique_regions = np.unique(atlas_data[label_mask])
            label_results = []
            
            for region_id in unique_regions:
                if region_id == 0:
                    continue
                
                overlap_mask = (atlas_data == region_id) & label_mask
                label_voxels = int(np.sum(overlap_mask))
                
                region_mask = atlas_data == region_id
                total_voxels = int(np.sum(region_mask))
                
                if total_voxels > 0:
                    percentage = (label_voxels / total_voxels) * 100
                    
                    label_results.append({
                        'region_id': int(region_id),
                        'region_name': self.region_names.get(region_id, f'Unknown_{region_id}'),
                        'voxels': label_voxels,
                        'percentage': round(percentage, 2),
                        'tumor_volume_in_region_mm3': round(label_voxels * voxel_vol, 1),
                        'total_region_volume_mm3': round(total_voxels * voxel_vol, 1)
                    })
            
            # Sort by percentage (descending)
            label_results.sort(key=lambda x: x['percentage'], reverse=True)
            results[label_name] = label_results[:10]  # Top 10 regions per component
        
        return results
    
    def process_segmentation(
        self,
        seg_path: str,
        t1_reference_path: Optional[str] = None,
        output_dir: str = './atlas_mapping_output'
    ) -> Dict:
        """
        Complete atlas mapping workflow.
        
        Args:
            seg_path: Path to segmentation mask
            t1_reference_path: Path to T1 scan (required for ANTs)
            output_dir: Directory for outputs
        
        Returns:
            Dictionary with complete anatomical analysis
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Registration
        if self.use_ants and t1_reference_path:
            seg_atlas = self.register_to_atlas_ants(
                seg_path,
                t1_reference_path,
                output_dir
            )
        else:
            seg_atlas = self.register_to_atlas_affine(seg_path)
        
        # Step 2: Multi-label subregion analysis
        subregion_analysis = self.analyze_tumor_subregions(seg_atlas)
        
        # Step 3: Overall tumor analysis
        overall_analysis = self.calculate_percentage_involvement(seg_atlas)
        
        # Step 4: Metadata
        seg_data = seg_atlas.get_fdata()
        voxel_vol = np.prod(seg_atlas.header.get_zooms())
        total_tumor_volume = int(np.sum(seg_data > 0) * voxel_vol)
        
        result = {
            'subregion_analysis': subregion_analysis,
            'overall_affected_regions': overall_analysis[:15],  # Top 15
            'metadata': {
                'atlas': self.atlas_name,
                'registration': 'ANTs' if self.use_ants else 'affine',
                'total_tumor_volume_mm3': total_tumor_volume
            }
        }
        
        logger.info(f"Atlas mapping complete: {len(overall_analysis)} regions affected")
        
        return result


# Utility functions

def remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove isolated voxel clusters from segmentation mask.
    
    Args:
        mask: Binary or multi-label mask
        min_size: Minimum component size in voxels
    
    Returns:
        Cleaned mask
    """
    from scipy.ndimage import label as ndlabel
    from skimage.morphology import remove_small_objects
    
    # Handle multi-label masks
    unique_labels = np.unique(mask)
    cleaned = np.zeros_like(mask)
    
    for label_val in unique_labels:
        if label_val == 0:
            continue
        
        binary_mask = mask == label_val
        labeled, num_features = ndlabel(binary_mask)
        
        # Remove small objects
        cleaned_binary = remove_small_objects(
            labeled.astype(bool),
            min_size=min_size
        )
        
        cleaned[cleaned_binary] = label_val
    
    return cleaned


if __name__ == '__main__':
    # Example usage
    mapper = BrainAtlasMapper(
        atlas_path='atlases/MNI152_T1_1mm.nii.gz',
        atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',
        use_ants=False
    )
    
    results = mapper.process_segmentation(
        seg_path='predictions/case_001_seg.nii.gz',
        t1_reference_path='inputs/case_001_t1.nii.gz'
    )
    
    print(f"Total tumor volume: {results['metadata']['total_tumor_volume_mm3']} mm³")
    print(f"Top affected region: {results['overall_affected_regions'][0]['region_name']}")
