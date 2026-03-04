"""
JSON Descriptor Generator

Creates structured JSON descriptors from segmentation masks and atlas mappings
following the schema defined in docs/JSON_SCHEMA_GUIDE.md
"""

import json
import numpy as np
import nibabel as nib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import logging

from .atlas_mapping import BrainAtlasMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TumorDescriptorGenerator:
    """
    Generate structured JSON descriptors from segmentation + atlas mapping.
    """
    
    def __init__(self, atlas_mapper: BrainAtlasMapper):
        """
        Initialize descriptor generator.
        
        Args:
            atlas_mapper: Configured BrainAtlasMapper instance
        """
        self.atlas_mapper = atlas_mapper
    
    def generate_descriptor(
        self,
        case_id: str,
        seg_path: str,
        t1_path: str,
        modalities: List[str] = None,
        patient_metadata: Optional[Dict] = None,
        model_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create complete JSON descriptor from segmentation.
        
        Args:
            case_id: Unique case identifier
            seg_path: Path to segmentation mask
            t1_path: Path to T1 MRI scan
            modalities: List of MRI modalities used
            patient_metadata: Additional patient info (age, sex, etc.)
            model_metadata: Model information (name, version, etc.)
        
        Returns:
            Dictionary conforming to tumor descriptor schema
        """
        if modalities is None:
            modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
        
        logger.info(f"Generating descriptor for case {case_id}")
        
        # Run atlas mapping
        atlas_results = self.atlas_mapper.process_segmentation(
            seg_path,
            t1_path
        )
        
        # Load segmentation for volumetric analysis
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        voxel_vol = np.prod(seg_img.header.get_zooms())
        
        # Build descriptor
        descriptor = {
            "patient_info": self._build_patient_info(case_id, patient_metadata),
            "imaging_metadata": self._build_imaging_metadata(
                seg_img,
                modalities
            ),
            "segmentation_results": self._extract_segmentation_results(
                seg_data,
                voxel_vol
            ),
            "anatomical_mapping": self._build_anatomical_mapping(
                atlas_results,
                seg_data,
                seg_img
            ),
            "model_metadata": self._build_model_metadata(model_metadata)
        }
        
        # Add clinical features if detectable
        descriptor["clinical_features"] = self._extract_clinical_features(
            seg_data,
            seg_img,
            atlas_results
        )
        
        # Validate schema
        self._validate_descriptor(descriptor)
        
        logger.info(f"Descriptor generated successfully for {case_id}")
        
        return descriptor
    
    def _build_patient_info(
        self,
        case_id: str,
        patient_metadata: Optional[Dict]
    ) -> Dict:
        """Build patient_info section."""
        info = {
            "case_id": case_id,
            "scan_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        if patient_metadata:
            if 'age' in patient_metadata:
                info['age'] = int(patient_metadata['age'])
            if 'sex' in patient_metadata:
                info['sex'] = patient_metadata['sex']
        
        return info
    
    def _build_imaging_metadata(
        self,
        seg_img: nib.Nifti1Image,
        modalities: List[str]
    ) -> Dict:
        """Build imaging_metadata section."""
        return {
            "modalities": modalities,
            "scanner_info": {
                "manufacturer": "Unknown",  # Extract from DICOM if available
                "field_strength": 3.0,       # Default, extract from metadata
                "resolution_mm": [float(x) for x in seg_img.header.get_zooms()]
            }
        }
    
    def _extract_segmentation_results(
        self,
        seg_data: np.ndarray,
        voxel_vol: float
    ) -> Dict:
        """
        Extract tumor component information from segmentation.
        
        BraTS labels:
        - 1: Necrotic and non-enhancing tumor core
        - 2: Peritumoral edema
        - 4: Enhancing tumor
        """
        # Analyze each component
        components = {
            "enhancing_tumor": self._analyze_component(seg_data, 4, voxel_vol),
            "necrotic_core": self._analyze_component(seg_data, 1, voxel_vol),
            "peritumoral_edema": self._analyze_component(seg_data, 2, voxel_vol)
        }
        
        # Volumetric analysis
        wt_mask = seg_data > 0                    # Whole tumor
        tc_mask = (seg_data == 1) | (seg_data == 4)  # Tumor core
        et_mask = seg_data == 4                   # Enhancing
        
        total_vol = float(np.sum(wt_mask) * voxel_vol)
        tc_vol = float(np.sum(tc_mask) * voxel_vol)
        et_vol = float(np.sum(et_mask) * voxel_vol)
        necrosis_vol = float(np.sum(seg_data == 1) * voxel_vol)
        
        necrosis_pct = (necrosis_vol / total_vol * 100) if total_vol > 0 else 0.0
        
        return {
            "tumor_components": components,
            "volumetric_analysis": {
                "total_tumor_volume_mm3": round(total_vol, 1),
                "whole_tumor_volume_mm3": round(total_vol, 1),
                "tumor_core_volume_mm3": round(tc_vol, 1),
                "enhancing_volume_mm3": round(et_vol, 1),
                "necrosis_percentage": round(necrosis_pct, 2)
            },
            "confidence_metrics": {
                "mean_dice_score": 0.88,  # Replace with actual model output
                "per_class_confidence": {
                    "enhancing_tumor": 0.92,
                    "necrotic_core": 0.88,
                    "edema": 0.85
                }
            }
        }
    
    def _analyze_component(
        self,
        seg_data: np.ndarray,
        label: int,
        voxel_vol: float
    ) -> Dict:
        """Analyze single tumor component."""
        mask = seg_data == label
        
        if not np.any(mask):
            return {
                "present": False,
                "volume_mm3": 0.0,
                "voxel_count": 0,
                "confidence_score": 0.0,
                "centroid_coords": [0.0, 0.0, 0.0]
            }
        
        voxel_count = int(np.sum(mask))
        volume = float(voxel_count * voxel_vol)
        
        # Calculate centroid
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0).tolist()
        
        return {
            "present": True,
            "volume_mm3": round(volume, 1),
            "voxel_count": voxel_count,
            "confidence_score": 0.90,  # Replace with actual confidence
            "centroid_coords": [round(float(c), 1) for c in centroid]
        }
    
    def _build_anatomical_mapping(
        self,
        atlas_results: Dict,
        seg_data: np.ndarray,
        seg_img: nib.Nifti1Image
    ) -> Dict:
        """Build anatomical_mapping section."""
        # Determine hemisphere
        hemisphere = self._determine_hemisphere(seg_data)
        
        # Check midline crossing
        crossing_midline = self._check_midline_crossing(seg_data)
        
        # Format subregion mapping
        subregion_mapping = {
            "enhancing_tumor": atlas_results['subregion_analysis'].get(
                'enhancing_tumor', []
            ),
            "necrotic_core": atlas_results['subregion_analysis'].get(
                'necrotic_and_non_enhancing_tumor_core', []
            ),
            "peritumoral_edema": atlas_results['subregion_analysis'].get(
                'peritumoral_edema', []
            )
        }
        
        # Add hemisphere to each region
        for regions in subregion_mapping.values():
            for region in regions:
                if 'hemisphere' not in region:
                    region['hemisphere'] = hemisphere
        
        return {
            "atlas_name": atlas_results['metadata']['atlas'],
            "registration_method": atlas_results['metadata']['registration'],
            "hemisphere": hemisphere,
            "crossing_midline": crossing_midline,
            "affected_regions": atlas_results['overall_affected_regions'],
            "subregion_mapping": subregion_mapping
        }
    
    def _determine_hemisphere(self, seg_data: np.ndarray) -> str:
        """Determine primary tumor hemisphere."""
        midline_x = seg_data.shape[0] // 2
        
        left_voxels = np.sum(seg_data[:midline_x, :, :] > 0)
        right_voxels = np.sum(seg_data[midline_x:, :, :] > 0)
        
        if left_voxels > 2 * right_voxels:
            return "left"
        elif right_voxels > 2 * left_voxels:
            return "right"
        else:
            return "bilateral"
    
    def _check_midline_crossing(self, seg_data: np.ndarray) -> bool:
        """Check if tumor crosses brain midline."""
        midline_x = seg_data.shape[0] // 2
        
        # Allow 2-voxel tolerance for midline definition
        left_present = np.any(seg_data[:midline_x-2, :, :] > 0)
        right_present = np.any(seg_data[midline_x+2:, :, :] > 0)
        
        return bool(left_present and right_present)
    
    def _build_model_metadata(self, model_metadata: Optional[Dict]) -> Dict:
        """Build model_metadata section."""
        metadata = {
            "model_name": "MoME+",
            "model_version": "v1.0.0",
            "training_datasets": ["BraTS2021"],
            "inference_timestamp": datetime.utcnow().isoformat() + 'Z',
            "processing_time_seconds": 15.2
        }
        
        if model_metadata:
            metadata.update(model_metadata)
        
        return metadata
    
    def _extract_clinical_features(
        self,
        seg_data: np.ndarray,
        seg_img: nib.Nifti1Image,
        atlas_results: Dict
    ) -> Dict:
        """Extract observable clinical features."""
        features = {}
        
        # Mass effect (simplified heuristic)
        total_vol = atlas_results['metadata']['total_tumor_volume_mm3']
        features['mass_effect'] = total_vol > 20000  # >20cc suggests mass effect
        
        # Check for eloquent area involvement
        eloquent_regions = {
            'motor_cortex': ['Precentral Gyrus'],
            'speech_area': ['Inferior Frontal Gyrus'],
            'visual_cortex': ['Occipital'],
        }
        
        eloquent_involved = []
        affected_names = [
            r['region_name']
            for r in atlas_results['overall_affected_regions']
        ]
        
        for area, region_keywords in eloquent_regions.items():
            for region_name in affected_names:
                if any(keyword in region_name for keyword in region_keywords):
                    eloquent_involved.append(area)
                    break
        
        features['eloquent_area_involvement'] = eloquent_involved
        
        # Estimated grade (based on enhancement pattern)
        enhancing_vol = np.sum(seg_data == 4) * np.prod(seg_img.header.get_zooms())
        necrotic_vol = np.sum(seg_data == 1) * np.prod(seg_img.header.get_zooms())
        
        if enhancing_vol > 1000 or necrotic_vol > 500:
            features['estimated_grade'] = 'high_grade'
        else:
            features['estimated_grade'] = 'low_grade'
        
        return features
    
    def _validate_descriptor(self, descriptor: Dict) -> bool:
        """
        Validate descriptor against schema.
        
        Args:
            descriptor: Generated descriptor
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If validation fails
        """
        # Basic validation (use jsonschema for production)
        required_keys = [
            'patient_info',
            'imaging_metadata',
            'segmentation_results',
            'anatomical_mapping',
            'model_metadata'
        ]
        
        for key in required_keys:
            if key not in descriptor:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate patient_info
        if 'case_id' not in descriptor['patient_info']:
            raise ValueError("Missing case_id in patient_info")
        
        # Validate volumetric data is positive
        vol_analysis = descriptor['segmentation_results']['volumetric_analysis']
        if vol_analysis['total_tumor_volume_mm3'] <= 0:
            raise ValueError("Total tumor volume must be positive")
        
        logger.info("Descriptor validation passed")
        return True
    
    def save_descriptor(
        self,
        descriptor: Dict,
        output_path: str
    ) -> None:
        """
        Save descriptor to JSON file.
        
        Args:
            descriptor: Generated descriptor
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(descriptor, f, indent=2)
        
        logger.info(f"Descriptor saved to {output_path}")


def batch_generate_descriptors(
    dataset_dir: str,
    atlas_mapper: BrainAtlasMapper,
    output_dir: str,
    max_samples: Optional[int] = None
) -> None:
    """
    Generate descriptors for an entire dataset.
    
    Args:
        dataset_dir: Directory containing segmentations
        atlas_mapper: Configured atlas mapper
        output_dir: Output directory for JSON files
        max_samples: Maximum number of samples to process
    """
    generator = TumorDescriptorGenerator(atlas_mapper)
    
    # Find segmentation files
    seg_files = list(Path(dataset_dir).rglob('*seg.nii.gz'))
    
    if max_samples:
        seg_files = seg_files[:max_samples]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(seg_files)} segmentations...")
    
    for i, seg_path in enumerate(seg_files):
        case_id = seg_path.stem.replace('_seg', '')
        
        # Find corresponding T1
        t1_path = seg_path.parent / f"{case_id}_t1.nii.gz"
        
        if not t1_path.exists():
            logger.warning(f"T1 not found for {case_id}, skipping")
            continue
        
        try:
            descriptor = generator.generate_descriptor(
                case_id=case_id,
                seg_path=str(seg_path),
                t1_path=str(t1_path)
            )
            
            output_file = output_path / f"{case_id}_descriptor.json"
            generator.save_descriptor(descriptor, str(output_file))
            
            logger.info(f"[{i+1}/{len(seg_files)}] Processed {case_id}")
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {e}")
            continue
    
    logger.info(f"Batch generation complete. Output: {output_dir}")


if __name__ == '__main__':
    # Example usage
    from atlas_mapping import BrainAtlasMapper
    
    atlas_mapper = BrainAtlasMapper(
        atlas_path='atlases/MNI152_T1_1mm.nii.gz',
        atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
    )
    
    generator = TumorDescriptorGenerator(atlas_mapper)
    
    descriptor = generator.generate_descriptor(
        case_id="BraTS2021_00123",
        seg_path="predictions/case_00123_seg.nii.gz",
        t1_path="inputs/case_00123_t1.nii.gz",
        patient_metadata={"age": 58, "sex": "M"}
    )
    
    generator.save_descriptor(
        descriptor,
        "descriptors/case_00123_descriptor.json"
    )
