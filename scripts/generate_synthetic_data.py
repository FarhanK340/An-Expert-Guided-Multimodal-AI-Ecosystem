"""
Synthetic Data Generation Script

Generates large-scale synthetic JSON→Report datasets for LLM fine-tuning.
Based on the methodology described in docs/SYNTHETIC_DATA_GUIDE.md
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticJSONGenerator:
    """Generate synthetic but clinically plausible tumor descriptors."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.age_distribution = (30, 80)
        self.volume_distribution = {
            'small': (500, 5000),
            'medium': (5000, 20000),
            'large': (20000, 50000)
        }
        
        self.brain_regions = [
            {"id": 1, "name": "Frontal Pole", "typical_volume": 8500},
            {"id": 2, "name": "Insular Cortex", "typical_volume": 7200},
            {"id": 3, "name": "Superior Frontal Gyrus", "typical_volume": 24000},
            {"id": 4, "name": "Middle Frontal Gyrus", "typical_volume": 26500},
            {"id": 5, "name": "Inferior Frontal Gyrus", "typical_volume": 18000},
            {"id": 6, "name": "Precentral Gyrus", "typical_volume": 25500},
            {"id": 7, "name": "Temporal Pole", "typical_volume": 9800},
            {"id": 8, "name": "Superior Temporal Gyrus", "typical_volume": 22000},
            {"id": 9, "name": "Middle Temporal Gyrus", "typical_volume": 24500},
            {"id": 10, "name": "Inferior Temporal Gyrus", "typical_volume": 21000},
            {"id": 11, "name": "Postcentral Gyrus", "typical_volume": 18500},
            {"id": 12, "name": "Superior Parietal Lobule", "typical_volume": 19000},
            {"id": 13, "name": "Angular Gyrus", "typical_volume": 13500},
            {"id": 14, "name": "Lateral Occipital Cortex", "typical_volume": 20000},
            {"id": 15, "name": "Precuneous Cortex", "typical_volume": 16000},
        ]
    
    def generate_synthetic_descriptor(self, case_id: str) -> Dict:
        """Generate a single synthetic tumor descriptor."""
        # Random tumor size
        size_category = random.choice(['small', 'medium', 'large'])
        total_volume = random.uniform(*self.volume_distribution[size_category])
        
        # Component breakdown
        enhancing_ratio = random.uniform(0.25, 0.45)
        necrotic_ratio = random.uniform(0.15, 0.35)
        edema_ratio = 1 - enhancing_ratio - necrotic_ratio
        
        enhancing_vol = total_volume * enhancing_ratio
        necrotic_vol = total_volume * necrotic_ratio
        edema_vol = total_volume * edema_ratio
        
        # Select affected regions
        num_regions = random.randint(3, 8)
        selected_regions = random.sample(self.brain_regions, num_regions)
        
        affected_regions = []
        remaining_volume = total_volume
        
        for i, region in enumerate(selected_regions):
            if i == num_regions - 1:
                region_tumor_vol = remaining_volume
            else:
                region_tumor_vol = random.uniform(
                    remaining_volume * 0.1,
                    remaining_volume * 0.6
                )
            
            percentage = min((region_tumor_vol / region['typical_volume']) * 100, 95)
            hemisphere = random.choice(['left', 'right'])
            
            affected_regions.append({
                "region_id": region['id'],
                "region_name": region['name'],
                "percentage_involvement": round(percentage, 2),
                "tumor_volume_in_region_mm3": round(region_tumor_vol, 1),
                "total_region_volume_mm3": region['typical_volume'],
                "hemisphere": hemisphere
            })
            
            remaining_volume -= region_tumor_vol
        
        affected_regions.sort(key=lambda x: x['percentage_involvement'], reverse=True)
        
        # Build descriptor
        descriptor = {
            "patient_info": {
                "case_id": case_id,
                "age": random.randint(*self.age_distribution),
                "sex": random.choice(["M", "F"]),
                "scan_date": self._random_date()
            },
            "imaging_metadata": {
                "modalities": ["T1", "T1ce", "T2", "FLAIR"],
                "scanner_info": {
                    "manufacturer": random.choice(["Siemens", "GE", "Philips"]),
                    "field_strength": random.choice([1.5, 3.0]),
                    "resolution_mm": [1.0, 1.0, 1.0]
                }
            },
            "segmentation_results": {
                "tumor_components": {
                    "enhancing_tumor": {
                        "present": True,
                        "volume_mm3": round(enhancing_vol, 1),
                        "voxel_count": int(enhancing_vol),
                        "confidence_score": round(random.uniform(0.85, 0.95), 2),
                        "centroid_coords": self._random_coords()
                    },
                    "necrotic_core": {
                        "present": necrotic_vol > 100,
                        "volume_mm3": round(necrotic_vol, 1),
                        "voxel_count": int(necrotic_vol),
                        "confidence_score": round(random.uniform(0.80, 0.92), 2),
                        "centroid_coords": self._random_coords()
                    },
                    "peritumoral_edema": {
                        "present": True,
                        "volume_mm3": round(edema_vol, 1),
                        "voxel_count": int(edema_vol),
                        "confidence_score": round(random.uniform(0.78, 0.90), 2),
                        "centroid_coords": self._random_coords()
                    }
                },
                "volumetric_analysis": {
                    "total_tumor_volume_mm3": round(total_volume, 1),
                    "whole_tumor_volume_mm3": round(total_volume, 1),
                    "tumor_core_volume_mm3": round(enhancing_vol + necrotic_vol, 1),
                    "enhancing_volume_mm3": round(enhancing_vol, 1),
                    "necrosis_percentage": round((necrotic_vol/total_volume)*100, 1)
                }
            },
            "anatomical_mapping": {
                "atlas_name": "harvard_oxford",
                "registration_method": random.choice(["affine", "ANTs_SyN"]),
                "hemisphere": self._determine_hemisphere(affected_regions),
                "crossing_midline": random.random() < 0.15,
                "affected_regions": affected_regions
            },
            "model_metadata": {
                "model_name": "MoME+",
                "model_version": "v1.0.0",
                "training_datasets": ["BraTS2021"],
                "inference_timestamp": datetime.utcnow().isoformat() + 'Z',
                "processing_time_seconds": round(random.uniform(10, 20), 1)
            },
            "clinical_features": {
                "mass_effect": random.random() < 0.4,
                "ventricular_involvement": random.random() < 0.2,
                "eloquent_area_involvement": self._sample_eloquent_areas(),
                "estimated_grade": random.choice(["low_grade", "high_grade"])
            }
        }
        
        return descriptor
    
    def _random_date(self):
        days_ago = random.randint(0, 730)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime("%Y-%m-%d")
    
    def _random_coords(self):
        return [
            round(random.uniform(-80, 80), 1),
            round(random.uniform(-110, 70), 1),
            round(random.uniform(-50, 85), 1)
        ]
    
    def _determine_hemisphere(self, regions):
        left_count = sum(1 for r in regions if r['hemisphere'] == 'left')
        right_count = len(regions) - left_count
        
        if left_count > 2 * right_count:
            return 'left'
        elif right_count > 2 * left_count:
            return 'right'
        else:
            return 'bilateral'
    
    def _sample_eloquent_areas(self):
        areas = []
        probabilities = {
            'motor_cortex': 0.25,
            'speech_area': 0.15,
            'visual_cortex': 0.10,
        }
        
        for area, prob in probabilities.items():
            if random.random() < prob:
                areas.append(area)
        
        return areas


class ReportTemplate:
    """Template-based medical report generator."""
    
    def __init__(self):
        self.intro_templates = [
            "Clinical MRI scan reveals {tumor_description} in the {primary_location}.",
            "Imaging demonstrates {tumor_description} involving the {primary_location}.",
            "MRI findings indicate {tumor_description} with primary involvement of the {primary_location}.",
        ]
        
        self.volume_templates = [
            "The lesion measures approximately {total_volume:.1f} mm³ in total volume.",
            "Total tumor volume is estimated at {total_volume:.1f} mm³.",
        ]
        
        self.component_templates = {
            'enhancing': [
                "The enhancing component measures {volume:.1f} mm³.",
                "Contrast-enhancing tumor accounts for {volume:.1f} mm³.",
            ],
            'necrotic': [
                "A necrotic core of {volume:.1f} mm³ is present.",
                "Central necrosis measures {volume:.1f} mm³.",
            ],
            'edema': [
                "Surrounding peritumoral edema extends {volume:.1f} mm³.",
                "Vasogenic edema measures {volume:.1f} mm³.",
            ]
        }
        
        self.conclusion_templates = [
            "Findings are consistent with a {grade} glioma.",
            "Imaging characteristics suggest a {grade} glial neoplasm.",
        ]
    
    def generate_report(self, descriptor: Dict) -> str:
        """Generate report from JSON descriptor."""
        sections = []
        
        sections.append("FINDINGS:")
        sections.append("")
        
        # Introduction
        tumor_desc = self._describe_tumor(descriptor)
        primary_location = descriptor['anatomical_mapping']['affected_regions'][0]['region_name']
        hemisphere = descriptor['anatomical_mapping']['hemisphere']
        
        intro = random.choice(self.intro_templates).format(
            tumor_description=tumor_desc,
            primary_location=f"{hemisphere} {primary_location}"
        )
        sections.append(intro)
        sections.append("")
        
        # Volume
        total_vol = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
        sections.append(random.choice(self.volume_templates).format(total_volume=total_vol))
        sections.append("")
        
        # Components
        components = descriptor['segmentation_results']['tumor_components']
        
        if components['enhancing_tumor']['present']:
            template = random.choice(self.component_templates['enhancing'])
            sections.append(template.format(volume=components['enhancing_tumor']['volume_mm3']))
        
        if components['necrotic_core']['present'] and components['necrotic_core']['volume_mm3'] > 100:
            template = random.choice(self.component_templates['necrotic'])
            sections.append(template.format(volume=components['necrotic_core']['volume_mm3']))
        
        if components['peritumoral_edema']['present']:
            template = random.choice(self.component_templates['edema'])
            sections.append(template.format(volume=components['peritumoral_edema']['volume_mm3']))
        
        sections.append("")
        sections.append("IMPRESSION:")
        sections.append("")
        
        # Conclusion
        grade_map = {'low_grade': 'low-grade', 'high_grade': 'high-grade'}
        grade = descriptor['clinical_features'].get('estimated_grade', 'unknown')
        grade_text = grade_map.get(grade, 'intermediate-grade')
        
        sections.append(random.choice(self.conclusion_templates).format(grade=grade_text))
        
        return '\n'.join(sections)
    
    def _describe_tumor(self, descriptor):
        vol = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
        
        if vol < 5000:
            size = random.choice(['small', 'focal'])
        elif vol < 20000:
            size = random.choice(['moderate-sized', 'well-defined'])
        else:
            size = random.choice(['large', 'extensive'])
        
        components = descriptor['segmentation_results']['tumor_components']
        
        if components['enhancing_tumor']['present'] and components['necrotic_core']['present']:
            characteristics = 'heterogeneous mass with enhancement and necrosis'
        elif components['enhancing_tumor']['present']:
            characteristics = 'enhancing mass lesion'
        else:
            characteristics = 'infiltrative lesion'
        
        return f"{size} {characteristics}"


def generate_dataset(
    num_samples: int,
    num_variations: int,
    output_dir: str,
    seed: int = 42
):
    """
    Generate complete synthetic dataset.
    
    Args:
        num_samples: Number of unique JSON descriptors
        num_variations: Report variations per JSON
        output_dir: Output directory
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_dir = output_path / 'json'
    json_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating {num_samples} synthetic JSON descriptors...")
    
    # Step 1: Generate JSONs
    json_gen = SyntheticJSONGenerator(seed=seed)
    
    for i in range(num_samples):
        case_id = f"SYNTHETIC_{i:06d}"
        descriptor = json_gen.generate_synthetic_descriptor(case_id)
        
        with open(json_dir / f"{case_id}.json", 'w') as f:
            json.dump(descriptor, f, indent=2)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i+1}/{num_samples} JSON descriptors")
    
    # Step 2: Generate reports
    logger.info(f"\nGenerating {num_samples * num_variations} reports...")
    
    template_gen = ReportTemplate()
    output_file = output_path / 'training_data.jsonl'
    
    with open(output_file, 'w') as outf:
        for json_file in json_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                descriptor = json.load(f)
            
            for var_idx in range(num_variations):
                report = template_gen.generate_report(descriptor)
                
                training_example = {
                    "descriptor": descriptor,
                    "report": report,
                    "metadata": {
                        "case_id": descriptor['patient_info']['case_id'],
                        "variation": var_idx
                    }
                }
                
                outf.write(json.dumps(training_example) + '\n')
    
    logger.info(f"\n✅ Dataset generation complete!")
    logger.info(f"   - JSON descriptors: {json_dir}")
    logger.info(f"   - Training data: {output_file}")
    logger.info(f"   - Total examples: {num_samples * num_variations}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic medical report dataset')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of unique JSON descriptors to generate')
    parser.add_argument('--num_variations', type=int, default=3,
                       help='Number of report variations per JSON')
    parser.add_argument('--output_dir', type=str, default='synthetic_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    logger.info("=== Synthetic Data Generation ===")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Variations: {args.num_variations}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 35)
    
    generate_dataset(
        num_samples=args.num_samples,
        num_variations=args.num_variations,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
