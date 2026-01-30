"""
Example script for generating atlas-based JSON descriptors from segmentations.

Usage:
    python scripts/generate_atlas_json.py --seg_path path/to/segmentation.nii.gz --case_id BraTS_00123
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.atlas_mapping import BrainAtlasMapper
from src.json_generation import TumorDescriptorGenerator, save_validated_descriptor


def main():
    parser = argparse.ArgumentParser(
        description='Generate JSON descriptors from brain tumor segmentations'
    )
    
    parser.add_argument(
        '--seg_path',
        type=str,
        required=True,
        help='Path to segmentation file (.nii.gz)'
    )
    
    parser.add_argument(
        '--case_id',
        type=str,
        required=True,
        help='Unique case identifier'
    )
    
    parser.add_argument(
        '--t1_path',
        type=str,
        default=None,
        help='Path to T1 reference scan (optional, for ANTs registration)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/descriptors',
        help='Output directory for JSON descriptors'
    )
    
    parser.add_argument(
        '--use_ants',
        action='store_true',
        help='Use ANTs for non-linear registration (requires ANTs installation)'
    )
    
    parser.add_argument(
        '--atlas_name',
        type=str,
        default='harvard_oxford',
        choices=['harvard_oxford', 'AAL3'],
        help='Brain atlas to use'
    )
    
    parser.add_argument(
        '--patient_age',
        type=int,
        default=None,
        help='Patient age (optional)'
    )
    
    parser.add_argument(
        '--patient_sex',
        type=str,
        default=None,
        choices=['M', 'F'],
        help='Patient sex (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize atlas mapper
    print("=" * 60)
    print("Atlas-Based JSON Descriptor Generation")
    print("=" * 60)
    
    atlas_mapper = BrainAtlasMapper(
        atlas_name=args.atlas_name,
        use_ants=args.use_ants
    )
    
    # Initialize descriptor generator
    descriptor_gen = TumorDescriptorGenerator(atlas_mapper)
    
    # Prepare patient metadata
    patient_metadata = {}
    if args.patient_age:
        patient_metadata['age'] = args.patient_age
    if args.patient_sex:
        patient_metadata['sex'] = args.patient_sex
    
    # Generate descriptor
    descriptor = descriptor_gen.generate_descriptor(
        case_id=args.case_id,
        seg_path=args.seg_path,
        t1_path=args.t1_path,
        patient_metadata=patient_metadata if patient_metadata else None
    )
    
    # Save descriptor
    output_path = Path(args.output_dir) / f'{args.case_id}_descriptor.json'
    save_validated_descriptor(descriptor, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Case ID: {descriptor['patient_info']['case_id']}")
    print(f"Total tumor volume: {descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']:.1f} mm³")
    print(f"Hemisphere: {descriptor['anatomical_mapping']['hemisphere']}")
    print(f"Crosses midline: {descriptor['anatomical_mapping']['crossing_midline']}")
    print(f"\nTop 5 affected regions:")
    for i, region in enumerate(descriptor['anatomical_mapping']['affected_regions'][:5], 1):
        print(f"  {i}. {region['region_name']}: {region['percentage_involvement']:.1f}%")
    
    print(f"\n✓ Descriptor saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
