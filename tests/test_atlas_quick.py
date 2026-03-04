"""
Quick test script for atlas-JSON mapping pipeline.
Creates a synthetic segmentation and runs the full pipeline.

Usage: python test_atlas_quick.py
"""

import numpy as np
import nibabel as nib
import tempfile
import json
from pathlib import Path

print("=" * 70)
print("Atlas-JSON Mapping Pipeline Test")
print("=" * 70)

# Step 1: Create synthetic segmentation
print("\n[1/5] Creating synthetic brain tumor segmentation...")
seg_data = np.zeros((100, 100, 100), dtype=np.uint8)

# Add tumor regions (BraTS labels)
seg_data[40:60, 40:60, 40:60] = 1  # Necrotic core (label 1)
seg_data[45:55, 45:55, 45:55] = 4  # Enhancing tumor (label 4)
seg_data[35:65, 35:65, 35:65] = 2  # Edema (label 2)

# Create NIfTI image with proper affine
affine = np.eye(4)
affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm isotropic
seg_img = nib.Nifti1Image(seg_data, affine)

# Save to temporary file
temp_dir = Path('./test_output')
temp_dir.mkdir(exist_ok=True)
seg_path = temp_dir / 'synthetic_segmentation.nii.gz'
nib.save(seg_img, seg_path)
print(f"   ✓ Synthetic segmentation created: {seg_path}")
print(f"   - Volume: {np.sum(seg_data > 0)} voxels")
print(f"   - Necrotic core: {np.sum(seg_data == 1)} voxels")
print(f"   - Edema: {np.sum(seg_data == 2)} voxels")
print(f"   - Enhancing: {np.sum(seg_data == 4)} voxels")

# Step 2: Import modules
print("\n[2/5] Importing atlas mapping modules...")
try:
    from src.atlas_mapping import BrainAtlasMapper
    from src.json_generation import TumorDescriptorGenerator
    print("   ✓ Modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\n   Make sure you're in the project root directory and dependencies are installed:")
    print("   pip install nibabel nilearn scipy scikit-image jsonschema")
    exit(1)

# Step 3: Initialize atlas mapper
print("\n[3/5] Initializing atlas mapper...")
print("   (This may download Harvard-Oxford atlas if not cached...)")
try:
    atlas_mapper = BrainAtlasMapper(
        atlas_name='harvard_oxford',
        use_ants=False  # Use fast affine registration for testing
    )
    print("   ✓ Atlas mapper initialized")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    exit(1)

# Step 4: Generate JSON descriptor
print("\n[4/5] Generating JSON descriptor...")
try:
    generator = TumorDescriptorGenerator(atlas_mapper)
    
    descriptor = generator.generate_descriptor(
        case_id='SYNTHETIC_TEST_001',
        seg_path=str(seg_path),
        patient_metadata={
            'age': 58,
            'sex': 'M'
        },
        model_name='MoME+',
        model_version='v1.0.0'
    )
    print("   ✓ JSON descriptor generated successfully")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Save and display results
print("\n[5/5] Saving results...")
output_path = temp_dir / 'SYNTHETIC_TEST_001_descriptor.json'
with open(output_path, 'w') as f:
    json.dump(descriptor, f, indent=2)
print(f"   ✓ JSON saved to: {output_path}")

# Display summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n📊 Patient Info:")
print(f"   Case ID: {descriptor['patient_info']['case_id']}")
print(f"   Age: {descriptor['patient_info'].get('age', 'N/A')}")
print(f"   Sex: {descriptor['patient_info'].get('sex', 'N/A')}")

print(f"\n📈 Volumetric Analysis:")
vol_analysis = descriptor['segmentation_results']['volumetric_analysis']
print(f"   Total tumor volume: {vol_analysis['total_tumor_volume_mm3']:.1f} mm³")
print(f"   Tumor core volume: {vol_analysis.get('tumor_core_volume_mm3', 0):.1f} mm³")
print(f"   Enhancing volume: {vol_analysis.get('enhancing_volume_mm3', 0):.1f} mm³")
print(f"   Necrosis percentage: {vol_analysis.get('necrosis_percentage', 0):.1f}%")

print(f"\n🧠 Anatomical Mapping:")
anat = descriptor['anatomical_mapping']
print(f"   Atlas: {anat['atlas_name']}")
print(f"   Registration: {anat['registration_method']}")
print(f"   Hemisphere: {anat.get('hemisphere', 'N/A')}")
print(f"   Crosses midline: {anat.get('crossing_midline', False)}")

print(f"\n🎯 Top 5 Affected Regions:")
for i, region in enumerate(anat['affected_regions'][:5], 1):
    print(f"   {i}. {region['region_name']}")
    print(f"      - Involvement: {region['percentage_involvement']:.1f}%")
    print(f"      - Volume: {region['tumor_volume_in_region_mm3']:.1f} mm³")

print(f"\n🔬 Model Info:")
model = descriptor['model_metadata']
print(f"   Model: {model['model_name']} {model['model_version']}")
print(f"   Timestamp: {model['inference_timestamp']}")

print("\n" + "=" * 70)
print("✅ TEST COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nOutput files:")
print(f"  - Segmentation: {seg_path}")
print(f"  - JSON Descriptor: {output_path}")
print("\nYou can now:")
print("  1. Inspect the JSON file")
print("  2. Use it as input for report generation")
print("  3. Test with real segmentation files")
print("=" * 70)
