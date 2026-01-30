"""
Test atlas-JSON mapping with BraTS-GLI-00006-100-seg.nii.gz
"""

import json
from pathlib import Path

print("=" * 70)
print("Atlas-JSON Mapping Test for BraTS-GLI-00006-100")
print("=" * 70)

# Your mask file
MASK_PATH = "BraTS-GLI-00006-100-seg.nii.gz"
CASE_ID = "BraTS-GLI-00006-100"

print(f"\n📁 Segmentation mask: {MASK_PATH}")
print(f"🆔 Case ID: {CASE_ID}")

# Check if file exists
if not Path(MASK_PATH).exists():
    print(f"\n❌ ERROR: Mask file not found at {MASK_PATH}")
    print("   Make sure you're running this script from the project root directory.")
    exit(1)

print("   ✓ Mask file found")

# Step 1: Import modules
print("\n[1/4] Importing modules...")
try:
    from src.atlas_mapping import BrainAtlasMapper
    from src.json_generation import TumorDescriptorGenerator
    print("   ✓ Modules imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\n   Install dependencies: pip install nibabel nilearn scipy scikit-image jsonschema")
    exit(1)

# Step 2: Initialize mapper
print("\n[2/4] Initializing atlas mapper...")
print("   (First run will download Harvard-Oxford atlas ~50MB)")
try:
    mapper = BrainAtlasMapper(
        atlas_name='harvard_oxford',
        use_ants=False  # Use fast affine registration
    )
    print("   ✓ Atlas mapper ready")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Generate descriptor
print("\n[3/4] Processing segmentation and generating JSON...")
print("   This may take 10-30 seconds...")
try:
    generator = TumorDescriptorGenerator(mapper)
    
    descriptor = generator.generate_descriptor(
        case_id=CASE_ID,
        seg_path=MASK_PATH,
        patient_metadata=None  # Add if you have: {'age': 65, 'sex': 'M'}
    )
    print("   ✓ JSON descriptor generated")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Save results
print("\n[4/4] Saving results...")
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f'{CASE_ID}_descriptor.json'

with open(output_path, 'w') as f:
    json.dump(descriptor, f, indent=2)
print(f"   ✓ Saved to: {output_path}")

# Display results
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY")
print("=" * 70)

print(f"\n🔹 Case Information:")
print(f"   Case ID: {descriptor['patient_info']['case_id']}")

print(f"\n🔹 Volumetric Analysis:")
vol = descriptor['segmentation_results']['volumetric_analysis']
print(f"   Total tumor volume: {vol['total_tumor_volume_mm3']:.1f} mm³")
print(f"   Tumor core volume: {vol.get('tumor_core_volume_mm3', 0):.1f} mm³")
print(f"   Enhancing volume: {vol.get('enhancing_volume_mm3', 0):.1f} mm³")
print(f"   Necrosis: {vol.get('necrosis_percentage', 0):.1f}%")

print(f"\n🔹 Tumor Components:")
components = descriptor['segmentation_results']['tumor_components']
if components['enhancing_tumor']['present']:
    print(f"   ✓ Enhancing tumor: {components['enhancing_tumor']['volume_mm3']:.1f} mm³")
if components['necrotic_core']['present']:
    print(f"   ✓ Necrotic core: {components['necrotic_core']['volume_mm3']:.1f} mm³")
if components['peritumoral_edema']['present']:
    print(f"   ✓ Peritumoral edema: {components['peritumoral_edema']['volume_mm3']:.1f} mm³")

print(f"\n🔹 Anatomical Mapping:")
anat = descriptor['anatomical_mapping']
print(f"   Atlas: {anat['atlas_name']}")
print(f"   Registration: {anat['registration_method']}")
print(f"   Hemisphere: {anat.get('hemisphere', 'N/A')}")
print(f"   Crosses midline: {anat.get('crossing_midline', False)}")
print(f"   Number of affected regions: {len(anat['affected_regions'])}")

print(f"\n🔹 Top 10 Affected Brain Regions:")
for i, region in enumerate(anat['affected_regions'][:10], 1):
    print(f"   {i:2d}. {region['region_name']:40s} - {region['percentage_involvement']:5.1f}% ({region['tumor_volume_in_region_mm3']:8.1f} mm³)")

if 'subregion_mapping' in anat:
    print(f"\n🔹 Subregion Analysis:")
    for component, regions in anat['subregion_mapping'].items():
        if regions:
            print(f"\n   {component.replace('_', ' ').title()}:")
            for i, region in enumerate(regions[:3], 1):
                print(f"      {i}. {region['region_name']}: {region['percentage_involvement']:.1f}%")

print("\n" + "=" * 70)
print("✅ ATLAS-JSON MAPPING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nOutput file: {output_path}")
print("\nNext steps:")
print("  1. Review the JSON file for accuracy")
print("  2. Use this descriptor for LLM-based report generation")
print("  3. Integrate into your pipeline")
print("=" * 70)
