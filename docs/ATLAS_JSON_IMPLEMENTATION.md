# Atlas-JSON Mapping Implementation

## Overview

This implementation provides **Stage 2 (Atlas Mapping)** and **Stage 3 (JSON Generation)** of the brain tumor analysis pipeline. It converts segmentation masks into structured JSON descriptors by mapping tumor regions to anatomical brain structures.

---

## Directory Structure

```
├── src/
│   ├── atlas_mapping/          # Atlas mapping module
│   │   ├── __init__.py
│   │   ├── atlas_mapper.py     # Main mapper class
│   │   ├── registration.py     # Registration utilities
│   │   ├── region_analysis.py  # Region overlap analysis
│   │   └── atlas_data.py       # Atlas data management
│   │
│   └── json_generation/        # JSON generation module
│       ├── __init__.py
│       ├── descriptor_generator.py  # Main generator class
│       └── schema_validator.py      # Schema validation
│
├── schemas/
│   └── tumor_descriptor_schema.json  # JSON schema definition
│
└── scripts/
    └── generate_atlas_json.py  # CLI tool
```

---

## Installation

### 1. Install Dependencies

```bash
# Install atlas mapping and JSON generation dependencies
pip install nibabel nilearn scipy scikit-image jsonschema

# Or install all dependencies
pip install -r requirements_atlas_llm.txt
```

### 2. (Optional) Install ANTs for Non-linear Registration

For production-quality registration, install ANTs:

```bash
# Option 1: Using conda (recommended)
conda install -c conda-forge ants

# Option 2: Using pip (Python wrapper)
pip install antspyx
```

Without ANTs, the system will use affine registration (faster but less accurate).

---

## Quick Start

### Basic Usage

```python
from src.atlas_mapping import BrainAtlasMapper
from src.json_generation import TumorDescriptorGenerator

# Initialize atlas mapper
atlas_mapper = BrainAtlasMapper(
    atlas_name='harvard_oxford',
    use_ants=False  # Set to True if ANTs is installed
)

# Initialize JSON generator
generator = TumorDescriptorGenerator(atlas_mapper)

# Generate descriptor
descriptor = generator.generate_descriptor(
    case_id='BraTS_00123',
    seg_path='path/to/segmentation.nii.gz',
    t1_path='path/to/t1.nii.gz',  # Optional, required for ANTs
    patient_metadata={'age': 58, 'sex': 'M'}
)

# Save descriptor
import json
with open('output_descriptor.json', 'w') as f:
    json.dump(descriptor, f, indent=2)
```

### CLI Usage

```bash
# Basic usage with affine registration
python scripts/generate_atlas_json.py \
    --seg_path data/BraTS_00123_seg.nii.gz \
    --case_id BraTS_00123 \
    --output_dir ./output/descriptors

# With ANTs registration
python scripts/generate_atlas_json.py \
    --seg_path data/BraTS_00123_seg.nii.gz \
    --t1_path data/BraTS_00123_t1.nii.gz \
    --case_id BraTS_00123 \
    --use_ants \
    --patient_age 58 \
    --patient_sex M \
    --output_dir ./output/descriptors
```

---

## Features

### Atlas Mapping

✅ **Multiple Atlases**
- Harvard-Oxford (cortical + subcortical)
- AAL3 (extensible)
  
✅ **Registration Methods**
- Affine (fast, ~10 seconds)
- ANTs SyN (accurate, ~2-3 minutes)

✅ **Multi-label Analysis**
- Separate analysis for edema, tumor core, enhancing regions
- Percentage involvement calculation
- Volume quantification

✅ **Anatomical Features**
- Hemisphere determination
- Midline crossing detection
- Top affected regions ranking

### JSON Generation

✅ **Schema-Compliant**
- Validates against JSON schema
- Consistent structure
- LLM-friendly format

✅ **Comprehensive Data**
- Patient metadata
- Imaging parameters
- Volumetric analysis
- Anatomical mapping
- Model metadata
- Clinical features

✅ **Extensible**
- Optional fields
- Future-proof design
- Version-controlled

---

## Output Format

The generated JSON includes:

```json
{
  "patient_info": {
    "case_id": "BraTS_00123",
    "age": 58,
    "sex": "M",
    "scan_date": "2026-01-29"
  },
  "segmentation_results": {
    "tumor_components": {...},
    "volumetric_analysis": {
      "total_tumor_volume_mm3": 27341.3,
      "necrosis_percentage": 27.5,
      ...
    }
  },
  "anatomical_mapping": {
    "atlas_name": "harvard_oxford",
    "hemisphere": "right",
    "crossing_midline": false,
    "affected_regions": [
      {
        "region_name": "Middle Frontal Gyrus",
        "percentage_involvement": 42.3,
        "tumor_volume_in_region_mm3": 11234.5
      },
      ...
    ],
    "subregion_mapping": {
      "enhancing_tumor": [...],
      "peritumoral_edema": [...]
    }
  },
  "model_metadata": {...},
  "clinical_features": {...}
}
```

---

## Advanced Usage

### Custom Atlas

```python
atlas_mapper = BrainAtlasMapper(
    atlas_path='custom_atlas/MNI152_T1_1mm.nii.gz',
    atlas_labels_path='custom_atlas/custom_labels.nii.gz',
    atlas_name='harvard_oxford',
    use_ants=True
)
```

### Batch Processing

```python
import glob
from pathlib import Path

seg_files = glob.glob('data/*_seg.nii.gz')

for seg_path in seg_files:
    case_id = Path(seg_path).stem.replace('_seg', '')
    
    try:
        descriptor = generator.generate_descriptor(
            case_id=case_id,
            seg_path=seg_path
        )
        
        output_path = f'output/{case_id}_descriptor.json'
        with open(output_path, 'w') as f:
            json.dump(descriptor, f, indent=2)
        
        print(f"✓ Processed {case_id}")
    except Exception as e:
        print(f"✗ Failed {case_id}: {e}")
```

### Integration with Segmentation Pipeline

```python
from src.models import load_model, predict  # Your existing code
from src.atlas_mapping import BrainAtlasMapper
from src.json_generation import TumorDescriptorGenerator

# Initialize
model = load_model('path/to/checkpoint.pth')
atlas_mapper = BrainAtlasMapper()
generator = TumorDescriptorGenerator(atlas_mapper)

# Full pipeline
for case in dataset:
    # Stage 1: Segmentation (existing)
    segmentation = predict(model, case['images'])
    
    # Save segmentation
    seg_path = f'predictions/{case["id"]}_seg.nii.gz'
    save_segmentation(segmentation, seg_path)
    
    # Stage 2 & 3: Atlas mapping + JSON generation (new)
    descriptor = generator.generate_descriptor(
        case_id=case['id'],
        seg_path=seg_path,
        t1_path=case['t1_path']
    )
    
    # Save descriptor
    json_path = f'descriptors/{case["id"]}_descriptor.json'
    with open(json_path, 'w') as f:
        json.dump(descriptor, f, indent=2)
```

---

## Performance

| Method | Registration Time | Accuracy | Recommendation |
|--------|------------------|----------|----------------|
| Affine | ~10 seconds | Good | Development, quick testing |
| ANTs SyN | ~2-3 minutes | Excellent | Production, research |

**Tips:**
- Use affine for rapid prototyping
- Use ANTs for final results
- Consider GPU acceleration for ANTs (requires compilation)

---

## Troubleshooting

### Issue: ANTs not found

**Solution:** Install ANTs or use affine registration
```python
atlas_mapper = BrainAtlasMapper(use_ants=False)
```

### Issue: Atlas download fails

**Solution:** Manual atlas download
```python
from nilearn import datasets
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
print(atlas.maps)  # Use this path
```

### Issue: Memory error with large datasets

**Solution:** Process in smaller batches or use affine registration

### Issue: Validation errors

**Solution:** Check schema compliance
```python
from src.json_generation import validate_descriptor

try:
    validate_descriptor(descriptor)
except Exception as e:
    print(f"Validation error: {e}")
```

---

## Next Steps

1. **Generate Synthetic Data** → See `SYNTHETIC_DATA_GUIDE.md`
2. **Train Report Generation LLM** → See `LLM_FINETUNING_GUIDE.md`
3. **Integrate into Production Pipeline** → See implementation examples above

---

## References

- [Harvard-Oxford Atlas](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases)
- [ANTs Registration](http://stnava.github.io/ANTs/)
- [Nilearn Documentation](https://nilearn.github.io/)
- [JSON Schema](https://json-schema.org/)

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Consult the reference guides in `docs/`
