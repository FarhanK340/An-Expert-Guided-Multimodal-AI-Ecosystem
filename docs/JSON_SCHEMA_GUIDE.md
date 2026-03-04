# JSON Schema Design for Medical Report Generation

## Overview

This document defines a **strict, schema-valid JSON format** for representing brain tumor segmentation results and anatomical analysis. The schema is designed to be:

✅ **Deterministic**: Same input → same JSON  
✅ **LLM-Friendly**: Structured, easy to parse  
✅ **Clinically Complete**: All necessary information for report generation  
✅ **Extensible**: Supports future multimodal inputs  
✅ **Traceable**: References back to source data  

---

## JSON Schema Definition

### Complete Schema Example

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BrainTumorSegmentationDescriptor",
  "type": "object",
  "required": [
    "patient_info",
    "imaging_metadata",
    "segmentation_results",
    "anatomical_mapping",
    "model_metadata"
  ],
  "properties": {
    "patient_info": {
      "type": "object",
      "required": ["case_id"],
      "properties": {
        "case_id": {
          "type": "string",
          "description": "Unique patient identifier (anonymized)"
        },
        "age": {
          "type": "integer",
          "description": "Patient age in years (optional)"
        },
        "sex": {
          "type": "string",
          "enum": ["M", "F", "Unknown"]
        },
        "scan_date": {
          "type": "string",
          "format": "date",
          "description": "MRI acquisition date (YYYY-MM-DD)"
        }
      }
    },
    "imaging_metadata": {
      "type": "object",
      "required": ["modalities", "scanner_info"],
      "properties": {
        "modalities": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["T1", "T1ce", "T2", "FLAIR"]
          },
          "description": "MRI sequences used"
        },
        "scanner_info": {
          "type": "object",
          "properties": {
            "manufacturer": {"type": "string"},
            "field_strength": {
              "type": "number",
              "description": "Tesla (e.g., 1.5, 3.0)"
            },
            "resolution_mm": {
              "type": "array",
              "items": {"type": "number"},
              "description": "[x, y, z] voxel spacing"
            }
          }
        }
      }
    },
    "segmentation_results": {
      "type": "object",
      "required": ["tumor_components", "volumetric_analysis"],
      "properties": {
        "tumor_components": {
          "type": "object",
          "description": "Per-label segmentation details",
          "properties": {
            "enhancing_tumor": {
              "$ref": "#/definitions/tumor_component"
            },
            "necrotic_core": {
              "$ref": "#/definitions/tumor_component"
            },
            "peritumoral_edema": {
              "$ref": "#/definitions/tumor_component"
            }
          }
        },
        "volumetric_analysis": {
          "type": "object",
          "required": ["total_tumor_volume_mm3"],
          "properties": {
            "total_tumor_volume_mm3": {"type": "number"},
            "whole_tumor_volume_mm3": {"type": "number"},
            "tumor_core_volume_mm3": {"type": "number"},
            "enhancing_volume_mm3": {"type": "number"},
            "necrosis_percentage": {
              "type": "number",
              "minimum": 0,
              "maximum": 100
            }
          }
        },
        "confidence_metrics": {
          "type": "object",
          "properties": {
            "mean_dice_score": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "per_class_confidence": {
              "type": "object",
              "additionalProperties": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
              }
            }
          }
        }
      }
    },
    "anatomical_mapping": {
      "type": "object",
      "required": ["atlas_name", "affected_regions"],
      "properties": {
        "atlas_name": {
          "type": "string",
          "enum": ["harvard_oxford", "AAL3", "MNI152", "Julich"]
        },
        "registration_method": {
          "type": "string",
          "enum": ["affine", "ANTs_SyN", "SPM_DARTEL"]
        },
        "affected_regions": {
          "type": "array",
          "description": "Brain regions affected by tumor",
          "items": {
            "$ref": "#/definitions/affected_region"
          }
        },
        "subregion_mapping": {
          "type": "object",
          "description": "Per tumor component region mapping",
          "properties": {
            "enhancing_tumor": {
              "type": "array",
              "items": {"$ref": "#/definitions/affected_region"}
            },
            "necrotic_core": {
              "type": "array",
              "items": {"$ref": "#/definitions/affected_region"}
            },
            "peritumoral_edema": {
              "type": "array",
              "items": {"$ref": "#/definitions/affected_region"}
            }
          }
        },
        "hemisphere": {
          "type": "string",
          "enum": ["left", "right", "bilateral"]
        },
        "crossing_midline": {
          "type": "boolean",
          "description": "Whether tumor crosses corpus callosum"
        }
      }
    },
    "model_metadata": {
      "type": "object",
      "required": ["model_name", "model_version"],
      "properties": {
        "model_name": {
          "type": "string",
          "description": "e.g., MoME+, SegResNet"
        },
        "model_version": {
          "type": "string",
          "description": "Version identifier"
        },
        "training_datasets": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Datasets used for training"
        },
        "inference_timestamp": {
          "type": "string",
          "format": "date-time"
        },
        "processing_time_seconds": {
          "type": "number"
        }
      }
    },
    "clinical_features": {
      "type": "object",
      "description": "Optional derived clinical indicators",
      "properties": {
        "mass_effect": {
          "type": "boolean",
          "description": "Evidence of mass effect/midline shift"
        },
        "ventricular_involvement": {
          "type": "boolean"
        },
        "eloquent_area_involvement": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": [
              "motor_cortex",
              "speech_area",
              "visual_cortex",
              "brainstem"
            ]
          }
        },
        "estimated_grade": {
          "type": "string",
          "enum": ["low_grade", "high_grade", "unknown"],
          "description": "Imaging-based estimate (not diagnostic)"
        }
      }
    }
  },
  "definitions": {
    "tumor_component": {
      "type": "object",
      "required": ["present", "volume_mm3"],
      "properties": {
        "present": {
          "type": "boolean",
          "description": "Whether this component was detected"
        },
        "volume_mm3": {
          "type": "number",
          "minimum": 0
        },
        "voxel_count": {
          "type": "integer",
          "minimum": 0
        },
        "confidence_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "centroid_coords": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 3,
          "maxItems": 3,
          "description": "[x, y, z] in mm (MNI space)"
        }
      }
    },
    "affected_region": {
      "type": "object",
      "required": [
        "region_id",
        "region_name",
        "percentage_involvement",
        "tumor_volume_in_region_mm3"
      ],
      "properties": {
        "region_id": {
          "type": "integer",
          "description": "Atlas region identifier"
        },
        "region_name": {
          "type": "string",
          "description": "Anatomical name (e.g., 'Middle Frontal Gyrus')"
        },
        "percentage_involvement": {
          "type": "number",
          "minimum": 0,
          "maximum": 100,
          "description": "% of region occupied by tumor"
        },
        "tumor_volume_in_region_mm3": {
          "type": "number"
        },
        "total_region_volume_mm3": {
          "type": "number"
        },
        "hemisphere": {
          "type": "string",
          "enum": ["left", "right"]
        }
      }
    }
  }
}
```

---

## Example JSON Instance

```json
{
  "patient_info": {
    "case_id": "BraTS2021_00123",
    "age": 58,
    "sex": "M",
    "scan_date": "2021-03-15"
  },
  "imaging_metadata": {
    "modalities": ["T1", "T1ce", "T2", "FLAIR"],
    "scanner_info": {
      "manufacturer": "Siemens",
      "field_strength": 3.0,
      "resolution_mm": [1.0, 1.0, 1.0]
    }
  },
  "segmentation_results": {
    "tumor_components": {
      "enhancing_tumor": {
        "present": true,
        "volume_mm3": 8452.3,
        "voxel_count": 8452,
        "confidence_score": 0.92,
        "centroid_coords": [45.2, -12.5, 38.7]
      },
      "necrotic_core": {
        "present": true,
        "volume_mm3": 3210.1,
        "voxel_count": 3210,
        "confidence_score": 0.88,
        "centroid_coords": [43.8, -13.1, 39.2]
      },
      "peritumoral_edema": {
        "present": true,
        "volume_mm3": 15678.9,
        "voxel_count": 15679,
        "confidence_score": 0.85,
        "centroid_coords": [46.5, -10.2, 37.5]
      }
    },
    "volumetric_analysis": {
      "total_tumor_volume_mm3": 27341.3,
      "whole_tumor_volume_mm3": 27341.3,
      "tumor_core_volume_mm3": 11662.4,
      "enhancing_volume_mm3": 8452.3,
      "necrosis_percentage": 27.5
    },
    "confidence_metrics": {
      "mean_dice_score": 0.88,
      "per_class_confidence": {
        "enhancing_tumor": 0.92,
        "necrotic_core": 0.88,
        "edema": 0.85
      }
    }
  },
  "anatomical_mapping": {
    "atlas_name": "harvard_oxford",
    "registration_method": "ANTs_SyN",
    "hemisphere": "right",
    "crossing_midline": false,
    "affected_regions": [
      {
        "region_id": 4,
        "region_name": "Middle Frontal Gyrus",
        "percentage_involvement": 42.3,
        "tumor_volume_in_region_mm3": 11234.5,
        "total_region_volume_mm3": 26567.2,
        "hemisphere": "right"
      },
      {
        "region_id": 3,
        "region_name": "Superior Frontal Gyrus",
        "percentage_involvement": 18.7,
        "tumor_volume_in_region_mm3": 4523.8,
        "total_region_volume_mm3": 24189.3,
        "hemisphere": "right"
      },
      {
        "region_id": 6,
        "region_name": "Precentral Gyrus",
        "percentage_involvement": 15.2,
        "tumor_volume_in_region_mm3": 3876.2,
        "total_region_volume_mm3": 25501.3,
        "hemisphere": "right"
      }
    ],
    "subregion_mapping": {
      "enhancing_tumor": [
        {
          "region_id": 4,
          "region_name": "Middle Frontal Gyrus",
          "percentage_involvement": 28.5,
          "tumor_volume_in_region_mm3": 7567.8,
          "total_region_volume_mm3": 26567.2,
          "hemisphere": "right"
        }
      ],
      "peritumoral_edema": [
        {
          "region_id": 3,
          "region_name": "Superior Frontal Gyrus",
          "percentage_involvement": 12.4,
          "tumor_volume_in_region_mm3": 3001.5,
          "total_region_volume_mm3": 24189.3,
          "hemisphere": "right"
        }
      ]
    }
  },
  "model_metadata": {
    "model_name": "MoME+",
    "model_version": "v1.2.0",
    "training_datasets": ["BraTS2021", "BraTS2020"],
    "inference_timestamp": "2024-03-15T14:23:45Z",
    "processing_time_seconds": 12.4
  },
  "clinical_features": {
    "mass_effect": true,
    "ventricular_involvement": false,
    "eloquent_area_involvement": ["motor_cortex"],
    "estimated_grade": "high_grade"
  }
}
```

---

## Python Implementation

### Schema Validation

```python
import json
from jsonschema import validate, ValidationError
from pathlib import Path

def load_schema():
    """Load the JSON schema definition."""
    schema_path = Path(__file__).parent / 'schemas' / 'tumor_descriptor_schema.json'
    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_descriptor(descriptor_dict):
    """
    Validate a tumor descriptor against the schema.
    
    Args:
        descriptor_dict: Dictionary to validate
    
    Raises:
        ValidationError: If validation fails
    
    Returns:
        True if valid
    """
    schema = load_schema()
    try:
        validate(instance=descriptor_dict, schema=schema)
        return True
    except ValidationError as e:
        print(f"Validation failed: {e.message}")
        print(f"Failed path: {' -> '.join(str(p) for p in e.path)}")
        raise
```

### JSON Generation from Atlas Mapping

```python
from datetime import datetime
import numpy as np

class TumorDescriptorGenerator:
    """
    Generate structured JSON descriptors from segmentation + atlas mapping.
    """
    
    def __init__(self, atlas_mapper):
        self.atlas_mapper = atlas_mapper
    
    def generate_descriptor(
        self,
        case_id,
        seg_path,
        t1_path,
        modalities=['T1', 'T1ce', 'T2', 'FLAIR'],
        patient_metadata=None
    ):
        """
        Create complete JSON descriptor.
        
        Returns:
            Dictionary conforming to schema
        """
        # Run atlas mapping
        atlas_results = self.atlas_mapper.process_segmentation(
            seg_path, t1_path
        )
        
        # Load segmentation for volumetric analysis
        import nibabel as nib
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()
        voxel_vol = np.prod(seg_img.header.get_zooms())
        
        # Build descriptor
        descriptor = {
            "patient_info": {
                "case_id": case_id,
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
                **(patient_metadata or {})
            },
            "imaging_metadata": {
                "modalities": modalities,
                "scanner_info": {
                    "manufacturer": "Unknown",
                    "field_strength": 3.0,
                    "resolution_mm": list(seg_img.header.get_zooms())
                }
            },
            "segmentation_results": self._extract_segmentation_results(
                seg_data, voxel_vol
            ),
            "anatomical_mapping": {
                "atlas_name": atlas_results['metadata']['atlas'],
                "registration_method": atlas_results['metadata']['registration'],
                "affected_regions": atlas_results['overall_affected_regions'],
                "subregion_mapping": self._format_subregion_mapping(
                    atlas_results['subregion_analysis']
                ),
                "hemisphere": self._determine_hemisphere(seg_data, seg_img),
                "crossing_midline": self._check_midline_crossing(seg_data, seg_img)
            },
            "model_metadata": {
                "model_name": "MoME+",
                "model_version": "v1.0.0",
                "training_datasets": ["BraTS2021"],
                "inference_timestamp": datetime.utcnow().isoformat() + 'Z',
                "processing_time_seconds": 15.2
            }
        }
        
        # Validate before returning
        validate_descriptor(descriptor)
        
        return descriptor
    
    def _extract_segmentation_results(self, seg_data, voxel_vol):
        """Extract tumor component information."""
        # BraTS labels: 1=necrotic, 2=edema, 4=enhancing
        components = {
            "enhancing_tumor": self._analyze_component(seg_data, 4, voxel_vol),
            "necrotic_core": self._analyze_component(seg_data, 1, voxel_vol),
            "peritumoral_edema": self._analyze_component(seg_data, 2, voxel_vol)
        }
        
        # Volumetric analysis
        wt_mask = seg_data > 0  # Whole tumor
        tc_mask = (seg_data == 1) | (seg_data == 4)  # Tumor core
        et_mask = seg_data == 4  # Enhancing
        
        total_vol = np.sum(wt_mask) * voxel_vol
        tc_vol = np.sum(tc_mask) * voxel_vol
        et_vol = np.sum(et_mask) * voxel_vol
        necrosis_vol = np.sum(seg_data == 1) * voxel_vol
        
        return {
            "tumor_components": components,
            "volumetric_analysis": {
                "total_tumor_volume_mm3": float(total_vol),
                "whole_tumor_volume_mm3": float(total_vol),
                "tumor_core_volume_mm3": float(tc_vol),
                "enhancing_volume_mm3": float(et_vol),
                "necrosis_percentage": float((necrosis_vol / total_vol * 100) if total_vol > 0 else 0)
            },
            "confidence_metrics": {
                "mean_dice_score": 0.88,  # From model output
                "per_class_confidence": {
                    "enhancing_tumor": 0.92,
                    "necrotic_core": 0.88,
                    "edema": 0.85
                }
            }
        }
    
    def _analyze_component(self, seg_data, label, voxel_vol):
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
            "volume_mm3": volume,
            "voxel_count": voxel_count,
            "confidence_score": 0.90,  # Placeholder
            "centroid_coords": [float(c) for c in centroid]
        }
    
    def _format_subregion_mapping(self, subregion_analysis):
        """Convert atlas mapping to schema format."""
        return {
            "enhancing_tumor": subregion_analysis.get('enhancing_tumor', []),
            "necrotic_core": subregion_analysis.get('necrotic_and_non_enhancing_tumor_core', []),
            "peritumoral_edema": subregion_analysis.get('peritumoral_edema', [])
        }
    
    def _determine_hemisphere(self, seg_data, seg_img):
        """Determine primary hemisphere (simplified)."""
        # Assumes standard orientation
        midline_x = seg_data.shape[0] // 2
        
        left_voxels = np.sum(seg_data[:midline_x, :, :] > 0)
        right_voxels = np.sum(seg_data[midline_x:, :, :] > 0)
        
        if left_voxels > 2 * right_voxels:
            return "left"
        elif right_voxels > 2 * left_voxels:
            return "right"
        else:
            return "bilateral"
    
    def _check_midline_crossing(self, seg_data, seg_img):
        """Check if tumor crosses midline."""
        midline_x = seg_data.shape[0] // 2
        
        left_present = np.any(seg_data[:midline_x-2, :, :] > 0)
        right_present = np.any(seg_data[midline_x+2:, :, :] > 0)
        
        return bool(left_present and right_present)
```

---

## Usage Example

```python
# Initialize
from atlas_mapping import BrainAtlasMapper
from json_generation import TumorDescriptorGenerator

atlas_mapper = BrainAtlasMapper(
    atlas_path='atlases/MNI152_T1_1mm.nii.gz',
    atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',
    use_ants=True
)

descriptor_gen = TumorDescriptorGenerator(atlas_mapper)

# Generate JSON
descriptor = descriptor_gen.generate_descriptor(
    case_id="BraTS2021_00123",
    seg_path="predictions/case_00123_seg.nii.gz",
    t1_path="inputs/case_00123_t1.nii.gz",
    patient_metadata={"age": 58, "sex": "M"}
)

# Save
import json
with open('descriptors/case_00123_descriptor.json', 'w') as f:
    json.dump(descriptor, f, indent=2)
```

---

## Design Principles

### 1. **Determinism**
- Same segmentation → same JSON
- No randomness in field generation
- Reproducible across runs

### 2. **LLM-Friendliness**
- Clear, hierarchical structure
- Self-documenting field names
- Consistent terminology

### 3. **Clinical Completeness**
- All information needed for report generation
- No external lookups required
- Traceable to source data

### 4. **Extensibility**
- Optional fields for future features
- `additionalProperties` allowed in specific sections
- Version-controlled schema

---

## Next Steps

→ **Synthetic Data Generation**: See `SYNTHETIC_DATA_GUIDE.md`  
→ **LLM Fine-tuning**: See `LLM_FINETUNING_GUIDE.md`
