# Brain Atlas Mapping Guide

## System Architecture Overview

The complete pipeline for converting brain tumor segmentation masks to clinical reports consists of four modular stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: SEGMENTATION (Existing)                                   │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  MRI Input → MoME+ Model → Segmentation Masks          │        │
│  │  Output: WT, TC, ET masks + confidence scores          │        │
│  └────────────────────────────────────────────────────────┘        │
│                            ↓                                         │
│  Stage 2: ATLAS MAPPING (New)                                       │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  1. Normalization to MNI152 standard space             │        │
│  │  2. Overlap calculation with brain regions             │        │
│  │  3. Affected region extraction                         │        │
│  │  4. Percentage involvement computation                 │        │
│  └────────────────────────────────────────────────────────┘        │
│                            ↓                                         │
│  Stage 3: JSON GENERATION (New)                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Structured anatomical descriptor creation             │        │
│  │  - Regions + involvement percentages                   │        │
│  │  - Tumor characteristics (core, edema, necrosis)       │        │
│  │  - Confidence scores + metadata                        │        │
│  └────────────────────────────────────────────────────────┘        │
│                            ↓                                         │
│  Stage 4: REPORT GENERATION (New)                                   │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  JSON → MedGemma-4B → Clinical Report                  │        │
│  │  - Instruction-tuned LLM                               │        │
│  │  - Template-guided generation                          │        │
│  │  - Factual consistency verification                    │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 2: Brain Atlas Mapping Implementation

### 2.1 Overview

Brain atlas mapping transforms voxel-level segmentation masks into anatomically meaningful region labels. We use the **Harvard-Oxford cortical and subcortical atlases** (available in FSL) for glioma analysis.

### 2.2 Prerequisites

**Required Libraries:**
```bash
pip install nibabel numpy scipy scikit-image nilearn
```

**Atlas Data:**
- Download Harvard-Oxford atlas from FSL or use `nilearn` datasets
- Alternative: AAL3, Julich-Brain, or MNI152 atlases

### 2.3 Registration to Atlas Space

**Two Approaches:**

#### Option A: Affine Registration (Faster, Less Accurate)
```python
from nilearn.image import resample_to_img
import nibabel as nib

def register_to_atlas_affine(segmentation_path, atlas_path):
    """
    Resample segmentation mask to atlas space using affine transformation.
    
    Args:
        segmentation_path: Path to segmentation mask (.nii.gz)
        atlas_path: Path to brain atlas (.nii.gz)
    
    Returns:
        Nibabel image object in atlas space
    """
    seg_img = nib.load(segmentation_path)
    atlas_img = nib.load(atlas_path)
    
    # Resample to atlas space
    seg_resampled = resample_to_img(
        seg_img, 
        atlas_img, 
        interpolation='nearest'  # Preserve label values
    )
    
    return seg_resampled
```

#### Option B: Non-linear Registration (ANTs - Production Quality)
```python
import subprocess
import os

def register_to_atlas_ants(
    segmentation_path, 
    t1_reference_path,
    atlas_template_path,
    output_dir
):
    """
    Use ANTs for precise non-linear registration.
    
    Assumes you have ANTsPy or call ANTs command-line tools.
    """
    # Step 1: Register T1 scan to MNI152 template
    transform_prefix = os.path.join(output_dir, 'transform_')
    
    cmd = [
        'antsRegistrationSyNQuick.sh',
        '-d', '3',
        '-f', atlas_template_path,  # Fixed: MNI152 template
        '-m', t1_reference_path,     # Moving: Patient T1
        '-o', transform_prefix,
        '-t', 's'  # Use SyN (symmetric normalization)
    ]
    
    subprocess.run(cmd, check=True)
    
    # Step 2: Apply transformation to segmentation mask
    output_seg = os.path.join(output_dir, 'seg_in_atlas_space.nii.gz')
    
    cmd_transform = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', segmentation_path,
        '-r', atlas_template_path,
        '-o', output_seg,
        '-n', 'NearestNeighbor',  # Preserve discrete labels
        '-t', f'{transform_prefix}1Warp.nii.gz',
        '-t', f'{transform_prefix}0GenericAffine.mat'
    ]
    
    subprocess.run(cmd_transform, check=True)
    
    return nib.load(output_seg)
```

### 2.4 Overlap Calculation with Brain Regions

```python
import numpy as np
from collections import defaultdict

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


def get_region_names(atlas_name='harvard_oxford'):
    """
    Load region names from atlas lookup table.
    
    Returns:
        Dictionary mapping region IDs to anatomical names
    """
    if atlas_name == 'harvard_oxford':
        # This is a simplified mapping - use actual FSL XML files
        return {
            0: 'Background',
            1: 'Frontal Pole',
            2: 'Insular Cortex',
            3: 'Superior Frontal Gyrus',
            4: 'Middle Frontal Gyrus',
            5: 'Inferior Frontal Gyrus',
            6: 'Precentral Gyrus',
            7: 'Temporal Pole',
            8: 'Superior Temporal Gyrus',
            9: 'Middle Temporal Gyrus',
            10: 'Inferior Temporal Gyrus',
            # ... Add all 48 cortical regions
            # Subcortical regions:
            11: 'Amygdala',
            12: 'Hippocampus',
            13: 'Thalamus',
            14: 'Caudate',
            15: 'Putamen',
            # ... etc
        }
    else:
        raise ValueError(f"Atlas {atlas_name} not supported")
```

### 2.5 Quantify Percentage Involvement

```python
def calculate_percentage_involvement(seg_atlas_space, atlas_labels, region_names):
    """
    Compute percentage of each brain region affected by tumor.
    
    Returns:
        List of dicts with region info sorted by involvement
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
            'volume_mm3': int(tumor_voxels * np.prod(seg_atlas_space.header.get_zooms()))
        })
    
    # Sort by percentage involvement (descending)
    results.sort(key=lambda x: x['percentage_involvement'], reverse=True)
    
    return results
```

### 2.6 Multi-Label Segmentation Analysis

For gliomas, we have three sub-regions:
- **Edema (Label 2)**: Peritumoral edema
- **Tumor Core (Label 1)**: Non-enhancing tumor
- **Enhancing Tumor (Label 4)**: Active tumor with contrast enhancement

```python
def analyze_tumor_subregions(seg_atlas_space, atlas_labels, region_names):
    """
    Separate analysis for each tumor sub-component.
    
    Returns:
        Dictionary with analysis for each label
    """
    seg_data = seg_atlas_space.get_fdata()
    atlas_data = atlas_labels.get_fdata().astype(int)
    
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
                    'percentage': round(percentage, 2)
                })
        
        label_results.sort(key=lambda x: x['percentage'], reverse=True)
        results[label_name] = label_results[:10]  # Top 10 regions
    
    return results
```

### 2.7 Production-Ready Integration

```python
class BrainAtlasMapper:
    """
    Production class for brain atlas mapping pipeline.
    """
    def __init__(self, atlas_path, atlas_labels_path, use_ants=False):
        self.atlas_img = nib.load(atlas_path)
        self.atlas_labels = nib.load(atlas_labels_path)
        self.region_names = get_region_names('harvard_oxford')
        self.use_ants = use_ants
    
    def process_segmentation(
        self, 
        seg_path, 
        t1_reference_path=None,
        output_dir='./atlas_mapping_output'
    ):
        """
        Complete atlas mapping workflow.
        
        Returns:
            Dictionary with anatomical analysis
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Registration
        if self.use_ants and t1_reference_path:
            seg_atlas = register_to_atlas_ants(
                seg_path, 
                t1_reference_path,
                self.atlas_img,
                output_dir
            )
        else:
            seg_atlas = register_to_atlas_affine(seg_path, self.atlas_img)
        
        # Step 2: Multi-label analysis
        subregion_analysis = analyze_tumor_subregions(
            seg_atlas,
            self.atlas_labels,
            self.region_names
        )
        
        # Step 3: Overall tumor analysis
        overall_analysis = calculate_percentage_involvement(
            seg_atlas,
            self.atlas_labels,
            self.region_names
        )
        
        return {
            'subregion_analysis': subregion_analysis,
            'overall_affected_regions': overall_analysis[:15],  # Top 15
            'metadata': {
                'atlas': 'harvard_oxford',
                'registration': 'ANTs' if self.use_ants else 'affine',
                'total_tumor_volume_mm3': int(np.sum(seg_atlas.get_fdata() > 0) * 
                                               np.prod(seg_atlas.header.get_zooms()))
            }
        }
```

### 2.8 Handling Edge Cases

**Partial Volume Effects:**
```python
def apply_confidence_threshold(seg_data, threshold=0.5):
    """
    Only consider voxels with high confidence.
    Assumes probabilistic segmentation output.
    """
    return seg_data > threshold
```

**Noise Filtering:**
```python
from scipy.ndimage import binary_opening, binary_closing

def remove_small_components(mask, min_size=50):
    """
    Remove isolated voxel clusters.
    """
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    
    labeled = label(mask)
    cleaned = remove_small_objects(labeled, min_size=min_size)
    return cleaned > 0
```

---

## Next Steps

1. **JSON Schema Design** → See `JSON_SCHEMA_GUIDE.md`
2. **Synthetic Data Generation** → See `SYNTHETIC_DATA_GUIDE.md`
3. **MedGemma Fine-tuning** → See `LLM_FINETUNING_GUIDE.md`

## References

- [Harvard-Oxford Atlas Documentation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases)
- [ANTs Registration](http://stnava.github.io/ANTs/)
- [Nilearn Documentation](https://nilearn.github.io/)
