# Atlas-JSON Mapping: Complete Guide

## 📋 Overview

The **Atlas-JSON Mapping Pipeline** converts brain tumor segmentation masks into structured, anatomically-enriched JSON descriptors by mapping tumor voxels to specific brain regions.

---

## 🔄 How It Works: Step-by-Step Pipeline

### **Stage 1: Input Processing**
```
Input: Brain tumor segmentation mask (NIfTI format)
- Example: BraTS-GLI-00006-100-seg.nii.gz
- Contains labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
- Original space: Patient-specific coordinates
- Dimensions: Typically 240x240x155 or similar
```

### **Stage 2: Atlas Registration**
```
Process: Align segmentation to standard brain atlas space

Methods:
1. AFFINE (Fast, ~10 seconds)
   - Linear transformation (rotation, translation, scaling)
   - Resamples segmentation to match atlas dimensions
   - Good for: Quick prototyping, clinical use

2. ANTs SyN (Accurate, ~2-3 minutes)  
   - Non-linear deformation
   - Better anatomical precision
   - Good for: Research, detailed analysis

Output: Segmentation in atlas space (182x218x182 for Harvard-Oxford)
```

### **Stage 3: Region Overlap Analysis**
```
Process: Calculate which brain regions contain tumor

For each atlas region:
1. Count tumor voxels in that region
2. Count total voxels in that region
3. Calculate percentage involvement
4. Compute volume in mm³

Algorithm:
- Voxel-wise comparison: tumor_mask AND region_mask
- Handles multi-label segmentations (edema, core, enhancing)
- Assigns anatomical names from atlas lookup table
```

### **Stage 4: JSON Generation**
```
Process: Structure results into schema-compliant JSON

Includes:
- Patient metadata (case ID, demographics)
- Volumetric analysis (total volume, tumor subcomponents)
- Anatomical mapping (affected regions with percentages)
- Model metadata (inference timestamp, version)
- Clinical features (hemisphere, midline crossing)
```

---

## 📊 What Results Does It Show?

### **1. Volumetric Measurements**

```json
"volumetric_analysis": {
  "total_tumor_volume_mm3": 39981.0,
  "tumor_core_volume_mm3": 22074.0,
  "enhancing_volume_mm3": 22074.0,
  "necrosis_percentage": 0.0
}
```

**Interpretation:**
- Total tumor = All labeled voxels combined
- Tumor core = Necrotic + Enhancing regions
- These volumes help assess tumor burden

---

### **2. Tumor Component Analysis**

```json
"tumor_components": {
  "enhancing_tumor": {
    "present": true,
    "volume_mm3": 22074.0,
    "voxel_count": 22074,
    "confidence_score": 0.9,
    "centroid_coords": [95.5, 102.0, 80.5]
  },
  "peritumoral_edema": {...},
  "necrotic_core": {...}
}
```

**Interpretation:**
- Separate analysis for each tumor type (edema, core, enhancing)
- Centroid = Center of mass coordinates
- Useful for surgical planning

---

### **3. Anatomical Region Mapping** (Most Important!)

```json
"affected_regions": [
  {
    "region_id": 28,
    "region_name": "Paracingulate Gyrus",
    "percentage_involvement": 15.1,
    "tumor_volume_in_region_mm3": 3517.0,
    "total_voxels": 23265
  },
  {
    "region_id": 1,
    "region_name": "Frontal Pole",
    "percentage_involvement": 10.6,
    "tumor_volume_in_region_mm3": 12768.0
  }
]
```

**Interpretation:**
- **Paracingulate Gyrus**: 15.1% of this region contains tumor (3,517 mm³)
- **Frontal Pole**: 10.6% affected (12,768 mm³ - larger region)
- Sorted by percentage involvement (most affected first)

**Clinical Significance:**
- High % in small region = locally aggressive
- Low % in large region = widespread infiltration
- Helps predict functional deficits

---

### **4. Subregion Mapping** (Advanced)

```json
"subregion_mapping": {
  "enhancing_tumor": [
    {"region_name": "Paracingulate Gyrus", "percentage": 13.2%},
    {"region_name": "Frontal Pole", "percentage": 7.8%}
  ],
  "peritumoral_edema": [
    {"region_name": "Frontal Operculum Cortex", "percentage": 4.1%},
    {"region_name": "Frontal Pole", "percentage": 2.7%}
  ]
}
```

**Interpretation:**
- Shows WHERE each tumor component is located
- Enhancing tumor mainly in Paracingulate Gyrus
- Edema spreading to Frontal Operculum
- Useful for treatment planning (edema vs. solid tumor)

---

### **5. Spatial Features**

```json
"anatomical_mapping": {
  "hemisphere": "right",
  "crossing_midline": true
}
```

**Interpretation:**
- Primarily right hemisphere
- Crosses midline = Potentially involves corpus callosum
- Midline crossing = Higher surgical complexity

---

## ✅ How to Cross-Verify the JSON Mappings

### **Method 1: Visual Inspection** (Most Reliable)

**Using 3D Slicer or ITK-SNAP:**

1. Load files:
   ```
   - Atlas labels: C:\Users\Farhan\nilearn_data\fsl\...\HarvardOxford-cort-maxprob-thr25-1mm.nii.gz
   - Your segmentation: BraTS-GLI-00006-100-seg.nii.gz
   - MNI152 template (optional): for anatomical context
   ```

2. Overlay segmentation on atlas

3. Manually identify regions:
   - Look at where the tumor appears
   - Compare region labels to JSON output
   - Use atlas lookup table to match region IDs

4. Verify specific claims:
   - JSON says "Paracingulate Gyrus 15.1%" → Check if tumor is there
   - JSON says "crossing midline: true" → Visually confirm

**Expected Workflow:**
```
Open ITK-SNAP
→ Load atlas as main image
→ Load segmentation as overlay (red)
→ Click on tumor voxels
→ Check which atlas region number appears
→ Cross-reference with region_names in JSON
```

---

### **Method 2: Quantitative Verification**

**Using Python:**

```python
import nibabel as nib
import numpy as np

# Load files
seg = nib.load('BraTS-GLI-00006-100-seg.nii.gz')
atlas = nib.load('path/to/HarvardOxford-atlas.nii.gz')

# Register segmentation to atlas space (same as pipeline)
from nilearn.image import resample_to_img
seg_resampled = resample_to_img(seg, atlas, interpolation='nearest')

# Get data
seg_data = seg_resampled.get_fdata()
atlas_data = atlas.get_fdata().astype(int)

# Manually count voxels in region 28 (Paracingulate Gyrus)
region_id = 28
tumor_mask = seg_data > 0
region_mask = atlas_data == region_id

overlap = tumor_mask & region_mask
tumor_voxels_in_region = np.sum(overlap)
total_voxels_in_region = np.sum(region_mask)
percentage = (tumor_voxels_in_region / total_voxels_in_region) * 100

print(f"Region 28 involvement: {percentage:.1f}%")
# Should match JSON: "percentage_involvement": 15.1
```

---

### **Method 3: Sanity Checks**

**Check 1: Volume Conservation**
```python
# Total tumor volume should equal sum of components
total = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
enhancing = descriptor['segmentation_results']['tumor_components']['enhancing_tumor']['volume_mm3']
edema = descriptor['segmentation_results']['tumor_components']['peritumoral_edema']['volume_mm3']
necrotic = descriptor['segmentation_results']['tumor_components']['necrotic_core']['volume_mm3']

assert abs(total - (enhancing + edema + necrotic)) < 10  # Allow small rounding error
```

**Check 2: Percentage Bounds**
```python
# All percentages should be 0-100%
for region in descriptor['anatomical_mapping']['affected_regions']:
    assert 0 <= region['percentage_involvement'] <= 100
```

**Check 3: Region Sum**
```python
# Sum of all regional volumes should ≈ total tumor volume
regional_sum = sum(r['tumor_volume_in_region_mm3'] 
                   for r in descriptor['anatomical_mapping']['affected_regions'])
total_volume = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']

# Should be close (some voxels may be in unlabeled atlas regions)
print(f"Regional coverage: {regional_sum / total_volume * 100:.1f}%")
# Expect 80-100% (some atlas background is normal)
```

**Check 4: Hemisphere Logic**
```python
# If "crossing_midline" is true, should have regions in both hemispheres
if descriptor['anatomical_mapping']['crossing_midline']:
    # Count left vs right regions
    left_regions = [r for r in affected_regions if 'Left' in r['region_name']]
    right_regions = [r for r in affected_regions if 'Right' in r['region_name']]
    
    # Should have both (though Harvard-Oxford doesn't always specify L/R in names)
```

---

### **Method 4: Expert Review (Clinical Validation)**

**Involve a radiologist or neurosurgeon to:**

1. **Review the JSON output**
   - Does "Paracingulate Gyrus" make sense anatomically?
   - Is "Frontal Pole" consistent with imaging?

2. **Compare to radiology report**
   - JSON: "right hemisphere" → MRI report should confirm
   - JSON: "crossing midline" → Should mention corpus callosum

3. **Validate functional implications**
   - If motor cortex (Precentral Gyrus) is involved → Expect motor deficits
   - If speech areas involved → Expect language issues

---

## 🔍 Common Issues & How to Spot Them

### **Issue 1: Registration Failure**

**Symptoms:**
- Unexpected regions (e.g., tumor in cerebellum when it should be frontal)
- Very low percentage involvements across all regions
- Many "Background" or "Unknown" regions

**Verification:**
```python
# Check if shapes match after registration
print(f"Seg shape: {seg_atlas.shape}")
print(f"Atlas shape: {atlas.shape}")
# Should be identical!
```

---

### **Issue 2: Wrong Atlas Loaded**

**Symptoms:**
- Region names don't match anatomy
- Region IDs seem random

**Verification:**
```python
# Check which atlas was actually used
print(descriptor['anatomical_mapping']['atlas_name'])
# Should be: "harvard_oxford"
```

---

### **Issue 3: Label Mismatch**

**Symptoms:**
- "Necrotic core" shows 0 volume but you know there's necrosis
- Component volumes don't add up

**Verification:**
```python
# Check what labels are actually in your segmentation
unique_labels = np.unique(seg_data)
print(f"Labels in segmentation: {unique_labels}")
# BraTS standard: [0, 1, 2, 4]
# If you see [0, 1, 2, 3], labels don't match!
```

---

## 📈 Interpretation Example: Your Results

```
Case: BraTS-GLI-00006-100

Key Findings:
✅ Total volume: 39,981 mm³ (≈ 40 mL) - Moderate-large tumor
✅ Enhancing: 22,074 mm³ (55%) - Active tumor component
✅ Edema: 17,907 mm³ (45%) - Peritumoral swelling
✅ Necrosis: 0% - No necrotic areas

✅ Location: Right frontal lobe
   - Paracingulate Gyrus (15.1% involvement) - Medial frontal
   - Frontal Pole (10.6%) - Anterior frontal
   - Frontal Operculum (4.1%) - Lateral frontal

✅ Crosses midline: YES - Suggests corpus callosum involvement

Clinical Implications:
⚠️ Right frontal location → May affect executive function
⚠️ Midline crossing → Increases surgical complexity  
⚠️ No necrosis → Possibly lower-grade (but enhancing suggests high-grade)
⚠️ 40mL volume → Significant mass effect likely
```

---

## 🎯 Summary

**What the pipeline does:**
1. Takes your segmentation mask
2. Aligns it to a standard brain atlas
3. Counts which brain regions contain tumor
4. Generates structured JSON with percentages and volumes

**What you get:**
- Precise anatomical localization
- Quantitative measurements per region
- Ready for LLM-based report generation
- Clinically interpretable data

**How to verify:**
1. **Visual**: Load in 3D Slicer, check if regions match
2. **Quantitative**: Re-compute percentages manually in Python
3. **Sanity checks**: Volume conservation, percentage bounds
4. **Clinical**: Expert review for anatomical plausibility

---

This JSON descriptor is now ready to be fed into an LLM for automated radiology report generation! 🚀
