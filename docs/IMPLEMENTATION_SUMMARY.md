# Implementation Summary: Brain Atlas Mapping & Report Generation Pipeline

## Overview

This document provides a comprehensive guide for implementing mask-to-brain-atlas mapping, JSON generation, synthetic data creation, and LLM fine-tuning for medical report generation in your FYP project.

---

## 📁 Project Structure

```
An-Expert-Guided-Multimodal-AI-Ecosystem/
├── docs/
│   ├── ATLAS_MAPPING_GUIDE.md           # Brain atlas mapping theory & methods
│   ├── JSON_SCHEMA_GUIDE.md             # JSON schema design & validation
│   ├── SYNTHETIC_DATA_GUIDE.md          # Synthetic dataset generation
│   ├── LLM_FINETUNING_GUIDE.md         # MedGemma-4B fine-tuning
│   └── IMPLEMENTATION_SUMMARY.md        # This file
│
├── src/
│   └── utils/
│       ├── atlas_mapping.py             # Brain atlas mapper (implemented)
│       └── json_descriptor_generator.py # JSON generator (implemented)
│
├── scripts/
│   ├── generate_synthetic_data.py       # To be created
│   ├── train_medgemma.py               # To be created
│   └── evaluate_reports.py             # To be created
│
└── atlases/                            # Download brain atlases here
    ├── MNI152_T1_1mm.nii.gz
    └── HarvardOxford-cort-maxprob-thr25-1mm.nii.gz
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install nibabel numpy scipy scikit-image nilearn

# For LLM fine-tuning
pip install torch transformers accelerate peft datasets bitsandbytes

# For evaluation
pip install nltk rouge bert-score jsonschema
```

### Step 2: Download Brain Atlases

Option A: Using FSL (if installed)
```bash
# Harvard-Oxford atlas comes with FSL
cp $FSLDIR/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz atlases/
cp $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz atlases/
```

Option B: Using nilearn (Python)
```python
from nilearn import datasets

# Download MNI152 template
mni = datasets.fetch_icbm152_2009()
# Save to atlases/

# Download Harvard-Oxford atlas
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
# Save to atlases/
```

Option C: Manual download
- MNI152: http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
- Harvard-Oxford: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases

### Step 3: Test Atlas Mapping

```python
from src.utils.atlas_mapping import BrainAtlasMapper

# Initialize mapper
mapper = BrainAtlasMapper(
    atlas_path='atlases/MNI152_T1_1mm.nii.gz',
    atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz',
    use_ants=False  # Use affine (faster, no ANTs required)
)

# Process a segmentation
results = mapper.process_segmentation(
    seg_path='path/to/your/segmentation.nii.gz',
    t1_reference_path='path/to/your/t1.nii.gz'
)

print(results['overall_affected_regions'][:5])  # Top 5 regions
```

### Step 4: Generate JSON Descriptors

```python
from src.utils.json_descriptor_generator import TumorDescriptorGenerator

# Initialize generator
generator = TumorDescriptorGenerator(mapper)

# Generate descriptor
descriptor = generator.generate_descriptor(
    case_id="test_001",
    seg_path='path/to/segmentation.nii.gz',
    t1_path='path/to/t1.nii.gz',
    patient_metadata={"age": 58, "sex": "M"}
)

# Save to file
generator.save_descriptor(descriptor, 'output/test_001_descriptor.json')
```

### Step 5: Generate Synthetic Dataset (see SYNTHETIC_DATA_GUIDE.md)

Create `scripts/generate_synthetic_data.py`:

```python
import sys
sys.path.append('.')

from docs.SYNTHETIC_DATA_GUIDE import (
    SyntheticJSONGenerator,
    ReportTemplate,
    MedicalSynonymAugmenter
)

# Generate 5000 synthetic JSON-report pairs
generator = SyntheticJSONGenerator(seed=42)
template = ReportTemplate()

for i in range(5000):
    # Generate JSON
    case_id = f"SYNTHETIC_{i:06d}"
    descriptor = generator.generate_synthetic_descriptor(case_id)
    
    # Generate report
    report = template.generate_report(descriptor, verbosity='standard')
    
    # Save
    # ... (see full implementation in guide)
```

### Step 6: Fine-Tune MedGemma-4B (see LLM_FINETUNING_GUIDE.md)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b",
    load_in_8bit=True,
    device_map='auto'
)

# Apply LoRA
# ... (see full implementation in guide)

# Train
# ... (see full implementation in guide)
```

---

## 📊 Pipeline Workflow

### End-to-End Process

1. **Input**: MRI segmentation mask (from MoME+ model)
   
2. **Atlas Mapping** (`atlas_mapping.py`)
   - Register mask to MNI152 standard space
   - Overlay with Harvard-Oxford brain atlas
   - Calculate region-wise tumor involvement
   - Extract top affected regions
   
3. **JSON Generation** (`json_descriptor_generator.py`)
   - Create structured anatomical descriptor
   - Include volumetric analysis
   - Add tumor component breakdown
   - Validate against schema
   
4. **Synthetic Data Generation** (optional if no real data)
   - Generate plausible JSON variations
   - Create template-based reports
   - Apply linguistic augmentation
   - Quality control & validation
   
5. **LLM Fine-Tuning**
   - Format data for instruction tuning
   - Apply LoRA to MedGemma-4B
   - Train on JSON→Report pairs
   - Evaluate factual consistency
   
6. **Report Generation**
   - Input: JSON descriptor
   - Output: Radiology-style report
   - Post-processing & safety checks

---

## 🔧 Integration with Existing MoME+ Pipeline

### Current MoME+ Workflow

```python
# Your existing inference code
from src.models.mome_segmenter import MoMESegmenter
from src.inference.inference_engine import InferenceEngine

model = MoMESegmenter(...)
seg_output = model.predict(mri_data)  # Returns segmentation mask
```

### Enhanced Workflow with Report Generation

```python
from src.models.mome_segmenter import MoMESegmenter
from src.utils.atlas_mapping import BrainAtlasMapper
from src.utils.json_descriptor_generator import TumorDescriptorGenerator
# from src.llm.report_generator import MedGemmaReportGenerator  # To be created

# Step 1: Segmentation (existing)
model = MoMESegmenter(...)
seg_output = model.predict(mri_data)

# Step 2: Atlas mapping (new)
atlas_mapper = BrainAtlasMapper(
    atlas_path='atlases/MNI152_T1_1mm.nii.gz',
    atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
)

# Step 3: Generate JSON descriptor (new)
descriptor_gen = TumorDescriptorGenerator(atlas_mapper)
descriptor = descriptor_gen.generate_descriptor(
    case_id=case_id,
    seg_path=seg_output['mask_path'],
    t1_path=mri_data['t1_path'],
    patient_metadata=patient_info,
    model_metadata={
        'model_name': 'MoME+',
        'dice_scores': seg_output['metrics']
    }
)

# Step 4: Generate report (new)
# report_gen = MedGemmaReportGenerator(model_path='./medgemma_finetuned')
# report = report_gen.generate_report(descriptor)

# print(report)
```

---

## 📈 Expected Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Setup atlases & test atlas mapping | 1-2 days | ⚪ Not started |
| 2 | Generate JSON descriptors from real data | 2-3 days | ⚪ Not started |
| 3 | Create synthetic data generator | 3-5 days | ⚪ Not started |
| 4 | Generate 10K synthetic pairs | 1 day | ⚪ Not started |
| 5 | Setup MedGemma & LoRA fine-tuning | 2-3 days | ⚪ Not started |
| 6 | Train model (3 epochs) | 1-2 days | ⚪ Not started |
| 7 | Evaluate & refine | 2-3 days | ⚪ Not started |
| 8 | Integration & API deployment | 2-3 days | ⚪ Not started |
| **Total** | | **14-22 days** | |

---

## 🎯 Key Milestones

- [ ] **Milestone 1**: Successfully map 10 real segmentations to brain atlas
- [ ] **Milestone 2**: Generate 100 validated JSON descriptors
- [ ] **Milestone 3**: Create 5,000 synthetic JSON-report pairs
- [ ] **Milestone 4**: Fine-tune MedGemma-4B with BLEU > 0.35
- [ ] **Milestone 5**: Deploy report generation API
- [ ] **Milestone 6**: Integrate with existing MoME+ pipeline

---

## 📝 Data Requirements

### For Real Data Approach

**Minimum:**
- 50-100 segmented cases (BraTS format)
- Corresponding T1 MRI scans
- Ground truth reports (optional, for evaluation)

**Ideal:**
- 200+ segmented cases
- Multi-modal MRI (T1, T1ce, T2, FLAIR)
- Expert-annotated reports

### For Synthetic Data Approach

**Minimum:**
- 5,000 synthetic JSON descriptors
- 3 report variations per JSON
- Total: 15,000 training examples

**Ideal:**
- 10,000 synthetic JSON descriptors
- 3-5 report variations per JSON
- Total: 30,000-50,000 training examples

---

## 🧪 Validation Strategy

### Atlas Mapping Validation

```python
# Visual inspection
import matplotlib.pyplot as plt
from nilearn import plotting

# Overlay tumor on atlas
plotting.plot_roi(
    roi_img=seg_atlas_space,
    bg_img=atlas_img,
    title='Tumor overlaid on MNI152'
)
plt.show()

# Check top regions make anatomical sense
print(results['overall_affected_regions'][:5])
```

### JSON Schema Validation

```python
from jsonschema import validate

# Load schema
with open('schemas/tumor_descriptor_schema.json', 'r') as f:
    schema = json.load(f)

# Validate descriptor
validate(instance=descriptor, schema=schema)
print("✓ Schema validation passed")
```

### Report Quality Validation

```python
from src.llm.evaluate_reports import ReportEvaluator

evaluator = ReportEvaluator()

# Factual consistency
is_consistent, errors = evaluator.factual_consistency_check(
    descriptor,
    generated_report
)

if not is_consistent:
    print("⚠️ Factual errors:", errors)
```

---

## 🔍 Troubleshooting

### Issue: Atlas registration fails

**Solution:**
- Ensure both segmentation and atlas are in NIfTI format
- Check orientation (RAS+ vs LAS+)
- Try affine registration first before ANTs
- Verify atlas files are not corrupted

### Issue: JSON validation errors

**Solution:**
- Check all required fields are present
- Ensure numeric fields are not NaN
- Verify region_id exists in atlas
- Check percentage values are 0-100

### Issue: MedGemma OOM (Out of Memory)

**Solution:**
- Reduce batch size to 1
- Increase gradient accumulation steps
- Use 8-bit quantization (`load_in_8bit=True`)
- Enable gradient checkpointing
- Reduce LoRA rank from 16 to 8

### Issue: Generated reports hallucinate facts

**Solution:**
- Lower temperature (0.1-0.3)
- Use nucleus sampling (top_p=0.9)
- Add safety filtering post-processing
- Fine-tune with more factually consistent examples
- Implement forced JSON-report alignment

---

## 📚 Additional Resources

### Brain Atlases
- [FSL Atlases](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases)
- [Nilearn Datasets](https://nilearn.github.io/dev/modules/datasets.html)
- [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/)

### MRI Processing
- [nibabel Documentation](https://nipy.org/nibabel/)
- [nilearn Tutorials](https://nilearn.github.io/stable/auto_examples/index.html)
- [ANTs Registration](http://stnava.github.io/ANTs/)

### LLM Fine-Tuning
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-4b)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Medical NLP
- [BioGPT](https://github.com/microsoft/BioGPT)
- [RadLex Terminology](https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon)
- [Medical Report Generation Survey](https://arxiv.org/abs/2203.02573)

---

## 🎓 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{expert_guided_multimodal_ai_2024,
  title={An Expert-Guided Multimodal AI Ecosystem for Diagnostic Intelligence},
  author={Your Name},
  year={2024},
  institution={Your University},
  type={Final Year Project}
}
```

And the original "From Segmentation to Explanation" paper:
```bibtex
@article{valerio2025segmentation,
  title={From Segmentation to Explanation: A Medical Imaging AI Pipeline},
  author={Valerio, Alberto et al.},
  journal={Computer Methods and Programs in Biomedicine},
  year={2025}
}
```

---

## 📞 Support

For issues or questions:
1. Check the detailed guides in `docs/`
2. Review troubleshooting section above
3. Examine example code in implemented modules
4. Refer to reference papers and documentation

---

**Last Updated**: 2024-12-15

**Version**: 1.0.0

**Status**: Implementation guides complete, awaiting deployment ✅
