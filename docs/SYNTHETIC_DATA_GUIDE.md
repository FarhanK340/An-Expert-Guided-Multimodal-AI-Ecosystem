# Synthetic Data Generation for Medical Report Training

## Overview

This guide provides a **production-grade system** for generating large-scale synthetic datasets of (JSON → Report) pairs to fine-tune LLMs for medical report generation.

**Goals:**
- Generate 10,000+ high-quality JSON-report pairs
- Maintain clinical plausibility
- Introduce linguistic variation without semantic drift
- Ensure factual consistency (no hallucinations)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│           SYNTHETIC DATA GENERATION PIPELINE                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. JSON Variation Generation                                │
│     ├─ Real segmentations (if available)                     │
│     ├─ Synthetic variations via parameter sampling           │
│     └─ JSON schema validation                                │
│                    ↓                                          │
│  2. Template-Based Report Generation                         │
│     ├─ Multiple verbosity levels                             │
│     ├─ Template randomization                                │
│     └─ Clinical phrasing library                             │
│                    ↓                                          │
│  3. Linguistic Augmentation                                  │
│     ├─ Synonym replacement                                   │
│     ├─ Sentence reordering                                   │
│     └─ Paraphrasing (via smaller LLM)                        │
│                    ↓                                          │
│  4. Quality Control                                          │
│     ├─ Factual consistency check                             │
│     ├─ Medical terminology validation                        │
│     └─ Data filtering                                        │
│                    ↓                                          │
│  5. Dataset Export                                           │
│     └─ JSONL format for LLM training                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 1: JSON Variation Generation

### 1.1 Using Real Segmentations (Recommended)

If you have BraTS or other segmented datasets:

```python
import os
import json
from pathlib import Path
from atlas_mapping import BrainAtlasMapper
from json_generation import TumorDescriptorGenerator

def generate_real_json_descriptors(
    dataset_dir,
    output_dir,
    max_samples=1000
):
    """
    Generate JSON descriptors from real segmentations.
    
    Args:
        dataset_dir: Path to BraTS-style dataset
        output_dir: Where to save JSON files
        max_samples: Number of samples to process
    """
    atlas_mapper = BrainAtlasMapper(
        atlas_path='atlases/MNI152_T1_1mm.nii.gz',
        atlas_labels_path='atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
    )
    
    descriptor_gen = TumorDescriptorGenerator(atlas_mapper)
    
    # Find all segmentation files
    seg_files = list(Path(dataset_dir).rglob('*seg.nii.gz'))[:max_samples]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, seg_path in enumerate(seg_files):
        case_id = seg_path.stem.replace('_seg', '')
        
        # Find corresponding T1
        t1_path = seg_path.parent / f"{case_id}_t1.nii.gz"
        
        if not t1_path.exists():
            continue
        
        try:
            descriptor = descriptor_gen.generate_descriptor(
                case_id=case_id,
                seg_path=str(seg_path),
                t1_path=str(t1_path)
            )
            
            output_file = Path(output_dir) / f"{case_id}_descriptor.json"
            with open(output_file, 'w') as f:
                json.dump(descriptor, f, indent=2)
            
            print(f"[{i+1}/{len(seg_files)}] Generated: {case_id}")
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    print(f"\nGenerated {len(list(Path(output_dir).glob('*.json')))} descriptors")
```

### 1.2 Synthetic JSON Generation (When Data is Limited)

Create plausible JSON descriptors via controlled randomization:

```python
import random
import numpy as np
from datetime import datetime, timedelta

class SyntheticJSONGenerator:
    """
    Generate synthetic but clinically plausible tumor descriptors.
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Clinical parameters distributions
        self.age_distribution = (30, 80)  # Mean age range for gliomas
        self.volume_distribution = {
            'small': (500, 5000),      # mm³
            'medium': (5000, 20000),
            'large': (20000, 50000)
        }
        
        # Brain regions (Harvard-Oxford cortical)
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
            {"id": 13, "name": "Supramarginal Gyrus", "typical_volume": 14000},
            {"id": 14, "name": "Angular Gyrus", "typical_volume": 13500},
            {"id": 15, "name": "Lateral Occipital Cortex", "typical_volume": 20000},
            # Add more regions...
        ]
    
    def generate_synthetic_descriptor(self, case_id):
        """
        Generate a single synthetic tumor descriptor.
        
        Returns:
            Dictionary conforming to schema
        """
        # Random tumor size category
        size_category = random.choice(['small', 'medium', 'large'])
        total_volume = random.uniform(*self.volume_distribution[size_category])
        
        # Component breakdown (realistic proportions)
        enhancing_ratio = random.uniform(0.25, 0.45)
        necrotic_ratio = random.uniform(0.15, 0.35)
        edema_ratio = 1 - enhancing_ratio - necrotic_ratio
        
        enhancing_vol = total_volume * enhancing_ratio
        necrotic_vol = total_volume * necrotic_ratio
        edema_vol = total_volume * edema_ratio
        
        # Select 3-8 affected regions
        num_regions = random.randint(3, 8)
        selected_regions = random.sample(self.brain_regions, num_regions)
        
        # Assign tumor volumes to regions (largest first)
        affected_regions = []
        remaining_volume = total_volume
        
        for i, region in enumerate(selected_regions):
            if i == num_regions - 1:
                region_tumor_vol = remaining_volume
            else:
                # Decreasing volumes (Zipf-like distribution)
                max_vol = remaining_volume * 0.6
                region_tumor_vol = random.uniform(
                    remaining_volume * 0.1, 
                    max_vol
                )
            
            percentage = (region_tumor_vol / region['typical_volume']) * 100
            percentage = min(percentage, 95)  # Cap at 95%
            
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
        
        # Sort by percentage (descending)
        affected_regions.sort(
            key=lambda x: x['percentage_involvement'], 
            reverse=True
        )
        
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
                },
                "confidence_metrics": {
                    "mean_dice_score": round(random.uniform(0.82, 0.92), 2)
                }
            },
            "anatomical_mapping": {
                "atlas_name": "harvard_oxford",
                "registration_method": random.choice(["affine", "ANTs_SyN"]),
                "hemisphere": self._determine_hemisphere(affected_regions),
                "crossing_midline": random.random() < 0.15,  # 15% cross midline
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
        """Generate random scan date within last 2 years."""
        days_ago = random.randint(0, 730)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime("%Y-%m-%d")
    
    def _random_coords(self):
        """Generate random MNI coordinates."""
        return [
            round(random.uniform(-80, 80), 1),
            round(random.uniform(-110, 70), 1),
            round(random.uniform(-50, 85), 1)
        ]
    
    def _determine_hemisphere(self, regions):
        """Determine hemisphere based on region distribution."""
        left_count = sum(1 for r in regions if r['hemisphere'] == 'left')
        right_count = len(regions) - left_count
        
        if left_count > 2 * right_count:
            return 'left'
        elif right_count > 2 * left_count:
            return 'right'
        else:
            return 'bilateral'
    
    def _sample_eloquent_areas(self):
        """Sample eloquent areas with realistic probability."""
        areas = []
        probabilities = {
            'motor_cortex': 0.25,
            'speech_area': 0.15,
            'visual_cortex': 0.10,
            'brainstem': 0.05
        }
        
        for area, prob in probabilities.items():
            if random.random() < prob:
                areas.append(area)
        
        return areas
```

### Generate batch of synthetic JSONs

```python
def generate_synthetic_dataset(num_samples=10000, output_dir='synthetic_json'):
    """Generate large synthetic JSON dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticJSONGenerator(seed=42)
    
    for i in range(num_samples):
        case_id = f"SYNTHETIC_{i:06d}"
        descriptor = generator.generate_synthetic_descriptor(case_id)
        
        output_file = Path(output_dir) / f"{case_id}.json"
        with open(output_file, 'w') as f:
            json.dump(descriptor, f, indent=2)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} descriptors")
    
    print(f"\n✓ Generated {num_samples} synthetic JSON descriptors")
```

---

## Stage 2: Template-Based Report Generation

### 2.1 Report Template System

```python
class ReportTemplate:
    """
    Template-based medical report generator.
    """
    
    def __init__(self):
        # Multiple template variations for each section
        self.intro_templates = [
            "Clinical MRI scan reveals {tumor_description} in the {primary_location}.",
            "Imaging demonstrates {tumor_description} involving the {primary_location}.",
            "MRI findings indicate {tumor_description} with primary involvement of the {primary_location}.",
            "A {tumor_description} is identified, predominantly affecting the {primary_location}."
        ]
        
        self.volume_templates = [
            "The lesion measures approximately {total_volume:.1f} mm³ in total volume.",
            "Total tumor volume is estimated at {total_volume:.1f} mm³.",
            "Volumetric analysis shows a lesion volume of {total_volume:.1f} mm³.",
        ]
        
        self.component_templates = {
            'enhancing': [
                "The enhancing component measures {volume:.1f} mm³ ({percentage:.1f}% of total).",
                "Contrast-enhancing tumor accounts for {volume:.1f} mm³.",
                "Enhancing regions constitute {volume:.1f} mm³ of the lesion.",
            ],
            'necrotic': [
                "A necrotic core of {volume:.1f} mm³ is present ({percentage:.1f}% necrosis).",
                "Central necrosis measures {volume:.1f} mm³.",
                "Necrotic regions account for {volume:.1f} mm³.",
            ],
            'edema': [
                "Surrounding peritumoral edema extends {volume:.1f} mm³.",
                "Vasogenic edema measures {volume:.1f} mm³.",
                "Edematous changes occupy {volume:.1f} mm³.",
            ]
        }
        
        self.region_templates = [
            "The tumor predominantly affects the {region_name} ({hemisphere} hemisphere), with {percentage:.1f}% regional involvement.",
            "{percentage:.1f}% of the {hemisphere} {region_name} is involved.",
            "Significant infiltration of the {hemisphere} {region_name} is noted ({percentage:.1f}% involvement).",
        ]
        
        self.conclusion_templates = [
            "Findings are consistent with a {grade} glioma.",
            "Imaging characteristics suggest a {grade} glial neoplasm.",
            "The appearance is most compatible with a {grade} glioma.",
        ]
    
    def generate_report(self, descriptor, verbosity='standard'):
        """
        Generate report from JSON descriptor.
        
        Args:
            descriptor: JSON descriptor dictionary
            verbosity: 'brief', 'standard', or 'detailed'
        
        Returns:
            str: Medical report
        """
        sections = []
        
        # FINDINGS section
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
        volume_desc = random.choice(self.volume_templates).format(
            total_volume=total_vol
        )
        sections.append(volume_desc)
        sections.append("")
        
        # Components
        components = descriptor['segmentation_results']['tumor_components']
        vol_analysis = descriptor['segmentation_results']['volumetric_analysis']
        
        if components['enhancing_tumor']['present']:
            template = random.choice(self.component_templates['enhancing'])
            sections.append(template.format(
                volume=components['enhancing_tumor']['volume_mm3'],
                percentage=(components['enhancing_tumor']['volume_mm3'] / total_vol * 100)
            ))
        
        if components['necrotic_core']['present'] and components['necrotic_core']['volume_mm3'] > 100:
            template = random.choice(self.component_templates['necrotic'])
            sections.append(template.format(
                volume=components['necrotic_core']['volume_mm3'],
                percentage=vol_analysis['necrosis_percentage']
            ))
        
        if components['peritumoral_edema']['present']:
            template = random.choice(self.component_templates['edema'])
            sections.append(template.format(
                volume=components['peritumoral_edema']['volume_mm3']
            ))
        
        sections.append("")
        
        # Anatomical regions (top 3-5)
        if verbosity in ['standard', 'detailed']:
            num_regions = 5 if verbosity == 'detailed' else 3
            affected_regions = descriptor['anatomical_mapping']['affected_regions'][:num_regions]
            
            sections.append("Anatomical distribution:")
            for region in affected_regions:
                template = random.choice(self.region_templates)
                sections.append("- " + template.format(
                    region_name=region['region_name'],
                    hemisphere=region.get('hemisphere', hemisphere),
                    percentage=region['percentage_involvement']
                ))
            sections.append("")
        
        # Clinical features
        clinical = descriptor.get('clinical_features', {})
        
        if clinical.get('mass_effect'):
            sections.append("Mass effect is present.")
        
        if clinical.get('eloquent_area_involvement'):
            eloquent = ', '.join(clinical['eloquent_area_involvement']).replace('_', ' ')
            sections.append(f"The lesion involves eloquent areas: {eloquent}.")
        
        if clinical.get('crossing_midline'):
            sections.append("The tumor crosses the midline.")
        
        if descriptor['anatomical_mapping'].get('crossing_midline'):
            sections.append("Extension across the corpus callosum is noted.")
        
        sections.append("")
        
        # IMPRESSION section
        sections.append("IMPRESSION:")
        sections.append("")
        
        grade_map = {
            'low_grade': 'low-grade',
            'high_grade': 'high-grade',
            'unknown': 'intermediate-grade'
        }
        grade = clinical.get('estimated_grade', 'unknown')
        grade_text = grade_map.get(grade, 'intermediate-grade')
        
        conclusion = random.choice(self.conclusion_templates).format(
            grade=grade_text
        )
        sections.append(conclusion)
        
        if clinical.get('mass_effect') or clinical.get('crossing_midline'):
            sections.append("Surgical evaluation is recommended.")
        
        return '\n'.join(sections)
    
    def _describe_tumor(self, descriptor):
        """Generate tumor description phrase."""
        vol = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
        
        if vol < 5000:
            size = random.choice(['small', 'focal'])
        elif vol < 20000:
            size = random.choice(['moderate-sized', 'well-defined'])
        else:
            size = random.choice(['large', 'extensive'])
        
        components = descriptor['segmentation_results']['tumor_components']
        
        if components['enhancing_tumor']['present'] and components['necrotic_core']['present']:
            characteristics = random.choice([
                'heterogeneous mass with enhancement and necrosis',
                'complex lesion demonstrating enhancement and necrotic change',
                'heterogeneously enhancing mass with central necrosis'
            ])
        elif components['enhancing_tumor']['present']:
            characteristics = random.choice([
                'enhancing mass lesion',
                'contrast-enhancing lesion',
                'heterogeneously enhancing mass'
            ])
        else:
            characteristics = random.choice([
                'infiltrative lesion',
                'non-enhancing mass',
                'diffuse infiltrative process'
            ])
        
        return f"{size} {characteristics}"
```

### 2.2 Generate Report Dataset

```python
def generate_json_report_pairs(
    json_dir,
    output_file,
    num_variations=3
):
    """
    Generate multiple report variations per JSON.
    
    Args:
        json_dir: Directory containing JSON descriptors
        output_file: Output JSONL file path
        num_variations: Reports per JSON (for diversity)
    """
    template_gen = ReportTemplate()
    json_files = list(Path(json_dir).glob('*.json'))
    
    with open(output_file, 'w') as outf:
        for json_path in json_files:
            with open(json_path, 'r') as f:
                descriptor = json.load(f)
            
            for i in range(num_variations):
                # Random verbosity
                verbosity = random.choice(['brief', 'standard', 'detailed'])
                
                report = template_gen.generate_report(
                    descriptor,
                    verbosity=verbosity
                )
                
                # Format for LLM training
                training_example = {
                    "descriptor": descriptor,
                    "report": report,
                    "metadata": {
                        "case_id": descriptor['patient_info']['case_id'],
                        "verbosity": verbosity,
                        "variation": i
                    }
                }
                
                outf.write(json.dumps(training_example) + '\n')
    
    print(f"✓ Generated {len(json_files) * num_variations} training examples")
```

---

## Stage 3: Linguistic Augmentation

### 3.1 Synonym Replacement (Medical-Safe)

```python
class MedicalSynonymAugmenter:
    """
    Safe synonym replacement for medical text.
    """
    
    def __init__(self):
        # Curated medical synonyms (preserve meaning)
        self.synonyms = {
            'lesion': ['mass', 'abnormality', 'lesion'],
            'demonstrates': ['shows', 'reveals', 'demonstrates', 'exhibits'],
            'approximately': ['approximately', 'roughly', 'about'],
            'involvement': ['involvement', 'infiltration', 'engagement'],
            'tumor': ['tumor', 'neoplasm', 'lesion', 'mass'],
            'measures': ['measures', 'extends', 'spans'],
            'present': ['present', 'evident', 'identified', 'noted'],
            'suggests': ['suggests', 'indicates', 'is consistent with'],
        }
    
    def augment(self, text, probability=0.3):
        """
        Replace words with synonyms.
        
        Args:
            text: Original report text
            probability: Chance of replacing each word
        
        Returns:
            Augmented text
        """
        import re
        
        words = text.split()
        augmented = []
        
        for word in words:
            word_lower = word.lower().strip('.,;:')
            
            if word_lower in self.synonyms and random.random() < probability:
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                augmented.append(synonym + word[len(word_lower):])  # Preserve punctuation
            else:
                augmented.append(word)
        
        return ' '.join(augmented)
```

### 3.2 Paraphrasing with Small LLM (Optional)

```python
# Using a small paraphrasing model
from transformers import pipeline

class LLMParaphraser:
    """
    Use a small LLM for controlled paraphrasing.
    """
    
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.paraphraser = pipeline('summarization', model=model_name)
    
    def paraphrase(self, text, max_length=512):
        """
        Paraphrase while preserving medical facts.
        
        Note: This is experimental - validate outputs carefully.
        """
        # Split into sentences
        sentences = text.split('.')
        paraphrased = []
        
        for sent in sentences:
            if len(sent.strip()) > 10:
                result = self.paraphraser(
                    sent,
                    max_length=max_length,
                    do_sample=True
                )
                paraphrased.append(result[0]['summary_text'])
            else:
                paraphrased.append(sent)
        
        return '. '.join(paraphrased)
```

---

## Stage 4: Quality Control

```python
class QualityValidator:
    """
    Validate factual consistency of generated reports.
    """
    
    def validate(self, descriptor, report):
        """
        Check if report accurately reflects JSON descriptor.
        
        Returns:
            (valid: bool, errors: list)
        """
        errors = []
        
        # Check volume mentions
        stated_volume = descriptor['segmentation_results']['volumetric_analysis']['total_tumor_volume_mm3']
        
        import re
        volume_match = re.search(r'(\d+\.?\d*)\s*mm³', report)
        if volume_match:
            report_volume = float(volume_match.group(1))
            # Allow 1% tolerance for rounding
            if abs(report_volume - stated_volume) / stated_volume > 0.01:
                errors.append(f"Volume mismatch: JSON={stated_volume}, Report={report_volume}")
        
        # Check region names
        for region in descriptor['anatomical_mapping']['affected_regions'][:3]:
            if region['region_name'].lower() not in report.lower():
                errors.append(f"Missing region: {region['region_name']}")
        
        # Check hemisphere
        hemisphere = descriptor['anatomical_mapping']['hemisphere']
        if hemisphere != 'bilateral' and hemisphere.lower() not in report.lower():
            errors.append(f"Missing hemisphere: {hemisphere}")
        
        # Check clinical features
        clinical = descriptor.get('clinical_features', {})
        if clinical.get('mass_effect') and 'mass effect' not in report.lower():
            errors.append("Missing mass effect mention")
        
        return len(errors) == 0, errors
```

---

## Complete Pipeline

```python
def generate_complete_synthetic_dataset(
    num_samples=10000,
    num_variations=3,
    output_dir='synthetic_dataset'
):
    """
    End-to-end synthetic dataset generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: Generate synthetic JSONs
    print("Stage 1: Generating synthetic JSON descriptors...")
    json_dir = os.path.join(output_dir, 'json')
    generate_synthetic_dataset(num_samples, json_dir)
    
    # Stage 2: Generate reports
    print("\nStage 2: Generating reports from templates...")
    raw_pairs_file = os.path.join(output_dir, 'raw_pairs.jsonl')
    generate_json_report_pairs(json_dir, raw_pairs_file, num_variations)
    
    # Stage 3: Linguistic augmentation
    print("\nStage 3: Applying linguistic augmentation...")
    augmenter = MedicalSynonymAugmenter()
    validator = QualityValidator()
    
    augmented_file = os.path.join(output_dir, 'augmented_pairs.jsonl')
    valid_count = 0
    
    with open(raw_pairs_file, 'r') as inf, open(augmented_file, 'w') as outf:
        for line in inf:
            example = json.loads(line)
            
            # Augment report
            original_report = example['report']
            augmented_report = augmenter.augment(original_report, probability=0.3)
            
            # Validate
            is_valid, errors = validator.validate(
                example['descriptor'],
                augmented_report
            )
            
            if is_valid:
                example['report'] = augmented_report
                outf.write(json.dumps(example) + '\n')
                valid_count += 1
    
    print(f"\n✓ Generated {valid_count} validated training examples")
    print(f"✓ Dataset saved to: {output_dir}")
    
    return augmented_file

# Run
if __name__ == '__main__':
    dataset_file = generate_complete_synthetic_dataset(
        num_samples=5000,
        num_variations=3
    )
    print(f"\n✅ Final dataset: {dataset_file}")
```

---

## Output Format for LLM Training

Each line in the JSONL file:

```json
{
  "descriptor": { ... full JSON descriptor ... },
  "report": "FINDINGS:\n\nClinical MRI scan reveals...",
  "metadata": {
    "case_id": "SYNTHETIC_000123",
    "verbosity": "standard",
    "variation": 0
  }
}
```

---

## Next: LLM Fine-Tuning

→ See `LLM_FINETUNING_GUIDE.md` for MedGemma-4B instruction tuning using this dataset.
