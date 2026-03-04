"""
Generate synthetic JSON-to-report pairs for training.
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generation.templates import generate_template_report
from src.report_generation.dataset import save_jsonl


def generate_synthetic_json() -> dict:
    """Generate a realistic synthetic JSON anatomical descriptor."""
    
    # Random laterality
    laterality_options = ['left', 'right', 'bilateral']
    laterality = random.choice(laterality_options)
    
    # Brain regions for sampling
    regions = [
        'Right Middle Frontal Gyrus', 'Left Temporal Lobe', 'Right Parietal Lobe',
        'Left Frontal Lobe', 'Right Temporal Lobe', 'Left Parietal Lobe',
        'Right Occipital Lobe', 'Left Occipital Lobe', 'Right Insular Cortex',
        'Left Thalamus', 'Right Thalamus', 'Corpus Callosum',
        'Right Putamen', 'Left Caudate', 'Brainstem', 'Cerebellum'
    ]
    
    # Generate tumor volumes
    et_volume = round(random.uniform(1.0, 25.0), 1)
    tc_volume = round(et_volume * random.uniform(1.2, 2.5), 1)
    wt_volume = round(tc_volume * random.uniform(1.3, 3.0), 1)
    
    # Generate confidence scores
    et_conf = round(random.uniform(0.85, 0.98), 2)
    tc_conf = round(random.uniform(0.82, 0.96), 2)
    wt_conf = round(random.uniform(0.88, 0.97), 2)
    
    # Select affected regions
    num_regions = random.randint(2, 4)
    affected_regions = random.sample(regions, num_regions)
    
    # Generate overlap percentages (decreasing)
    overlaps = sorted([round(random.uniform(10, 60), 1) for _ in range(num_regions)], reverse=True)
    
    et_regions = [
        {
            'name': region,
            'overlap_percent': overlap,
            'volume_ml': round(et_volume * overlap / 100, 2)
        }
        for region, overlap in zip(affected_regions, overlaps)
    ]
    
    # Create JSON structure
    json_data = {
        'case_id': f'BraTS-GLI-{random.randint(10000, 99999):05d}',
        'imaging_parameters': {
            'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'scanner': '3T MRI',
            'acquisition_date': '2024-01-15'
        },
        'segmentation_results': {
            'enhancing_tumor': {
                'volume_cm3': et_volume,
                'confidence': et_conf,
                'centroid': [random.randint(80, 120), random.randint(100, 140), random.randint(60, 100)]
            },
            'tumor_core': {
                'volume_cm3': tc_volume,
                'confidence': tc_conf,
                'centroid': [random.randint(80, 120), random.randint(100, 140), random.randint(60, 100)]
            },
            'whole_tumor': {
                'volume_cm3': wt_volume,
                'confidence': wt_conf,
                'centroid': [random.randint(80, 120), random.randint(100, 140), random.randint(60, 100)]
            }
        },
        'anatomical_mapping': {
            'laterality': laterality,
            'enhancing_tumor_regions': et_regions[:3],  # Top 3 regions
            'primary_location': et_regions[0]['name']
        },
        'mass_effect': random.choice([True, False]),
        'clinical_history': random.choice([
            'Brain tumor evaluation',
            'Neurological symptoms, rule out mass lesion',
            'Follow-up for known glioma',
            'Headache and visual changes',
            'Seizure disorder workup'
        ]),
        'model_metadata': {
            'architecture': 'MoME+',
            'inference_time_sec': round(random.uniform(2.5, 4.5), 2),
            'cuda_available': True
        }
    }
    
    return json_data


def generate_dataset(num_samples: int, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Generate complete synthetic dataset.
    
    Args:
        num_samples: Total number of samples to generate
        output_dir: Directory to save JSONL files
        train_ratio: Proportion for training (default 70%)
        val_ratio: Proportion for validation (default 15%, rest is test)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples...")
    
    all_data = []
    for i in range(num_samples):
        # Generate JSON
        json_data = generate_synthetic_json()
        
        # Generate report using random template variation
        template_idx = random.randint(0, 4)
        report = generate_template_report(json_data, template_idx)
        
        # Store pair
        all_data.append({
            'id': i,
            'json_data': json_data,
            'report': report,
            'template_used': template_idx
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    # Shuffle
    random.shuffle(all_data)
    
    # Split
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save
    print(f"\nSaving datasets...")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    save_jsonl(train_data, str(output_path / 'train.jsonl'))
    save_jsonl(val_data, str(output_path / 'val.jsonl'))
    save_jsonl(test_data, str(output_path / 'test.jsonl'))
    
    # Save summary
    summary = {
        'total_samples': num_samples,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1 - train_ratio - val_ratio
    }
    
    with open(output_path / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"   Files: train.jsonl, val.jsonl, test.jsonl, dataset_summary.json")
    
    # Show example
    print(f"\n--- Example JSON-Report Pair ---")
    example = train_data[0]
    print(f"\nJSON (excerpt):")
    print(json.dumps(example['json_data']['segmentation_results'], indent=2))
    print(f"\nREPORT:")
    print(example['report'][:300] + "...")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic report training data")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='data/reports',
                        help='Output directory (default: data/reports)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Report Generation Dataset Creation")
    print("=" * 60)
    
    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == '__main__':
    main()
