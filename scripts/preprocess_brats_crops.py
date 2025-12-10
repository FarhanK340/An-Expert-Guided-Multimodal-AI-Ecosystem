"""
Preprocessing script to create 64x64x64 crops from BraTS 2024 dataset.
Saves preprocessed crops to local data/preprocessed/ for efficient training.
"""

import os
import json
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse


def create_crops_from_volume(volume, mask, crop_size=(64, 64, 64), num_crops=10, foreground_prob=0.7):
    """
    Extract random crops from a full volume.
    
    Args:
        volume: (C, H, W, D) - multi-modal volume
        mask: (3, H, W, D) - segmentation mask (WT, TC, ET)
        crop_size: Size of crops to extract
        num_crops: Number of crops per volume
        foreground_prob: Probability of sampling crops containing tumor
        
    Returns:
        List of (crop_volume, crop_mask) tuples
    """
    C, H, W, D = volume.shape
    ch, cw, cd = crop_size
    
    crops = []
    
    # Get tumor locations (any non-zero in mask)
    tumor_mask = (mask.sum(axis=0) > 0)  # Shape: (H, W, D)
    tumor_indices = np.argwhere(tumor_mask)
    
    for i in range(num_crops):
        # Decide whether to sample from tumor region or random
        if len(tumor_indices) > 0 and np.random.rand() < foreground_prob:
            # Sample around tumor
            center = tumor_indices[np.random.randint(len(tumor_indices))]
            h_start = max(0, center[0] - ch//2)
            w_start = max(0, center[1] - cw//2)
            d_start = max(0, center[2] - cd//2)
        else:
            # Random crop
            h_start = np.random.randint(0, max(1, H - ch))
            w_start = np.random.randint(0, max(1, W - cw))
            d_start = np.random.randint(0, max(1, D - cd))
        
        # Ensure crop doesn't exceed bounds
        h_start = min(h_start, H - ch)
        w_start = min(w_start, W - cw)
        d_start = min(d_start, D - cd)
        
        # Extract crop
        crop_vol = volume[:, h_start:h_start+ch, w_start:w_start+cw, d_start:d_start+cd]
        crop_mask = mask[:, h_start:h_start+ch, w_start:w_start+cw, d_start:d_start+cd]
        
        crops.append((crop_vol, crop_mask))
    
    return crops


def normalize_volume(volume):
    """Z-score normalization per modality."""
    normalized = np.zeros_like(volume)
    C = volume.shape[0]
    
    for c in range(C):
        mod_data = volume[c]
        # Only normalize non-zero voxels (brain region)
        nonzero_mask = mod_data > 0
        if nonzero_mask.sum() > 0:
            mean_val = mod_data[nonzero_mask].mean()
            std_val = mod_data[nonzero_mask].std()
            if std_val > 0:
                normalized[c][nonzero_mask] = (mod_data[nonzero_mask] - mean_val) / std_val
    
    return normalized


def process_case(case_dir, modalities=["t1n", "t1c", "t2w", "t2f"], crop_size=(64, 64, 64), num_crops=10):
    """Process a single BraTS case."""
    case_name = case_dir.name
    
    # Load modalities
    images = []
    for mod in modalities:
        mod_files = list(case_dir.glob(f"*-{mod}.nii.gz"))
        if not mod_files:
            raise FileNotFoundError(f"No file found for {mod} in {case_dir}")
        
        img = nib.load(str(mod_files[0]))
        img_data = img.get_fdata().astype(np.float32)
        images.append(img_data)
    
    # Stack: (4, H, W, D)
    volume = np.stack(images, axis=0)
    volume = normalize_volume(volume)
    
    # Load mask
    mask_files = list(case_dir.glob("*-seg.nii.gz"))
    if mask_files:
        mask_img = nib.load(str(mask_files[0]))
        mask_data = mask_img.get_fdata().astype(np.uint8)
        
        # Convert to WT, TC, ET channels
        wt = (mask_data > 0).astype(np.float32)
        tc = np.logical_or(mask_data == 1, np.logical_or(mask_data == 3, mask_data == 4)).astype(np.float32)
        et = np.logical_or(mask_data == 3, mask_data == 4).astype(np.float32)
        mask = np.stack([wt, tc, et], axis=0)
    else:
        # Validation data (no masks)
        mask = np.zeros((3,) + volume.shape[1:], dtype=np.float32)
        num_crops = 5  # Fewer crops for validation
    
    # Extract crops
    crops = create_crops_from_volume(volume, mask, crop_size, num_crops)
    
    return crops


def preprocess_dataset(input_dirs, output_dir, split="train", crop_size=(64, 64, 64), num_crops_per_case=10):
    """Preprocess entire dataset."""
    output_path = Path(output_dir) / f"brats2024_gli_{split}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all case directories
    all_cases = []
    for input_dir in input_dirs:
        cases = sorted([d for d in Path(input_dir).iterdir() if d.is_dir()])
        all_cases.extend(cases)
    
    print(f"Found {len(all_cases)} cases for {split} split")
    
    # Create HDF5 file
    with h5py.File(str(output_path), 'w') as h5f:
        crop_idx = 0
        
        for case_dir in tqdm(all_cases, desc=f"Processing {split}"):
            try:
                crops = process_case(case_dir, crop_size=crop_size, num_crops=num_crops_per_case)
                
                for crop_vol, crop_mask in crops:
                    grp = h5f.create_group(f"crop_{crop_idx:06d}")
                    grp.create_dataset("image", data=crop_vol, compression="gzip", compression_opts=4)
                    grp.create_dataset("mask", data=crop_mask, compression="gzip", compression_opts=4)
                    grp.attrs["case_name"] = str(case_dir.name)
                    crop_idx += 1
                    
            except Exception as e:
                print(f"Error processing {case_dir.name}: {e}")
                continue
        
        h5f.attrs["num_crops"] = crop_idx
        h5f.attrs["crop_size"] = crop_size
        h5f.attrs["num_cases"] = len(all_cases)
    
    print(f"✅ Saved {crop_idx} crops to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS2024 dataset")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed",
                        help="Output directory")
    parser.add_argument("--crops_per_case", type=int, default=10,
                        help="Number of crops per training case")
    args = parser.parse_args()
    
    # Training data (combined folders)
    train_dirs = [
        "h:/FYP/synapsedownloads/Brats2024/BratsGLI/training_data1_v2",
        "h:/FYP/synapsedownloads/Brats2024/BratsGLI/training_data_additional"
    ]
    
    # Validation data
    val_dir = ["h:/FYP/synapsedownloads/Brats2024/BratsGLI/validation_data"]
    
    print("=" * 60)
    print("BraTS 2024 GLI Preprocessing Pipeline")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Crops per case: {args.crops_per_case}")
    print()
    
    # Process training data
    print("Processing training data...")
    preprocess_dataset(
        input_dirs=train_dirs,
        output_dir=args.output_dir,
        split="train",
        crop_size=(64, 64, 64),
        num_crops_per_case=args.crops_per_case
    )
    
    # Process validation data
    print("\nProcessing validation data...")
    preprocess_dataset(
        input_dirs=val_dir,
        output_dir=args.output_dir,
        split="val",
        crop_size=(64, 64, 64),
        num_crops_per_case=5
    )
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
