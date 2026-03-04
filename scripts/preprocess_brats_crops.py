"""
Preprocessing script to create crops from BraTS 2024 dataset.
Saves preprocessed crops to local data/preprocessed/ for efficient training.
Default crop size: 128x128x128 for individual expert training.
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


MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]

def preprocess_dataset_with_split(input_dirs, output_dir, train_ratio=0.8, crop_size=(64, 64, 64), num_crops_per_case=10, seed=42):
    """
    Preprocess dataset with train/val split, creating SEPARATE files per modality.
    
    Creates 8 HDF5 files:
    - brats2024_gli_{T1,T1ce,T2,FLAIR}_train.h5
    - brats2024_gli_{T1,T1ce,T2,FLAIR}_val.h5
    
    Each file contains single-channel crops for ~4x faster loading.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all case directories
    all_cases = []
    for input_dir in input_dirs:
        cases = sorted([d for d in Path(input_dir).iterdir() if d.is_dir()])
        all_cases.extend(cases)
    
    # Shuffle and split cases
    np.random.seed(seed)
    np.random.shuffle(all_cases)
    split_idx = int(len(all_cases) * train_ratio)
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    
    print(f"Total cases: {len(all_cases)}")
    print(f"  Training: {len(train_cases)} ({100*train_ratio:.0f}%)")
    print(f"  Validation: {len(val_cases)} ({100*(1-train_ratio):.0f}%)")
    print(f"\nCreating SEPARATE HDF5 files per modality for faster training!")
    
    def process_split(cases, split_name):
        """Process a train/val split and save to separate modality files."""
        # Open 4 HDF5 files (one per modality)
        h5_files = {}
        crop_counts = {}
        
        for mod in MODALITIES:
            filepath = output_path / f"brats2024_gli_{mod}_{split_name}.h5"
            h5_files[mod] = h5py.File(str(filepath), 'w')
            crop_counts[mod] = 0
        
        try:
            for case_dir in tqdm(cases, desc=f"Processing {split_name}"):
                try:
                    crops = process_case(case_dir, crop_size=crop_size, num_crops=num_crops_per_case)
                    
                    for crop_vol, crop_mask in crops:
                        # crop_vol shape: (4, D, H, W) - all 4 modalities
                        # crop_mask shape: (3, D, H, W) - WT, TC, ET
                        
                        # Save each modality to its own file
                        for mod_idx, mod in enumerate(MODALITIES):
                            h5f = h5_files[mod]
                            idx = crop_counts[mod]
                            
                            grp = h5f.create_group(f"crop_{idx:06d}")
                            # Save SINGLE channel (1, D, H, W)
                            single_channel = crop_vol[mod_idx:mod_idx+1]
                            grp.create_dataset("image", data=single_channel, compression="gzip", compression_opts=4)
                            grp.create_dataset("mask", data=crop_mask, compression="gzip", compression_opts=4)
                            grp.attrs["case_name"] = str(case_dir.name)
                            
                            crop_counts[mod] += 1
                            
                except Exception as e:
                    print(f"Error processing {case_dir.name}: {e}")
                    continue
            
            # Save metadata to each file
            for mod in MODALITIES:
                h5_files[mod].attrs["num_crops"] = crop_counts[mod]
                h5_files[mod].attrs["crop_size"] = crop_size
                h5_files[mod].attrs["num_cases"] = len(cases)
                h5_files[mod].attrs["modality"] = mod
                
        finally:
            # Close all files
            for mod in MODALITIES:
                filepath = output_path / f"brats2024_gli_{mod}_{split_name}.h5"
                h5_files[mod].close()
                print(f"✅ Saved {crop_counts[mod]} {mod} crops to {filepath}")
    
    # Process training data
    print(f"\n=== Creating training files ===")
    process_split(train_cases, "train")
    
    # Process validation data
    print(f"\n=== Creating validation files ===")
    process_split(val_cases, "val")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS2024 dataset with train/val split")
    parser.add_argument("--output_dir", type=str, default="./../data/preprocessed",
                        help="Output directory")
    parser.add_argument("--crop_size", type=int, default=128,
                        help="Crop size (default: 128 for 128x128x128)")
    parser.add_argument("--crops_per_case", type=int, default=10,
                        help="Number of crops per case")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of cases for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()
    
    # Training data directories (BraTS val data has NO masks, so we split training data)
    train_dirs = [
        "E:/Farhan FYP/dataset_script/brats_data/brats2024/brats2024-brats-gli-trainingdata/training_data1_v2",
        "E:/Farhan FYP/dataset_script/brats_data/brats2024/brats2024-brats-gli-additionaltrainingdata/training_data_additional"
    ]
    
    crop_size = (args.crop_size, args.crop_size, args.crop_size)
    
    print("=" * 60)
    print("BraTS 2024 GLI Preprocessing Pipeline")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Crop size: {args.crop_size}³")
    print(f"Crops per case: {args.crops_per_case}")
    print(f"Train/Val ratio: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print()
    print("NOTE: Using training data only for both train/val splits")
    print("      (BraTS 2024 validation data has no segmentation masks)")
    print()
    
    preprocess_dataset_with_split(
        input_dirs=train_dirs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        crop_size=crop_size,
        num_crops_per_case=args.crops_per_case,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

