"""
Preprocessing script for BratsMEN (Meningioma) dataset.

Key differences from BratsGLI:
- Only T1c modality (not 4 modalities)
- GTV mask (Gross Tumor Volume) - single class
- 1000 training cases, 70 validation cases
"""

import os
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
        volume: (1, H, W, D) - T1c modality only
        mask: (1, H, W, D) - GTV mask (single class)
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


def process_case(case_dir, crop_size=(64, 64, 64), num_crops=10):
    """Process a single BratsMEN case (T1c only).
    
    Output mask has 4 channels for compatibility with GLI model:
    - Channel 0: WT (Whole Tumor) - zeros for MEN
    - Channel 1: TC (Tumor Core) - zeros for MEN  
    - Channel 2: ET (Enhancing Tumor) - zeros for MEN
    - Channel 3: GTV (Gross Tumor Volume) - MEN target
    """
    case_name = case_dir.name
    
    # Load T1c modality only
    t1c_files = list(case_dir.glob("*_t1c.nii.gz"))
    if not t1c_files:
        raise FileNotFoundError(f"No T1c file found in {case_dir}")
    
    img = nib.load(str(t1c_files[0]))
    img_data = img.get_fdata().astype(np.float32)
    
    # Stack as single channel: (1, H, W, D)
    volume = np.expand_dims(img_data, axis=0)
    volume = normalize_volume(volume)
    
    # Load GTV mask
    mask_files = list(case_dir.glob("*_gtv.nii.gz"))
    if mask_files:
        mask_img = nib.load(str(mask_files[0]))
        mask_data = mask_img.get_fdata().astype(np.float32)
        gtv = (mask_data > 0).astype(np.float32)
        
        # Create 4-channel mask: [WT, TC, ET, GTV]
        # Channels 0-2 are zeros (GLI classes), channel 3 is GTV
        H, W, D = volume.shape[1:]
        mask = np.zeros((4, H, W, D), dtype=np.float32)
        mask[3] = gtv  # GTV in channel 3
    else:
        # No mask found
        mask = np.zeros((4,) + volume.shape[1:], dtype=np.float32)
        num_crops = 5
    
    # Extract crops
    crops = create_crops_from_volume(volume, mask, crop_size, num_crops)
    
    return crops


def preprocess_bratsmen_with_split(input_dirs, output_dir, train_ratio=0.8, crop_size=(64, 64, 64), num_crops_per_case=10, seed=42):
    """
    Preprocess BratsMEN dataset with train/val split.
    """
    output_path_train = Path(output_dir) / "bratsmen_train.h5"
    output_path_val = Path(output_dir) / "bratsmen_val.h5"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all case directories
    all_cases = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            cases = sorted([d for d in input_path.iterdir() if d.is_dir()])
            all_cases.extend(cases)
    
    if len(all_cases) == 0:
        print("ERROR: No cases found!")
        return
    
    # Shuffle and split cases
    np.random.seed(seed)
    np.random.shuffle(all_cases)
    split_idx = int(len(all_cases) * train_ratio)
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    
    print(f"Total cases: {len(all_cases)}")
    print(f"  Training: {len(train_cases)} ({100*train_ratio:.0f}%)")
    print(f"  Validation: {len(val_cases)} ({100*(1-train_ratio):.0f}%)")
    
    # Process training data
    print(f"\n=== Creating training file ===")
    with h5py.File(str(output_path_train), 'w') as h5f:
        crop_idx = 0
        
        for case_dir in tqdm(train_cases, desc="Training"):
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
        h5f.attrs["num_cases"] = len(train_cases)
        h5f.attrs["num_classes"] = 4  # WT, TC, ET, GTV
        h5f.attrs["modalities"] = ["t1c"]
        h5f.attrs["class_names"] = ["WT", "TC", "ET", "GTV"]
    
    print(f"✅ Saved {crop_idx} training crops to {output_path_train}")
    
    # Process validation data
    print(f"\n=== Creating validation file ===")
    with h5py.File(str(output_path_val), 'w') as h5f:
        crop_idx = 0
        
        for case_dir in tqdm(val_cases, desc="Validation"):
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
        h5f.attrs["num_cases"] = len(val_cases)
        h5f.attrs["num_classes"] = 4
        h5f.attrs["modalities"] = ["t1c"]
        h5f.attrs["class_names"] = ["WT", "TC", "ET", "GTV"]
    
    print(f"✅ Saved {crop_idx} validation crops to {output_path_val}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BratsMEN dataset")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed",
                        help="Output directory")
    parser.add_argument("--crops_per_case", type=int, default=10,
                        help="Number of crops per case")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of cases for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()
    
    # BratsMEN data directories
    train_dirs = [
        "h:/FYP/synapsedownloads/Brats2024/BratsMEN/BraTS-MEN-RT-Train-v2"
    ]
    
    print("=" * 60)
    print("BraTS 2024 MEN (Meningioma) Preprocessing Pipeline")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Crops per case: {args.crops_per_case}")
    print(f"Train/Val ratio: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print()
    print("NOTE: BratsMEN has only T1c modality (1 channel)")
    print("      Output: GTV (Gross Tumor Volume) - 1 class")
    print()
    
    preprocess_bratsmen_with_split(
        input_dirs=train_dirs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        crop_size=(64, 64, 64),
        num_crops_per_case=args.crops_per_case,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
