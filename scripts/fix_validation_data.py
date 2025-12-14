"""
Fix validation data by splitting training data 80/20.

This script:
1. Deletes useless old val HDF5 (has no masks)
2. Reads existing training HDF5
3. Splits 80% train / 20% val
4. Creates new files (no backups to save space)
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_training_data(
    train_h5_path: str = "data/preprocessed/brats2024_gli_train.h5",
    val_h5_path: str = "data/preprocessed/brats2024_gli_val.h5",
    train_ratio: float = 0.8,
    seed: int = 42
):
    train_h5_path = Path(train_h5_path)
    val_h5_path = Path(val_h5_path)
    
    # Delete old useless validation file (no masks)
    print("=== Removing old validation file (no masks) ===")
    if val_h5_path.exists():
        val_h5_path.unlink()
        print(f"  Deleted {val_h5_path}")
    
    # Read original training data
    print("\n=== Reading original training data ===")
    with h5py.File(str(train_h5_path), 'r') as h5f:
        num_crops = h5f.attrs["num_crops"]
        crop_size = tuple(h5f.attrs["crop_size"])
        print(f"  Total crops: {num_crops}")
        print(f"  Crop size: {crop_size}")
    
    # Generate random split
    np.random.seed(seed)
    all_indices = np.arange(num_crops)
    np.random.shuffle(all_indices)
    
    split_idx = int(num_crops * train_ratio)
    train_indices = sorted(all_indices[:split_idx])
    val_indices = sorted(all_indices[split_idx:])
    
    print(f"\n=== Split Statistics ===")
    print(f"  Training crops: {len(train_indices)} ({100*train_ratio:.0f}%)")
    print(f"  Validation crops: {len(val_indices)} ({100*(1-train_ratio):.0f}%)")
    
    # Create new training HDF5
    new_train_path = Path("data/preprocessed/brats2024_gli_train_new.h5")
    print(f"\n=== Creating new training file ===")
    
    with h5py.File(str(train_h5_path), 'r') as src_h5:
        with h5py.File(str(new_train_path), 'w') as dst_h5:
            dst_h5.attrs["num_crops"] = len(train_indices)
            dst_h5.attrs["crop_size"] = crop_size
            
            for new_idx, old_idx in enumerate(tqdm(train_indices, desc="  Copying train crops")):
                src_group = src_h5[f"crop_{old_idx:06d}"]
                dst_group = dst_h5.create_group(f"crop_{new_idx:06d}")
                
                dst_group.create_dataset("image", data=src_group["image"][:], compression="gzip", compression_opts=4)
                dst_group.create_dataset("mask", data=src_group["mask"][:], compression="gzip", compression_opts=4)
                dst_group.attrs["case_name"] = src_group.attrs["case_name"]
    
    # Create new validation HDF5
    new_val_path = Path("data/preprocessed/brats2024_gli_val_new.h5")
    print(f"\n=== Creating new validation file ===")
    
    with h5py.File(str(train_h5_path), 'r') as src_h5:
        with h5py.File(str(new_val_path), 'w') as dst_h5:
            dst_h5.attrs["num_crops"] = len(val_indices)
            dst_h5.attrs["crop_size"] = crop_size
            
            for new_idx, old_idx in enumerate(tqdm(val_indices, desc="  Copying val crops")):
                src_group = src_h5[f"crop_{old_idx:06d}"]
                dst_group = dst_h5.create_group(f"crop_{new_idx:06d}")
                
                dst_group.create_dataset("image", data=src_group["image"][:], compression="gzip", compression_opts=4)
                dst_group.create_dataset("mask", data=src_group["mask"][:], compression="gzip", compression_opts=4)
                dst_group.attrs["case_name"] = src_group.attrs["case_name"]
    
    # Replace original train file with new one
    print(f"\n=== Replacing original training file ===")
    
    # Remove old train file
    train_h5_path.unlink()
    
    # Rename new files
    new_train_path.rename(train_h5_path)
    new_val_path.rename(val_h5_path)
    
    print(f"  Replaced {train_h5_path.name}")
    print(f"  Replaced {val_h5_path.name}")
    
    # Verify
    print(f"\n=== Verification ===")
    with h5py.File(str(train_h5_path), 'r') as h5f:
        print(f"  New training crops: {h5f.attrs['num_crops']}")
        # Check a sample has mask
        mask_sum = h5f["crop_000000"]["mask"][:].sum()
        print(f"  Sample train mask sum: {mask_sum:.0f}")
    
    with h5py.File(str(val_h5_path), 'r') as h5f:
        print(f"  New validation crops: {h5f.attrs['num_crops']}")
        # Check a sample has mask
        mask_sum = h5f["crop_000000"]["mask"][:].sum()
        print(f"  Sample val mask sum: {mask_sum:.0f}")
    
    print(f"\n=== DONE! ===")
    print(f"You can now restart training and should see proper Dice scores!")
    print(f"\nIMPORTANT: Update max_crops in config if needed:")
    print(f"  Old max_crops: 6000")
    print(f"  New training size: {len(train_indices)}")
    print(f"  Suggested max_crops: {min(6000, len(train_indices))}")

if __name__ == "__main__":
    split_training_data()
