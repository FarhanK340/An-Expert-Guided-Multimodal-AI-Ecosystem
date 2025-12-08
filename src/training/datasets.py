import os
import json
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Optional, Tuple, Dict

class BraTSDataset(Dataset):
    """
    Dataset for BraTS data, supporting both processed and raw data with on-the-fly loading.
    """
    def __init__(self, 
                 data_dir: str, 
                 mode: str = "train", 
                 modalities: List[str] = ["T1", "T1ce", "T2", "FLAIR"],
                 transform=None,
                 split_file: Optional[str] = None):
        """
        Args:
            data_dir: Directory containing the data (raw or processed)
            mode: "train" or "val"
            modalities: List of modalities to load
            transform: Transforms to apply
            split_file: Path to JSON file containing train/val/test splits (required for raw data)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.modalities = modalities
        self.transform = transform
        self.split_file = split_file
        
        self.case_map = {} # Map case name to path
        self.case_names = []
        
        if self.split_file:
            # Loading from raw data with split file
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            
            # Get list of cases for this mode
            # split_file keys are "train_cases", "val_cases", "test_cases"
            key = f"{mode}_cases"
            self.case_names = split_data.get(key, [])
            
            # Index the data directory to find case paths
            # This handles the nested structure of BraTS 2024
            # We assume case names in split_json match folder names
            print(f"Indexing data directory: {self.data_dir}")
            for root, dirs, files in os.walk(self.data_dir):
                for d in dirs:
                    if d in self.case_names:
                        self.case_map[d] = Path(root) / d
                        
            # Check if all cases were found
            found_count = len(self.case_map)
            expected_count = len(self.case_names)
            if found_count < expected_count:
                print(f"Warning: Found {found_count} cases out of {expected_count} expected for {mode} split.")
                # Filter out missing cases
                self.case_names = [c for c in self.case_names if c in self.case_map]
                
        else:
            # Assume processed structure: data_dir/mode/case_name
            search_dir = self.data_dir / mode
            self.case_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])
            self.case_names = [d.name for d in self.case_dirs]
            self.case_map = {d.name: d for d in self.case_dirs}

    def __len__(self):
        return len(self.case_names)
    
    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        case_dir = self.case_map[case_name]
        
        images = []
        for mod in self.modalities:
            img_path = self._find_modality_file(case_dir, mod, case_name)
            
            if not img_path or not img_path.exists():
                raise FileNotFoundError(f"Modality {mod} not found for case {case_name} in {case_dir}")
            
            try:
                nii = nib.load(str(img_path))
                img_data = nii.get_fdata().astype(np.float32)
                images.append(img_data)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                raise
            
        # Stack modalities: (C, H, W, D)
        image_volume = np.stack(images, axis=0)
        
        sample = {"image": image_volume, "case_name": case_name}
        
        # Load mask if available and not test mode
        mask_path = self._find_mask_file(case_dir, case_name)
        if mask_path and mask_path.exists():
            nii_mask = nib.load(str(mask_path))
            mask_data = nii_mask.get_fdata().astype(np.uint8)
            
            # Convert to BraTS regions (WT, TC, ET)
            # WT: Whole Tumor (all labels > 0)
            # TC: Tumor Core (labels 1 and 4/3)
            # ET: Enhancing Tumor (label 4/3)
            # Check if using label 4 or 3 for ET (BraTS versions vary)
            
            wt = mask_data > 0
            # TC includes 1 (NCR) and 4 (ET) (sometimes 3)
            tc = np.logical_or(mask_data == 1, mask_data == 4)
            tc = np.logical_or(tc, mask_data == 3) # Handle 3 just in case
            
            # ET is 4 (or 3)
            et = np.logical_or(mask_data == 4, mask_data == 3)
            
            # Stack channels: (3, H, W, D)
            mask_data = np.stack([wt, tc, et], axis=0).astype(np.float32)
            
            sample["mask"] = mask_data
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def _find_modality_file(self, case_dir: Path, modality: str, case_name: str) -> Optional[Path]:
        """Find modality file handling different naming conventions."""
        # Check standard processed name first
        std_path = case_dir / f"{modality}.nii.gz"
        if std_path.exists():
            return std_path
            
        # Check BraTS 2024 raw naming
        # T1 -> t1n, T1ce -> t1c, T2 -> t2w, FLAIR -> t2f
        suffix_map = {
            "T1": "t1n",
            "T1ce": "t1c",
            "T2": "t2w",
            "FLAIR": "t2f"
        }
        
        if modality in suffix_map:
            suffix = suffix_map[modality]
            # Files are named like BraTS-GLI-00005-100-t1n.nii.gz
            # We search for the file ending with the suffix
            # Use specific pattern to match exact suffix
            files = list(case_dir.glob(f"*-{suffix}.nii.gz"))
            if files:
                return files[0]
                
        return None

    def _find_mask_file(self, case_dir: Path, case_name: str) -> Optional[Path]:
        """Find mask file handling different naming conventions."""
        # Check standard processed name
        std_path = case_dir / "mask.nii.gz"
        if std_path.exists():
            return std_path
            
        # Check BraTS 2024 raw naming
        files = list(case_dir.glob("*-seg.nii.gz"))
        if files:
            return files[0]
            
        return None
