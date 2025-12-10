"""
Dataset loader for preprocessed BraTS crops stored in HDF5 format.
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any
from pathlib import Path


class PreprocessedBraTSDataset(Dataset):
    """
    Dataset for loading preprocessed 64x64x64 crops from HDF5 files.
    Much faster and more memory-efficient than loading full volumes.
    """
    
    def __init__(self, h5_path: str, transform=None):
        """
        Args:
            h5_path: Path to HDF5 file (e.g., data/preprocessed/brats2024_gli_train.h5)
            transform: Optional MONAI transforms to apply
        """
        self.h5_path = Path(h5_path)
        self.transform = transform
        
        # Open HDF5 file to get metadata
        with h5py.File(str(self.h5_path), 'r') as h5f:
            self.num_crops = h5f.attrs["num_crops"]
            self.crop_size = tuple(h5f.attrs["crop_size"])
        
        # Keep file open for duration (faster access)
        self.h5_file = None
    
    def __len__(self):
        return self.num_crops
    
    def _open_h5(self):
        """Lazy loading of HDF5 file."""
        if self.h5_file is None:
            self.h5_file = h5py.File(str(self.h5_path), 'r')
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        self._open_h5()
        
        # Load crop from HDF5
        crop_group = self.h5_file[f"crop_{idx:06d}"]
        image = crop_group["image"][:]  # Shape: (4, 64, 64, 64)
        mask = crop_group["mask"][:]    # Shape: (3, 64, 64, 64)
        case_name = crop_group.attrs["case_name"]
        
        # Convert to dict format expected by training loop
        sample = {
            "image": image.astype(np.float32),
            "mask": mask.astype(np.float32),
            "case_name": case_name
        }
        
        # Apply transforms if any (e.g., random flips, noise)
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()
