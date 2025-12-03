"""
Data preprocessing pipeline for medical imaging datasets.

Handles loading, normalization, skull stripping, and augmentation
of multi-modal MRI data for training and inference.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandSpatialCropd,
    RandFlipd, RandRotate90d, RandGaussianNoised, RandGaussianSmoothd,
    ToTensord, EnsureChannelFirstd
)
from monai.data import Dataset, DataLoader
import torch

from .utils import load_nifti, save_nifti, normalize_volume
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Main data preprocessing class for medical imaging datasets.
    
    Handles preprocessing of BraTS, OASIS, and ISLES datasets with
    normalization, skull stripping, and augmentation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.modalities = config.get("modalities", ["T1", "T1ce", "T2", "FLAIR"])
        self.target_spacing = config.get("target_spacing", [1.0, 1.0, 1.0])
        self.crop_size = config.get("crop_size", [128, 128, 128])
        
    def get_transforms(self, mode: str = "train") -> Compose:
        """
        Get preprocessing transforms for the specified mode.
        
        Args:
            mode: "train", "val", or "test"
            
        Returns:
            MONAI Compose transform
        """
        keys = ["image"] + [f"image_{mod}" for mod in self.modalities]
        
        if mode == "train":
            transforms = Compose([
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                Spacingd(keys=keys, pixdim=self.target_spacing, mode="bilinear"),
                Orientationd(keys=keys, axcodes="RAS"),
                ScaleIntensityRanged(keys=keys, a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
                CropForegroundd(keys=keys, source_key="image"),
                RandSpatialCropd(keys=keys, roi_size=self.crop_size, random_size=False),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
                RandGaussianNoised(keys=keys, prob=0.1, mean=0.0, std=0.1),
                RandGaussianSmoothd(keys=keys, prob=0.1, sigma_x=(0.5, 1.0)),
                ToTensord(keys=keys)
            ])
        else:
            transforms = Compose([
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                Spacingd(keys=keys, pixdim=self.target_spacing, mode="bilinear"),
                Orientationd(keys=keys, axcodes="RAS"),
                ScaleIntensityRanged(keys=keys, a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
                CropForegroundd(keys=keys, source_key="image"),
                ToTensord(keys=keys)
            ])
            
        return transforms
    
    def preprocess_dataset(self, 
                          input_dir: str, 
                          output_dir: str,
                          dataset_name: str,
                          split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> None:
        """
        Preprocess an entire dataset and split into train/val/test.
        
        Args:
            input_dir: Path to raw dataset
            output_dir: Path to save processed data
            dataset_name: Name of the dataset (BraTS2021, OASIS, ISLES)
            split_ratio: Train/val/test split ratios
        """
        logger.info(f"Preprocessing {dataset_name} dataset...")
        
        # Find all case directories
        case_dirs = self._find_case_directories(input_dir, dataset_name)
        logger.info(f"Found {len(case_dirs)} cases")
        
        # Split cases
        train_cases, val_cases, test_cases = self._split_cases(case_dirs, split_ratio)
        
        # Process each split
        for split_name, cases in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
            self._process_split(cases, output_dir, split_name, dataset_name)
            
        # Save split information
        self._save_split_info(dataset_name, train_cases, val_cases, test_cases, output_dir)
        
        logger.info(f"Preprocessing completed for {dataset_name}")
    
    def _find_case_directories(self, input_dir: str, dataset_name: str) -> List[str]:
        """Find all case directories in the dataset."""
        case_dirs = []
        
        if dataset_name == "BraTS2021":
            # BraTS structure: BraTS2021_Training_XXX/
            pattern = "BraTS2021_Training_*"
        elif dataset_name == "OASIS":
            # OASIS structure: OASIS_XXX/
            pattern = "OASIS_*"
        elif dataset_name == "ISLES":
            # ISLES structure: case_XXX/
            pattern = "case_*"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        for item in Path(input_dir).glob(pattern):
            if item.is_dir():
                case_dirs.append(str(item))
                
        return sorted(case_dirs)
    
    def _split_cases(self, case_dirs: List[str], split_ratio: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
        """Split cases into train/val/test sets."""
        np.random.seed(42)
        indices = np.random.permutation(len(case_dirs))
        
        train_end = int(len(case_dirs) * split_ratio[0])
        val_end = train_end + int(len(case_dirs) * split_ratio[1])
        
        train_cases = [case_dirs[i] for i in indices[:train_end]]
        val_cases = [case_dirs[i] for i in indices[train_end:val_end]]
        test_cases = [case_dirs[i] for i in indices[val_end:]]
        
        return train_cases, val_cases, test_cases
    
    def _process_split(self, cases: List[str], output_dir: str, split_name: str, dataset_name: str) -> None:
        """Process a single split of the dataset."""
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for case_path in cases:
            case_name = Path(case_path).name
            output_case_dir = split_dir / case_name
            output_case_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                self._process_single_case(case_path, str(output_case_dir), dataset_name)
            except Exception as e:
                logger.error(f"Error processing case {case_name}: {e}")
                continue
    
    def _process_single_case(self, case_dir: str, output_dir: str, dataset_name: str) -> None:
        """Process a single case directory."""
        case_path = Path(case_dir)
        
        # Load all modalities
        modality_data = {}
        for modality in self.modalities:
            modality_file = self._find_modality_file(case_path, modality, dataset_name)
            if modality_file:
                data, affine = load_nifti(str(modality_file))
                modality_data[modality] = data
            else:
                logger.warning(f"Modality {modality} not found in {case_dir}")
        
        # Load segmentation mask if available
        mask_file = self._find_mask_file(case_path, dataset_name)
        if mask_file:
            mask_data, _ = load_nifti(str(mask_file))
            modality_data["mask"] = mask_data
        
        # Normalize and save
        for modality, data in modality_data.items():
            normalized_data = normalize_volume(data, modality)
            output_file = Path(output_dir) / f"{modality}.nii.gz"
            save_nifti(normalized_data, affine, str(output_file))
    
    def _find_modality_file(self, case_path: Path, modality: str, dataset_name: str) -> Optional[Path]:
        """Find the file for a specific modality."""
        if dataset_name == "BraTS2021":
            return case_path / f"{case_path.name}_{modality}.nii.gz"
        elif dataset_name == "OASIS":
            return case_path / f"mpr-1_{modality}.nii.gz"
        elif dataset_name == "ISLES":
            return case_path / f"{modality}.nii.gz"
        return None
    
    def _find_mask_file(self, case_path: Path, dataset_name: str) -> Optional[Path]:
        """Find the segmentation mask file."""
        if dataset_name == "BraTS2021":
            return case_path / f"{case_path.name}_seg.nii.gz"
        elif dataset_name == "OASIS":
            return case_path / "mpr-1_seg.nii.gz"
        elif dataset_name == "ISLES":
            return case_path / "mask.nii.gz"
        return None
    
    def _save_split_info(self, dataset_name: str, train_cases: List[str], 
                        val_cases: List[str], test_cases: List[str], output_dir: str) -> None:
        """Save split information to JSON file."""
        split_info = {
            "dataset": dataset_name,
            "train_cases": [Path(c).name for c in train_cases],
            "val_cases": [Path(c).name for c in val_cases],
            "test_cases": [Path(c).name for c in test_cases],
            "total_cases": len(train_cases) + len(val_cases) + len(test_cases)
        }
        
        output_file = Path(output_dir) / f"{dataset_name}_split.json"
        with open(output_file, 'w') as f:
            json.dump(split_info, f, indent=2)


def main():
    """Main function for command-line usage."""
    import argparse
    from ..utils.config_parser import load_config
    
    parser = argparse.ArgumentParser(description="Preprocess medical imaging datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    preprocessor = DataPreprocessor(config["preprocessing"])
    
    preprocessor.preprocess_dataset(
        args.input_dir, 
        args.output_dir, 
        args.dataset
    )


if __name__ == "__main__":
    main()

