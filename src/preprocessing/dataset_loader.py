"""
PyTorch Dataset and DataLoader classes for medical imaging data.

Provides efficient loading and batching of multi-modal MRI data
for training and inference.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import Compose
from monai.data import list_data_collate

from .utils import load_nifti
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for medical imaging data.
    
    Handles loading of multi-modal MRI data with optional segmentation masks
    for training and inference.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 modalities: List[str] = None,
                 transform: Optional[Compose] = None,
                 load_masks: bool = True,
                 case_list: Optional[List[str]] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to processed data directory
            split: Dataset split ("train", "val", "test")
            modalities: List of modalities to load
            transform: MONAI transforms to apply
            load_masks: Whether to load segmentation masks
            case_list: Optional list of specific cases to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ["T1", "T1ce", "T2", "FLAIR"]
        self.transform = transform
        self.load_masks = load_masks
        
        # Get list of cases
        if case_list is None:
            self.cases = self._get_case_list()
        else:
            self.cases = case_list
            
        logger.info(f"Loaded {len(self.cases)} cases for {split} split")
    
    def _get_case_list(self) -> List[str]:
        """Get list of case directories."""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        cases = []
        for item in split_dir.iterdir():
            if item.is_dir():
                cases.append(item.name)
                
        return sorted(cases)
    
    def __len__(self) -> int:
        """Return the number of cases in the dataset."""
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing image data and optional mask
        """
        case_name = self.cases[idx]
        case_dir = self.data_dir / self.split / case_name
        
        # Load modality data
        data_dict = {}
        for modality in self.modalities:
            modality_file = case_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                data, affine = load_nifti(str(modality_file))
                data_dict[f"image_{modality}"] = data
            else:
                logger.warning(f"Modality {modality} not found for case {case_name}")
                # Create zero tensor as placeholder
                data_dict[f"image_{modality}"] = np.zeros((128, 128, 128), dtype=np.float32)
        
        # Load segmentation mask if requested
        if self.load_masks:
            mask_file = case_dir / "mask.nii.gz"
            if mask_file.exists():
                mask_data, _ = load_nifti(str(mask_file))
                data_dict["mask"] = mask_data
            else:
                logger.warning(f"Mask not found for case {case_name}")
                # Create zero tensor as placeholder
                data_dict["mask"] = np.zeros((128, 128, 128), dtype=np.int32)
        
        # Add metadata
        data_dict["case_name"] = case_name
        data_dict["affine"] = affine
        
        # Apply transforms
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def get_case_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a specific case.
        
        Args:
            idx: Index of the case
            
        Returns:
            Dictionary with case information
        """
        case_name = self.cases[idx]
        case_dir = self.data_dir / self.split / case_name
        
        info = {
            "case_name": case_name,
            "split": self.split,
            "modalities": [],
            "has_mask": False
        }
        
        # Check available modalities
        for modality in self.modalities:
            modality_file = case_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                info["modalities"].append(modality)
        
        # Check for mask
        mask_file = case_dir / "mask.nii.gz"
        if mask_file.exists():
            info["has_mask"] = True
            
        return info


class ContinualLearningDataset(MedicalImageDataset):
    """
    Dataset for continual learning scenarios.
    
    Extends MedicalImageDataset to handle task-based and lesion-based
    continual learning setups.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 modalities: List[str] = None,
                 transform: Optional[Compose] = None,
                 load_masks: bool = True,
                 task_id: Optional[int] = None,
                 task_name: Optional[str] = None):
        """
        Initialize the continual learning dataset.
        
        Args:
            data_dir: Path to processed data directory
            split: Dataset split ("train", "val", "test")
            modalities: List of modalities to load
            transform: MONAI transforms to apply
            load_masks: Whether to load segmentation masks
            task_id: Task identifier for continual learning
            task_name: Name of the task
        """
        super().__init__(data_dir, split, modalities, transform, load_masks)
        self.task_id = task_id
        self.task_name = task_name
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with task information."""
        data_dict = super().__getitem__(idx)
        
        # Add task information
        if self.task_id is not None:
            data_dict["task_id"] = self.task_id
        if self.task_name is not None:
            data_dict["task_name"] = self.task_name
            
        return data_dict


def get_dataloader(dataset: Dataset,
                  batch_size: int = 2,
                  shuffle: bool = True,
                  num_workers: int = 4,
                  pin_memory: bool = True,
                  drop_last: bool = True) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=list_data_collate
    )


def create_datasets(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from .data_preprocessing import DataPreprocessor
    
    # Get preprocessing configuration
    preprocessing_config = config.get("preprocessing", {})
    modalities = config.get("dataset", {}).get("modalities", ["T1", "T1ce", "T2", "FLAIR"])
    
    # Create preprocessor for transforms
    preprocessor = DataPreprocessor(preprocessing_config)
    
    # Get transforms for each split
    train_transforms = preprocessor.get_transforms("train")
    val_transforms = preprocessor.get_transforms("val")
    test_transforms = preprocessor.get_transforms("test")
    
    # Create datasets
    data_dir = config.get("dataset", {}).get("data_dir", "data/processed")
    
    train_dataset = MedicalImageDataset(
        data_dir=data_dir,
        split="train",
        modalities=modalities,
        transform=train_transforms,
        load_masks=True
    )
    
    val_dataset = MedicalImageDataset(
        data_dir=data_dir,
        split="val",
        modalities=modalities,
        transform=val_transforms,
        load_masks=True
    )
    
    test_dataset = MedicalImageDataset(
        data_dir=data_dir,
        split="test",
        modalities=modalities,
        transform=test_transforms,
        load_masks=True
    )
    
    return train_dataset, val_dataset, test_dataset


def create_continual_learning_datasets(config: Dict) -> List[ContinualLearningDataset]:
    """
    Create datasets for continual learning scenarios.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of datasets for each task
    """
    from .data_preprocessing import DataPreprocessor
    
    # Get continual learning configuration
    cl_config = config.get("continual_learning", {})
    task_configs = cl_config.get("tasks", {}).get("task_based", {}).get("task_order", [])
    
    if not task_configs:
        raise ValueError("No task configuration found for continual learning")
    
    # Get preprocessing configuration
    preprocessing_config = config.get("preprocessing", {})
    modalities = config.get("dataset", {}).get("modalities", ["T1", "T1ce", "T2", "FLAIR"])
    
    # Create preprocessor for transforms
    preprocessor = DataPreprocessor(preprocessing_config)
    train_transforms = preprocessor.get_transforms("train")
    
    datasets = []
    data_dir = config.get("dataset", {}).get("data_dir", "data/processed")
    
    for task_id, task_name in enumerate(task_configs):
        dataset = ContinualLearningDataset(
            data_dir=data_dir,
            split="train",
            modalities=modalities,
            transform=train_transforms,
            load_masks=True,
            task_id=task_id,
            task_name=task_name
        )
        datasets.append(dataset)
    
    return datasets

