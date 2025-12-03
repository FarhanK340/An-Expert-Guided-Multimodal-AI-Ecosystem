"""
Tests for data preprocessing module.

Tests data loading, normalization, augmentation, and other
preprocessing operations.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from src.preprocessing.data_preprocessing import DataPreprocessor
from src.preprocessing.dataset_loader import MedicalImageDataset
from src.preprocessing.utils import load_nifti, save_nifti, normalize_volume


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample data
        self.sample_data = np.random.rand(64, 64, 64).astype(np.float32)
        self.sample_affine = np.eye(4)
        
        # Save sample NIfTI file
        self.sample_file = self.temp_path / "sample.nii.gz"
        save_nifti(self.sample_data, self.sample_affine, str(self.sample_file))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_nifti(self):
        """Test NIfTI file loading."""
        data, affine = load_nifti(str(self.sample_file))
        
        assert data.shape == self.sample_data.shape
        assert np.allclose(data, self.sample_data)
        assert np.allclose(affine, self.sample_affine)
    
    def test_save_nifti(self):
        """Test NIfTI file saving."""
        output_file = self.temp_path / "output.nii.gz"
        save_nifti(self.sample_data, self.sample_affine, str(output_file))
        
        assert output_file.exists()
        
        # Load and verify
        loaded_data, loaded_affine = load_nifti(str(output_file))
        assert np.allclose(loaded_data, self.sample_data)
        assert np.allclose(loaded_affine, self.sample_affine)
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        # Test z-score normalization
        normalized = normalize_volume(self.sample_data, "T1", "zscore")
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-6)
        
        # Test min-max normalization
        normalized = normalize_volume(self.sample_data, "T1", "minmax")
        assert np.isclose(np.min(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.max(normalized), 1.0, atol=1e-6)
    
    def test_data_preprocessor(self):
        """Test data preprocessor."""
        preprocessor = DataPreprocessor()
        
        # Test preprocessing
        processed_data = preprocessor.preprocess(self.sample_data, "T1")
        
        assert processed_data.shape == self.sample_data.shape
        assert processed_data.dtype == np.float32
    
    def test_medical_image_dataset(self):
        """Test medical image dataset."""
        # Create sample dataset structure
        dataset_dir = self.temp_path / "dataset"
        dataset_dir.mkdir()
        
        # Create sample case
        case_dir = dataset_dir / "case_001"
        case_dir.mkdir()
        
        # Save sample files
        for modality in ["T1", "T1ce", "T2", "FLAIR"]:
            modality_file = case_dir / f"{modality}.nii.gz"
            save_nifti(self.sample_data, self.sample_affine, str(modality_file))
        
        # Create segmentation file
        segmentation_file = case_dir / "seg.nii.gz"
        segmentation_data = np.random.randint(0, 4, self.sample_data.shape).astype(np.uint8)
        save_nifti(segmentation_data, self.sample_affine, str(segmentation_file))
        
        # Test dataset
        dataset = MedicalImageDataset(str(dataset_dir))
        
        assert len(dataset) == 1
        
        # Test data loading
        sample = dataset[0]
        assert 'T1' in sample
        assert 'T1ce' in sample
        assert 'T2' in sample
        assert 'FLAIR' in sample
        assert 'segmentation' in sample
        
        for modality in ["T1", "T1ce", "T2", "FLAIR"]:
            assert sample[modality].shape == self.sample_data.shape
            assert sample[modality].dtype == torch.float32
        
        assert sample['segmentation'].shape == self.sample_data.shape
        assert sample['segmentation'].dtype == torch.long


class TestAugmentation:
    """Test data augmentation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = np.random.rand(64, 64, 64).astype(np.float32)
        self.sample_segmentation = np.random.randint(0, 4, (64, 64, 64)).astype(np.uint8)
    
    def test_rotation_augmentation(self):
        """Test rotation augmentation."""
        from src.preprocessing.data_preprocessing import RotationAugmentation
        
        aug = RotationAugmentation(rotation_range=15)
        
        augmented_data, augmented_seg = aug(self.sample_data, self.sample_segmentation)
        
        assert augmented_data.shape == self.sample_data.shape
        assert augmented_seg.shape == self.sample_segmentation.shape
    
    def test_flip_augmentation(self):
        """Test flip augmentation."""
        from src.preprocessing.data_preprocessing import FlipAugmentation
        
        aug = FlipAugmentation(flip_probability=0.5)
        
        augmented_data, augmented_seg = aug(self.sample_data, self.sample_segmentation)
        
        assert augmented_data.shape == self.sample_data.shape
        assert augmented_seg.shape == self.sample_segmentation.shape
    
    def test_intensity_augmentation(self):
        """Test intensity augmentation."""
        from src.preprocessing.data_preprocessing import IntensityAugmentation
        
        aug = IntensityAugmentation(
            brightness_range=0.1,
            contrast_range=0.1,
            noise_std=0.01
        )
        
        augmented_data, augmented_seg = aug(self.sample_data, self.sample_segmentation)
        
        assert augmented_data.shape == self.sample_data.shape
        assert augmented_seg.shape == self.sample_segmentation.shape


class TestDataValidation:
    """Test data validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = np.random.rand(64, 64, 64).astype(np.float32)
        self.sample_segmentation = np.random.randint(0, 4, (64, 64, 64)).astype(np.uint8)
    
    def test_data_validation(self):
        """Test data validation."""
        from src.preprocessing.data_preprocessing import DataValidator
        
        validator = DataValidator()
        
        # Test valid data
        is_valid, issues = validator.validate(self.sample_data, self.sample_segmentation)
        assert is_valid
        assert len(issues) == 0
        
        # Test invalid data (negative values)
        invalid_data = self.sample_data.copy()
        invalid_data[0, 0, 0] = -1.0
        
        is_valid, issues = validator.validate(invalid_data, self.sample_segmentation)
        assert not is_valid
        assert len(issues) > 0
    
    def test_segmentation_validation(self):
        """Test segmentation validation."""
        from src.preprocessing.data_preprocessing import SegmentationValidator
        
        validator = SegmentationValidator()
        
        # Test valid segmentation
        is_valid, issues = validator.validate(self.sample_segmentation)
        assert is_valid
        assert len(issues) == 0
        
        # Test invalid segmentation (out of range values)
        invalid_seg = self.sample_segmentation.copy()
        invalid_seg[0, 0, 0] = 10  # Invalid class ID
        
        is_valid, issues = validator.validate(invalid_seg)
        assert not is_valid
        assert len(issues) > 0


if __name__ == "__main__":
    pytest.main([__file__])


