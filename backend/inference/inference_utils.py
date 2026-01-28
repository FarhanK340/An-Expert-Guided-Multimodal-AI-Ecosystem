"""
Inference utilities for brain tumor segmentation.

Handles model loading, inference execution, and result saving.
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

from django.conf import settings

# Add project root directory to path to import model modules from src
project_root = Path(settings.BASE_DIR).parent
sys.path.insert(0, str(project_root))

from src.models.mome_segmenter import MoMESegmenter
from cases.models import Case, MRIImage, SegmentationResult
from .preprocessing_pipeline import InferencePreprocessor


class InferenceEngine:
    """
    Main inference engine for brain tumor segmentation.
    
    Handles model loading, preprocessing, inference, and result storage.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint (uses default if None)
            device: Device to use (cuda/cpu, auto-detects if None)
        """
        self.model = None
        self.device = self._setup_device(device)
        self.model_path = model_path or self._get_default_model_path()
        self.preprocessor = InferencePreprocessor()
        
    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """
        Setup compute device with GPU priority.
        
        Args:
            device: Device string or None for auto-detection
            
        Returns:
            torch.device object
        """
        if device is not None:
            return torch.device(device)
        
        # Auto-detect: prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU")
        
        return device
    
    def _get_default_model_path(self) -> str:
        """Get default model checkpoint path."""
        return os.path.join(
            settings.BASE_DIR,
            'models',
            'checkpoints',
            'mome_segmenter.pth'
        )
    
    def load_model(self) -> None:
        """
        Load the segmentation model from checkpoint.
        
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                f"Please upload the trained model to backend/models/checkpoints/mome_segmenter.pth"
            )
        
        print(f"Loading model from {self.model_path}...")
        
        # Initialize model architecture
        self.model = MoMESegmenter(
            modalities=['T1', 'T1ce', 'T2', 'FLAIR'],
            in_channels=1,
            num_classes=3,
            base_channels=32,
            depth=4,
            attention_type='cbam',
            gating_hidden_channels=[64, 32, 16],
            fusion_method='weighted',
            use_batch_norm=True,
            dropout=0.1
        )
        
        # Load checkpoint (weights_only=False is safe for trusted model checkpoints)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def run_inference(self, case_id: str) -> Dict:
        """
        Run inference on a case.
        
        Args:
            case_id: UUID of the case
            
        Returns:
            Dictionary containing segmentation results and metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get case and verify all modalities exist
        case = Case.objects.get(case_id=case_id)
        mri_images = MRIImage.objects.filter(case=case)
        
        if mri_images.count() < 4:
            raise ValueError(
                f"Case requires all 4 modalities. Found {mri_images.count()}/4."
            )
        
        # Get case directory
        case_dir = Path(settings.MEDIA_ROOT) / 'cases' / str(case_id)
        
        # Preprocess MRI volumes
        print(f"Preprocessing case {case_id}...")
        preprocessed_data = self.preprocessor.load_and_preprocess_case(case_dir)
        
        # Move to device
        for modality in preprocessed_data:
            preprocessed_data[modality] = preprocessed_data[modality].to(self.device)
        
        # Run inference
        print(f"Running inference on {self.device}...")
        with torch.no_grad():
            outputs = self.model(preprocessed_data)
        
        # Get segmentation output
        segmentation = outputs['segmentation']  # [B, 3, H, W, D]
        
        # Convert to binary masks and numpy
        segmentation_np = segmentation.cpu().numpy()[0]  # [3, H, W, D]
        
        # Apply softmax and threshold
        segmentation_probs = torch.softmax(segmentation, dim=1).cpu().numpy()[0]
        segmentation_binary = (segmentation_probs > 0.5).astype(np.uint8)
        
        # Calculate volumes (in voxels, convert to mm³ later with spacing)
        volumes = {
            'whole_tumor': float(np.sum(segmentation_binary[0])),
            'tumor_core': float(np.sum(segmentation_binary[1])),
            'enhancing_tumor': float(np.sum(segmentation_binary[2]))
        }
        
        # Calculate confidence scores (mean probability in predicted regions)
        confidence_scores = {
            'whole_tumor': float(np.mean(segmentation_probs[0][segmentation_binary[0] > 0])) if volumes['whole_tumor'] > 0 else 0.0,
            'tumor_core': float(np.mean(segmentation_probs[1][segmentation_binary[1] > 0])) if volumes['tumor_core'] > 0 else 0.0,
            'enhancing_tumor': float(np.mean(segmentation_probs[2][segmentation_binary[2] > 0])) if volumes['enhancing_tumor'] > 0 else 0.0
        }
        
        # Save results
        result_data = self.save_segmentation_result(
            case_id=case_id,
            segmentation_masks=segmentation_binary,
            volumes=volumes,
            confidence_scores=confidence_scores,
            expert_weights=outputs.get('expert_weights'),
            spatial_attention=outputs.get('spatial_attention')
        )
        
        return result_data
    
    def save_segmentation_result(self, 
                                 case_id: str,
                                 segmentation_masks: np.ndarray,
                                 volumes: Dict[str, float],
                                 confidence_scores: Dict[str, float],
                                 expert_weights: Optional[torch.Tensor] = None,
                                 spatial_attention: Optional[torch.Tensor] = None) -> Dict:
        """
        Save segmentation results to database and files.
        
        Args:
            case_id: UUID of the case
            segmentation_masks: Binary segmentation masks [3, H, W, D]
            volumes: Dictionary of tumor volumes
            confidence_scores: Dictionary of confidence scores
            expert_weights: Expert weights from gating network
            spatial_attention: Spatial attention maps
            
        Returns:
            Dictionary with saved result information
        """
        case = Case.objects.get(case_id=case_id)
        case_dir = Path(settings.MEDIA_ROOT) / 'cases' / str(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masks as NIfTI files
        mask_files = {}
        affine = np.eye(4)  # Identity affine (can be improved with actual affine from input)
        
        for idx, mask_name in enumerate(['whole_tumor', 'tumor_core', 'enhancing_tumor']):
            mask_path = case_dir / f'{mask_name}_mask.nii.gz'
            nifti_img = nib.Nifti1Image(segmentation_masks[idx], affine)
            nib.save(nifti_img, str(mask_path))
            mask_files[mask_name] = f'cases/{case_id}/{mask_name}_mask.nii.gz'
        
        # Create or update SegmentationResult
        structured_findings = {
            'volumes': volumes,
            'confidence_scores': confidence_scores,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'MoME+ v1.0',
            'device': str(self.device)
        }
        
        # Convert volumes from voxels to mm³ (assuming 1mm³ voxels)
        volume_mm3 = {k: v for k, v in volumes.items()}
        
        result, created = SegmentationResult.objects.update_or_create(
            case=case,
            defaults={
                'whole_tumor_mask': mask_files['whole_tumor'],
                'tumor_core_mask': mask_files['tumor_core'],
                'enhancing_tumor_mask': mask_files['enhancing_tumor'],
                'whole_tumor_volume': volume_mm3['whole_tumor'],
                'tumor_core_volume': volume_mm3['tumor_core'],
                'enhancing_tumor_volume': volume_mm3['enhancing_tumor'],
                'whole_tumor_confidence': confidence_scores['whole_tumor'],
                'tumor_core_confidence': confidence_scores['tumor_core'],
                'enhancing_tumor_confidence': confidence_scores['enhancing_tumor'],
                'structured_findings': structured_findings
            }
        )
        
        # Update case status
        case.status = 'completed'
        case.save()
        
        return {
            'case_id': str(case_id),
            'volumes': volume_mm3,
            'confidence_scores': confidence_scores,
            'mask_files': mask_files,
            'created': created
        }


def run_case_inference(case_id: str) -> Dict:
    """
    Convenience function to run inference on a case.
    
    Args:
        case_id: UUID of the case
        
    Returns:
        Dictionary with inference results
    """
    engine = InferenceEngine()
    return engine.run_inference(case_id)
