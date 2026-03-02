"""
MoME+ Inference Engine

Performs full-volume inference on BraTS MRI scans using sliding window
approach with overlap for smooth predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from monai.inferers import sliding_window_inference

from ..models.mome_segmenter import MoMESegmenter
from ..models.mome_expert import ModalityExpert
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Inference engine for MoME+ brain tumor segmentation.
    
    Supports both single-expert inference and full MoME+ fusion inference.
    Uses sliding window approach to handle full-resolution volumes.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        sw_batch_size: int = 4,
        use_amp: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model (MoMESegmenter or ModalityExpert)
            device: Device to run inference on
            roi_size: Size of sliding window (should match training crop size)
            overlap: Overlap ratio between windows (0.5 = 50% overlap)
            sw_batch_size: Number of windows to process in parallel
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.use_amp = use_amp
        
        logger.info(f"InferenceEngine initialized on {device}")
        logger.info(f"ROI size: {roi_size}, Overlap: {overlap*100}%")
    
    @staticmethod
    def normalize_zscore(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score normalize a volume.
        
        Args:
            volume: Input volume
            mask: Optional brain mask (normalize only within mask)
            
        Returns:
            Normalized volume
        """
        if mask is not None:
            # Normalize only within brain region
            brain_voxels = volume[mask > 0]
            mean = brain_voxels.mean()
            std = brain_voxels.std() + 1e-8
        else:
            # Normalize entire volume (excluding zeros)
            non_zero = volume[volume > 0]
            if len(non_zero) > 0:
                mean = non_zero.mean()
                std = non_zero.std() + 1e-8
            else:
                mean, std = 0.0, 1.0
        
        return (volume - mean) / std
    
    def load_nifti(self, path: Union[str, Path]) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """
        Load and normalize a NIfTI file.
        
        Args:
            path: Path to NIfTI file
            
        Returns:
            Tuple of (normalized array, original nifti object for affine)
        """
        nifti = nib.load(str(path))
        data = nifti.get_fdata().astype(np.float32)
        normalized = self.normalize_zscore(data)
        return normalized, nifti
    
    def predict_single_expert(
        self,
        volume: np.ndarray,
        expert: ModalityExpert
    ) -> np.ndarray:
        """
        Run inference with a single modality expert.
        
        Args:
            volume: Normalized volume (D, H, W)
            expert: Trained ModalityExpert
            
        Returns:
            Predicted segmentation (D, H, W) with class labels
        """
        # Prepare input: (1, 1, D, H, W)
        input_tensor = torch.from_numpy(volume).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        def predictor(x):
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output, _ = expert(x)
                else:
                    output, _ = expert(x)
                return output
        
        # Sliding window inference
        prediction = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=predictor,
            overlap=self.overlap,
            mode='gaussian'  # Gaussian weighting for smooth boundaries
        )
        
        # Convert to class labels
        # prediction shape: (1, 3, D, H, W) -> (D, H, W)
        probs = F.softmax(prediction, dim=1)
        labels = torch.argmax(probs, dim=1).squeeze(0)
        
        return labels.cpu().numpy()
    
    def predict_full_mome(
        self,
        modality_volumes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with full MoME+ model (all experts + fusion).
        
        Args:
            modality_volumes: Dict of {modality_name: normalized_volume}
                Expected keys: ["T1", "T1ce", "T2", "FLAIR"]
                
        Returns:
            Dict containing:
                - "segmentation": Final fused segmentation (D, H, W)
                - "probabilities": Per-class probabilities (3, D, H, W)
                - "expert_weights": Per-voxel expert weights (optional)
        """
        # Verify we have a MoMESegmenter
        if not isinstance(self.model, MoMESegmenter):
            raise ValueError("Full MoME prediction requires MoMESegmenter model")
        
        # Get volume shape from first modality
        first_key = list(modality_volumes.keys())[0]
        volume_shape = modality_volumes[first_key].shape
        
        # Prepare inputs
        inputs = {}
        for modality, volume in modality_volumes.items():
            tensor = torch.from_numpy(volume).float()
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            inputs[modality] = tensor
        
        def predictor(stacked_input):
            """Predictor that unpacks stacked input back to dict."""
            # stacked_input: (B, 4, D, H, W) -> split into modality dict
            with torch.no_grad():
                modality_dict = {
                    "T1": stacked_input[:, 0:1],
                    "T1ce": stacked_input[:, 1:2],
                    "T2": stacked_input[:, 2:3],
                    "FLAIR": stacked_input[:, 3:4]
                }
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(modality_dict)
                else:
                    output = self.model(modality_dict)
                return output["segmentation"]
        
        # Stack all modalities: (1, 4, D, H, W)
        stacked = torch.cat([
            inputs.get("T1", torch.zeros(1, 1, *volume_shape, device=self.device)),
            inputs.get("T1ce", torch.zeros(1, 1, *volume_shape, device=self.device)),
            inputs.get("T2", torch.zeros(1, 1, *volume_shape, device=self.device)),
            inputs.get("FLAIR", torch.zeros(1, 1, *volume_shape, device=self.device))
        ], dim=1)
        
        # Sliding window inference
        prediction = sliding_window_inference(
            inputs=stacked,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=predictor,
            overlap=self.overlap,
            mode='gaussian'
        )
        
        # Process outputs
        probs = F.softmax(prediction, dim=1)
        labels = torch.argmax(probs, dim=1).squeeze(0)
        
        return {
            "segmentation": labels.cpu().numpy(),
            "probabilities": probs.squeeze(0).cpu().numpy()
        }
    
    def save_prediction(
        self,
        prediction: np.ndarray,
        reference_nifti: nib.Nifti1Image,
        output_path: Union[str, Path]
    ):
        """
        Save prediction as NIfTI file with same affine as reference.
        
        Args:
            prediction: Predicted segmentation (D, H, W)
            reference_nifti: Reference NIfTI for affine/header
            output_path: Path to save output
        """
        output_nifti = nib.Nifti1Image(
            prediction.astype(np.uint8),
            affine=reference_nifti.affine,
            header=reference_nifti.header
        )
        nib.save(output_nifti, str(output_path))
        logger.info(f"Saved prediction to {output_path}")


def run_single_expert_inference(
    checkpoint_path: str,
    modality: str,
    input_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Convenience function to run single-expert inference.
    
    Args:
        checkpoint_path: Path to trained expert checkpoint
        modality: Modality name (T1, T1ce, T2, FLAIR)
        input_path: Path to input NIfTI file
        output_path: Path for output segmentation
        device: Device to run on
    """
    # Load model
    model = ModalityExpert(
        modality=modality,
        in_channels=1,
        num_classes=3,
        base_channels=32,
        depth=4
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded {modality} expert from {checkpoint_path}")
    
    # Create engine
    engine = InferenceEngine(model, device=device)
    
    # Load and predict
    volume, reference = engine.load_nifti(input_path)
    prediction = engine.predict_single_expert(volume, model)
    
    # Save
    engine.save_prediction(prediction, reference, output_path)
    return prediction


def run_full_inference(
    checkpoint_path: str,
    input_paths: Dict[str, str],
    output_path: str,
    device: str = "cuda"
):
    """
    Convenience function to run full MoME+ inference.
    
    Args:
        checkpoint_path: Path to trained MoME+ checkpoint
        input_paths: Dict of {modality: path_to_nifti}
        output_path: Path for output segmentation
        device: Device to run on
    """
    # Load model
    model = MoMESegmenter()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded MoME+ model from {checkpoint_path}")
    
    # Create engine
    engine = InferenceEngine(model, device=device)
    
    # Load all modalities
    volumes = {}
    reference = None
    for modality, path in input_paths.items():
        volume, nifti = engine.load_nifti(path)
        volumes[modality] = volume
        if reference is None:
            reference = nifti
    
    # Predict
    result = engine.predict_full_mome(volumes)
    
    # Save
    engine.save_prediction(result["segmentation"], reference, output_path)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MoME+ Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["single", "full"], default="single",
                        help="Inference mode: 'single' for one expert, 'full' for MoME+")
    parser.add_argument("--modality", type=str, choices=["T1", "T1ce", "T2", "FLAIR"],
                        help="Modality for single-expert mode")
    parser.add_argument("--input", type=str, required=True,
                        help="Input NIfTI path (for single) or directory (for full)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output segmentation path")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.modality:
            raise ValueError("--modality required for single-expert mode")
        run_single_expert_inference(
            args.checkpoint, args.modality, args.input, args.output, args.device
        )
    else:
        # For full mode, expect input to be a directory with modality files
        input_dir = Path(args.input)
        input_paths = {
            "T1": str(input_dir / "*t1n*.nii.gz"),
            "T1ce": str(input_dir / "*t1c*.nii.gz"),
            "T2": str(input_dir / "*t2w*.nii.gz"),
            "FLAIR": str(input_dir / "*t2f*.nii.gz")
        }
        run_full_inference(args.checkpoint, input_paths, args.output, args.device)
