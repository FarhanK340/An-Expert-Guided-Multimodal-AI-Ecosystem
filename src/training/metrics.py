"""
Metrics for medical image segmentation evaluation.

Implements Dice score, IoU, precision, recall, and other segmentation metrics
for evaluating the MoME+ model performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DiceScore:
    """
    Dice coefficient computation for medical image segmentation.
    """
    
    def __init__(self, 
                 smooth: float = 1e-6,
                 ignore_index: int = -1):
        """
        Initialize Dice score calculator.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in computation
        """
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def compute(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                class_wise: bool = True) -> Union[float, Dict[int, float]]:
        """
        Compute Dice score.
        """
        # Check for multi-label/one-hot targets
        if targets.shape == predictions.shape:
             targets_one_hot = targets.float()
             # Use Sigmoid for multi-label
             predictions = torch.sigmoid(predictions)
             # Threshold predictions for binary segmentation
             predictions_binary = (predictions > 0.5).float()
        else:
             # Convert targets to one-hot encoding
             targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
             # Use Softmax for multi-class
             predictions = F.softmax(predictions, dim=1)
             # Convert to one-hot by taking argmax
             predictions_binary = F.one_hot(predictions.argmax(dim=1), num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Compute intersection and union using BINARY predictions
        intersection = (predictions_binary * targets_one_hot).sum(dim=(2, 3, 4))
        union = predictions_binary.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        if class_wise:
            return {i: dice[:, i].mean().item() for i in range(predictions.size(1))}
        else:
            return dice.mean().item()


class IoUScore:
    """
    Intersection over Union (IoU) computation for medical image segmentation.
    """
    
    def __init__(self, 
                 smooth: float = 1e-6,
                 ignore_index: int = -1):
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def compute(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                class_wise: bool = True) -> Union[float, Dict[int, float]]:
        """
        Compute IoU score.
        """
        # Check for multi-label/one-hot targets
        if targets.shape == predictions.shape:
             targets_one_hot = targets.float()
             predictions = torch.sigmoid(predictions)
             predictions_binary = (predictions > 0.5).float()
        else:
             targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
             predictions = F.softmax(predictions, dim=1)
             predictions_binary = F.one_hot(predictions.argmax(dim=1), num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Compute intersection and union using BINARY predictions
        intersection = (predictions_binary * targets_one_hot).sum(dim=(2, 3, 4))
        union = predictions_binary.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4)) - intersection
        
        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        if class_wise:
            return {i: iou[:, i].mean().item() for i in range(predictions.size(1))}
        else:
            return iou.mean().item()


class PrecisionRecall:
    """
    Precision and recall computation for medical image segmentation.
    """
    
    def __init__(self, 
                 smooth: float = 1e-6,
                 ignore_index: int = -1):
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def compute(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                class_wise: bool = True) -> Dict[str, Union[float, Dict[int, float]]]:
        """
        Compute precision and recall.
        """
        # Check for multi-label/one-hot targets
        if targets.shape == predictions.shape:
             targets_one_hot = targets.float()
             predictions = torch.sigmoid(predictions)
             predictions_binary = (predictions > 0.5).float()
        else:
             targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
             predictions = F.softmax(predictions, dim=1)
             predictions_binary = F.one_hot(predictions.argmax(dim=1), num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Compute true positives, false positives, and false negatives using BINARY predictions
        tp = (predictions_binary * targets_one_hot).sum(dim=(2, 3, 4))
        fp = (predictions_binary * (1 - targets_one_hot)).sum(dim=(2, 3, 4))
        fn = ((1 - predictions_binary) * targets_one_hot).sum(dim=(2, 3, 4))
        
        # Compute precision and recall
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        if class_wise:
            return {
                "precision": {i: precision[:, i].mean().item() for i in range(predictions.size(1))},
                "recall": {i: recall[:, i].mean().item() for i in range(predictions.size(1))}
            }
        else:
            return {
                "precision": precision.mean().item(),
                "recall": recall.mean().item()
            }


class HausdorffDistance:
    """
    Hausdorff distance computation for medical image segmentation.
    """
    
    def __init__(self, 
                 percentile: float = 95.0,
                 ignore_index: int = -1):
        """
        Initialize Hausdorff distance calculator.
        
        Args:
            percentile: Percentile for robust Hausdorff distance
            ignore_index: Index to ignore in computation
        """
        self.percentile = percentile
        self.ignore_index = ignore_index
    
    def compute(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                class_wise: bool = True) -> Union[float, Dict[int, float]]:
        """
        Compute Hausdorff distance.
        
        Args:
            predictions: Predicted segmentation masks (B, C, D, H, W)
            targets: Ground truth masks (B, D, H, W)
            class_wise: Whether to compute class-wise scores
            
        Returns:
            Hausdorff distance(s)
        """
        # Check for multi-label/one-hot targets
        if targets.shape == predictions.shape:
             # Multi-label case (BraTS)
             # predictions: (B, C, D, H, W), targets: (B, C, D, H, W)
             num_classes = predictions.size(1)
             predictions_binary = (torch.sigmoid(predictions) > 0.5).float()
             targets_binary = targets.float()
        else:
             # Multi-class case
             num_classes = predictions.size(1)
             predictions = F.softmax(predictions, dim=1)
             predictions_binary = torch.zeros_like(predictions)
             predictions_argmax = torch.argmax(predictions, dim=1)
             for c in range(num_classes):
                 predictions_binary[:, c] = (predictions_argmax == c).float()
             
             targets_binary = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        # Compute Hausdorff distance for each class
        hausdorff_distances = {}
        
        for class_id in range(num_classes):
            # Get binary masks for this class
            pred_mask = predictions_binary[:, class_id]
            target_mask = targets_binary[:, class_id]
            
            # Compute Hausdorff distance
            hd = self._compute_hausdorff_distance(pred_mask, target_mask)
            hausdorff_distances[class_id] = hd
        
        if class_wise:
            return hausdorff_distances
        else:
            return np.mean([v for v in hausdorff_distances.values() if v != float('inf')])
    
    def _compute_hausdorff_distance(self, 
                                  pred_mask: torch.Tensor, 
                                  target_mask: torch.Tensor) -> float:
        """
        Compute Hausdorff distance using fast distance transform.
        
        Uses scipy's distance_transform_edt which is O(n) instead of O(n*m).
        """
        from scipy.ndimage import distance_transform_edt
        
        # Convert to numpy
        pred_mask = pred_mask.cpu().numpy().astype(bool)
        target_mask = target_mask.cpu().numpy().astype(bool)
        
        # Check for empty masks
        if not np.any(pred_mask) or not np.any(target_mask):
            return float('inf')
        
        # Compute distance transforms (distance from each voxel to nearest foreground)
        # For pred->target: distance transform of ~target, sampled at pred locations
        dist_pred_to_target = distance_transform_edt(~target_mask)
        distances_p2t = dist_pred_to_target[pred_mask]
        
        # For target->pred: distance transform of ~pred, sampled at target locations  
        dist_target_to_pred = distance_transform_edt(~pred_mask)
        distances_t2p = dist_target_to_pred[target_mask]
        
        # Hausdorff 95 = max of 95th percentile of both directions
        hd95_p2t = np.percentile(distances_p2t, self.percentile) if len(distances_p2t) > 0 else 0
        hd95_t2p = np.percentile(distances_t2p, self.percentile) if len(distances_t2p) > 0 else 0
        
        return float(max(hd95_p2t, hd95_t2p))


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    
    Computes multiple metrics for medical image segmentation evaluation.
    """
    
    def __init__(self, 
                 num_classes: int = 3,
                 class_names: Optional[List[str]] = None,
                 ignore_index: int = -1):
        """
        Initialize segmentation metrics calculator.
        
        Args:
            num_classes: Number of segmentation classes
            class_names: Names of the classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # Initialize metric calculators
        self.dice_calculator = DiceScore(ignore_index=ignore_index)
        self.iou_calculator = IoUScore(ignore_index=ignore_index)
        self.precision_recall_calculator = PrecisionRecall(ignore_index=ignore_index)
        self.hausdorff_calculator = HausdorffDistance(ignore_index=ignore_index)
    
    def compute_metrics(self, 
                       outputs: Dict[str, torch.Tensor], 
                       batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute comprehensive segmentation metrics.
        
        Args:
            outputs: Model outputs dictionary
            batch: Batch dictionary with ground truth
            
        Returns:
            Dictionary with computed metrics
        """
        predictions = outputs["segmentation"]
        targets = batch["mask"]
        
        metrics = {}
        
        # Dice score
        dice_scores = self.dice_calculator.compute(predictions, targets, class_wise=True)
        for i, score in dice_scores.items():
            metrics[f"dice_{self.class_names[i]}"] = score
        metrics["dice_mean"] = np.mean(list(dice_scores.values()))
        
        # IoU score
        iou_scores = self.iou_calculator.compute(predictions, targets, class_wise=True)
        for i, score in iou_scores.items():
            metrics[f"iou_{self.class_names[i]}"] = score
        metrics["iou_mean"] = np.mean(list(iou_scores.values()))
        
        # Precision and recall
        precision_recall = self.precision_recall_calculator.compute(predictions, targets, class_wise=True)
        for i, score in precision_recall["precision"].items():
            metrics[f"precision_{self.class_names[i]}"] = score
        for i, score in precision_recall["recall"].items():
            metrics[f"recall_{self.class_names[i]}"] = score
        metrics["precision_mean"] = np.mean(list(precision_recall["precision"].values()))
        metrics["recall_mean"] = np.mean(list(precision_recall["recall"].values()))
        
        # Hausdorff distance (using fast distance transform)
        hausdorff_distances = self.hausdorff_calculator.compute(predictions, targets, class_wise=True)
        for i, distance in hausdorff_distances.items():
            metrics[f"hausdorff_{self.class_names[i]}"] = distance
        valid_hd = [d for d in hausdorff_distances.values() if d != float('inf')]
        metrics["hausdorff_mean"] = np.mean(valid_hd) if valid_hd else float('nan')
        
        # Overall metrics
        metrics["dice_score"] = metrics["dice_mean"]  # For compatibility
        metrics["iou_score"] = metrics["iou_mean"]    # For compatibility
        
        return metrics
    
    def compute_class_metrics(self, 
                            predictions: torch.Tensor, 
                            targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Compute class-wise metrics.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth masks
            
        Returns:
            Dictionary with class-wise metrics
        """
        class_metrics = {}
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            # Get binary masks for this class
            pred_mask = (predictions == class_id).float()
            target_mask = (targets == class_id).float()
            
            # Compute metrics
            dice = self.dice_calculator.compute(pred_mask.unsqueeze(1), target_mask, class_wise=False)
            iou = self.iou_calculator.compute(pred_mask.unsqueeze(1), target_mask, class_wise=False)
            precision_recall = self.precision_recall_calculator.compute(pred_mask.unsqueeze(1), target_mask, class_wise=False)
            hausdorff = self.hausdorff_calculator.compute(pred_mask.unsqueeze(1), target_mask, class_wise=False)
            
            class_metrics[class_name] = {
                "dice": dice,
                "iou": iou,
                "precision": precision_recall["precision"],
                "recall": precision_recall["recall"],
                "hausdorff": hausdorff
            }
        
        return class_metrics




