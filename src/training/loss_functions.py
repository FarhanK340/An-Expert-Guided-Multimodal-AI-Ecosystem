"""
Loss functions for medical image segmentation.

Implements Dice loss, Focal loss, Cross-entropy loss, and combined losses
for training the MoME+ segmentation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    Dice loss for medical image segmentation.
    
    Computes the Dice coefficient loss between predicted and ground truth masks.
    """
    
    def __init__(self, 
                 smooth: float = 1e-6,
                 ignore_index: int = -1,
                 reduction: str = "mean",
                 class_weights: Optional[List[float]] = None):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ("mean", "sum", "none")
            class_weights: Per-class weights for handling imbalance (e.g., [1.0, 2.0, 3.0])
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weights = torch.tensor(class_weights) if class_weights else None
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        Handles both multi-class (softmax) and multi-label (sigmoid) cases automatically
        based on target shape.
        """
        # Check if targets are already one-hot/multi-label (same channels as predictions)
        if targets.shape == predictions.shape:
            # Multi-label case (e.g. BraTS overlapping regions)
            targets_one_hot = targets.float()
            # Use Sigmoid for multi-label
            predictions = torch.sigmoid(predictions)
        else:
            # Multi-class case (e.g. integer labels)
            targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
            # Use Softmax for multi-class
            predictions = F.softmax(predictions, dim=1)
        
        # Compute intersection and union
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3, 4))
        union = predictions.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Compute Dice loss
        dice_loss = 1.0 - dice
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(dice_loss.device)
            # dice_loss shape: (B, C), class_weights shape: (C,)
            dice_loss = dice_loss * class_weights.unsqueeze(0)
        
        # Handle reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in medical image segmentation.
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 ignore_index: int = -1,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        """
        if targets.shape == predictions.shape:
             # Multi-label case: Use Binary Cross Entropy
             # BCEWithLogitsLoss is more stable
             bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
             pt = torch.exp(-bce_loss)
             focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        else:
            # Multi-class case: Use Cross Entropy
            ce_loss = F.cross_entropy(predictions, targets, 
                                    ignore_index=self.ignore_index, 
                                    reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Handle reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss for medical image segmentation.
    """
    
    def __init__(self, 
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 smooth: float = 1e-6,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        """
        if targets.shape == predictions.shape:
            # Multi-label
            targets_one_hot = targets.float()
            predictions = torch.sigmoid(predictions)
        else:
            # Multi-class
            targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1)).permute(0, 4, 1, 2, 3).float()
            predictions = F.softmax(predictions, dim=1)
        
        # Compute true positives, false positives, and false negatives
        tp = (predictions * targets_one_hot).sum(dim=(2, 3, 4))
        fp = (predictions * (1 - targets_one_hot)).sum(dim=(2, 3, 4))
        fn = ((1 - predictions) * targets_one_hot).sum(dim=(2, 3, 4))
        
        # Compute Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Compute Tversky loss
        tversky_loss = 1.0 - tversky
        
        # Handle reduction
        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for medical image segmentation.
    """
    
    def __init__(self, 
                 loss_config: Dict,
                 dice_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 focal_weight: float = 0.5,
                 tversky_weight: float = 0.0):
        super().__init__()
        
        # Get weights from config or use defaults
        self.dice_weight = loss_config.get("dice_weight", dice_weight)
        self.ce_weight = loss_config.get("ce_weight", ce_weight)
        self.focal_weight = loss_config.get("focal_weight", focal_weight)
        self.tversky_weight = loss_config.get("tversky_weight", tversky_weight)
        
        # Get class weights for handling imbalance
        class_weights = loss_config.get("class_weights", None)
        
        # Initialize loss functions
        self.dice_loss = DiceLoss(
            smooth=loss_config.get("dice_smooth", 1e-6),
            reduction="mean",
            class_weights=class_weights
        )
        
        # Note: ce_loss used for multi-class, for multi-label we handle internally or via Focal
        self.ce_loss_fn = nn.CrossEntropyLoss(
            ignore_index=loss_config.get("ignore_index", -1),
            reduction="mean"
        )
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        
        self.focal_loss = FocalLoss(
            alpha=loss_config.get("focal_alpha", 0.25),
            gamma=loss_config.get("focal_gamma", 2.0),
            reduction="mean"
        )
        
        if self.tversky_weight > 0:
            self.tversky_loss = TverskyLoss(
                alpha=loss_config.get("tversky_alpha", 0.3),
                beta=loss_config.get("tversky_beta", 0.7),
                reduction="mean"
            )
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute combined loss.
        """
        predictions = outputs["segmentation"]
        targets = batch["mask"]
        
        # Determine strict or soft matching based on shape
        if targets.dim() == 5 and targets.shape[1] > 1:
            # Multi-label case (B, C, D, H, W)
            targets = targets.float()
            is_multilabel = True
        else:
            # Multi-class case (index labels)
            targets = targets.long()
            if targets.dim() == 5 and targets.shape[1] == 1:
                 targets = targets.squeeze(1)
            is_multilabel = False
        
        total_loss = 0.0
        loss_components = {}
        
        # Dice loss (handles both modes)
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(predictions, targets)
            total_loss += self.dice_weight * dice_loss
            loss_components["dice_loss"] = dice_loss.item()
        
        # Cross-entropy / BCE loss
        if self.ce_weight > 0:
            if is_multilabel:
                ce_val = self.bce_loss_fn(predictions, targets)
            else:
                ce_val = self.ce_loss_fn(predictions, targets)
            
            total_loss += self.ce_weight * ce_val
            loss_components["ce_loss"] = ce_val.item()
        
        # Focal loss (handles both modes)
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(predictions, targets)
            total_loss += self.focal_weight * focal_loss
            loss_components["focal_loss"] = focal_loss.item()
        
        # Tversky loss (handles both modes)
        if self.tversky_weight > 0:
            tversky_loss = self.tversky_loss(predictions, targets)
            total_loss += self.tversky_weight * tversky_loss
            loss_components["tversky_loss"] = tversky_loss.item()
            
        # Store loss components for logging
        outputs["loss_components"] = loss_components
        
        return total_loss


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) loss for continual learning.
    
    Penalizes changes to important weights from previous tasks.
    """
    
    def __init__(self, 
                 ewc_lambda: float = 1000.0,
                 fisher_matrices: Optional[Dict] = None,
                 optimal_params: Optional[Dict] = None):
        """
        Initialize EWC loss.
        
        Args:
            ewc_lambda: EWC regularization strength
            fisher_matrices: Fisher information matrices for each task
            optimal_params: Optimal parameters for each task
        """
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices = fisher_matrices or {}
        self.optimal_params = optimal_params or {}
    
    def forward(self, 
                model: nn.Module, 
                task_id: int) -> torch.Tensor:
        """
        Compute EWC loss.
        
        Args:
            model: The model to compute EWC loss for
            task_id: Current task identifier
            
        Returns:
            EWC loss
        """
        if task_id not in self.fisher_matrices:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = 0.0
        fisher_info = self.fisher_matrices[task_id]
        optimal_params = self.optimal_params[task_id]
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_info:
                ewc_loss += (fisher_info[name] * (param - optimal_params[name]) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def update_fisher_info(self, 
                          model: nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          task_id: int,
                          num_samples: int = 1000):
        """
        Update Fisher information matrix for a task.
        
        Args:
            model: The model
            dataloader: Data loader for the task
            task_id: Task identifier
            num_samples: Number of samples to use
        """
        model.eval()
        
        # Initialize Fisher information
        fisher_info = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        # Compute Fisher information
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                # Forward pass
                outputs = model(batch)
                
                # Compute gradients
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", 0)
                else:
                    loss = outputs
                
                if loss.requires_grad:
                    loss.backward()
                    
                    # Accumulate Fisher information
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            fisher_info[name] += param.grad.data ** 2
                    
                    # Clear gradients
                    model.zero_grad()
                
                sample_count += 1
        
        # Normalize Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        # Store Fisher information and optimal parameters
        self.fisher_matrices[task_id] = fisher_info
        self.optimal_params[task_id] = {
            name: param.data.clone() for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        logger.info(f"Updated Fisher information for task {task_id}")

