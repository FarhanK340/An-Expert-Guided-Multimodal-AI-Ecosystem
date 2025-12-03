"""
Training module for the MoME+ segmentation system.

This module contains:
- Main training loop and trainer
- Loss functions (Dice, Cross-entropy, EWC)
- Metrics computation (Dice, IoU, precision, recall)
- Learning rate schedulers and optimizers
- Checkpoint utilities
- Training visualization
"""

from .trainer import Trainer
from .loss_functions import DiceLoss, FocalLoss, CombinedLoss, EWCLoss
from .metrics import DiceScore, IoUScore, PrecisionRecall, SegmentationMetrics
from .scheduler import get_scheduler, get_optimizer
from .checkpoint_utils import CheckpointManager, save_checkpoint, load_checkpoint
from .visualization import TrainingVisualizer, plot_training_curves

__all__ = [
    "Trainer",
    "DiceLoss",
    "FocalLoss", 
    "CombinedLoss",
    "EWCLoss",
    "DiceScore",
    "IoUScore",
    "PrecisionRecall",
    "SegmentationMetrics",
    "get_scheduler",
    "get_optimizer",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "TrainingVisualizer",
    "plot_training_curves"
]

