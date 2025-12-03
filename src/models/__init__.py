"""
Model architecture modules for MoME+ segmentation system.

This module contains:
- Base UNet building blocks
- Attention mechanisms (SE, CBAM, Transformer)
- Individual modality experts
- Hierarchical gating network
- Full MoME+ segmenter
- Continual learning modules (EWC, Replay)
"""

from .base_unet import UNet3D, UNetBlock, DownBlock, UpBlock
from .attention_modules import SEBlock, CBAMBlock, TransformerBlock
from .mome_expert import ModalityExpert
from .gating_network import HierarchicalGatingNetwork
from .mome_segmenter import MoMESegmenter
from .continual_learning import EWC, ReplayBuffer, ContinualLearningWrapper

__all__ = [
    "UNet3D",
    "UNetBlock", 
    "DownBlock",
    "UpBlock",
    "SEBlock",
    "CBAMBlock", 
    "TransformerBlock",
    "ModalityExpert",
    "HierarchicalGatingNetwork",
    "MoMESegmenter",
    "EWC",
    "ReplayBuffer",
    "ContinualLearningWrapper"
]

