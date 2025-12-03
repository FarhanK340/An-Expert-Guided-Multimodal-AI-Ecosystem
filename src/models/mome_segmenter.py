"""
Main MoME+ segmenter architecture.

Combines modality experts with hierarchical gating network for
multi-modal brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .mome_expert import ModalityExpert, ExpertEnsemble
from .gating_network import HierarchicalGatingNetwork, ExpertFusion
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MoMESegmenter(nn.Module):
    """
    Mixture of Modality Experts (MoME+) segmenter.
    
    Main architecture that combines multiple modality experts with
    a hierarchical gating network for dynamic expert fusion.
    """
    
    def __init__(self, 
                 modalities: List[str] = ["T1", "T1ce", "T2", "FLAIR"],
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 attention_type: str = "cbam",
                 gating_hidden_channels: List[int] = [64, 32, 16],
                 fusion_method: str = "weighted",
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize MoME+ segmenter.
        
        Args:
            modalities: List of modality names
            in_channels: Number of input channels per modality
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            attention_type: Type of attention mechanism
            gating_hidden_channels: Hidden layer sizes for gating network
            fusion_method: Method for fusing expert outputs
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modalities = modalities
        self.num_experts = len(modalities)
        self.num_classes = num_classes
        
        # Create modality experts
        self.experts = nn.ModuleDict()
        for modality in modalities:
            self.experts[modality] = ModalityExpert(
                modality=modality,
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                depth=depth,
                attention_type=attention_type,
                use_batch_norm=use_batch_norm,
                dropout=dropout
            )
        
        # Create gating network
        self.gating_network = HierarchicalGatingNetwork(
            input_channels=len(modalities),
            num_experts=self.num_experts,
            hidden_channels=gating_hidden_channels,
            dropout=dropout
        )
        
        # Create expert fusion module
        self.expert_fusion = ExpertFusion(
            num_experts=self.num_experts,
            num_classes=num_classes,
            fusion_method=fusion_method
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MoME+ segmenter.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Dictionary containing segmentation output and intermediate results
        """
        # Process each modality through its expert
        expert_outputs = []
        expert_features = []
        
        for modality in self.modalities:
            if modality in x:
                expert_output, features = self.experts[modality](x[modality])
                expert_outputs.append(expert_output)
                expert_features.extend(features)
            else:
                # Create zero tensor if modality is missing
                batch_size = next(iter(x.values())).size(0)
                device = next(iter(x.values())).device
                zero_output = torch.zeros(
                    batch_size, self.num_classes, 128, 128, 128,
                    device=device, dtype=torch.float32
                )
                expert_outputs.append(zero_output)
        
        # Create input for gating network (concatenate all modalities)
        gating_input = torch.cat([x[mod] for mod in self.modalities if mod in x], dim=1)
        
        # Get gating weights and spatial attention
        expert_weights, spatial_attention = self.gating_network(gating_input)
        
        # Fuse expert outputs
        fused_output = self.expert_fusion(expert_outputs, expert_weights, spatial_attention)
        
        return {
            "segmentation": fused_output,
            "expert_outputs": expert_outputs,
            "expert_weights": expert_weights,
            "spatial_attention": spatial_attention,
            "expert_features": expert_features
        }
    
    def get_expert_outputs(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get individual expert outputs without fusion.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Dictionary of expert outputs
        """
        expert_outputs = {}
        
        for modality in self.modalities:
            if modality in x:
                expert_output, _ = self.experts[modality](x[modality])
                expert_outputs[modality] = expert_output
        
        return expert_outputs
    
    def get_gating_weights(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get gating weights for analysis.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Tuple of (expert_weights, spatial_attention)
        """
        # Create input for gating network
        gating_input = torch.cat([x[mod] for mod in self.modalities if mod in x], dim=1)
        
        # Get gating weights
        expert_weights, spatial_attention = self.gating_network(gating_input)
        
        return expert_weights, spatial_attention
    
    def freeze_experts(self, expert_names: Optional[List[str]] = None):
        """
        Freeze specific experts for transfer learning.
        
        Args:
            expert_names: List of expert names to freeze (if None, freeze all)
        """
        if expert_names is None:
            expert_names = self.modalities
        
        for name in expert_names:
            if name in self.experts:
                self.experts[name].freeze_encoder()
                logger.info(f"Frozen expert: {name}")
    
    def unfreeze_experts(self, expert_names: Optional[List[str]] = None):
        """
        Unfreeze specific experts.
        
        Args:
            expert_names: List of expert names to unfreeze (if None, unfreeze all)
        """
        if expert_names is None:
            expert_names = self.modalities
        
        for name in expert_names:
            if name in self.experts:
                self.experts[name].unfreeze_encoder()
                logger.info(f"Unfrozen expert: {name}")
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size information.
        
        Returns:
            Dictionary with model size statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        expert_params = {}
        for name, expert in self.experts.items():
            expert_params[name] = sum(p.numel() for p in expert.parameters())
        
        gating_params = sum(p.numel() for p in self.gating_network.parameters())
        fusion_params = sum(p.numel() for p in self.expert_fusion.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "expert_parameters": expert_params,
            "gating_parameters": gating_params,
            "fusion_parameters": fusion_params
        }


class MoMESegmenterWithContinualLearning(nn.Module):
    """
    MoME+ segmenter with continual learning capabilities.
    
    Extends the base MoME+ segmenter with continual learning modules
    for knowledge retention across tasks.
    """
    
    def __init__(self, 
                 modalities: List[str] = ["T1", "T1ce", "T2", "FLAIR"],
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 attention_type: str = "cbam",
                 gating_hidden_channels: List[int] = [64, 32, 16],
                 fusion_method: str = "weighted",
                 use_batch_norm: bool = True,
                 dropout: float = 0.1,
                 continual_learning_config: Optional[Dict] = None):
        """
        Initialize MoME+ segmenter with continual learning.
        
        Args:
            modalities: List of modality names
            in_channels: Number of input channels per modality
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            attention_type: Type of attention mechanism
            gating_hidden_channels: Hidden layer sizes for gating network
            fusion_method: Method for fusing expert outputs
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            continual_learning_config: Configuration for continual learning
        """
        super().__init__()
        
        # Create base MoME+ segmenter
        self.mome_segmenter = MoMESegmenter(
            modalities=modalities,
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            attention_type=attention_type,
            gating_hidden_channels=gating_hidden_channels,
            fusion_method=fusion_method,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
        
        # Continual learning modules will be added here
        self.continual_learning_config = continual_learning_config or {}
        
        # Task-specific heads for continual learning
        self.task_heads = nn.ModuleDict()
        
        # Current task ID
        self.current_task_id = 0
    
    def forward(self, x: Dict[str, torch.Tensor], task_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific processing.
        
        Args:
            x: Dictionary of input tensors for each modality
            task_id: Task identifier for continual learning
            
        Returns:
            Dictionary containing segmentation output and task-specific results
        """
        # Forward through base segmenter
        base_output = self.mome_segmenter(x)
        
        # Add task-specific processing if needed
        if task_id is not None and str(task_id) in self.task_heads:
            task_output = self.task_heads[str(task_id)](base_output["segmentation"])
            base_output["task_output"] = task_output
        
        return base_output
    
    def add_task_head(self, task_id: int, num_classes: int):
        """
        Add a task-specific head for continual learning.
        
        Args:
            task_id: Task identifier
            num_classes: Number of classes for this task
        """
        task_head = nn.Sequential(
            nn.Conv3d(self.mome_segmenter.num_classes, num_classes, 1),
            nn.Softmax(dim=1)
        )
        
        self.task_heads[str(task_id)] = task_head
        logger.info(f"Added task head for task {task_id} with {num_classes} classes")
    
    def set_current_task(self, task_id: int):
        """Set the current task for continual learning."""
        self.current_task_id = task_id
        logger.info(f"Set current task to {task_id}")
    
    def get_current_task(self) -> int:
        """Get the current task ID."""
        return self.current_task_id

