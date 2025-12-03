"""
Hierarchical gating network for the MoME+ architecture.

Implements a gating mechanism that dynamically combines outputs from
different modality experts based on input characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalGatingNetwork(nn.Module):
    """
    Hierarchical gating network for expert fusion.
    
    Implements a multi-level gating mechanism that learns to dynamically
    weight and combine outputs from different modality experts.
    """
    
    def __init__(self, 
                 input_channels: int = 4,  # Number of modalities
                 num_experts: int = 4,
                 hidden_channels: List[int] = [64, 32, 16],
                 temperature: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize hierarchical gating network.
        
        Args:
            input_channels: Number of input channels (modalities)
            num_experts: Number of expert networks
            hidden_channels: List of hidden layer channel sizes
            temperature: Temperature for gating softmax
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_experts = num_experts
        self.temperature = temperature
        
        # Build gating layers
        self._build_gating_layers(hidden_channels, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_gating_layers(self, hidden_channels: List[int], dropout: float):
        """Build the gating network layers."""
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv3d(self.input_channels, hidden_channels[0], 3, padding=1),
            nn.BatchNorm3d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        # Hierarchical gating layers
        self.gating_layers = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            layer = nn.Sequential(
                nn.Conv3d(hidden_channels[i], hidden_channels[i + 1], 3, padding=1),
                nn.BatchNorm3d(hidden_channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)
            )
            self.gating_layers.append(layer)
        
        # Final gating layer
        self.final_gating = nn.Sequential(
            nn.Conv3d(hidden_channels[-1], self.num_experts, 1),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Attention mechanism for spatial gating
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(self.input_channels, self.num_experts, 1),
            nn.Sigmoid()
        )
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gating network.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Tuple of (expert_weights, spatial_attention)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Hierarchical gating
        for gating_layer in self.gating_layers:
            x = gating_layer(x)
        
        # Final gating weights
        expert_weights = self.final_gating(x)  # (B, num_experts)
        expert_weights = F.softmax(expert_weights / self.temperature, dim=1)
        
        # Spatial attention
        spatial_attention = self.spatial_attention(x)
        
        return expert_weights, spatial_attention


class AdaptiveGatingNetwork(nn.Module):
    """
    Adaptive gating network with learnable temperature and attention.
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 num_experts: int = 4,
                 hidden_channels: List[int] = [64, 32, 16],
                 dropout: float = 0.1):
        """
        Initialize adaptive gating network.
        
        Args:
            input_channels: Number of input channels
            num_experts: Number of expert networks
            hidden_channels: List of hidden layer channel sizes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_experts = num_experts
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Build gating layers
        self._build_gating_layers(hidden_channels, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_gating_layers(self, hidden_channels: List[int], dropout: float):
        """Build the gating network layers."""
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv3d(self.input_channels, hidden_channels[0], 3, padding=1),
            nn.BatchNorm3d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        # Hierarchical gating layers
        self.gating_layers = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            layer = nn.Sequential(
                nn.Conv3d(hidden_channels[i], hidden_channels[i + 1], 3, padding=1),
                nn.BatchNorm3d(hidden_channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)
            )
            self.gating_layers.append(layer)
        
        # Final gating layer
        self.final_gating = nn.Sequential(
            nn.Conv3d(hidden_channels[-1], self.num_experts, 1),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        
        # Multi-scale attention
        self.multi_scale_attention = nn.ModuleList([
            nn.Conv3d(self.input_channels, self.num_experts, 1),
            nn.Conv3d(self.input_channels, self.num_experts, 3, padding=1),
            nn.Conv3d(self.input_channels, self.num_experts, 5, padding=2)
        ])
        
        self.attention_fusion = nn.Conv3d(self.num_experts * 3, self.num_experts, 1)
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the adaptive gating network.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Tuple of (expert_weights, spatial_attention)
        """
        # Input projection
        x_proj = self.input_projection(x)
        
        # Hierarchical gating
        for gating_layer in self.gating_layers:
            x_proj = gating_layer(x_proj)
        
        # Final gating weights
        expert_weights = self.final_gating(x_proj)
        expert_weights = F.softmax(expert_weights / self.temperature, dim=1)
        
        # Multi-scale spatial attention
        attention_maps = []
        for attention_layer in self.multi_scale_attention:
            attention_maps.append(attention_layer(x))
        
        # Fuse attention maps
        spatial_attention = torch.cat(attention_maps, dim=1)
        spatial_attention = self.attention_fusion(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention)
        
        return expert_weights, spatial_attention


class ExpertFusion(nn.Module):
    """
    Expert fusion module that combines outputs from multiple experts.
    """
    
    def __init__(self, 
                 num_experts: int = 4,
                 num_classes: int = 3,
                 fusion_method: str = "weighted"):
        """
        Initialize expert fusion module.
        
        Args:
            num_experts: Number of expert networks
            num_classes: Number of output classes
            fusion_method: Fusion method ("weighted", "attention", "concat")
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            self.attention_fusion = nn.Sequential(
                nn.Conv3d(num_classes * num_experts, num_classes, 1),
                nn.Sigmoid()
            )
        elif fusion_method == "concat":
            self.fusion_conv = nn.Conv3d(num_classes * num_experts, num_classes, 1)
    
    def forward(self, 
                expert_outputs: List[torch.Tensor], 
                expert_weights: torch.Tensor,
                spatial_attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse expert outputs.
        
        Args:
            expert_outputs: List of expert output tensors
            expert_weights: Expert weight tensor
            spatial_attention: Optional spatial attention tensor
            
        Returns:
            Fused output tensor
        """
        if self.fusion_method == "weighted":
            # Weighted combination
            fused_output = torch.zeros_like(expert_outputs[0])
            for i, output in enumerate(expert_outputs):
                weight = expert_weights[:, i:i+1, None, None, None]
                fused_output += weight * output
            
            # Apply spatial attention if provided
            if spatial_attention is not None:
                fused_output = fused_output * spatial_attention
            
        elif self.fusion_method == "attention":
            # Concatenate all outputs
            concat_outputs = torch.cat(expert_outputs, dim=1)
            
            # Apply attention
            attention_weights = self.attention_fusion(concat_outputs)
            fused_output = concat_outputs * attention_weights
            
            # Final fusion
            fused_output = torch.sum(fused_output.view(
                fused_output.size(0), self.num_experts, self.num_classes, 
                *fused_output.shape[2:]
            ), dim=1)
            
        elif self.fusion_method == "concat":
            # Concatenate and fuse
            concat_outputs = torch.cat(expert_outputs, dim=1)
            fused_output = self.fusion_conv(concat_outputs)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_output

