"""
Individual modality expert networks for the MoME+ architecture.

Each expert is specialized for a specific MRI modality (T1, T1ce, T2, FLAIR)
and learns modality-specific features for brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .base_unet import UNet3D, UNetBlock
from .attention_modules import SEBlock, CBAMBlock, AdaptiveAttention
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModalityExpert(nn.Module):
    """
    Individual modality expert network.
    
    Each expert is a specialized 3D U-Net with attention mechanisms
    designed to process a specific MRI modality.
    """
    
    def __init__(self, 
                 modality: str,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 attention_type: str = "cbam",
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize modality expert.
        
        Args:
            modality: Modality name (T1, T1ce, T2, FLAIR)
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            attention_type: Type of attention mechanism
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modality = modality
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        
        # Build the expert network
        self._build_network(attention_type, use_batch_norm, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self, 
                      attention_type: str, 
                      use_batch_norm: bool, 
                      dropout: float):
        """Build the expert network architecture."""
        
        # Calculate channel sizes
        self.channels = [self.base_channels * (2 ** i) for i in range(self.depth + 1)]
        
        # Initial convolution
        self.initial_conv = UNetBlock(
            self.in_channels, 
            self.channels[0], 
            use_batch_norm=use_batch_norm, 
            dropout=dropout
        )
        
        # Encoder with attention
        self.encoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(self.depth):
            # Encoder block
            encoder_block = nn.Sequential(
                nn.MaxPool3d(2),
                UNetBlock(
                    self.channels[i], 
                    self.channels[i + 1], 
                    use_batch_norm=use_batch_norm, 
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(encoder_block)
            
            # Attention block
            if attention_type == "se":
                attention = SEBlock(self.channels[i + 1])
            elif attention_type == "cbam":
                attention = CBAMBlock(self.channels[i + 1])
            elif attention_type == "adaptive":
                attention = AdaptiveAttention(self.channels[i + 1])
            else:
                attention = nn.Identity()
            
            self.attention_blocks.append(attention)
        
        # Decoder
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            self.upsample_blocks.append(
                nn.ConvTranspose3d(self.channels[i + 1], self.channels[i], 2, 2)
            )
            self.decoder_blocks.append(
                UNetBlock(
                    self.channels[i + 1],  # After concatenation
                    self.channels[i], 
                    use_batch_norm=use_batch_norm, 
                    dropout=dropout
                )
            )
        
        # Final classification
        self.final_conv = nn.Conv3d(self.channels[0], self.num_classes, 1)
        
        # Modality-specific feature extraction
        self.modality_features = nn.ModuleDict({
            "T1": self._create_modality_specific_layers(),
            "T1ce": self._create_modality_specific_layers(),
            "T2": self._create_modality_specific_layers(),
            "FLAIR": self._create_modality_specific_layers()
        })
    
    def _create_modality_specific_layers(self) -> nn.Module:
        """Create modality-specific feature extraction layers."""
        return nn.Sequential(
            nn.Conv3d(self.channels[0], self.channels[0] // 2, 3, padding=1),
            nn.BatchNorm3d(self.channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.channels[0] // 2, self.channels[0] // 4, 3, padding=1),
            nn.BatchNorm3d(self.channels[0] // 4),
            nn.ReLU(inplace=True)
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the modality expert.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Tuple of (segmentation_output, feature_maps)
        """
        # Initial convolution
        x = self.initial_conv(x)
        skip_connections = [x]
        
        # Encoder with attention
        for encoder_block, attention_block in zip(self.encoder_blocks, self.attention_blocks):
            x = encoder_block(x)
            x = attention_block(x)
            skip_connections.append(x)
        
        # Decoder
        skip_connections = skip_connections[:-1]  # Remove the last skip connection
        
        for upsample, decoder_block in zip(self.upsample_blocks, self.decoder_blocks):
            skip = skip_connections.pop()
            
            x = upsample(x)
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Extract modality-specific features
        modality_features = self.modality_features[self.modality](x)
        
        # Final classification
        segmentation = self.final_conv(x)
        
        return segmentation, [modality_features]
    
    def get_expert_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract expert-specific features without segmentation output.
        
        Args:
            x: Input tensor
            
        Returns:
            Expert features
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder with attention
        for encoder_block, attention_block in zip(self.encoder_blocks, self.attention_blocks):
            x = encoder_block(x)
            x = attention_block(x)
        
        # Extract modality-specific features
        modality_features = self.modality_features[self.modality](x)
        
        return modality_features
    
    def freeze_encoder(self):
        """Freeze encoder weights for transfer learning."""
        for param in self.initial_conv.parameters():
            param.requires_grad = False
        
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
        
        for attention_block in self.attention_blocks:
            for param in attention_block.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.initial_conv.parameters():
            param.requires_grad = True
        
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = True
        
        for attention_block in self.attention_blocks:
            for param in attention_block.parameters():
                param.requires_grad = True


class ExpertEnsemble(nn.Module):
    """
    Ensemble of modality experts for multi-modal processing.
    """
    
    def __init__(self, 
                 modalities: List[str],
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 attention_type: str = "cbam",
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize expert ensemble.
        
        Args:
            modalities: List of modality names
            in_channels: Number of input channels per modality
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            attention_type: Type of attention mechanism
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modalities = modalities
        self.num_experts = len(modalities)
        
        # Create individual experts
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
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all experts.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Dictionary of expert outputs
        """
        outputs = {}
        
        for modality, expert in self.experts.items():
            if modality in x:
                segmentation, features = expert(x[modality])
                outputs[modality] = {
                    "segmentation": segmentation,
                    "features": features
                }
        
        return outputs
    
    def get_expert_features(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract features from all experts.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Dictionary of expert features
        """
        features = {}
        
        for modality, expert in self.experts.items():
            if modality in x:
                features[modality] = expert.get_expert_features(x[modality])
        
        return features

