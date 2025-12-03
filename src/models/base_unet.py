"""
Base UNet architecture building blocks for 3D medical image segmentation.

Provides the fundamental building blocks for constructing 3D U-Net architectures
used in the modality experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class UNetBlock(nn.Module):
    """
    Basic UNet block with two convolutions and optional batch normalization.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_batch_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize UNet block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        
        return x


class DownBlock(nn.Module):
    """
    Downsampling block with max pooling and UNet block.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 pool_size: int = 2,
                 use_batch_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            pool_size: Max pooling kernel size
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.pool = nn.MaxPool3d(pool_size)
        self.conv_block = UNetBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling block."""
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with transposed convolution and UNet block.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 use_batch_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Transposed convolution kernel size
            stride: Transposed convolution stride
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.conv_block = UNetBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, dropout=dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x: Input tensor from previous layer
            skip: Skip connection tensor
            
        Returns:
            Upsampled and concatenated tensor
        """
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net architecture for medical image segmentation.
    
    Standard U-Net with encoder-decoder structure and skip connections.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 32,
                 depth: int = 4,
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize 3D U-Net.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.depth = depth
        self.channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Encoder
        self.initial_conv = UNetBlock(in_channels, self.channels[0], use_batch_norm=use_batch_norm, dropout=dropout)
        
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                DownBlock(self.channels[i], self.channels[i + 1], use_batch_norm=use_batch_norm, dropout=dropout)
            )
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up_blocks.append(
                UpBlock(self.channels[i + 1], self.channels[i], use_batch_norm=use_batch_norm, dropout=dropout)
            )
        
        # Final classification layer
        self.final_conv = nn.Conv3d(self.channels[0], num_classes, kernel_size=1)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
        """
        # Encoder
        x = self.initial_conv(x)
        skip_connections = [x]
        
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        
        # Decoder
        skip_connections = skip_connections[:-1]  # Remove the last skip connection
        
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop())
        
        # Final classification
        x = self.final_conv(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from different levels of the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors from each encoder level
        """
        features = []
        
        # Encoder
        x = self.initial_conv(x)
        features.append(x)
        
        for down_block in self.down_blocks:
            x = down_block(x)
            features.append(x)
        
        return features


class ResidualUNetBlock(nn.Module):
    """
    Residual UNet block with skip connection.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_batch_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize residual UNet block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        identity = x
        
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        
        # Skip connection
        if self.skip_conv:
            identity = self.skip_conv(identity)
        
        x = x + identity
        x = self.relu(x)
        
        return x

