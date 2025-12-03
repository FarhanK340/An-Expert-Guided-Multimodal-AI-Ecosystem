"""
Attention mechanism modules for the MoME+ architecture.

Implements SE (Squeeze-and-Excitation), CBAM (Convolutional Block Attention Module),
and Transformer-based attention mechanisms for enhanced feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Implements the SE-Net attention mechanism that adaptively recalibrates
    channel-wise feature responses.
    """
    
    def __init__(self, 
                 channels: int, 
                 reduction_ratio: int = 16):
        """
        Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super().__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Squeeze and excitation layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SE block.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Attention-weighted tensor
        """
        b, c, d, h, w = x.size()
        
        # Global average pooling
        y = self.global_pool(x).view(b, c)
        
        # Squeeze and excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        
        # Scale the input
        return x * y


class ChannelAttention(nn.Module):
    """
    Channel attention module for CBAM.
    """
    
    def __init__(self, 
                 channels: int, 
                 reduction_ratio: int = 16):
        """
        Initialize channel attention.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention."""
        b, c, d, h, w = x.size()
        
        # Global average pooling
        avg_out = self.global_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial attention module for CBAM.
    """
    
    def __init__(self, 
                 kernel_size: int = 7):
        """
        Initialize spatial attention.
        
        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial attention."""
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention mechanisms.
    """
    
    def __init__(self, 
                 channels: int, 
                 reduction_ratio: int = 16,
                 spatial_kernel_size: int = 7):
        """
        Initialize CBAM block.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            spatial_kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CBAM block.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention-weighted tensor
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for 3D data.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, N, C) where N is sequence length
            
        Returns:
            Attention output tensor
        """
        b, n, c = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 8,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer blocks.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that combines different attention types.
    """
    
    def __init__(self, 
                 channels: int,
                 attention_types: list = ["se", "cbam"],
                 reduction_ratio: int = 16):
        """
        Initialize adaptive attention.
        
        Args:
            channels: Number of input channels
            attention_types: List of attention types to use
            reduction_ratio: Reduction ratio for attention modules
        """
        super().__init__()
        
        self.attention_modules = nn.ModuleDict()
        
        if "se" in attention_types:
            self.attention_modules["se"] = SEBlock(channels, reduction_ratio)
        
        if "cbam" in attention_types:
            self.attention_modules["cbam"] = CBAMBlock(channels, reduction_ratio)
        
        # Learnable weights for combining attention outputs
        self.attention_weights = nn.Parameter(torch.ones(len(self.attention_modules)))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined attention output
        """
        if len(self.attention_modules) == 1:
            return list(self.attention_modules.values())[0](x)
        
        # Apply each attention module
        attention_outputs = []
        for attention_module in self.attention_modules.values():
            attention_outputs.append(attention_module(x))
        
        # Combine with learnable weights
        weights = self.softmax(self.attention_weights)
        combined = sum(w * out for w, out in zip(weights, attention_outputs))
        
        return combined

