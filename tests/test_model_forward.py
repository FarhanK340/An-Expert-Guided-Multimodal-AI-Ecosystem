"""
Tests for model forward pass and architecture.

Tests the MoME+ model architecture, forward pass, and
basic functionality.
"""

import pytest
import torch
import numpy as np

from src.models.mome_segmenter import MoMESegmenter
from src.models.mome_expert import ModalityExpert
from src.models.gating_network import GatingNetwork
from src.models.attention_modules import SEBlock, CBAM, TransformerBlock


class TestMoMESegmenter:
    """Test MoME+ segmenter architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modalities = ["T1", "T1ce", "T2", "FLAIR"]
        self.input_shape = (1, 1, 64, 64, 64)
        self.num_classes = 3
        
        # Create model
        self.model = MoMESegmenter(
            modalities=self.modalities,
            in_channels=1,
            num_classes=self.num_classes,
            base_channels=32,
            depth=4
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.num_modalities == len(self.modalities)
        assert self.model.num_classes == self.num_classes
        assert len(self.model.experts) == len(self.modalities)
        assert self.model.gating_network is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create dummy input
        input_data = {}
        for modality in self.modalities:
            input_data[modality] = torch.randn(self.input_shape)
        
        # Forward pass
        output = self.model(input_data)
        
        # Check output
        assert 'segmentation' in output
        assert output['segmentation'].shape == (1, self.num_classes, 64, 64, 64)
        assert 'expert_weights' in output
        assert 'spatial_attention' in output
    
    def test_expert_weights(self):
        """Test expert weight computation."""
        input_data = {}
        for modality in self.modalities:
            input_data[modality] = torch.randn(self.input_shape)
        
        output = self.model(input_data)
        expert_weights = output['expert_weights']
        
        # Check expert weights
        assert expert_weights.shape == (1, len(self.modalities))
        assert torch.allclose(torch.sum(expert_weights, dim=1), torch.ones(1))
        assert torch.all(expert_weights >= 0)
    
    def test_spatial_attention(self):
        """Test spatial attention computation."""
        input_data = {}
        for modality in self.modalities:
            input_data[modality] = torch.randn(self.input_shape)
        
        output = self.model(input_data)
        spatial_attention = output['spatial_attention']
        
        # Check spatial attention
        assert spatial_attention.shape == (1, 1, 64, 64, 64)
        assert torch.all(spatial_attention >= 0)
        assert torch.all(spatial_attention <= 1)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        input_data = {}
        for modality in self.modalities:
            input_data[modality] = torch.randn(self.input_shape, requires_grad=True)
        
        output = self.model(input_data)
        loss = output['segmentation'].sum()
        loss.backward()
        
        # Check gradients
        for modality in self.modalities:
            assert input_data[modality].grad is not None
            assert not torch.allclose(input_data[modality].grad, torch.zeros_like(input_data[modality].grad))


class TestModalityExpert:
    """Test modality expert architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_shape = (1, 1, 64, 64, 64)
        self.num_classes = 3
        
        # Create expert
        self.expert = ModalityExpert(
            in_channels=1,
            num_classes=self.num_classes,
            base_channels=32,
            depth=4
        )
    
    def test_expert_initialization(self):
        """Test expert initialization."""
        assert self.expert.in_channels == 1
        assert self.expert.num_classes == self.num_classes
        assert self.expert.base_channels == 32
        assert self.expert.depth == 4
    
    def test_expert_forward(self):
        """Test expert forward pass."""
        input_tensor = torch.randn(self.input_shape)
        
        output = self.expert(input_tensor)
        
        assert output.shape == (1, self.num_classes, 64, 64, 64)
    
    def test_expert_attention(self):
        """Test expert attention mechanism."""
        input_tensor = torch.randn(self.input_shape)
        
        output, attention = self.expert(input_tensor, return_attention=True)
        
        assert output.shape == (1, self.num_classes, 64, 64, 64)
        assert attention.shape == (1, 1, 64, 64, 64)
        assert torch.all(attention >= 0)
        assert torch.all(attention <= 1)


class TestGatingNetwork:
    """Test gating network architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_modalities = 4
        self.input_shape = (1, 1, 64, 64, 64)
        
        # Create gating network
        self.gating_network = GatingNetwork(
            num_modalities=self.num_modalities,
            input_shape=self.input_shape[2:]  # Remove batch and channel dimensions
        )
    
    def test_gating_initialization(self):
        """Test gating network initialization."""
        assert self.gating_network.num_modalities == self.num_modalities
        assert self.gating_network.input_shape == self.input_shape[2:]
    
    def test_gating_forward(self):
        """Test gating network forward pass."""
        input_data = {}
        for i in range(self.num_modalities):
            input_data[f"modality_{i}"] = torch.randn(self.input_shape)
        
        weights = self.gating_network(input_data)
        
        assert weights.shape == (1, self.num_modalities)
        assert torch.allclose(torch.sum(weights, dim=1), torch.ones(1))
        assert torch.all(weights >= 0)
    
    def test_gating_consistency(self):
        """Test gating network consistency."""
        input_data = {}
        for i in range(self.num_modalities):
            input_data[f"modality_{i}"] = torch.randn(self.input_shape)
        
        # Multiple forward passes should give consistent results
        weights1 = self.gating_network(input_data)
        weights2 = self.gating_network(input_data)
        
        assert torch.allclose(weights1, weights2)


class TestAttentionModules:
    """Test attention modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_shape = (1, 32, 64, 64, 64)
    
    def test_se_block(self):
        """Test SE (Squeeze-and-Excitation) block."""
        se_block = SEBlock(channels=32, reduction=16)
        
        input_tensor = torch.randn(self.input_shape)
        output = se_block(input_tensor)
        
        assert output.shape == self.input_shape
    
    def test_cbam(self):
        """Test CBAM (Convolutional Block Attention Module)."""
        cbam = CBAM(channels=32, reduction=16)
        
        input_tensor = torch.randn(self.input_shape)
        output = cbam(input_tensor)
        
        assert output.shape == self.input_shape
    
    def test_transformer_block(self):
        """Test Transformer block."""
        transformer = TransformerBlock(
            dim=32,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1
        )
        
        input_tensor = torch.randn(self.input_shape)
        output = transformer(input_tensor)
        
        assert output.shape == self.input_shape


class TestModelIntegration:
    """Test model integration and end-to-end functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modalities = ["T1", "T1ce", "T2", "FLAIR"]
        self.input_shape = (1, 1, 64, 64, 64)
        self.num_classes = 3
        
        # Create model
        self.model = MoMESegmenter(
            modalities=self.modalities,
            in_channels=1,
            num_classes=self.num_classes,
            base_channels=32,
            depth=4
        )
    
    def test_end_to_end_forward(self):
        """Test end-to-end forward pass."""
        # Create dummy input
        input_data = {}
        for modality in self.modalities:
            input_data[modality] = torch.randn(self.input_shape)
        
        # Forward pass
        output = self.model(input_data)
        
        # Check all outputs
        assert 'segmentation' in output
        assert 'expert_weights' in output
        assert 'spatial_attention' in output
        
        # Check shapes
        assert output['segmentation'].shape == (1, self.num_classes, 64, 64, 64)
        assert output['expert_weights'].shape == (1, len(self.modalities))
        assert output['spatial_attention'].shape == (1, 1, 64, 64, 64)
    
    def test_model_parameters(self):
        """Test model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_model_device_placement(self):
        """Test model device placement."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model = self.model.to(device)
            
            # Create input on GPU
            input_data = {}
            for modality in self.modalities:
                input_data[modality] = torch.randn(self.input_shape).to(device)
            
            # Forward pass
            output = self.model(input_data)
            
            # Check device placement
            assert output['segmentation'].device == device
            assert output['expert_weights'].device == device
            assert output['spatial_attention'].device == device


if __name__ == "__main__":
    pytest.main([__file__])


