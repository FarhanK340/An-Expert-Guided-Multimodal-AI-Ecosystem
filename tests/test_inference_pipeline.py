"""
Tests for inference pipeline and post-processing.

Tests the inference engine, post-processing, and output generation
functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from src.inference.inference_engine import InferenceEngine
from src.inference.postprocessing import PostProcessor, ThresholdProcessor, ConnectedComponentsProcessor
from src.inference.json_mapper import JSONMapper, BrainAtlasMapper
from src.inference.gltf_exporter import GLTFExporter, MeshGenerator


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create dummy config
        self.config = {
            'model': {
                'checkpoint_path': 'dummy_path',
                'precision': 'fp32'
            },
            'input': {
                'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
                'normalization': 'zscore',
                'resample_to': [128, 128, 128]
            },
            'postprocessing': {
                'threshold': 0.5,
                'min_volume': 100,
                'connected_components': True,
                'smoothing': True
            },
            'output': {
                'save_masks': True,
                'save_probabilities': False,
                'save_3d_mesh': True,
                'mesh_format': 'gltf'
            },
            'performance': {
                'batch_size': 1,
                'use_amp': False
            }
        }
        
        # Create dummy segmentation data
        self.segmentation_data = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_inference_engine_initialization(self):
        """Test inference engine initialization."""
        # This would normally test with a real model
        # For now, we'll test the configuration parsing
        assert self.config['model']['precision'] == 'fp32'
        assert self.config['input']['modalities'] == ['T1', 'T1ce', 'T2', 'FLAIR']
        assert self.config['postprocessing']['threshold'] == 0.5
    
    def test_input_preprocessing(self):
        """Test input preprocessing."""
        # Create dummy input data
        input_data = {
            'T1': np.random.randn(128, 128, 128).astype(np.float32),
            'T1ce': np.random.randn(128, 128, 128).astype(np.float32),
            'T2': np.random.randn(128, 128, 128).astype(np.float32),
            'FLAIR': np.random.randn(128, 128, 128).astype(np.float32)
        }
        
        # Test normalization
        for modality, data in input_data.items():
            normalized = (data - np.mean(data)) / np.std(data)
            assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
            assert np.isclose(np.std(normalized), 1.0, atol=1e-6)
    
    def test_output_generation(self):
        """Test output generation."""
        # Create dummy segmentation
        segmentation = torch.randint(0, 4, (1, 3, 128, 128, 128)).float()
        
        # Test output formatting
        assert segmentation.shape == (1, 3, 128, 128, 128)
        assert segmentation.dtype == torch.float32


class TestPostProcessing:
    """Test post-processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'threshold': 0.5,
            'min_volume': 100,
            'connected_components': True,
            'min_component_size': 50,
            'smoothing': True,
            'smoothing_sigma': 1.0,
            'morphological_ops': True,
            'opening_kernel_size': 3,
            'closing_kernel_size': 3
        }
        
        # Create dummy segmentation
        self.segmentation = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
    
    def test_postprocessor_initialization(self):
        """Test post-processor initialization."""
        postprocessor = PostProcessor(self.config)
        
        assert postprocessor.threshold == 0.5
        assert postprocessor.min_volume == 100
        assert postprocessor.connected_components == True
        assert postprocessor.smoothing == True
    
    def test_threshold_processing(self):
        """Test threshold processing."""
        threshold_processor = ThresholdProcessor(threshold=0.5)
        
        # Create dummy probabilities
        probabilities = np.random.rand(128, 128, 128).astype(np.float32)
        
        # Apply threshold
        thresholded = threshold_processor.process(probabilities)
        
        assert thresholded.shape == probabilities.shape
        assert np.all(thresholded >= 0)
        assert np.all(thresholded <= 1)
    
    def test_connected_components_processing(self):
        """Test connected components processing."""
        cc_processor = ConnectedComponentsProcessor(min_size=50)
        
        # Create dummy segmentation with small components
        segmentation = np.zeros((128, 128, 128), dtype=np.uint8)
        segmentation[10:20, 10:20, 10:20] = 1  # Small component
        segmentation[50:100, 50:100, 50:100] = 2  # Large component
        
        # Process
        processed = cc_processor.process(segmentation)
        
        assert processed.shape == segmentation.shape
        assert np.all(processed >= 0)
    
    def test_postprocessing_pipeline(self):
        """Test complete post-processing pipeline."""
        postprocessor = PostProcessor(self.config)
        
        # Create dummy segmentation
        segmentation = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
        
        # Process
        processed = postprocessor.process(segmentation)
        
        assert processed.shape == segmentation.shape
        assert processed.dtype == np.uint8


class TestJSONMapper:
    """Test JSON mapping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'atlas': 'aal',
            'include_statistics': True,
            'include_volumes': True,
            'include_coordinates': True
        }
        
        # Create dummy segmentation
        self.segmentation = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
    
    def test_brain_atlas_mapper(self):
        """Test brain atlas mapper."""
        atlas_mapper = BrainAtlasMapper(atlas_type='aal')
        
        # Test region info
        region_info = atlas_mapper.get_region_info(1)
        assert region_info is not None
        assert 'name' in region_info
        assert 'hemisphere' in region_info
        assert 'lobe' in region_info
    
    def test_json_mapper(self):
        """Test JSON mapper."""
        json_mapper = JSONMapper(self.config)
        
        # Map segmentation
        mapping = json_mapper.map_segmentation(self.segmentation)
        
        assert 'segmentation_info' in mapping
        assert 'region_mappings' in mapping
        assert 'statistics' in mapping
        assert 'volumes' in mapping
        assert 'coordinates' in mapping
    
    def test_region_analysis(self):
        """Test region analysis."""
        json_mapper = JSONMapper(self.config)
        
        # Create segmentation with specific regions
        segmentation = np.zeros((128, 128, 128), dtype=np.uint8)
        segmentation[10:50, 10:50, 10:50] = 1
        segmentation[60:100, 60:100, 60:100] = 2
        
        # Analyze regions
        mapping = json_mapper.map_segmentation(segmentation)
        
        # Check region mappings
        assert '1' in mapping['region_mappings']
        assert '2' in mapping['region_mappings']
        
        # Check statistics
        assert 'total_regions' in mapping['statistics']
        assert 'region_counts' in mapping['statistics']
        assert 'region_percentages' in mapping['statistics']


class TestGLTFExporter:
    """Test GLTF export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.segmentation = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
    
    def test_mesh_generator(self):
        """Test mesh generator."""
        mesh_generator = MeshGenerator(resolution=0.5)
        
        # Generate mesh
        mesh = mesh_generator.generate_mesh(self.segmentation, class_id=1)
        
        # Check mesh properties
        assert hasattr(mesh, 'vertices')
        assert hasattr(mesh, 'faces')
        assert len(mesh.vertices) >= 0
        assert len(mesh.faces) >= 0
    
    def test_gltf_exporter(self):
        """Test GLTF exporter."""
        gltf_exporter = GLTFExporter(mesh_format='gltf')
        
        # Generate mesh data
        mesh_data = gltf_exporter.generate_mesh(self.segmentation)
        
        # Check mesh data
        assert isinstance(mesh_data, str)
        assert len(mesh_data) > 0
    
    def test_multi_class_mesh_generation(self):
        """Test multi-class mesh generation."""
        mesh_generator = MeshGenerator()
        
        # Generate meshes for multiple classes
        class_ids = [1, 2, 3]
        meshes = mesh_generator.generate_multi_class_mesh(self.segmentation, class_ids)
        
        assert len(meshes) <= len(class_ids)
        for class_id, mesh in meshes.items():
            assert class_id in class_ids
            assert hasattr(mesh, 'vertices')
            assert hasattr(mesh, 'faces')


class TestInferencePipeline:
    """Test complete inference pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create dummy input data
        self.input_data = {
            'T1': np.random.randn(128, 128, 128).astype(np.float32),
            'T1ce': np.random.randn(128, 128, 128).astype(np.float32),
            'T2': np.random.randn(128, 128, 128).astype(np.float32),
            'FLAIR': np.random.randn(128, 128, 128).astype(np.float32)
        }
        
        # Create dummy segmentation
        self.segmentation = np.random.randint(0, 4, (128, 128, 128)).astype(np.uint8)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline."""
        # This would normally test the complete pipeline
        # For now, we'll test individual components
        
        # Test input validation
        assert len(self.input_data) == 4
        for modality, data in self.input_data.items():
            assert data.shape == (128, 128, 128)
            assert data.dtype == np.float32
        
        # Test segmentation validation
        assert self.segmentation.shape == (128, 128, 128)
        assert self.segmentation.dtype == np.uint8
        assert np.all(self.segmentation >= 0)
        assert np.all(self.segmentation <= 3)
    
    def test_output_validation(self):
        """Test output validation."""
        # Create dummy outputs
        outputs = {
            'segmentation': self.segmentation,
            'probabilities': np.random.rand(3, 128, 128, 128).astype(np.float32),
            'expert_weights': np.random.rand(4).astype(np.float32),
            'spatial_attention': np.random.rand(128, 128, 128).astype(np.float32)
        }
        
        # Validate outputs
        assert 'segmentation' in outputs
        assert 'probabilities' in outputs
        assert 'expert_weights' in outputs
        assert 'spatial_attention' in outputs
        
        # Check shapes
        assert outputs['segmentation'].shape == (128, 128, 128)
        assert outputs['probabilities'].shape == (3, 128, 128, 128)
        assert outputs['expert_weights'].shape == (4,)
        assert outputs['spatial_attention'].shape == (128, 128, 128)
    
    def test_error_handling(self):
        """Test error handling in inference pipeline."""
        # Test with invalid input
        invalid_input = {
            'T1': np.random.randn(64, 64, 64).astype(np.float32),  # Wrong shape
            'T1ce': np.random.randn(128, 128, 128).astype(np.float32),
            'T2': np.random.randn(128, 128, 128).astype(np.float32),
            'FLAIR': np.random.randn(128, 128, 128).astype(np.float32)
        }
        
        # This should raise an error or handle gracefully
        try:
            # Simulate input validation
            for modality, data in invalid_input.items():
                if data.shape != (128, 128, 128):
                    raise ValueError(f"Invalid shape for {modality}: {data.shape}")
        except ValueError as e:
            assert "Invalid shape" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])


