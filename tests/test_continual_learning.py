"""
Tests for continual learning functionality.

Tests EWC (Elastic Weight Consolidation), replay buffer,
and continual learning mechanisms.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.continual_learning import EWC, ReplayBuffer, ContinualLearner
from src.training.loss_functions import EWCLoss, ReplayLoss


class TestEWC:
    """Test Elastic Weight Consolidation (EWC)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = torch.nn.Linear(10, 5)
        self.ewc = EWC(model=self.model, lambda_ewc=1000.0)
        
        # Create dummy data
        self.task1_data = torch.randn(100, 10)
        self.task1_targets = torch.randint(0, 5, (100,))
        self.task2_data = torch.randn(100, 10)
        self.task2_targets = torch.randint(0, 5, (100,))
    
    def test_ewc_initialization(self):
        """Test EWC initialization."""
        assert self.ewc.lambda_ewc == 1000.0
        assert self.ewc.model == self.model
        assert self.ewc.fisher_information is None
        assert self.ewc.optimal_params is None
    
    def test_compute_fisher_information(self):
        """Test Fisher information computation."""
        # Train on task 1
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.model(self.task1_data)
            loss = criterion(outputs, self.task1_targets)
            loss.backward()
            optimizer.step()
        
        # Compute Fisher information
        self.ewc.compute_fisher_information(self.task1_data, self.task1_targets)
        
        assert self.ewc.fisher_information is not None
        assert self.ewc.optimal_params is not None
        
        # Check Fisher information values
        for param_name, fisher_info in self.ewc.fisher_information.items():
            assert fisher_info.shape == self.model.state_dict()[param_name].shape
            assert torch.all(fisher_info >= 0)
    
    def test_ewc_loss(self):
        """Test EWC loss computation."""
        # Compute Fisher information first
        self.ewc.compute_fisher_information(self.task1_data, self.task1_targets)
        
        # Compute EWC loss
        ewc_loss = self.ewc.compute_ewc_loss()
        
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.item() >= 0
    
    def test_ewc_regularization(self):
        """Test EWC regularization effect."""
        # Train on task 1
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.model(self.task1_data)
            loss = criterion(outputs, self.task1_targets)
            loss.backward()
            optimizer.step()
        
        # Store task 1 parameters
        task1_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Compute Fisher information
        self.ewc.compute_fisher_information(self.task1_data, self.task1_targets)
        
        # Train on task 2 with EWC
        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.model(self.task2_data)
            task_loss = criterion(outputs, self.task2_targets)
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss = task_loss + ewc_loss
            total_loss.backward()
            optimizer.step()
        
        # Check that parameters didn't change too much
        for name, param in self.model.named_parameters():
            param_diff = torch.norm(param - task1_params[name])
            assert param_diff < 10.0  # Parameters should be relatively stable


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer_size = 1000
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size)
        
        # Create dummy data
        self.sample_data = {
            'T1': torch.randn(1, 1, 64, 64, 64),
            'T1ce': torch.randn(1, 1, 64, 64, 64),
            'T2': torch.randn(1, 1, 64, 64, 64),
            'FLAIR': torch.randn(1, 1, 64, 64, 64)
        }
        self.sample_target = torch.randint(0, 3, (1, 64, 64, 64))
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization."""
        assert self.buffer.buffer_size == self.buffer_size
        assert len(self.buffer.buffer) == 0
        assert self.buffer.current_size == 0
    
    def test_add_sample(self):
        """Test adding samples to buffer."""
        self.buffer.add_sample(self.sample_data, self.sample_target)
        
        assert len(self.buffer.buffer) == 1
        assert self.buffer.current_size == 1
        
        # Check sample structure
        sample = self.buffer.buffer[0]
        assert 'data' in sample
        assert 'target' in sample
        assert sample['data']['T1'].shape == self.sample_data['T1'].shape
        assert sample['target'].shape == self.sample_target.shape
    
    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        # Add more samples than buffer size
        for i in range(self.buffer_size + 100):
            sample_data = {
                'T1': torch.randn(1, 1, 64, 64, 64),
                'T1ce': torch.randn(1, 1, 64, 64, 64),
                'T2': torch.randn(1, 1, 64, 64, 64),
                'FLAIR': torch.randn(1, 1, 64, 64, 64)
            }
            sample_target = torch.randint(0, 3, (1, 64, 64, 64))
            self.buffer.add_sample(sample_data, sample_target)
        
        assert len(self.buffer.buffer) == self.buffer_size
        assert self.buffer.current_size == self.buffer_size
    
    def test_sample_batch(self):
        """Test sampling batch from buffer."""
        # Add samples to buffer
        for i in range(50):
            sample_data = {
                'T1': torch.randn(1, 1, 64, 64, 64),
                'T1ce': torch.randn(1, 1, 64, 64, 64),
                'T2': torch.randn(1, 1, 64, 64, 64),
                'FLAIR': torch.randn(1, 1, 64, 64, 64)
            }
            sample_target = torch.randint(0, 3, (1, 64, 64, 64))
            self.buffer.add_sample(sample_data, sample_target)
        
        # Sample batch
        batch_size = 10
        batch = self.buffer.sample_batch(batch_size)
        
        assert len(batch) == batch_size
        assert 'data' in batch
        assert 'target' in batch
        assert batch['data']['T1'].shape == (batch_size, 1, 64, 64, 64)
        assert batch['target'].shape == (batch_size, 64, 64, 64)
    
    def test_buffer_clear(self):
        """Test clearing buffer."""
        # Add samples
        for i in range(10):
            sample_data = {
                'T1': torch.randn(1, 1, 64, 64, 64),
                'T1ce': torch.randn(1, 1, 64, 64, 64),
                'T2': torch.randn(1, 1, 64, 64, 64),
                'FLAIR': torch.randn(1, 1, 64, 64, 64)
            }
            sample_target = torch.randint(0, 3, (1, 64, 64, 64))
            self.buffer.add_sample(sample_data, sample_target)
        
        assert len(self.buffer.buffer) == 10
        
        # Clear buffer
        self.buffer.clear()
        
        assert len(self.buffer.buffer) == 0
        assert self.buffer.current_size == 0


class TestContinualLearner:
    """Test continual learning system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = torch.nn.Linear(10, 5)
        self.continual_learner = ContinualLearner(
            model=self.model,
            ewc_lambda=1000.0,
            replay_buffer_size=1000
        )
        
        # Create dummy data for different tasks
        self.task1_data = torch.randn(100, 10)
        self.task1_targets = torch.randint(0, 5, (100,))
        self.task2_data = torch.randn(100, 10)
        self.task2_targets = torch.randint(0, 5, (100,))
    
    def test_continual_learner_initialization(self):
        """Test continual learner initialization."""
        assert self.continual_learner.model == self.model
        assert self.continual_learner.ewc is not None
        assert self.continual_learner.replay_buffer is not None
        assert self.continual_learner.current_task == 0
    
    def test_learn_task(self):
        """Test learning a new task."""
        # Learn task 1
        self.continual_learner.learn_task(
            self.task1_data, self.task1_targets, num_epochs=5
        )
        
        assert self.continual_learner.current_task == 1
        assert self.continual_learner.ewc.fisher_information is not None
        assert self.continual_learner.ewc.optimal_params is not None
    
    def test_continual_learning(self):
        """Test continual learning across multiple tasks."""
        # Learn task 1
        self.continual_learner.learn_task(
            self.task1_data, self.task1_targets, num_epochs=5
        )
        
        # Store task 1 performance
        task1_outputs = self.model(self.task1_data)
        task1_loss = torch.nn.functional.cross_entropy(task1_outputs, self.task1_targets)
        
        # Learn task 2
        self.continual_learner.learn_task(
            self.task2_data, self.task2_targets, num_epochs=5
        )
        
        # Check that task 1 performance is maintained
        task1_outputs_after = self.model(self.task1_data)
        task1_loss_after = torch.nn.functional.cross_entropy(task1_outputs_after, self.task1_targets)
        
        # Task 1 loss should not increase dramatically
        assert task1_loss_after.item() < task1_loss.item() * 2.0
    
    def test_replay_learning(self):
        """Test replay learning mechanism."""
        # Learn task 1
        self.continual_learner.learn_task(
            self.task1_data, self.task1_targets, num_epochs=5
        )
        
        # Learn task 2 with replay
        self.continual_learner.learn_task(
            self.task2_data, self.task2_targets, num_epochs=5, use_replay=True
        )
        
        # Check that replay buffer has samples
        assert len(self.continual_learner.replay_buffer.buffer) > 0
    
    def test_task_switching(self):
        """Test task switching functionality."""
        # Learn multiple tasks
        for task_id in range(3):
            task_data = torch.randn(100, 10)
            task_targets = torch.randint(0, 5, (100,))
            
            self.continual_learner.learn_task(
                task_data, task_targets, num_epochs=3
            )
        
        assert self.continual_learner.current_task == 3
        
        # Switch to previous task
        self.continual_learner.switch_to_task(1)
        assert self.continual_learner.current_task == 1


class TestLossFunctions:
    """Test continual learning loss functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = torch.nn.Linear(10, 5)
        self.ewc_loss = EWCLoss(model=self.model, lambda_ewc=1000.0)
        self.replay_loss = ReplayLoss(alpha=0.5)
    
    def test_ewc_loss_computation(self):
        """Test EWC loss computation."""
        # Create dummy data
        task_data = torch.randn(100, 10)
        task_targets = torch.randint(0, 5, (100,))
        
        # Compute Fisher information
        self.ewc_loss.compute_fisher_information(task_data, task_targets)
        
        # Compute EWC loss
        ewc_loss = self.ewc_loss.compute_ewc_loss()
        
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.item() >= 0
    
    def test_replay_loss_computation(self):
        """Test replay loss computation."""
        # Create dummy outputs and targets
        current_outputs = torch.randn(10, 5)
        current_targets = torch.randint(0, 5, (10,))
        replay_outputs = torch.randn(10, 5)
        replay_targets = torch.randint(0, 5, (10,))
        
        # Compute replay loss
        total_loss = self.replay_loss(
            current_outputs, current_targets,
            replay_outputs, replay_targets
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])


