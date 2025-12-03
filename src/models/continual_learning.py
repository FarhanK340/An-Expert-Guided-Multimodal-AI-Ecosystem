"""
Continual learning modules for the MoME+ architecture.

Implements Elastic Weight Consolidation (EWC) and Experience Replay
for knowledge retention across different tasks and datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EWC(nn.Module):
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    
    Prevents catastrophic forgetting by penalizing changes to important weights
    from previous tasks.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 ewc_lambda: float = 1000.0,
                 fisher_samples: int = 1000):
        """
        Initialize EWC module.
        
        Args:
            model: The model to apply EWC to
            ewc_lambda: EWC regularization strength
            fisher_samples: Number of samples for Fisher information estimation
        """
        super().__init__()
        
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        
        # Store Fisher information matrices for each task
        self.fisher_matrices = {}
        self.optimal_params = {}
        
        # Current task ID
        self.current_task_id = 0
    
    def compute_fisher_information(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 task_id: int,
                                 num_samples: Optional[int] = None) -> None:
        """
        Compute Fisher information matrix for the current task.
        
        Args:
            dataloader: DataLoader for the current task
            task_id: Task identifier
            num_samples: Number of samples to use (if None, use all)
        """
        if num_samples is None:
            num_samples = self.fisher_samples
        
        logger.info(f"Computing Fisher information for task {task_id}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Fisher information
        fisher_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        # Compute Fisher information
        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                
                # Get input data
                if isinstance(batch, dict):
                    inputs = batch
                else:
                    inputs = batch[0]
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute gradients
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", 0)
                else:
                    loss = outputs
                
                if loss.requires_grad:
                    loss.backward()
                    
                    # Accumulate Fisher information
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            fisher_info[name] += param.grad.data ** 2
                    
                    # Clear gradients
                    self.model.zero_grad()
                
                sample_count += 1
        
        # Normalize Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        # Store Fisher information and optimal parameters
        self.fisher_matrices[task_id] = fisher_info
        self.optimal_params[task_id] = {
            name: param.data.clone() for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        logger.info(f"Stored Fisher information for task {task_id}")
    
    def compute_ewc_loss(self, task_id: int) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            task_id: Task identifier
            
        Returns:
            EWC regularization loss
        """
        if task_id not in self.fisher_matrices:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = 0.0
        fisher_info = self.fisher_matrices[task_id]
        optimal_params = self.optimal_params[task_id]
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in fisher_info:
                ewc_loss += (fisher_info[name] * (param - optimal_params[name]) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def update_task(self, task_id: int):
        """Update current task ID."""
        self.current_task_id = task_id
        logger.info(f"Updated EWC to task {task_id}")
    
    def get_ewc_loss(self) -> torch.Tensor:
        """Get total EWC loss for all previous tasks."""
        total_ewc_loss = 0.0
        
        for task_id in range(self.current_task_id):
            total_ewc_loss += self.compute_ewc_loss(task_id)
        
        return total_ewc_loss


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    Stores and samples examples from previous tasks to prevent catastrophic forgetting.
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 sample_size: int = 100,
                 strategy: str = "random"):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of examples to store
            sample_size: Number of examples to sample for replay
            strategy: Sampling strategy ("random", "herding", "reservoir")
        """
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.strategy = strategy
        
        # Store examples from different tasks
        self.task_buffers = defaultdict(list)
        self.task_counts = defaultdict(int)
        
        # Reservoir sampling for reservoir strategy
        self.reservoir_samples = {}
        self.sample_counts = {}
    
    def add_examples(self, 
                    examples: List[Dict[str, Any]], 
                    task_id: int) -> None:
        """
        Add examples to the replay buffer.
        
        Args:
            examples: List of example dictionaries
            task_id: Task identifier
        """
        if self.strategy == "random":
            self._add_random(examples, task_id)
        elif self.strategy == "herding":
            self._add_herding(examples, task_id)
        elif self.strategy == "reservoir":
            self._add_reservoir(examples, task_id)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def _add_random(self, examples: List[Dict[str, Any]], task_id: int) -> None:
        """Add examples using random sampling."""
        for example in examples:
            if len(self.task_buffers[task_id]) < self.buffer_size:
                self.task_buffers[task_id].append(example)
            else:
                # Replace random example
                idx = np.random.randint(0, len(self.task_buffers[task_id]))
                self.task_buffers[task_id][idx] = example
    
    def _add_herding(self, examples: List[Dict[str, Any]], task_id: int) -> None:
        """Add examples using herding strategy."""
        # Simplified herding - just add examples in order
        for example in examples:
            if len(self.task_buffers[task_id]) < self.buffer_size:
                self.task_buffers[task_id].append(example)
    
    def _add_reservoir(self, examples: List[Dict[str, Any]], task_id: int) -> None:
        """Add examples using reservoir sampling."""
        if task_id not in self.reservoir_samples:
            self.reservoir_samples[task_id] = []
            self.sample_counts[task_id] = 0
        
        for example in examples:
            self.sample_counts[task_id] += 1
            
            if len(self.reservoir_samples[task_id]) < self.buffer_size:
                self.reservoir_samples[task_id].append(example)
            else:
                # Reservoir sampling
                j = np.random.randint(0, self.sample_counts[task_id])
                if j < self.buffer_size:
                    self.reservoir_samples[task_id][j] = example
    
    def sample_examples(self, task_id: int) -> List[Dict[str, Any]]:
        """
        Sample examples from the replay buffer.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of sampled examples
        """
        if self.strategy == "reservoir":
            buffer = self.reservoir_samples.get(task_id, [])
        else:
            buffer = self.task_buffers.get(task_id, [])
        
        if len(buffer) == 0:
            return []
        
        # Sample examples
        sample_size = min(self.sample_size, len(buffer))
        sampled_indices = np.random.choice(len(buffer), sample_size, replace=False)
        
        return [buffer[i] for i in sampled_indices]
    
    def get_buffer_size(self, task_id: int) -> int:
        """Get buffer size for a specific task."""
        if self.strategy == "reservoir":
            return len(self.reservoir_samples.get(task_id, []))
        else:
            return len(self.task_buffers.get(task_id, []))
    
    def clear_task(self, task_id: int) -> None:
        """Clear buffer for a specific task."""
        if task_id in self.task_buffers:
            del self.task_buffers[task_id]
        if task_id in self.reservoir_samples:
            del self.reservoir_samples[task_id]
        if task_id in self.sample_counts:
            del self.sample_counts[task_id]


class ContinualLearningWrapper(nn.Module):
    """
    Wrapper for continual learning that combines EWC and experience replay.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 ewc_lambda: float = 1000.0,
                 fisher_samples: int = 1000,
                 replay_buffer_size: int = 1000,
                 replay_sample_size: int = 100,
                 replay_strategy: str = "random"):
        """
        Initialize continual learning wrapper.
        
        Args:
            model: The model to apply continual learning to
            ewc_lambda: EWC regularization strength
            fisher_samples: Number of samples for Fisher information estimation
            replay_buffer_size: Size of replay buffer
            replay_sample_size: Number of examples to sample for replay
            replay_strategy: Replay sampling strategy
        """
        super().__init__()
        
        self.model = model
        self.ewc = EWC(model, ewc_lambda, fisher_samples)
        self.replay_buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            sample_size=replay_sample_size,
            strategy=replay_strategy
        )
        
        self.current_task_id = 0
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)
    
    def compute_continual_learning_loss(self, 
                                      current_loss: torch.Tensor,
                                      task_id: int) -> torch.Tensor:
        """
        Compute total continual learning loss.
        
        Args:
            current_loss: Current task loss
            task_id: Current task identifier
            
        Returns:
            Total loss including continual learning regularization
        """
        # EWC loss
        ewc_loss = self.ewc.get_ewc_loss()
        
        # Total loss
        total_loss = current_loss + ewc_loss
        
        return total_loss
    
    def update_task(self, task_id: int):
        """Update current task."""
        self.current_task_id = task_id
        self.ewc.update_task(task_id)
        logger.info(f"Updated continual learning to task {task_id}")
    
    def add_replay_examples(self, 
                          examples: List[Dict[str, Any]], 
                          task_id: int) -> None:
        """Add examples to replay buffer."""
        self.replay_buffer.add_examples(examples, task_id)
    
    def sample_replay_examples(self, task_id: int) -> List[Dict[str, Any]]:
        """Sample examples from replay buffer."""
        return self.replay_buffer.sample_examples(task_id)
    
    def compute_fisher_information(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 task_id: int) -> None:
        """Compute Fisher information for EWC."""
        self.ewc.compute_fisher_information(dataloader, task_id)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about continual learning."""
        stats = {
            "current_task_id": self.current_task_id,
            "ewc_tasks": len(self.ewc.fisher_matrices),
            "replay_tasks": len(self.replay_buffer.task_buffers),
            "total_ewc_loss": self.ewc.get_ewc_loss().item()
        }
        
        # Add buffer sizes for each task
        for task_id in range(self.current_task_id):
            stats[f"replay_buffer_size_task_{task_id}"] = self.replay_buffer.get_buffer_size(task_id)
        
        return stats

