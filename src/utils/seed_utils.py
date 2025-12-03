"""
Seed utilities for reproducibility.

Provides utilities for setting random seeds across different libraries
to ensure reproducible results.
"""

import random
import numpy as np
import torch
import os
from typing import Optional, Dict, Any

from .logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    try:
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy random seed
        np.random.seed(seed)
        
        # Set PyTorch random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set PyTorch deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Random seed set to: {seed}")
        
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise


def get_seed() -> Optional[int]:
    """
    Get current random seed.
    
    Returns:
        Current seed value or None
    """
    try:
        # Get Python random state
        python_seed = random.getstate()[1][0]
        
        # Get NumPy random state
        numpy_seed = np.random.get_state()[1][0]
        
        # Get PyTorch random state
        torch_seed = torch.initial_seed()
        
        # Check if all seeds are the same
        if python_seed == numpy_seed == torch_seed:
            return python_seed
        else:
            logger.warning("Seeds are not synchronized across libraries")
            return None
            
    except Exception as e:
        logger.error(f"Error getting seed: {e}")
        return None


def set_deterministic(seed: int = 42) -> None:
    """
    Set deterministic behavior for all libraries.
    
    Args:
        seed: Random seed value
    """
    try:
        # Set seed
        set_seed(seed)
        
        # Set additional deterministic settings
        torch.use_deterministic_algorithms(True)
        
        # Set environment variables
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.info(f"Deterministic behavior enabled with seed: {seed}")
        
    except Exception as e:
        logger.error(f"Error setting deterministic behavior: {e}")
        raise


def get_random_state() -> Dict[str, Any]:
    """
    Get random state for all libraries.
    
    Returns:
        Dictionary containing random states
    """
    try:
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
    except Exception as e:
        logger.error(f"Error getting random state: {e}")
        return {}


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Set random state for all libraries.
    
    Args:
        state: Dictionary containing random states
    """
    try:
        if 'python' in state:
            random.setstate(state['python'])
        
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        
        if 'torch_cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['torch_cuda'])
        
        logger.info("Random state restored")
        
    except Exception as e:
        logger.error(f"Error setting random state: {e}")
        raise


def create_seed_generator(seed: int = 42) -> 'SeedGenerator':
    """
    Create a seed generator for reproducible random numbers.
    
    Args:
        seed: Initial seed value
        
    Returns:
        Seed generator instance
    """
    return SeedGenerator(seed)


class SeedGenerator:
    """
    Generator for reproducible random seeds.
    """
    
    def __init__(self, 
                 initial_seed: int = 42):
        """
        Initialize seed generator.
        
        Args:
            initial_seed: Initial seed value
        """
        self.initial_seed = initial_seed
        self.current_seed = initial_seed
        self.seed_history = [initial_seed]
    
    def next_seed(self) -> int:
        """
        Get next seed value.
        
        Returns:
            Next seed value
        """
        self.current_seed += 1
        self.seed_history.append(self.current_seed)
        return self.current_seed
    
    def reset(self) -> None:
        """Reset to initial seed."""
        self.current_seed = self.initial_seed
        self.seed_history = [self.initial_seed]
    
    def get_seed_history(self) -> list:
        """
        Get seed history.
        
        Returns:
            List of used seeds
        """
        return self.seed_history.copy()
    
    def set_seed(self, 
                 seed: int) -> None:
        """
        Set current seed.
        
        Args:
            seed: Seed value to set
        """
        self.current_seed = seed
        self.seed_history.append(seed)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get generator status.
        
        Returns:
            Status information
        """
        return {
            'initial_seed': self.initial_seed,
            'current_seed': self.current_seed,
            'seed_count': len(self.seed_history),
            'seed_history': self.seed_history
        }




