"""
Device utilities for GPU/CPU handling.

Provides utilities for device management, memory monitoring, and
GPU/CPU optimization for the MoME+ segmentation system.
"""

import torch
import psutil
import GPUtil
from typing import Optional, Dict, Any, List
import logging

from .logger import get_logger

logger = get_logger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Specific device to use (optional)
        
    Returns:
        PyTorch device
    """
    try:
        if device is not None:
            return torch.device(device)
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
        
    except Exception as e:
        logger.error(f"Error getting device: {e}")
        return torch.device('cpu')


def set_device(device: str) -> torch.device:
    """
    Set the device for computation.
    
    Args:
        device: Device to set
        
    Returns:
        PyTorch device
    """
    try:
        device = torch.device(device)
        
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        
        logger.info(f"Device set to: {device}")
        return device
        
    except Exception as e:
        logger.error(f"Error setting device: {e}")
        return torch.device('cpu')


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dictionary containing GPU information
    """
    try:
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            gpu_info['devices'].append(device_info)
        
        return gpu_info
        
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {'available': False, 'error': str(e)}


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information.
    
    Returns:
        Dictionary containing CPU information
    """
    try:
        cpu_info = {
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq(),
            'usage': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_used': psutil.virtual_memory().used,
            'memory_percent': psutil.virtual_memory().percent
        }
        
        return cpu_info
        
    except Exception as e:
        logger.error(f"Error getting CPU info: {e}")
        return {'error': str(e)}


def get_memory_usage() -> Dict[str, Any]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary containing memory usage
    """
    try:
        memory_info = {
            'cpu': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            memory_info['gpu'] = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_cached': torch.cuda.max_memory_reserved()
            }
        
        return memory_info
        
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {'error': str(e)}


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared")
        else:
            logger.info("No GPU available to clear memory")
            
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")


def optimize_gpu_memory() -> None:
    """Optimize GPU memory usage."""
    try:
        if torch.cuda.is_available():
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            logger.info("GPU memory optimized")
        else:
            logger.info("No GPU available to optimize")
            
    except Exception as e:
        logger.error(f"Error optimizing GPU memory: {e}")


def get_device_status() -> Dict[str, Any]:
    """
    Get device status information.
    
    Returns:
        Dictionary containing device status
    """
    try:
        status = {
            'device': str(get_device()),
            'gpu_info': get_gpu_info(),
            'cpu_info': get_cpu_info(),
            'memory_usage': get_memory_usage()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting device status: {e}")
        return {'error': str(e)}


def monitor_device_usage(interval: float = 1.0) -> Dict[str, Any]:
    """
    Monitor device usage over time.
    
    Args:
        interval: Monitoring interval in seconds
        
    Returns:
        Dictionary containing usage statistics
    """
    try:
        import time
        
        # Get initial state
        initial_memory = get_memory_usage()
        
        # Wait for interval
        time.sleep(interval)
        
        # Get final state
        final_memory = get_memory_usage()
        
        # Calculate differences
        usage_stats = {
            'interval': interval,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_delta': {}
        }
        
        # Calculate memory deltas
        if 'gpu' in initial_memory and 'gpu' in final_memory:
            usage_stats['memory_delta']['gpu'] = {
                'allocated_delta': final_memory['gpu']['allocated'] - initial_memory['gpu']['allocated'],
                'cached_delta': final_memory['gpu']['cached'] - initial_memory['gpu']['cached']
            }
        
        if 'cpu' in initial_memory and 'cpu' in final_memory:
            usage_stats['memory_delta']['cpu'] = {
                'used_delta': final_memory['cpu']['used'] - initial_memory['cpu']['used']
            }
        
        return usage_stats
        
    except Exception as e:
        logger.error(f"Error monitoring device usage: {e}")
        return {'error': str(e)}


def get_optimal_batch_size(model: torch.nn.Module,
                          input_shape: tuple,
                          device: torch.device,
                          max_memory_gb: float = 8.0) -> int:
    """
    Find optimal batch size for given model and device.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to use
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Optimal batch size
    """
    try:
        if device.type == 'cpu':
            # For CPU, use a reasonable default
            return 4
        
        # For GPU, find optimal batch size
        model = model.to(device)
        model.eval()
        
        batch_size = 1
        max_batch_size = 64
        
        while batch_size <= max_batch_size:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                
                if memory_used > max_memory_gb:
                    break
                
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise
        
        # Clear memory
        clear_gpu_memory()
        
        # Return previous batch size
        optimal_batch_size = max(1, batch_size // 2)
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        logger.error(f"Error finding optimal batch size: {e}")
        return 1


def get_device_recommendations() -> Dict[str, Any]:
    """
    Get device recommendations based on system capabilities.
    
    Returns:
        Dictionary containing device recommendations
    """
    try:
        recommendations = {
            'recommended_device': 'cpu',
            'reason': 'No GPU available',
            'optimizations': []
        }
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            
            if gpu_info['available']:
                recommendations['recommended_device'] = 'cuda'
                recommendations['reason'] = 'GPU available and suitable'
                
                # Add GPU-specific optimizations
                recommendations['optimizations'].extend([
                    'Use mixed precision training',
                    'Enable gradient checkpointing',
                    'Use data parallel for multi-GPU',
                    'Optimize batch size for GPU memory'
                ])
        
        # Add general optimizations
        recommendations['optimizations'].extend([
            'Use appropriate batch size',
            'Enable memory efficient attention',
            'Clear GPU cache regularly',
            'Monitor memory usage'
        ])
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting device recommendations: {e}")
        return {'error': str(e)}




