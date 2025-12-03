"""
Checkpoint utilities for saving and loading model states.

Provides functionality for saving, loading, and managing model checkpoints
during training and inference.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup and organization.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "experiments/checkpoints",
                 keep_last_n: int = 5,
                 max_checkpoints: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_last_n: Number of recent checkpoints to keep
            max_checkpoints: Maximum number of checkpoints to store
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoint files
        self.checkpoint_files = []
    
    def save_checkpoint(self, 
                       checkpoint: Dict[str, Any], 
                       filename: str) -> str:
        """
        Save a checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint dictionary
            filename: Filename for the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint file
        self.checkpoint_files.append(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       filename: str) -> Dict[str, Any]:
        """
        Load a checkpoint from disk.
        
        Args:
            filename: Filename of the checkpoint to load
            
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint filenames
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
        return [f.name for f in sorted(checkpoint_files)]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint filename.
        
        Returns:
            Latest checkpoint filename or None
        """
        checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            return None
        
        latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return latest_file.name
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Remove oldest checkpoints
            files_to_remove = self.checkpoint_files[:-self.max_checkpoints]
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed old checkpoint: {file_path}")
            
            # Update tracked files
            self.checkpoint_files = self.checkpoint_files[-self.max_checkpoints:]


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   checkpoint_dir: str,
                   filename: str,
                   additional_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a complete training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The scheduler to save (optional)
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Filename for checkpoint
        additional_info: Additional information to save
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "metrics": metrics,
        "additional_info": additional_info or {}
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


def load_checkpoint(model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   checkpoint_path: str,
                   device: torch.device = torch.device("cpu"),
                   strict: bool = True) -> Dict[str, Any]:
    """
    Load a complete training checkpoint.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Loaded checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint


def save_model_only(model: nn.Module,
                   checkpoint_dir: str,
                   filename: str,
                   additional_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Save only the model state (for inference).
    
    Args:
        model: The model to save
        checkpoint_dir: Directory to save checkpoint
        filename: Filename for checkpoint
        additional_info: Additional information to save
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": additional_info or {}
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    logger.info(f"Saved model: {checkpoint_path}")
    return str(checkpoint_path)


def load_model_only(model: nn.Module,
                   checkpoint_path: str,
                   device: torch.device = torch.device("cpu"),
                   strict: bool = True) -> Dict[str, Any]:
    """
    Load only the model state (for inference).
    
    Args:
        model: The model to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Loaded checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    logger.info(f"Loaded model: {checkpoint_path}")
    return checkpoint


def copy_checkpoint(source_path: str, 
                   dest_path: str) -> str:
    """
    Copy a checkpoint to a new location.
    
    Args:
        source_path: Source checkpoint path
        dest_path: Destination checkpoint path
        
    Returns:
        Destination path
    """
    shutil.copy2(source_path, dest_path)
    logger.info(f"Copied checkpoint: {source_path} -> {dest_path}")
    return dest_path


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading it.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    info = {
        "epoch": checkpoint.get("epoch", "Unknown"),
        "loss": checkpoint.get("loss", "Unknown"),
        "metrics": checkpoint.get("metrics", {}),
        "file_size": os.path.getsize(checkpoint_path),
        "additional_info": checkpoint.get("additional_info", {})
    }
    
    return info


def cleanup_checkpoints(checkpoint_dir: str, 
                      keep_last_n: int = 5) -> None:
    """
    Clean up old checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pth"))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[:-keep_last_n]
    for file_path in files_to_remove:
        file_path.unlink()
        logger.info(f"Removed old checkpoint: {file_path}")


def create_checkpoint_backup(checkpoint_path: str, 
                           backup_dir: str) -> str:
    """
    Create a backup of a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint to backup
        backup_dir: Directory to store backup
        
    Returns:
        Path to backup file
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(checkpoint_path)
    backup_path = backup_dir / f"{checkpoint_path.stem}_backup{checkpoint_path.suffix}"
    
    shutil.copy2(checkpoint_path, backup_path)
    logger.info(f"Created checkpoint backup: {backup_path}")
    
    return str(backup_path)




