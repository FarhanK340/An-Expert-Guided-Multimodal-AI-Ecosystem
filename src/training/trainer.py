"""
Main training loop for the MoME+ segmentation system.

Handles training, validation, and continual learning scenarios
with support for EWC and experience replay.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from .loss_functions import CombinedLoss
from .metrics import SegmentationMetrics
from .scheduler import get_scheduler, get_optimizer
from .checkpoint_utils import CheckpointManager
from .visualization import TrainingVisualizer
from ..models.mome_segmenter import MoMESegmenter
from ..models.continual_learning import ContinualLearningWrapper
from ..utils.logger import get_logger
from ..utils.config_parser import load_config

logger = get_logger(__name__)


class Trainer:
    """
    Main trainer class for the MoME+ segmentation system.
    
    Handles training, validation, and continual learning scenarios.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 device: torch.device,
                 continual_learning: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on
            continual_learning: Whether to use continual learning
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.continual_learning = continual_learning
        
        # Training configuration
        self.epochs = config.get("training", {}).get("epochs", 100)
        self.learning_rate = config.get("training", {}).get("learning_rate", 1e-4)
        self.weight_decay = config.get("training", {}).get("weight_decay", 1e-5)
        self.gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm", 1.0)
        
        # Validation configuration
        self.val_frequency = config.get("validation", {}).get("frequency", 5)
        self.save_best = config.get("validation", {}).get("save_best", True)
        self.best_metric = config.get("validation", {}).get("metric", "dice_score")
        
        # Checkpoint configuration
        self.checkpoint_dir = config.get("checkpoint", {}).get("save_dir", "experiments/checkpoints")
        self.save_frequency = config.get("checkpoint", {}).get("save_frequency", 10)
        self.keep_last_n = config.get("checkpoint", {}).get("keep_last_n", 5)
        
        # Logging configuration
        self.log_dir = config.get("logging", {}).get("log_dir", "experiments/logs")
        self.log_frequency = config.get("logging", {}).get("log_frequency", 100)
        self.use_tensorboard = config.get("logging", {}).get("use_tensorboard", True)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
    
    def _initialize_components(self):
        """Initialize training components."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(self.config.get("loss", {}))
        
        # Initialize optimizer
        self.optimizer = get_optimizer(self.model, self.config.get("optimizer", {}))
        
        # Initialize scheduler
        self.scheduler = get_scheduler(self.optimizer, self.config.get("scheduler", {}))
        
        # Initialize metrics
        self.metrics = SegmentationMetrics()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            keep_last_n=self.keep_last_n
        )
        
        # Initialize visualization
        self.visualizer = TrainingVisualizer()
        
        # Initialize logging
        if self.use_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # Initialize continual learning if enabled
        if self.continual_learning:
            cl_config = self.config.get("continual_learning", {})
            self.model = ContinualLearningWrapper(self.model, **cl_config)
            self.model = self.model.to(self.device)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results and history
        """
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            if (epoch + 1) % self.val_frequency == 0:
                val_loss, val_metrics = self._validate_epoch()
                
                # Update best score
                if val_metrics[self.best_metric] > self.best_score:
                    self.best_score = val_metrics[self.best_metric]
                    if self.save_best:
                        self._save_checkpoint(is_best=True)
                
                # Log validation results
                self._log_validation_results(val_loss, val_metrics)
                
                # Update training history
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_metrics"].append(val_metrics)
            else:
                val_loss, val_metrics = None, None
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint()
            
            # Log training results
            self._log_training_results(train_loss, val_loss, val_metrics)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        # Final save
        self._save_checkpoint(is_final=True)
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
        
        return {
            "training_history": self.training_history,
            "best_score": self.best_score,
            "final_epoch": self.current_epoch
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss = self.criterion(outputs, batch)
            
            # Add continual learning loss if enabled
            if self.continual_learning:
                cl_loss = self.model.compute_continual_learning_loss(loss, self.model.current_task_id)
                loss = cl_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log training progress
            if self.global_step % self.log_frequency == 0:
                self._log_training_step(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = self.criterion(outputs, batch)
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self.metrics.compute_metrics(outputs, batch)
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return total_loss / len(self.val_dataloader), avg_metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        return device_batch
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_score": self.best_score,
            "training_history": self.training_history,
            "config": self.config
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint, "best_model.pth"
            )
            logger.info(f"Saved best model checkpoint: {checkpoint_path}")
        elif is_final:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint, "final_model.pth"
            )
            logger.info(f"Saved final model checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint, f"checkpoint_epoch_{self.current_epoch + 1}.pth"
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _log_training_step(self, loss: float):
        """Log training step."""
        if self.writer:
            self.writer.add_scalar("Train/Loss", loss, self.global_step)
            self.writer.add_scalar("Train/LearningRate", 
                                 self.optimizer.param_groups[0]["lr"], 
                                 self.global_step)
    
    def _log_training_results(self, train_loss: float, val_loss: Optional[float], 
                            val_metrics: Optional[Dict[str, float]]):
        """Log training results."""
        if self.writer:
            self.writer.add_scalar("Epoch/TrainLoss", train_loss, self.current_epoch)
            
            if val_loss is not None:
                self.writer.add_scalar("Epoch/ValLoss", val_loss, self.current_epoch)
            
            if val_metrics is not None:
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f"Epoch/Val{metric_name}", metric_value, self.current_epoch)
    
    def _log_validation_results(self, val_loss: float, val_metrics: Dict[str, float]):
        """Log validation results."""
        logger.info(f"Epoch {self.current_epoch + 1} - Validation Loss: {val_loss:.4f}")
        for metric_name, metric_value in val_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def train_continual_learning(self, task_dataloaders: List[DataLoader]) -> Dict[str, Any]:
        """
        Train with continual learning across multiple tasks.
        
        Args:
            task_dataloaders: List of data loaders for each task
            
        Returns:
            Training results for all tasks
        """
        if not self.continual_learning:
            raise ValueError("Continual learning not enabled")
        
        logger.info("Starting continual learning training...")
        
        all_results = {}
        
        for task_id, task_dataloader in enumerate(task_dataloaders):
            logger.info(f"Training task {task_id + 1}/{len(task_dataloaders)}")
            
            # Update current task
            self.model.update_task(task_id)
            
            # Train on current task
            task_results = self.train()
            all_results[f"task_{task_id}"] = task_results
            
            # Compute Fisher information for EWC
            if task_id < len(task_dataloaders) - 1:  # Not the last task
                logger.info(f"Computing Fisher information for task {task_id}")
                self.model.compute_fisher_information(task_dataloader, task_id)
        
        return all_results


def main():
    """Main function for command-line training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MoME+ segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--continual_learning", action="store_true", help="Enable continual learning")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MoMESegmenter(**config.get("model", {}))
    
    # Create data loaders (placeholder - implement based on your data)
    train_dataloader = None  # Implement based on your data
    val_dataloader = None    # Implement based on your data
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        continual_learning=args.continual_learning
    )
    
    # Train
    results = trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

