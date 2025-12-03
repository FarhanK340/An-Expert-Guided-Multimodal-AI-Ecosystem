"""
Learning rate schedulers and optimizers for training.

Provides various learning rate scheduling strategies and optimizer configurations
for training the MoME+ segmentation model.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
)
from typing import Dict, Optional, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = config.get("type", "AdamW").lower()
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get("betas", [0.9, 0.999]),
            eps=config.get("eps", 1e-8)
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get("betas", [0.9, 0.999]),
            eps=config.get("eps", 1e-8)
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", True)
        )
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9),
            alpha=config.get("alpha", 0.99),
            eps=config.get("eps", 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler or None
    """
    if not config:
        return None
    
    scheduler_type = config.get("type", "CosineAnnealingLR").lower()
    
    if scheduler_type == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1)
        )
    elif scheduler_type == "multisteplr":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [30, 60, 90]),
            gamma=config.get("gamma", 0.1)
        )
    elif scheduler_type == "exponentiallr":
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.get("gamma", 0.95)
        )
    elif scheduler_type == "cosineannealinglr":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 100),
            eta_min=config.get("eta_min", 1e-6)
        )
    elif scheduler_type == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 1),
            eta_min=config.get("eta_min", 1e-6)
        )
    elif scheduler_type == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
            threshold=config.get("threshold", 1e-4),
            min_lr=config.get("min_lr", 1e-6)
        )
    elif scheduler_type == "onecyclelr":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 1e-3),
            total_steps=config.get("total_steps", 1000),
            pct_start=config.get("pct_start", 0.3),
            anneal_strategy=config.get("anneal_strategy", "cos")
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler with warmup.
    
    Implements a warmup phase followed by the main scheduling strategy.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 10,
                 warmup_start_lr: float = 1e-6,
                 main_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            main_scheduler: Main scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.main_scheduler = main_scheduler
        
        # Store initial learning rate
        self.initial_lr = optimizer.param_groups[0]["lr"]
        
        # Current epoch
        self.current_epoch = 0
    
    def step(self):
        """Step the scheduler."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # Main scheduling phase
            if self.main_scheduler:
                self.main_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class PolynomialScheduler:
    """
    Polynomial learning rate scheduler.
    
    Implements a polynomial decay schedule for learning rate.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 max_epochs: int,
                 power: float = 0.9,
                 min_lr: float = 1e-6):
        """
        Initialize polynomial scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            max_epochs: Maximum number of epochs
            power: Power of the polynomial
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        
        # Store initial learning rate
        self.initial_lr = optimizer.param_groups[0]["lr"]
        
        # Current epoch
        self.current_epoch = 0
    
    def step(self):
        """Step the scheduler."""
        if self.current_epoch < self.max_epochs:
            # Polynomial decay
            lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 - self.current_epoch / self.max_epochs) ** self.power
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class CustomScheduler:
    """
    Custom learning rate scheduler with multiple phases.
    
    Allows defining different learning rate schedules for different phases of training.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 phases: list):
        """
        Initialize custom scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            phases: List of phase configurations
        """
        self.optimizer = optimizer
        self.phases = phases
        
        # Current phase and epoch
        self.current_phase = 0
        self.current_epoch = 0
        
        # Store initial learning rate
        self.initial_lr = optimizer.param_groups[0]["lr"]
    
    def step(self):
        """Step the scheduler."""
        if self.current_phase < len(self.phases):
            phase = self.phases[self.current_phase]
            phase_epochs = phase.get("epochs", 1)
            
            if self.current_epoch < phase_epochs:
                # Apply current phase
                lr = phase.get("lr", self.initial_lr)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                # Move to next phase
                self.current_phase += 1
                self.current_epoch = 0
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Any:
    """
    Create a scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler
    """
    scheduler_type = config.get("type", "CosineAnnealingLR").lower()
    
    if scheduler_type == "warmup":
        return WarmupScheduler(
            optimizer,
            warmup_epochs=config.get("warmup_epochs", 10),
            warmup_start_lr=config.get("warmup_start_lr", 1e-6),
            main_scheduler=get_scheduler(optimizer, config.get("main_scheduler", {}))
        )
    elif scheduler_type == "polynomial":
        return PolynomialScheduler(
            optimizer,
            max_epochs=config.get("max_epochs", 100),
            power=config.get("power", 0.9),
            min_lr=config.get("min_lr", 1e-6)
        )
    elif scheduler_type == "custom":
        return CustomScheduler(
            optimizer,
            phases=config.get("phases", [])
        )
    else:
        return get_scheduler(optimizer, config)




