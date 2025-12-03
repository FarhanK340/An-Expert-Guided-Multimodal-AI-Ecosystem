"""
Training visualization utilities.

Provides functionality for plotting training curves, metrics, and
other visualizations for monitoring training progress.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrainingVisualizer:
    """
    Training visualization class for plotting training progress.
    """
    
    def __init__(self, 
                 save_dir: str = "experiments/plots",
                 style: str = "seaborn-v0_8"):
        """
        Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_training_curves(self, 
                           training_history: Dict[str, List[float]],
                           save_path: Optional[str] = None,
                           show: bool = False) -> str:
        """
        Plot training curves (loss, metrics, etc.).
        
        Args:
            training_history: Dictionary with training history
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        
        # Create figure with subplots
        num_metrics = len(training_history)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(training_history.items()):
            ax = axes[idx]
            
            # Plot metric
            ax.plot(values, label=metric_name, linewidth=2)
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved training curves: {save_path}")
        return str(save_path)
    
    def plot_loss_curves(self, 
                        train_losses: List[float],
                        val_losses: Optional[List[float]] = None,
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
        """
        Plot loss curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses (optional)
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "loss_curves.png"
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        plt.plot(train_losses, label="Training Loss", linewidth=2, color="blue")
        
        # Plot validation loss if provided
        if val_losses:
            plt.plot(val_losses, label="Validation Loss", linewidth=2, color="red")
        
        plt.title("Training and Validation Loss", fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved loss curves: {save_path}")
        return str(save_path)
    
    def plot_metrics(self, 
                    metrics_history: List[Dict[str, float]],
                    save_path: Optional[str] = None,
                    show: bool = False) -> str:
        """
        Plot metrics over time.
        
        Args:
            metrics_history: List of metric dictionaries
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "metrics.png"
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        
        # Create figure
        num_metrics = len(df.columns)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(df.columns):
            ax = axes[idx]
            
            # Plot metric
            ax.plot(df.index, df[metric_name], label=metric_name, linewidth=2)
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved metrics plot: {save_path}")
        return str(save_path)
    
    def plot_learning_rate(self, 
                          learning_rates: List[float],
                          save_path: Optional[str] = None,
                          show: bool = False) -> str:
        """
        Plot learning rate schedule.
        
        Args:
            learning_rates: Learning rates over time
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "learning_rate.png"
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(learning_rates, linewidth=2, color="green")
        plt.title("Learning Rate Schedule", fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved learning rate plot: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None,
                            show: bool = False) -> str:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "confusion_matrix.png"
        
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt=".2f", 
                   cmap="Blues",
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title("Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved confusion matrix: {save_path}")
        return str(save_path)
    
    def plot_class_distribution(self, 
                              class_counts: Dict[str, int],
                              save_path: Optional[str] = None,
                              show: bool = False) -> str:
        """
        Plot class distribution.
        
        Args:
            class_counts: Dictionary with class counts
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "class_distribution.png"
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(classes, counts, color="skyblue", edgecolor="navy", alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha="center", va="bottom")
        
        plt.title("Class Distribution", fontsize=16)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved class distribution: {save_path}")
        return str(save_path)
    
    def plot_continual_learning_metrics(self, 
                                      task_metrics: Dict[str, Dict[str, List[float]]],
                                      save_path: Optional[str] = None,
                                      show: bool = False) -> str:
        """
        Plot continual learning metrics.
        
        Args:
            task_metrics: Dictionary with metrics for each task
            save_path: Path to save plot (optional)
            show: Whether to show the plot
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.save_dir / "continual_learning_metrics.png"
        
        # Get all unique metrics
        all_metrics = set()
        for task_data in task_metrics.values():
            all_metrics.update(task_data.keys())
        
        # Create subplots
        num_metrics = len(all_metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(all_metrics):
            ax = axes[idx]
            
            # Plot metric for each task
            for task_name, task_data in task_metrics.items():
                if metric_name in task_data:
                    ax.plot(task_data[metric_name], label=task_name, linewidth=2)
            
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Saved continual learning metrics: {save_path}")
        return str(save_path)


def plot_training_curves(training_history: Dict[str, List[float]], 
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
    """
    Convenience function to plot training curves.
    
    Args:
        training_history: Dictionary with training history
        save_path: Path to save plot (optional)
        show: Whether to show the plot
        
    Returns:
        Path to saved plot
    """
    visualizer = TrainingVisualizer()
    return visualizer.plot_training_curves(training_history, save_path, show)


def plot_loss_curves(train_losses: List[float],
                    val_losses: Optional[List[float]] = None,
                    save_path: Optional[str] = None,
                    show: bool = False) -> str:
    """
    Convenience function to plot loss curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses (optional)
        save_path: Path to save plot (optional)
        show: Whether to show the plot
        
    Returns:
        Path to saved plot
    """
    visualizer = TrainingVisualizer()
    return visualizer.plot_loss_curves(train_losses, val_losses, save_path, show)


def plot_metrics(metrics_history: List[Dict[str, float]],
                save_path: Optional[str] = None,
                show: bool = False) -> str:
    """
    Convenience function to plot metrics.
    
    Args:
        metrics_history: List of metric dictionaries
        save_path: Path to save plot (optional)
        show: Whether to show the plot
        
    Returns:
        Path to saved plot
    """
    visualizer = TrainingVisualizer()
    return visualizer.plot_metrics(metrics_history, save_path, show)




