"""
Logging utilities for experiment tracking and debugging.

Provides structured logging for the MoME+ segmentation system with
support for different log levels and output formats.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/mome_segmentation.log')
    ]
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(log_level: str = 'INFO',
                  log_file: Optional[str] = None,
                  log_format: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path
        log_format: Log format string
    """
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


class ExperimentLogger:
    """
    Logger for experiment tracking and metrics.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = 'logs/experiments'):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Setup logging
        setup_logging(log_file=str(self.log_file))
        self.logger = get_logger(f"experiment.{experiment_name}")
        
        # Initialize metrics storage
        self.metrics = {}
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.json"
    
    def log_metric(self, 
                   metric_name: str,
                   value: float,
                   step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Step number (optional)
        """
        if step is None:
            step = len(self.metrics.get(metric_name, []))
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info(f"Metric {metric_name}: {value} (step {step})")
    
    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (optional)
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step)
    
    def log_config(self, 
                   config: Dict[str, Any]) -> None:
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_info(self, 
                 message: str) -> None:
        """
        Log info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def log_warning(self, 
                   message: str) -> None:
        """
        Log warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def log_error(self, 
                 message: str) -> None:
        """
        Log error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def save_metrics(self) -> None:
        """Save metrics to file."""
        import json
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def get_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get logged metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def get_metric_history(self, 
                          metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return self.metrics.get(metric_name, []).copy()
    
    def get_latest_metric(self, 
                         metric_name: str) -> Optional[float]:
        """
        Get latest value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest metric value or None
        """
        history = self.get_metric_history(metric_name)
        if history:
            return history[-1]['value']
        return None
    
    def close(self) -> None:
        """Close the logger and save metrics."""
        self.save_metrics()
        self.logger.info("Experiment logger closed")


def create_experiment_logger(experiment_name: str,
                           log_dir: str = 'logs/experiments') -> ExperimentLogger:
    """
    Create an experiment logger.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        
    Returns:
        Experiment logger instance
    """
    return ExperimentLogger(experiment_name, log_dir)




