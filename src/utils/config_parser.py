"""
Configuration parser for YAML configuration files.

Provides utilities for loading, parsing, and managing configuration files
for the MoME+ segmentation system.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"JSON configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading JSON configuration: {e}")
        raise


def save_json_config(config: Dict[str, Any], 
                     config_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"JSON configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving JSON configuration: {e}")
        raise


class ConfigManager:
    """
    Manages configuration files and settings.
    """
    
    def __init__(self, 
                 config_dir: str = 'configs'):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files from the config directory."""
        try:
            for config_file in self.config_dir.glob('*.yaml'):
                config_name = config_file.stem
                self.configs[config_name] = load_config(config_file)
                logger.info(f"Loaded configuration: {config_name}")
            
            for config_file in self.config_dir.glob('*.yml'):
                config_name = config_file.stem
                self.configs[config_name] = load_config(config_file)
                logger.info(f"Loaded configuration: {config_name}")
                
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def get_config(self, 
                   config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration not found: {config_name}")
        
        return self.configs[config_name].copy()
    
    def set_config(self, 
                   config_name: str,
                   config: Dict[str, Any]) -> None:
        """
        Set configuration by name.
        
        Args:
            config_name: Name of the configuration
            config: Configuration dictionary
        """
        self.configs[config_name] = config.copy()
        logger.info(f"Configuration set: {config_name}")
    
    def save_config(self, 
                   config_name: str,
                   config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_name: Name of the configuration
            config_path: Path to save configuration (optional)
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration not found: {config_name}")
        
        if config_path is None:
            config_path = self.config_dir / f"{config_name}.yaml"
        
        save_config(self.configs[config_name], config_path)
        logger.info(f"Configuration saved: {config_name}")
    
    def list_configs(self) -> List[str]:
        """
        List available configurations.
        
        Returns:
            List of configuration names
        """
        return list(self.configs.keys())
    
    def reload_config(self, 
                     config_name: str) -> None:
        """
        Reload configuration from file.
        
        Args:
            config_name: Name of the configuration
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            self.configs[config_name] = load_config(config_path)
            logger.info(f"Configuration reloaded: {config_name}")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def merge_configs(self, 
                     base_config: str,
                     override_config: str) -> Dict[str, Any]:
        """
        Merge two configurations.
        
        Args:
            base_config: Base configuration name
            override_config: Override configuration name
            
        Returns:
            Merged configuration
        """
        base = self.get_config(base_config)
        override = self.get_config(override_config)
        
        merged = self._deep_merge(base, override)
        logger.info(f"Configurations merged: {base_config} + {override_config}")
        
        return merged
    
    def _deep_merge(self, 
                   base: Dict[str, Any],
                   override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, 
                       config_name: str,
                       schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config_name: Name of the configuration
            schema: Validation schema (optional)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config = self.get_config(config_name)
            
            if schema is None:
                # Basic validation
                return self._basic_validation(config)
            else:
                # Schema-based validation
                return self._schema_validation(config, schema)
                
        except Exception as e:
            logger.error(f"Error validating configuration {config_name}: {e}")
            return False
    
    def _basic_validation(self, 
                         config: Dict[str, Any]) -> bool:
        """
        Basic configuration validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for required fields
        required_fields = ['model', 'training', 'inference']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Required field missing: {field}")
                return False
        
        return True
    
    def _schema_validation(self, 
                          config: Dict[str, Any],
                          schema: Dict[str, Any]) -> bool:
        """
        Schema-based configuration validation.
        
        Args:
            config: Configuration to validate
            schema: Validation schema
            
        Returns:
            True if valid, False otherwise
        """
        # This would normally implement schema validation
        # For now, we'll return True
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get configuration manager status.
        
        Returns:
            Status information
        """
        return {
            'config_dir': str(self.config_dir),
            'loaded_configs': list(self.configs.keys()),
            'config_count': len(self.configs)
        }




