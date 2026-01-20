"""
Configuration loader for YAML-based settings.

This module provides a centralized configuration loader that reads
YAML files and converts them to type-safe settings objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar, overload

import yaml

from src.settings import (
    EmbedSettings,
    IndexBuildSettings,
    LoggingSettings,
    ServiceSettings,
)

# Type variable for generic settings loading
T = TypeVar("T")


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    
    pass


class ConfigLoader:
    """
    Configuration loader for YAML files.
    
    This class handles loading and parsing of YAML configuration files,
    converting them to type-safe settings objects.
    
    Attributes:
        config_directory: Path to the directory containing config files.
    
    Example:
        loader = ConfigLoader("config")
        settings = loader.load_service_settings("serve")
    """
    
    def __init__(self, config_directory: str | Path = "config") -> None:
        """
        Initialize the configuration loader.
        
        Args:
            config_directory: Path to the directory containing config files.
        """
        self._config_directory = Path(config_directory)
    
    @property
    def config_directory(self) -> Path:
        """Get the configuration directory path."""
        return self._config_directory
    
    def _get_config_path(self, config_name: str) -> Path:
        """
        Get the full path to a configuration file.
        
        Args:
            config_name: Name of the configuration (without .yaml extension).
            
        Returns:
            Full path to the configuration file.
        """
        return self._config_directory / f"{config_name}.yaml"
    
    def load_raw(self, config_name: str) -> dict[str, Any]:
        """
        Load raw configuration dictionary from YAML file.
        
        Args:
            config_name: Name of the configuration file (without extension).
            
        Returns:
            Configuration dictionary.
            
        Raises:
            ConfigurationError: If the configuration file is not found or invalid.
        """
        config_path = self._get_config_path(config_name)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )
        
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                
            if config is None:
                return {}
            
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Invalid configuration format in {config_path}: "
                    f"expected dictionary, got {type(config).__name__}"
                )
            
            return config
            
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML configuration {config_path}: {exc}"
            ) from exc
    
    def load_service_settings(self, config_name: str = "serve") -> ServiceSettings:
        """
        Load service settings for the retrieval service.
        
        Args:
            config_name: Name of the configuration file.
            
        Returns:
            ServiceSettings instance.
        """
        raw_config = self.load_raw(config_name)
        return ServiceSettings.from_dict(raw_config)
    
    def load_embed_settings(self, config_name: str = "embed") -> EmbedSettings:
        """
        Load settings for the embedding processor.
        
        Args:
            config_name: Name of the configuration file.
            
        Returns:
            EmbedSettings instance.
        """
        raw_config = self.load_raw(config_name)
        return EmbedSettings.from_dict(raw_config)
    
    def load_index_settings(self, config_name: str = "index") -> IndexBuildSettings:
        """
        Load settings for the index builder.
        
        Args:
            config_name: Name of the configuration file.
            
        Returns:
            IndexBuildSettings instance.
        """
        raw_config = self.load_raw(config_name)
        return IndexBuildSettings.from_dict(raw_config)
    
    def load_logging_settings(self, config_name: str = "log") -> LoggingSettings:
        """
        Load logging settings.
        
        Args:
            config_name: Name of the configuration file.
            
        Returns:
            LoggingSettings instance.
        """
        raw_config = self.load_raw(config_name)
        return LoggingSettings(**raw_config)
    
    # Legacy method for backward compatibility
    def load_config(self, config_name: str) -> dict[str, Any]:
        """
        Load configuration as raw dictionary (legacy method).
        
        This method is provided for backward compatibility.
        Consider using the typed load methods instead.
        
        Args:
            config_name: Name of the configuration file.
            
        Returns:
            Configuration dictionary.
        """
        return self.load_raw(config_name)


# =============================================================================
# Global Configuration Loader Instance
# =============================================================================

# Default configuration loader instance
config_loader = ConfigLoader()


def get_config_loader(config_directory: str | Path = "config") -> ConfigLoader:
    """
    Get or create a configuration loader for the specified directory.
    
    Args:
        config_directory: Path to the configuration directory.
        
    Returns:
        ConfigLoader instance.
    """
    return ConfigLoader(config_directory)
