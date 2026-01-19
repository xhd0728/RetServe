import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader for YAML files"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the config loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Dict containing configuration
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config


# Global config loader instance
config_loader = ConfigLoader()
