import logging
import os
from logging.handlers import RotatingFileHandler
from src.config_loader import config_loader


class LoggerConfig:
    """Logger configuration manager"""
    
    def __init__(self):
        """Initialize logger with configuration"""
        self.config = config_loader.load_config("log")
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.config["file"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Set log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config["level"])
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config["level"])
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.config["file"],
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.config["level"])
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Initialize logging
logger_config = LoggerConfig()
logger = logging.getLogger(__name__)
