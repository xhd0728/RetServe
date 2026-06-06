"""
Logging configuration and utilities.

This module provides centralized logging setup with support for
console and file output, log rotation, and structured formatting.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.settings import LoggingSettings


class LoggerManager:
    """
    Centralized logger manager for the application.

    This class handles the setup and configuration of the logging system,
    including console and file handlers with rotation support.

    Attributes:
        settings: Logging configuration settings.

    Example:
        manager = LoggerManager(LoggingSettings(level="DEBUG"))
        manager.setup()
        logger = manager.get_logger("my_module")
    """

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEBUG_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    def __init__(self, settings: LoggingSettings | None = None) -> None:
        """
        Initialize the logger manager.

        Args:
            settings: Logging configuration settings. If None, loads from config.
        """
        if settings is None:
            settings = self._load_default_settings()

        self._settings = settings
        self._is_configured = False

    @staticmethod
    def _load_default_settings() -> LoggingSettings:
        """
        Load default logging settings from configuration file.

        Returns:
            LoggingSettings instance.
        """
        try:
            from src.config_loader import config_loader

            return config_loader.load_logging_settings("log")
        except Exception:
            return LoggingSettings()

    @property
    def settings(self) -> LoggingSettings:
        """Get the logging settings."""
        return self._settings

    def _get_log_level(self) -> int:
        """
        Get the numeric logging level.

        Returns:
            Numeric log level.
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(self._settings.level.upper(), logging.INFO)

    def _create_formatter(self, detailed: bool = False) -> logging.Formatter:
        """
        Create a log formatter.

        Args:
            detailed: Whether to use detailed format with file info.

        Returns:
            Configured Formatter instance.
        """
        format_string = self.DEBUG_FORMAT if detailed else self.DEFAULT_FORMAT
        return logging.Formatter(format_string)

    def _create_console_handler(self) -> logging.StreamHandler:
        """
        Create a console log handler.

        Returns:
            Configured StreamHandler instance.
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._get_log_level())
        handler.setFormatter(self._create_formatter())
        return handler

    def _create_file_handler(self) -> RotatingFileHandler:
        """
        Create a rotating file log handler.

        Returns:
            Configured RotatingFileHandler instance.
        """
        log_path = self._settings.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=self._settings.max_bytes,
            backupCount=self._settings.backup_count,
            encoding="utf-8",
        )
        handler.setLevel(self._get_log_level())
        handler.setFormatter(self._create_formatter(detailed=True))
        return handler

    def setup(self) -> None:
        """
        Set up the logging configuration.

        This method configures the root logger with console and file handlers.
        It should be called once at application startup.
        """
        if self._is_configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(self._get_log_level())
        root_logger.handlers.clear()

        root_logger.addHandler(self._create_console_handler())
        try:
            root_logger.addHandler(self._create_file_handler())
        except (OSError, PermissionError) as exc:
            root_logger.warning(f"Failed to create file handler: {exc}")

        self._is_configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.

        Args:
            name: Logger name (typically __name__ of the module).

        Returns:
            Configured Logger instance.
        """
        if not self._is_configured:
            self.setup()

        return logging.getLogger(name)

    def reconfigure(self, settings: LoggingSettings) -> None:
        """
        Reconfigure logging with new settings.

        Args:
            settings: New logging settings.
        """
        self._settings = settings
        self._is_configured = False
        self.setup()


_logger_manager: LoggerManager | None = None


def get_logger_manager() -> LoggerManager:
    """
    Get the global logger manager instance.

    Returns:
        LoggerManager instance.
    """
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
        _logger_manager.setup()
    return _logger_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function that uses the global logger manager.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        Configured Logger instance.
    """
    return get_logger_manager().get_logger(name)


logger = get_logger(__name__)
