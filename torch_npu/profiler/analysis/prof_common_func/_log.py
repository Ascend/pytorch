import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Optional

from torch_npu.utils._path_manager import PathManager


class ProfilerLogger:
    """
    Profiler Logger class for managing log operations.

    This class provides a centralized logging mechanism for the profiler,
    writing logs to file with rotation support.

    Attributes:
        LOG_FORMAT: The format string for log messages
        DATE_FORMAT: The format string for timestamps in log messages
        DEFAULT_LOGGER_NAME: Default name for the logger instance
        DEFAULT_LOG_DIR: Default directory name for log files
        MAX_BYTES: Maximum size of each log file
        BACKUP_COUNT: Number of backup files to keep
    """

    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d-%H:%M:%S"
    DEFAULT_LOGGER_NAME = "AscendProfiler"
    DEFAULT_LOG_LEVEL = logging.INFO
    DEFAULT_LOG_DIR = "logs"
    # 10MB per file
    MAX_BYTES = 10 * 1024 * 1024
    # Keep 3 backup files
    BACKUP_COUNT = 3
    # logger instance
    _instance = None

    @classmethod
    def get_instance(cls) -> logging.Logger:
        """Get the singleton logger instance."""
        if cls._instance is None:
            raise RuntimeError("Logger not initialized. Call init first.")
        return cls._instance

    @classmethod
    def init(cls, output_dir: str, custom_name: Optional[str] = None) -> None:
        """
        Initialize the logger with rotating file handler.

        Args:
            output_dir (str): Directory where log files will be stored

        Raises:
            RuntimeError: If logger initialization fails
        """
        if cls._instance is not None:
            return

        # Create logs directory
        log_dir = os.path.join(output_dir, cls.DEFAULT_LOG_DIR)
        PathManager.make_dir_safety(log_dir)

        # Create logger
        logger = logging.getLogger(cls.DEFAULT_LOGGER_NAME)
        logger.setLevel(cls.DEFAULT_LOG_LEVEL)
        logger.propagate = False

        # Create formatters
        formatter = logging.Formatter(fmt=cls.LOG_FORMAT, datefmt=cls.DATE_FORMAT)

        # Add rotating file handler
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        log_file = os.path.join(
            log_dir,
            (
                f"profiler_{timestamp}_{os.getpid()}_{custom_name}.log"
                if custom_name
                else f"profiler_{timestamp}_{os.getpid()}.log"
            ),
        )
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=cls.MAX_BYTES,
            backupCount=cls.BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(cls.DEFAULT_LOG_LEVEL)
        logger.addHandler(file_handler)

        cls._instance = logger
        logger.info("Profiler logger initialized at: %s", log_file)

    @classmethod
    def set_level(cls, level: int) -> None:
        """
        Set the logging level for both file and console handlers.

        Args:
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        logger = cls.get_instance()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    @classmethod
    def destroy(cls) -> None:
        """Close and cleanup the logger."""
        if cls._instance:
            for handler in cls._instance.handlers[:]:
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None
