import logging
from typing import Any, Callable, Optional


class FlightRecorderLogger:
    _instance: Optional[Any] = None
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger("Flight Recorder")

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger("Flight Recorder")
            cls._instance.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            cls._instance.logger.addHandler(ch)
        return cls._instance

    def set_log_level(self, level: int) -> None:
        self.logger.setLevel(level)

    @property
    def debug(self) -> Callable[..., None]:
        return self.logger.debug

    @property
    def info(self) -> Callable[..., None]:
        return self.logger.info

    @property
    def warning(self) -> Callable[..., None]:
        return self.logger.warning

    @property
    def error(self) -> Callable[..., None]:
        return self.logger.error

    @property
    def critical(self) -> Callable[..., None]:
        return self.logger.critical
