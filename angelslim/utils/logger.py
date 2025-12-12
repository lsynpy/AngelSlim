import functools
import logging
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

_default_log_level = logging.INFO


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # Angelslim logger has already been configured.
            return

        _default_handler = logging.StreamHandler(sys.stderr)

        # Apply our default configuration to the angelslim logger.
        angelslim_logger = logging.getLogger("angelslim")
        angelslim_logger.addHandler(_default_handler)
        angelslim_logger.setLevel(_default_log_level)
        # Prevent propagation to avoid duplicate logs from other handlers
        angelslim_logger.propagate = False

        formatter = ColoredCombinedFormatter(
            fmt="%(asctime)s - %(combined_info)-21s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        _default_handler.setFormatter(formatter)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = "angelslim"

    _configure_library_root_logger()
    return logging.getLogger(name)


def reset_format() -> None:
    angelslim_logger = logging.getLogger("angelslim")
    for handler in angelslim_logger.handlers:
        handler.setFormatter(None)


@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    self.info(*args, **kwargs)


logging.Logger.info_once = info_once


class CombinedFormatter(logging.Formatter):
    def format(self, record):
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"
        return super().format(record)


class ColoredCombinedFormatter(CombinedFormatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"

        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]
        colored_levelname = f"{level_color}{record.levelname:<5}{reset_color}"

        original_levelname = record.levelname
        record.levelname = colored_levelname
        formatted_message = super().format(record)
        record.levelname = original_levelname

        return formatted_message
