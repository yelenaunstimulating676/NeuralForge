"""
Configurazione logging per NeuralForge.

Output: console (formato compatto) + file rotante (10 MB x 5 backup).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import settings

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | "
    "%(funcName)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configura il root logger. Idempotente."""
    settings.ensure_directories()

    root = logging.getLogger()
    root.setLevel(settings.log_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.log_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        settings.log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.log_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    root.info(
        "Logging inizializzato: level=%s file=%s",
        settings.log_level,
        settings.log_file_path,
    )