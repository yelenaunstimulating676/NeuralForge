"""Estrattori per ogni formato file supportato."""

from core.dataset.extractors.router import (
    UnsupportedFormatError,
    extract_file,
    get_extractor_for_path,
    is_supported_extension,
    supported_extensions,
)

__all__ = [
    "UnsupportedFormatError",
    "extract_file",
    "get_extractor_for_path",
    "is_supported_extension",
    "supported_extensions",
]