"""
Router degli estrattori: in base all'estensione, sceglie quello giusto.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.dataset.extracted import ExtractedDocument
from core.dataset.extractors.base import Extractor
from core.dataset.extractors.csv_extractor import CsvExtractor
from core.dataset.extractors.docx import DocxExtractor
from core.dataset.extractors.json_extractor import JsonExtractor
from core.dataset.extractors.pdf import PdfExtractor
from core.dataset.extractors.txt import TxtExtractor

logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """Estensione file non supportata da nessun estrattore."""


# Lista in ordine di registrazione. La selezione passa da `supports()`.
_EXTRACTORS: list[Extractor] = [
    TxtExtractor(),
    PdfExtractor(),
    CsvExtractor(),
    JsonExtractor(),
    DocxExtractor(),
]


def get_extractor_for_path(path: Path) -> Extractor:
    """
    Ritorna l'estrattore adatto al file. Solleva UnsupportedFormatError
    se nessuno gestisce quell'estensione.
    """
    for extractor in _EXTRACTORS:
        if extractor.supports(path):
            return extractor
    raise UnsupportedFormatError(
        f"Estensione non supportata: {path.suffix!r} (file: {path.name})."
    )


def extract_file(path: Path) -> ExtractedDocument:
    """
    API top-level: dato un path, estrai il documento.

    Raises:
        UnsupportedFormatError: estensione non gestita.
        ExtractorError: errori di lettura/parsing del file.
    """
    extractor = get_extractor_for_path(path)
    logger.info(
        "Estrazione %s con %s", path.name, type(extractor).__name__
    )
    return extractor.extract(path)


def supported_extensions() -> set[str]:
    """Set di tutte le estensioni gestite (con il punto: .pdf, .csv, ...)."""
    exts: set[str] = set()
    for e in _EXTRACTORS:
        exts.update(e.SUPPORTED_EXTENSIONS)
    return exts


def is_supported_extension(path: Path) -> bool:
    """True se l'estensione del path è gestita."""
    return path.suffix.lower() in supported_extensions()