"""
Classe astratta `Extractor`. Ogni implementazione concreta deve definire:
  - SUPPORTED_EXTENSIONS: tuple di estensioni gestite (es. (".txt",))
  - extract(path) → ExtractedDocument
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from core.dataset.extracted import ExtractedDocument

logger = logging.getLogger(__name__)


class ExtractorError(Exception):
    """Errore generico durante l'estrazione."""


class Extractor(ABC):
    """Interfaccia comune per tutti gli estrattori."""

    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """
        Estrae il contenuto da un file. Ogni estrattore implementa la
        propria logica (PyMuPDF, pandas, ecc.).

        Raises:
            ExtractorError: se il file non è leggibile / corrotto.
        """
        ...

    def supports(self, path: Path) -> bool:
        """True se questo estrattore può gestire il file."""
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _validate_file(self, path: Path) -> None:
        """Helper: controlli base prima di iniziare a leggere."""
        if not path.exists():
            raise ExtractorError(f"File non trovato: {path}")
        if not path.is_file():
            raise ExtractorError(f"Non è un file: {path}")
        if path.stat().st_size == 0:
            raise ExtractorError(f"File vuoto: {path}")