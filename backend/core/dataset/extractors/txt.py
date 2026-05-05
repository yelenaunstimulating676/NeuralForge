"""
Estrattore .txt (e formati testuali generici).

Auto-detect encoding via chardet. Una sola Section con tutto il contenuto.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chardet

from core.dataset.extracted import ExtractedDocument, Section
from core.dataset.extractors.base import Extractor, ExtractorError

logger = logging.getLogger(__name__)


class TxtExtractor(Extractor):
    """Estrattore per file di testo plain."""

    SUPPORTED_EXTENSIONS = (".txt", ".md", ".rst", ".log")

    def extract(self, path: Path) -> ExtractedDocument:
        self._validate_file(path)

        # 1. Detect encoding leggendo i primi 64 KB (sufficiente per detection)
        sample = path.read_bytes()[:65536]
        detected = chardet.detect(sample)
        encoding = detected.get("encoding") or "utf-8"
        confidence = detected.get("confidence") or 0.0

        warnings: list[str] = []
        if confidence < 0.7:
            warnings.append(
                f"Encoding detection a bassa confidenza ({confidence:.2f}), "
                f"uso {encoding!r} con fallback errors=replace."
            )

        # 2. Lettura
        try:
            text = path.read_text(encoding=encoding, errors="replace")
        except UnicodeDecodeError as exc:
            raise ExtractorError(f"Errore decoding di {path}: {exc}") from exc

        sections = [Section(title=None, text=text, metadata={})]

        return ExtractedDocument(
            text=text,
            source_format=path.suffix.lower().lstrip("."),
            sections=sections,
            metadata={
                "encoding": encoding,
                "encoding_confidence": confidence,
                "warnings": warnings,
                "file_size_bytes": path.stat().st_size,
            },
        )