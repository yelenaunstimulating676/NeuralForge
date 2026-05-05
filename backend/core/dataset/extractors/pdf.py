"""
Estrattore .pdf via PyMuPDF (fitz).

Ogni pagina diventa una Section. Se l'intero PDF estrae < SCANNED_THRESHOLD
caratteri, segnaliamo possibile PDF scansionato.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from core.dataset.extracted import ExtractedDocument, Section
from core.dataset.extractors.base import Extractor, ExtractorError

logger = logging.getLogger(__name__)

# Sotto questa soglia totale → probabile PDF scansionato (no layer testo)
SCANNED_PDF_THRESHOLD = 100


class PdfExtractor(Extractor):
    """Estrattore per file PDF basato su PyMuPDF."""

    SUPPORTED_EXTENSIONS = (".pdf",)

    def extract(self, path: Path) -> ExtractedDocument:
        self._validate_file(path)

        warnings: list[str] = []
        sections: list[Section] = []
        full_text_parts: list[str] = []

        try:
            doc = fitz.open(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ExtractorError(f"PDF non leggibile {path}: {exc}") from exc

        try:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                page_text = page_text.strip()

                if page_text:
                    sections.append(
                        Section(
                            title=f"Pagina {page_num}",
                            text=page_text,
                            metadata={"page_number": page_num},
                        )
                    )
                    full_text_parts.append(page_text)

            page_count = doc.page_count
        finally:
            doc.close()

        full_text = "\n\n".join(full_text_parts)

        # Detection PDF scansionato
        if len(full_text) < SCANNED_PDF_THRESHOLD and page_count > 0:
            warnings.append(
                f"Estratti solo {len(full_text)} caratteri su {page_count} pagine. "
                "Il PDF potrebbe essere scansionato (immagine senza layer testo). "
                "OCR non è supportato in questa versione."
            )
            logger.warning("Possibile PDF scansionato: %s", path)

        return ExtractedDocument(
            text=full_text,
            source_format="pdf",
            sections=sections,
            metadata={
                "page_count": page_count,
                "extracted_pages": len(sections),
                "warnings": warnings,
                "file_size_bytes": path.stat().st_size,
            },
        )