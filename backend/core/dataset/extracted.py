"""
Dataclass condivise dal Dataset Engine.

`ExtractedDocument` è l'output comune di tutti gli estrattori. Da qui in
poi (Detector, Chunker, Converter) la pipeline lavora SOLO su questa
struttura, senza sapere se il file originale era PDF, CSV o DOCX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    """
    Una "sezione" di un documento estratto.

    Esempi di cosa rappresenta:
      - una pagina di PDF
      - un capitolo / heading di DOCX
      - una riga di CSV (per CSV abbiamo molte sezioni piccole)
      - un singolo oggetto di un file JSON

    Avere sezioni esplicite aiuta il Chunker a rispettare confini
    semantici naturali invece di tagliare a metà.
    """

    title: str | None             # "Capitolo 3", "Pagina 12", o None
    text: str                     # contenuto testuale grezzo
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """
    Output unificato di un Extractor.

    Attrs:
        text: tutto il contenuto testuale concatenato (con \n\n tra sezioni).
        sections: lista di sezioni (può essere vuota se l'estrattore
            non ha info strutturali, es. un .txt).
        metadata: info sull'estrazione (es. {"source_format": "pdf",
            "pages": 12, "warnings": [...]}).
        source_format: estensione/formato originale ("pdf", "txt", "csv", ...).
    """

    text: str
    source_format: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def to_dict(self) -> dict[str, Any]:
        """Serializzazione JSON-safe (per response API o debug)."""
        return {
            "source_format": self.source_format,
            "char_count": self.char_count,
            "section_count": self.section_count,
            "metadata": self.metadata,
            "sections": [
                {"title": s.title, "text_preview": s.text[:200], "metadata": s.metadata}
                for s in self.sections[:5]  # solo prime 5 in preview
            ],
        }