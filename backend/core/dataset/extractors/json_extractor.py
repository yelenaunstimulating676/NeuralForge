"""
Estrattore .json e .jsonl.

Strategia:
  - Se array di oggetti → ogni oggetto è una Section
  - Se singolo oggetto → una Section
  - Se .jsonl (un oggetto per riga) → ogni riga è una Section
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any

import chardet

from core.dataset.extracted import ExtractedDocument, Section
from core.dataset.extractors.base import Extractor, ExtractorError

logger = logging.getLogger(__name__)

MAX_ITEMS = 50_000


class JsonExtractor(Extractor):
    """Estrattore per JSON e JSONL."""

    SUPPORTED_EXTENSIONS = (".json", ".jsonl")

    def extract(self, path: Path) -> ExtractedDocument:
        self._validate_file(path)
        warnings: list[str] = []

        # Detect encoding
        sample = path.read_bytes()[:65536]
        encoding = chardet.detect(sample).get("encoding") or "utf-8"

        try:
            raw = path.read_text(encoding=encoding, errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise ExtractorError(f"JSON non leggibile {path}: {exc}") from exc

        is_jsonl = path.suffix.lower() == ".jsonl"

        items: list[Any] = []
        if is_jsonl:
            for line_no, line in enumerate(raw.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(_json.loads(line))
                except _json.JSONDecodeError as exc:
                    warnings.append(f"Riga {line_no} non valida: {exc}")
        else:
            try:
                parsed = _json.loads(raw)
            except _json.JSONDecodeError as exc:
                raise ExtractorError(
                    f"JSON malformato in {path}: {exc}"
                ) from exc
            if isinstance(parsed, list):
                items = parsed
            else:
                items = [parsed]

        if len(items) > MAX_ITEMS:
            warnings.append(f"JSON troncato a {MAX_ITEMS} elementi.")
            items = items[:MAX_ITEMS]

        sections: list[Section] = []
        text_parts: list[str] = []

        for idx, item in enumerate(items):
            section_text = _serialize_item(item)
            sections.append(
                Section(
                    title=f"Item {idx + 1}",
                    text=section_text,
                    metadata={"item_index": idx, "raw_keys": _list_keys(item)},
                )
            )
            text_parts.append(section_text)

        full_text = "\n\n".join(text_parts)

        return ExtractedDocument(
            text=full_text,
            source_format="jsonl" if is_jsonl else "json",
            sections=sections,
            metadata={
                "item_count": len(items),
                "is_jsonl": is_jsonl,
                "encoding": encoding,
                "warnings": warnings,
                "file_size_bytes": path.stat().st_size,
            },
        )


def _serialize_item(item: Any) -> str:
    """
    Converte un valore JSON in testo leggibile.
    - Dict → "key1: value1\nkey2: value2..."
    - List → un elemento per riga
    - Altro → str()
    """
    if isinstance(item, dict):
        parts = []
        for k, v in item.items():
            v_str = v if isinstance(v, str) else _json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}: {v_str}")
        return "\n".join(parts)
    if isinstance(item, list):
        return "\n".join(
            x if isinstance(x, str) else _json.dumps(x, ensure_ascii=False)
            for x in item
        )
    return str(item)


def _list_keys(item: Any) -> list[str]:
    """Per dict, ritorna la lista delle chiavi top-level. Per altro, []."""
    if isinstance(item, dict):
        return list(item.keys())
    return []