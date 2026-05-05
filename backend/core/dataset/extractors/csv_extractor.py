"""
Estrattore .csv (e .tsv) via pandas.

Sniffing automatico del delimitatore. Ogni riga del DataFrame diventa
una Section. Il testo della riga è "col1: val1 | col2: val2 | ...".

Limite righe: per CSV grossi (> MAX_ROWS) tagliamo per non saturare RAM.
"""

from __future__ import annotations

import csv as _csv  # rinominato per evitare shadowing
import logging
from pathlib import Path

import chardet
import pandas as pd

from core.dataset.extracted import ExtractedDocument, Section
from core.dataset.extractors.base import Extractor, ExtractorError

logger = logging.getLogger(__name__)

MAX_ROWS = 50_000


class CsvExtractor(Extractor):
    """Estrattore per CSV/TSV via pandas."""

    SUPPORTED_EXTENSIONS = (".csv", ".tsv")

    def extract(self, path: Path) -> ExtractedDocument:
        self._validate_file(path)
        warnings: list[str] = []

        # 1. Detect encoding
        sample = path.read_bytes()[:65536]
        encoding = chardet.detect(sample).get("encoding") or "utf-8"

        # 2. Detect delimitatore con csv.Sniffer (su sample testuale)
        try:
            text_sample = sample.decode(encoding, errors="replace")
            dialect = _csv.Sniffer().sniff(text_sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except Exception:  # noqa: BLE001
            # Fallback: se .tsv usa tab, altrimenti virgola
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            warnings.append(f"Sniff delimitatore fallito, uso {delimiter!r}.")

        # 3. Leggi con pandas
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                encoding_errors="replace",
                nrows=MAX_ROWS + 1,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ExtractorError(f"CSV non leggibile {path}: {exc}") from exc

        if len(df) > MAX_ROWS:
            df = df.iloc[:MAX_ROWS]
            warnings.append(
                f"CSV troncato a {MAX_ROWS} righe (limite per evitare OOM)."
            )

        # 4. Converti righe in Section
        columns = list(df.columns)
        sections: list[Section] = []
        text_parts: list[str] = []

        for idx, row in df.iterrows():
            row_text = " | ".join(f"{col}: {row[col]}" for col in columns)
            sections.append(
                Section(
                    title=f"Riga {idx + 1}",
                    text=row_text,
                    metadata={"row_index": int(idx), "columns": columns},
                )
            )
            text_parts.append(row_text)

        full_text = "\n".join(text_parts)

        return ExtractedDocument(
            text=full_text,
            source_format=path.suffix.lower().lstrip("."),
            sections=sections,
            metadata={
                "row_count": len(df),
                "column_count": len(columns),
                "columns": columns,
                "delimiter": delimiter,
                "encoding": encoding,
                "warnings": warnings,
                "file_size_bytes": path.stat().st_size,
            },
        )