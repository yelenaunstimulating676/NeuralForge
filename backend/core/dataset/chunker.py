"""
Smart Chunker — divide testi lunghi in blocchi gestibili rispettando
i confini semantici naturali (sezioni → paragrafi → frasi → parole).

Strategia:
  - Per content type a "unità discrete" (qa_pairs, tabular, dialogue),
    ogni Section diventa un Chunk.
  - Per content type narrativi (narrative, mixed, code), aggreghiamo
    sezioni piccole e splittiamo sezioni grandi rispettando paragrafi
    e frasi.

Unità di misura: caratteri (non token). Stima tokens = chars // 4.

Funzioni pure: input → output, niente side effect, niente network.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.dataset.detector import ContentType
from core.dataset.extracted import ExtractedDocument, Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Un blocco di testo prodotto dal chunker."""

    text: str
    index: int                              # 0-based, ordine globale
    char_count: int
    estimated_tokens: int                   # chars // 4
    source_section: str | None              # titolo Section origine, se nota
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "source_section": self.source_section,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkerConfig:
    """
    Parametri del chunker. Tutti in CARATTERI (non token).

    Defaults pensati per:
      - target_chars=2048 → ~512 token
      - overlap_chars=200 → ~50 token (10%)
      - min_chunk_chars=200 → scarta chunk troppo piccoli
    """

    target_chars: int = 2048
    overlap_chars: int = 200
    min_chunk_chars: int = 200
    max_chunk_chars: int = 4096            # hard cap, evita chunk patologici

    def __post_init__(self) -> None:
        if self.target_chars < 100:
            raise ValueError("target_chars deve essere ≥ 100")
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars deve essere < target_chars")
        if self.min_chunk_chars > self.target_chars:
            raise ValueError("min_chunk_chars deve essere ≤ target_chars")


# ---------------------------------------------------------------------------
# Helpers di splitting
# ---------------------------------------------------------------------------


# Frase terminata: ., !, ? seguiti da spazio/newline, oppure fine testo
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_CODE_BLOCK_BOUNDARY_RE = re.compile(
    r"(?m)^(?:def |class |function |fn |pub fn |async def |async function )"
)


def _split_paragraphs(text: str) -> list[str]:
    """Spezza il testo su righe vuote (\\n\\n+)."""
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Spezza il testo su confini di frase (.!?)."""
    if not text.strip():
        return []
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str) -> list[str]:
    """Ultimo fallback: split su whitespace mantenendo le parole."""
    return text.split()


def _estimate_tokens(text: str) -> int:
    """Stima molto approssimativa: 4 caratteri per token (testo latino)."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Aggregator: prende pezzi e li unisce fino a target_chars
# ---------------------------------------------------------------------------


def _aggregate_pieces(
    pieces: list[str],
    config: ChunkerConfig,
    section_title: str | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Prende pezzi di testo (paragrafi o frasi) e li unisce in chunk
    di ~target_chars caratteri. Aggiunge overlap tra chunk consecutivi.

    Args:
        pieces: lista di stringhe già "atomiche" (paragrafi o frasi).
        config: parametri chunker.
        section_title: titolo Section di origine, da copiare nei chunk.
        start_index: offset per il campo `index` dei chunk creati.

    Returns:
        Lista di Chunk con `index` da `start_index` in su.
    """
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_chars = 0
    sep = "\n\n"  # separatore tra paragrafi nel buffer

    def flush() -> None:
        nonlocal buffer, buffer_chars
        if buffer_chars < config.min_chunk_chars and not chunks:
            # Primo chunk troppo piccolo ma è anche l'unico finora: tienilo
            pass
        if buffer_chars == 0:
            return
        text = sep.join(buffer).strip()
        if not text:
            buffer.clear()
            buffer_chars = 0
            return

        idx = start_index + len(chunks)
        chunks.append(
            Chunk(
                text=text,
                index=idx,
                char_count=len(text),
                estimated_tokens=_estimate_tokens(text),
                source_section=section_title,
                metadata={},
            )
        )
        buffer.clear()
        buffer_chars = 0

    for piece in pieces:
        piece_len = len(piece)

        # Se il singolo pezzo è già troppo grande, splittalo a frasi
        if piece_len > config.max_chunk_chars:
            # Flush quello che abbiamo
            flush()
            # Spezziamo il pezzone a frasi
            sentences = _split_sentences(piece)
            if not sentences or all(len(s) > config.max_chunk_chars for s in sentences):
                # Fallback estremo: split a parole
                words = _split_words(piece)
                sub_pieces = _pack_words(words, config.target_chars)
            else:
                sub_pieces = sentences
            sub_chunks = _aggregate_pieces(
                sub_pieces, config, section_title, start_index + len(chunks)
            )
            chunks.extend(sub_chunks)
            continue

        # Se aggiungerlo supera target_chars, flush prima
        prospective_chars = buffer_chars + (len(sep) if buffer else 0) + piece_len
        if prospective_chars > config.target_chars and buffer:
            flush()

            # Aggiungi overlap dall'ultimo chunk se config lo prevede
            if config.overlap_chars > 0 and chunks:
                tail = chunks[-1].text[-config.overlap_chars :]
                buffer.append(tail)
                buffer_chars += len(tail)

        buffer.append(piece)
        buffer_chars += piece_len + (len(sep) if len(buffer) > 1 else 0)

    flush()
    return chunks


def _pack_words(words: list[str], target_chars: int) -> list[str]:
    """
    Raggruppa parole in stringhe lunghe ≤ target_chars. Usato come ultimo
    fallback per pezzi mostruosamente lunghi senza punteggiatura.
    """
    out: list[str] = []
    cur: list[str] = []
    cur_chars = 0
    for w in words:
        if cur_chars + len(w) + 1 > target_chars and cur:
            out.append(" ".join(cur))
            cur = [w]
            cur_chars = len(w)
        else:
            cur.append(w)
            cur_chars += len(w) + 1
    if cur:
        out.append(" ".join(cur))
    return out


# ---------------------------------------------------------------------------
# Strategie per ContentType
# ---------------------------------------------------------------------------


def _chunk_unit_per_section(
    doc: ExtractedDocument, config: ChunkerConfig
) -> list[Chunk]:
    """
    Strategia "una sezione = un chunk".
    Usata per qa_pairs, tabular, dialogue dove ogni Section è una unità
    semantica completa che non vogliamo splittare.

    Sezioni più grandi di max_chunk_chars vengono comunque tagliate.
    """
    chunks: list[Chunk] = []
    for section in doc.sections:
        text = section.text.strip()
        if len(text) < config.min_chunk_chars:
            continue
        if len(text) <= config.max_chunk_chars:
            chunks.append(
                Chunk(
                    text=text,
                    index=len(chunks),
                    char_count=len(text),
                    estimated_tokens=_estimate_tokens(text),
                    source_section=section.title,
                    metadata=dict(section.metadata),
                )
            )
        else:
            # Sezione troppo grande: aggregator interno
            paragraphs = _split_paragraphs(text)
            sub_chunks = _aggregate_pieces(
                paragraphs, config, section.title, len(chunks)
            )
            chunks.extend(sub_chunks)
    return chunks


def _chunk_narrative(
    doc: ExtractedDocument, config: ChunkerConfig
) -> list[Chunk]:
    """
    Strategia narrative/mixed: rispetta sezioni come boundary "soft", poi
    aggrega paragrafi fino a target_chars con overlap.
    """
    chunks: list[Chunk] = []

    # Se non ci sono sezioni, tratta tutto il testo come una sezione unica
    sections = doc.sections if doc.sections else [
        Section(title=None, text=doc.text)
    ]

    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text]
        sub_chunks = _aggregate_pieces(
            paragraphs, config, section.title, len(chunks)
        )
        chunks.extend(sub_chunks)

    return chunks


def _chunk_code(
    doc: ExtractedDocument, config: ChunkerConfig
) -> list[Chunk]:
    """
    Strategia code: split su confini di funzione/classe.
    Se il pattern non viene trovato, fallback a narrative.
    """
    text = doc.text
    boundaries = [m.start() for m in _CODE_BLOCK_BOUNDARY_RE.finditer(text)]

    if not boundaries:
        return _chunk_narrative(doc, config)

    # Split su boundaries
    pieces: list[str] = []
    prev = 0
    for b in boundaries:
        if b - prev > 0:
            piece = text[prev:b].strip()
            if piece:
                pieces.append(piece)
        prev = b
    # Ultimo pezzo
    if prev < len(text):
        last = text[prev:].strip()
        if last:
            pieces.append(last)

    return _aggregate_pieces(pieces, config, None, 0)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def chunk_document(
    doc: ExtractedDocument,
    content_type: ContentType,
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    """
    Divide il documento in chunk in base al ContentType.

    Args:
        doc: documento estratto (da Extractor).
        content_type: tipo determinato dal Detector.
        config: parametri chunker. Default = ChunkerConfig().

    Returns:
        Lista di Chunk pronti per il Converter (M3.4).
    """
    config = config or ChunkerConfig()

    if content_type in {
        ContentType.QA_PAIRS,
        ContentType.TABULAR,
        ContentType.DIALOGUE,
    }:
        chunks = _chunk_unit_per_section(doc, config)
    elif content_type == ContentType.CODE:
        chunks = _chunk_code(doc, config)
    else:
        # NARRATIVE, MIXED, fallback
        chunks = _chunk_narrative(doc, config)

    logger.info(
        "Chunking: %s → %d chunks (target=%d chars, overlap=%d)",
        content_type.value, len(chunks), config.target_chars, config.overlap_chars,
    )
    return chunks