"""
Converter — trasforma Chunk in InstructionExample (formato Alpaca).

Ogni ContentType ha la sua strategia di conversione. L'obiettivo è
generare esempi {instruction, input, output} usabili per fine-tuning.

Approccio onesto:
  - Per casi "estraibili" (qa_pairs, dialogue, code-with-docstring),
    estraiamo dati reali dal testo
  - Per casi "generati" (narrative, tabular), produciamo esempi-scheletro
    con template fissi. La qualità non è massima ma è prevedibile e
    veloce. In M8/v2 valutiamo "smart mode" con LLM locale.

Funzioni pure: input → output deterministico.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.dataset.chunker import Chunk
from core.dataset.detector import ContentType
from core.dataset.extracted import ExtractedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstructionExample:
    """Esempio di instruction tuning in formato Alpaca."""

    instruction: str
    input: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConverterConfig:
    """Parametri del converter."""

    # Per narrative: quante "varianti" di esempi generare per ogni chunk
    examples_per_narrative_chunk: int = 1
    # Lingua dei template (per narrative). Per ora solo "it" o "en".
    template_language: str = "it"
    # Lunghezza minima del chunk in chars per generare esempi
    min_chars: int = 100
    # Skip esempi con output troppo corto (probabile estrazione fallita)
    min_output_chars: int = 20

    def __post_init__(self) -> None:
        if self.examples_per_narrative_chunk < 1:
            raise ValueError("examples_per_narrative_chunk deve essere ≥ 1")
        if self.template_language not in {"it", "en"}:
            raise ValueError("template_language deve essere 'it' o 'en'")


# ---------------------------------------------------------------------------
# Pattern utili
# ---------------------------------------------------------------------------

# Prefissi Q/A (case-insensitive, all'inizio riga)
_QA_PREFIX_RE = re.compile(
    r"(?im)^\s*(q|a|r|d|domanda|risposta|question|answer)\s*[:.\-)]\s*",
    re.MULTILINE,
)

# Speaker dialogue: "Nome: testo"
_DIALOGUE_TURN_RE = re.compile(
    r"(?m)^\s*([A-Za-z][A-Za-z0-9_ ]{0,30})\s*:\s+(.+)$"
)

# Docstring Python
_PY_DOCSTRING_RE = re.compile(
    r'(?:def|class)\s+\w[\w_]*\s*(?:\([^)]*\))?\s*(?:->.*?)?:\s*\n\s*"""([\s\S]+?)"""',
)


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

_TEMPLATES_IT = {
    "summarize": "Riassumi il seguente testo in poche frasi.",
    "key_points": "Quali sono i punti chiave del seguente testo?",
    "continue": "Continua il seguente testo in modo coerente.",
    "explain_code": "Spiega cosa fa il seguente codice.",
    "describe_record": "Descrivi in linguaggio naturale i seguenti dati.",
}

_TEMPLATES_EN = {
    "summarize": "Summarize the following text in a few sentences.",
    "key_points": "What are the key points of the following text?",
    "continue": "Continue the following text coherently.",
    "explain_code": "Explain what the following code does.",
    "describe_record": "Describe the following data in natural language.",
}


def _templates(config: ConverterConfig) -> dict[str, str]:
    return _TEMPLATES_IT if config.template_language == "it" else _TEMPLATES_EN


# ---------------------------------------------------------------------------
# QA Pairs strategy
# ---------------------------------------------------------------------------


def _convert_qa_pairs(
    chunk: Chunk,
    doc: ExtractedDocument,
    config: ConverterConfig,
) -> list[InstructionExample]:
    """
    Estrae coppie Q/A. Tre fonti possibili (in ordine di priorità):
      1. Sezione JSON con campi instruction-style → usa direttamente
      2. Sezione CSV con colonne Q/A → costruisci da colonne
      3. Testo con prefissi Q:/A: → regex extract

    Ritorna 0 o più esempi (un chunk può contenere più Q/A).
    """
    examples: list[InstructionExample] = []
    text = chunk.text

    # Caso 1: chunk da JSON con campi instruction-style
    if doc.source_format in {"json", "jsonl"}:
        # Cerchiamo di parsare il chunk come "key: value\nkey: value"
        parsed = _parse_kv_block(text)
        if parsed:
            ex = _build_from_kv(parsed, source_strategy="json_instruction_keys")
            if ex:
                examples.append(ex)
                return examples

    # Caso 2: chunk da CSV con colonne Q/A
    if doc.source_format in {"csv", "tsv"}:
        cols = {c.lower() for c in (doc.metadata.get("columns") or [])}
        qa_cols = {"question", "answer", "domanda", "risposta", "q", "a", "prompt", "completion"}
        if cols & qa_cols:
            parsed = _parse_kv_block(text, separator=" | ")
            if parsed:
                ex = _build_from_kv(parsed, source_strategy="csv_qa_columns")
                if ex:
                    examples.append(ex)
                    return examples

    # Caso 3: pattern Q:/A: nel testo
    pairs = _extract_qa_pattern(text)
    for q, a in pairs:
        if len(a) >= config.min_output_chars:
            examples.append(
                InstructionExample(
                    instruction=q.strip(),
                    input="",
                    output=a.strip(),
                    metadata={"strategy": "qa_pattern_extract", "chunk_index": chunk.index},
                )
            )
    return examples


def _parse_kv_block(text: str, separator: str = "\n") -> dict[str, str]:
    """
    Parse di un blocco "key: value" → dict.
    Separator può essere "\\n" (per JSON serialization) o " | " (CSV).
    """
    parts = text.split(separator)
    out: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        out[k.strip().lower()] = v.strip()
    return out


def _build_from_kv(
    kv: dict[str, str], source_strategy: str
) -> InstructionExample | None:
    """
    Da un dict di chiavi-valori, costruisci un InstructionExample
    se trova chiavi instruction-style note.
    """
    instruction = (
        kv.get("instruction")
        or kv.get("question")
        or kv.get("prompt")
        or kv.get("domanda")
        or kv.get("q")
    )
    output = (
        kv.get("output")
        or kv.get("answer")
        or kv.get("completion")
        or kv.get("response")
        or kv.get("risposta")
        or kv.get("a")
    )
    input_ = kv.get("input") or kv.get("context") or ""

    if not instruction or not output:
        return None

    # Spesso JSON ha campi serializzati come JSON dentro le stringhe (liste, dict)
    # Lasciamo grezzi: durante il training verranno tokenizzati comunque.
    return InstructionExample(
        instruction=instruction,
        input=input_,
        output=output,
        metadata={"strategy": source_strategy},
    )


def _extract_qa_pattern(text: str) -> list[tuple[str, str]]:
    """
    Estrae coppie (Q, A) da testo con prefissi tipo:
        Q: ...
        A: ...
        Domanda: ...
        Risposta: ...

    Strategia: split sui prefissi, raccoglie segmenti, accoppia
    consecutivi marcati come Q→A o D→R.
    """
    if not _QA_PREFIX_RE.search(text):
        return []

    # Tokenizza: lista di (prefix_lower, content)
    matches = list(_QA_PREFIX_RE.finditer(text))
    if len(matches) < 2:
        return []

    segments: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        prefix = m.group(1).lower()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        segments.append((prefix, content))

    # Accoppia consecutivi: prima il "domanda-like", poi il "risposta-like"
    question_prefixes = {"q", "domanda", "question", "d"}
    answer_prefixes = {"a", "r", "risposta", "answer"}

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(segments) - 1:
        p_cur, c_cur = segments[i]
        p_next, c_next = segments[i + 1]
        if p_cur in question_prefixes and p_next in answer_prefixes:
            pairs.append((c_cur, c_next))
            i += 2
        else:
            i += 1

    return pairs


# ---------------------------------------------------------------------------
# Tabular strategy
# ---------------------------------------------------------------------------


def _convert_tabular(
    chunk: Chunk,
    doc: ExtractedDocument,
    config: ConverterConfig,
) -> list[InstructionExample]:
    """..."""
    text = chunk.text.strip()
    # Per tabular il threshold è più basso: un record CSV tipico
    # ha pochi campi e può essere corto di natura.
    tabular_min = max(30, config.min_chars // 3)
    if len(text) < tabular_min:
        return []

    templates = _templates(config)
    kv = _parse_kv_block(text, separator=" | ")

    if not kv:
        return []

    # Output: concatenazione semplice di valori non vuoti, separati da virgola
    values = [v for v in kv.values() if v]
    if not values:
        return []
    output = ", ".join(values)

    if len(output) < config.min_output_chars:
        return []

    return [
        InstructionExample(
            instruction=templates["describe_record"],
            input=text,
            output=output,
            metadata={"strategy": "tabular_concat", "chunk_index": chunk.index},
        )
    ]


# ---------------------------------------------------------------------------
# Code strategy
# ---------------------------------------------------------------------------


def _convert_code(
    chunk: Chunk,
    doc: ExtractedDocument,
    config: ConverterConfig,
) -> list[InstructionExample]:
    """
    Code: genera "Spiega questo codice" SOLO se trova una docstring
    nel chunk (così l'output è reale, non inventato).

    Strategia conservativa per evitare di produrre rumore.
    """
    text = chunk.text
    templates = _templates(config)
    examples: list[InstructionExample] = []

    for match in _PY_DOCSTRING_RE.finditer(text):
        docstring = match.group(1).strip()
        if len(docstring) < config.min_output_chars:
            continue
        # Estrai la "definizione completa" che precede la docstring
        # (la funzione/classe stessa)
        func_start = text.rfind("\ndef ", 0, match.start())
        if func_start == -1:
            func_start = text.rfind("\nclass ", 0, match.start())
        if func_start == -1:
            func_start = max(0, match.start() - 200)

        func_text = text[func_start:match.end()].strip()

        examples.append(
            InstructionExample(
                instruction=templates["explain_code"],
                input=func_text,
                output=docstring,
                metadata={"strategy": "code_docstring", "chunk_index": chunk.index},
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Dialogue strategy
# ---------------------------------------------------------------------------


def _convert_dialogue(
    chunk: Chunk,
    doc: ExtractedDocument,
    config: ConverterConfig,
) -> list[InstructionExample]:
    """
    Dialogue: estrae turn dal chunk e accoppia consecutivi.
    Da [A, B, A, B] genera 3 esempi: A→B, B→A, A→B.
    """
    turns: list[tuple[str, str]] = []
    for match in _DIALOGUE_TURN_RE.finditer(chunk.text):
        speaker = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            turns.append((speaker, content))

    if len(turns) < 2:
        return []

    examples: list[InstructionExample] = []
    for i in range(len(turns) - 1):
        prev_speaker, prev_content = turns[i]
        next_speaker, next_content = turns[i + 1]
        # Skip se i due turn sono dello stesso speaker
        if prev_speaker.lower() == next_speaker.lower():
            continue
        if len(next_content) < config.min_output_chars:
            continue
        examples.append(
            InstructionExample(
                instruction=prev_content,
                input="",
                output=next_content,
                metadata={
                    "strategy": "dialogue_turn",
                    "chunk_index": chunk.index,
                    "speakers": f"{prev_speaker} → {next_speaker}",
                },
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Narrative strategy
# ---------------------------------------------------------------------------


def _convert_narrative(
    chunk: Chunk,
    doc: ExtractedDocument,
    config: ConverterConfig,
) -> list[InstructionExample]:
    """
    Narrative: genera esempi sintetici dal chunk usando template fissi.

    Strategie disponibili (cycliamo tra le prime N):
      - summarize: input=tutto il chunk, output=prima+ultima frase
      - continue: input=prime N parole, output=resto del chunk
      - key_points: input=tutto, output=prime frasi di ogni paragrafo
    """
    text = chunk.text.strip()
    if len(text) < config.min_chars:
        return []

    templates = _templates(config)
    examples: list[InstructionExample] = []

    strategies = [
        _strategy_summarize,
        _strategy_continue,
        _strategy_key_points,
    ]
    n = min(config.examples_per_narrative_chunk, len(strategies))

    for strategy in strategies[:n]:
        ex = strategy(chunk, text, templates, config)
        if ex:
            examples.append(ex)

    return examples


def _strategy_summarize(
    chunk: Chunk, text: str, templates: dict[str, str], config: ConverterConfig
) -> InstructionExample | None:
    """Pseudo-summary: prima frase + ultima frase. Veloce e prevedibile."""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return None
    summary = sentences[0]
    if len(sentences) > 4:
        summary += " " + sentences[-1]
    if len(summary) < config.min_output_chars:
        return None
    return InstructionExample(
        instruction=templates["summarize"],
        input=text,
        output=summary,
        metadata={"strategy": "narrative_summary", "chunk_index": chunk.index},
    )


def _strategy_continue(
    chunk: Chunk, text: str, templates: dict[str, str], config: ConverterConfig
) -> InstructionExample | None:
    """Continuation: prime ~30% del testo come input, resto come output."""
    if len(text) < 200:
        return None
    split_point = max(100, len(text) // 3)
    # Cerca la fine di frase più vicina dopo split_point
    look_ahead = text[split_point : split_point + 200]
    end_match = re.search(r"[.!?]\s", look_ahead)
    if end_match:
        split_point += end_match.end()
    input_text = text[:split_point].strip()
    output_text = text[split_point:].strip()

    if len(output_text) < config.min_output_chars:
        return None
    return InstructionExample(
        instruction=templates["continue"],
        input=input_text,
        output=output_text,
        metadata={"strategy": "narrative_continue", "chunk_index": chunk.index},
    )


def _strategy_key_points(
    chunk: Chunk, text: str, templates: dict[str, str], config: ConverterConfig
) -> InstructionExample | None:
    """Key points: prima frase di ogni paragrafo (max 5)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return None
    points = []
    for p in paragraphs[:5]:
        sents = _split_sentences(p)
        if sents:
            points.append(f"- {sents[0]}")
    output = "\n".join(points)
    if len(output) < config.min_output_chars:
        return None
    return InstructionExample(
        instruction=templates["key_points"],
        input=text,
        output=output,
        metadata={"strategy": "narrative_key_points", "chunk_index": chunk.index},
    )


def _split_sentences(text: str) -> list[str]:
    """Split frasi semplice (replica del chunker)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def convert_chunks(
    chunks: list[Chunk],
    doc: ExtractedDocument,
    content_type: ContentType,
    config: ConverterConfig | None = None,
) -> list[InstructionExample]:
    """
    Converte una lista di Chunk in esempi di instruction tuning.

    Args:
        chunks: chunk prodotti dal Chunker.
        doc: documento originale (serve per metadata di colonne CSV, ecc.).
        content_type: tipo determinato dal Detector.
        config: parametri converter.

    Returns:
        Lista di InstructionExample. Può essere più corta o più lunga di
        chunks (alcuni chunk producono 0 esempi, altri 2+).
    """
    config = config or ConverterConfig()

    strategy_map = {
        ContentType.QA_PAIRS: _convert_qa_pairs,
        ContentType.TABULAR: _convert_tabular,
        ContentType.CODE: _convert_code,
        ContentType.DIALOGUE: _convert_dialogue,
        ContentType.NARRATIVE: _convert_narrative,
        ContentType.MIXED: _convert_narrative,  # mixed → narrative fallback
    }

    strategy = strategy_map.get(content_type, _convert_narrative)

    examples: list[InstructionExample] = []
    for chunk in chunks:
        try:
            chunk_examples = strategy(chunk, doc, config)
            examples.extend(chunk_examples)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Errore conversione chunk %d (%s): %s",
                chunk.index, content_type.value, exc,
            )

    logger.info(
        "Converter: %s, %d chunks → %d esempi",
        content_type.value, len(chunks), len(examples),
    )
    return examples