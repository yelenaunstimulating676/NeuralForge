"""
Content Type Detector — classifica un ExtractedDocument in una di queste
famiglie di contenuto, decidendo la strategia di conversione in M3.4:

    - qa_pairs   → coppie domanda/risposta esplicite
    - narrative  → prosa continua (libri, articoli, blog)
    - code       → codice sorgente
    - dialogue   → conversazioni multi-turn (chat, trascrizioni)
    - tabular    → dati strutturati (CSV, righe omogenee)
    - mixed      → contenuto eterogeneo che non rientra nelle altre

Approccio euristico (no ML). Per ogni tipo calcoliamo uno score [0, 1].
Vince quello con score più alto. La `confidence` è la distanza tra il
top score e il secondo. Confidence bassa → mixed.

Note di design:
  - Funzioni pure, deterministiche, senza side effect
  - Nessuna dipendenza esterna (no torch, no sklearn)
  - Esecuzione O(n) sul testo, no embedding o LLM
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from core.dataset.extracted import ExtractedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum + dataclass output
# ---------------------------------------------------------------------------


class ContentType(str, Enum):
    QA_PAIRS = "qa_pairs"
    NARRATIVE = "narrative"
    CODE = "code"
    DIALOGUE = "dialogue"
    TABULAR = "tabular"
    MIXED = "mixed"


@dataclass(frozen=True)
class DetectionResult:
    """Output del detector: tipo + scores per tutti i tipi + confidence."""

    content_type: ContentType
    confidence: float                       # 0 (boh) → 1 (sicuro)
    scores: dict[ContentType, float]        # score di ogni tipo
    indicators: list[str]                   # spiegazioni human-readable

    def to_dict(self) -> dict:
        return {
            "content_type": self.content_type.value,
            "confidence": round(self.confidence, 3),
            "scores": {k.value: round(v, 3) for k, v in self.scores.items()},
            "indicators": self.indicators,
        }


# ---------------------------------------------------------------------------
# Pattern e regole
# ---------------------------------------------------------------------------


# === QA pairs patterns ===
# Cerchiamo prefissi tipo "Q:", "Domanda:", "A:", "R:", "Risposta:" all'inizio
# di una riga (case-insensitive).
_QA_PREFIXES = re.compile(
    r"(?im)^\s*(?:q|a|r|d|domanda|risposta|question|answer)\s*[:.\-)]",
    re.MULTILINE,
)

# Numerazione tipo "1.", "Q1.", "1)" all'inizio di righe — segnale debole
_NUMBERED_LINES = re.compile(r"(?m)^\s*\d+[\.\)]\s+")

# === Code patterns ===
_CODE_FENCES = re.compile(r"```[\s\S]*?```")
_CODE_KEYWORDS = re.compile(
    r"\b(def |class |function |import |from |return |const |let |var |"
    r"public |private |#include |using |fn |pub |async )"
)
_CODE_SYMBOLS_RE = re.compile(r"[{};()\[\]<>=]")
_INDENT_LINES_RE = re.compile(r"(?m)^(?: {2,}|\t+)")

# === Dialogue patterns ===
# "Mario: ciao", "Speaker A: ...", "User: ..."
_DIALOGUE_LINE = re.compile(
    r"(?m)^\s*([A-Z][A-Za-z0-9_ ]{0,30}|user|assistant|system|bot|host|guest)"
    r"\s*:\s+\S",
    re.MULTILINE,
)

# === Sentence splitter (rough) per narrative ===
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


# ---------------------------------------------------------------------------
# Helpers di scoring (ognuno ritorna un valore 0..1)
# ---------------------------------------------------------------------------


def _score_qa_pairs(text: str, doc: ExtractedDocument) -> tuple[float, list[str]]:
    """QA score: alto se tante righe iniziano con prefissi Q/A."""
    indicators: list[str] = []
    if not text:
        return 0.0, indicators

    qa_matches = _QA_PREFIXES.findall(text)
    line_count = max(1, text.count("\n") + 1)
    qa_density = len(qa_matches) / line_count

    # Bonus: JSON con chiavi instruction/question/answer
    json_bonus = 0.0
    if doc.source_format in {"json", "jsonl"}:
        sample_keys: set[str] = set()
        for s in doc.sections[:20]:
            sample_keys.update(k.lower() for k in (s.metadata.get("raw_keys") or []))
        instruction_keys = {"instruction", "input", "output", "question", "answer", "prompt", "completion"}
        if sample_keys & instruction_keys:
            json_bonus = 0.6
            indicators.append(
                f"JSON contiene chiavi instruction-style: {sample_keys & instruction_keys}"
            )

    # CSV con colonne question/answer
    csv_bonus = 0.0
    if doc.source_format in {"csv", "tsv"}:
        cols = {c.lower() for c in (doc.metadata.get("columns") or [])}
        qa_cols = {"question", "answer", "domanda", "risposta", "q", "a", "prompt", "completion"}
        if cols & qa_cols:
            csv_bonus = 0.7
            indicators.append(f"CSV con colonne Q/A: {cols & qa_cols}")

    score = min(1.0, qa_density * 4 + json_bonus + csv_bonus)
    if qa_density > 0.05:
        indicators.append(
            f"{len(qa_matches)} prefissi Q/A trovati ({qa_density:.1%} delle righe)"
        )
    return score, indicators


def _score_code(text: str, doc: ExtractedDocument) -> tuple[float, list[str]]:
    """Code score: alto con code fences, keywords, simboli, indentazione."""
    indicators: list[str] = []
    if not text:
        return 0.0, indicators

    text_len = len(text)
    fences = _CODE_FENCES.findall(text)
    keywords = _CODE_KEYWORDS.findall(text)
    symbol_count = len(_CODE_SYMBOLS_RE.findall(text))
    indent_lines = len(_INDENT_LINES_RE.findall(text))
    line_count = max(1, text.count("\n") + 1)

    # Score componenti
    fence_score = min(1.0, len(fences) * 0.25)
    keyword_density = len(keywords) / max(1, line_count) * 2  # normalizzato
    symbol_density = symbol_count / text_len * 30  # ~3% simboli → score 1
    indent_density = indent_lines / line_count

    score = min(
        1.0,
        fence_score * 0.4
        + min(1.0, keyword_density) * 0.25
        + min(1.0, symbol_density) * 0.2
        + min(1.0, indent_density) * 0.15,
    )

    if fences:
        indicators.append(f"{len(fences)} blocchi code-fence")
    if keywords:
        indicators.append(f"{len(keywords)} keyword di linguaggi noti")
    if symbol_density > 0.5:
        indicators.append(f"alta densità simboli code ({symbol_count} caratteri)")
    return score, indicators


def _score_dialogue(text: str, doc: ExtractedDocument) -> tuple[float, list[str]]:
    """Dialogue score: alto se molte righe sono `Nome:` con risposta inline."""
    indicators: list[str] = []
    if not text:
        return 0.0, indicators

    matches = _DIALOGUE_LINE.findall(text)
    line_count = max(1, text.count("\n") + 1)

    # Per essere "dialogo" servono ALMENO 2 speaker diversi
    speakers = {m.lower().strip() for m in matches if m.strip()}
    speaker_count = len(speakers)

    density = len(matches) / line_count
    score = 0.0
    if speaker_count >= 2:
        score = min(1.0, density * 3)
        indicators.append(
            f"{speaker_count} speaker distinti, "
            f"{len(matches)} turn ({density:.1%} righe)"
        )
    return score, indicators


def _score_tabular(text: str, doc: ExtractedDocument) -> tuple[float, list[str]]:
    """Tabular score: alto se source è CSV/TSV e non ha colonne Q/A."""
    indicators: list[str] = []

    if doc.source_format in {"csv", "tsv"}:
        cols = {c.lower() for c in (doc.metadata.get("columns") or [])}
        qa_cols = {"question", "answer", "domanda", "risposta", "q", "a", "prompt", "completion"}
        if cols & qa_cols:
            # È QA, non puro tabular
            return 0.2, indicators
        score = 0.85
        indicators.append(f"CSV/TSV con {len(cols)} colonne strutturate")
        return score, indicators

    # Fallback: testo con righe regolari separate da | o tab
    lines = text.split("\n")[:200]
    if not lines:
        return 0.0, indicators
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    tab_lines = sum(1 for line in lines if "\t" in line)
    density = max(pipe_lines, tab_lines) / max(1, len(lines))
    if density > 0.5:
        indicators.append(f"{density:.0%} righe con separatori tabular (|, tab)")
    return min(1.0, density), indicators


def _score_narrative(text: str, doc: ExtractedDocument) -> tuple[float, list[str]]:
    """
    Narrative score: alto se prosa continua → frasi lunghe, basso rapporto
    di simboli code, poche righe Q/A o dialogo.

    Pensata come "default" che aumenta quando GLI ALTRI segnali sono assenti.
    """
    indicators: list[str] = []
    if not text:
        return 0.0, indicators

    # Lunghezza media frase
    sentences = _SENTENCE_END.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0, indicators

    avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
    sentence_count = len(sentences)

    # Densità simboli code (penalizzante)
    symbol_density = len(_CODE_SYMBOLS_RE.findall(text)) / max(1, len(text))

    # Score: sweet spot frasi 60-300 caratteri (prosa standard)
    if 40 <= avg_sentence_len <= 350 and sentence_count >= 5:
        length_score = 1.0
    elif sentence_count >= 3:
        length_score = 0.6
    else:
        length_score = 0.2

    # Penalizza testo "rumoroso" di simboli
    symbol_penalty = max(0.0, 1.0 - symbol_density * 30)

    score = length_score * symbol_penalty
    indicators.append(
        f"{sentence_count} frasi, lunghezza media {avg_sentence_len:.0f} caratteri"
    )
    return score, indicators


# ---------------------------------------------------------------------------
# Detection top-level
# ---------------------------------------------------------------------------


def detect_content_type(doc: ExtractedDocument) -> DetectionResult:
    """
    Analizza un ExtractedDocument e ritorna il ContentType più probabile.

    Args:
        doc: documento estratto da un Extractor.

    Returns:
        DetectionResult con tipo, confidence, scores per tipo, indicatori.
    """
    text = doc.text or ""
    indicators: list[str] = []

    # Calcola tutti gli score
    scores: dict[ContentType, float] = {}
    all_indicators: dict[ContentType, list[str]] = {}

    for ct, scorer in (
        (ContentType.QA_PAIRS, _score_qa_pairs),
        (ContentType.CODE, _score_code),
        (ContentType.DIALOGUE, _score_dialogue),
        (ContentType.TABULAR, _score_tabular),
        (ContentType.NARRATIVE, _score_narrative),
    ):
        s, ind = scorer(text, doc)
        scores[ct] = s
        all_indicators[ct] = ind

    # Mixed non viene scorato direttamente: è il fallback se confidence è bassa
    scores[ContentType.MIXED] = 0.0

    # Trova top
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type, top_score = sorted_types[0]
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

    # Confidence = top - secondo (margine)
    confidence = max(0.0, top_score - second_score)

    # Threshold: se top score è basso → mixed
    if top_score < 0.3:
        winner = ContentType.MIXED
        indicators.append(
            f"Nessun tipo dominante (top score {top_score:.2f} < 0.3) → mixed"
        )
        confidence = 0.0
    else:
        winner = top_type
        indicators.extend(all_indicators[winner])
        if confidence < 0.15 and top_type != ContentType.NARRATIVE:
            # Margine basso: verifichiamo se siamo davvero sicuri
            indicators.append(
                f"Confidence bassa ({confidence:.2f}): seconda scelta "
                f"{sorted_types[1][0].value} con score {second_score:.2f}"
            )

    logger.info(
        "Detection: type=%s confidence=%.2f scores=%s",
        winner.value, confidence,
        {k.value: round(v, 2) for k, v in scores.items() if v > 0},
    )

    return DetectionResult(
        content_type=winner,
        confidence=confidence,
        scores=scores,
        indicators=indicators,
    )