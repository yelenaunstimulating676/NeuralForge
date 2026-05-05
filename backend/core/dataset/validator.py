"""
Validator — pulisce gli InstructionExample prodotti dal Converter e
calcola statistiche per la UI.

Filtri applicati in ordine:
  1. Skip esempi vuoti o con output troppo corto
  2. Skip esempi con totale chars > max_chars (esplodono tokenizer)
  3. Deduplica esatta (hash su instruction+input+output normalizzati)
  4. (Opzionale) Deduplica fuzzy (similarity > threshold)

Output: ValidatedDataset con esempi puliti + statistiche complete.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from core.dataset.converter import InstructionExample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatorConfig:
    """Parametri del validator."""

    # Lunghezza output minima (in chars)
    min_output_chars: int = 20
    # Lunghezza output massima
    max_output_chars: int = 8192
    # Lunghezza totale massima (instruction + input + output)
    max_total_chars: int = 16384
    # Deduplica fuzzy: confronta primi N chars di instruction+output
    enable_fuzzy_dedup: bool = False
    fuzzy_threshold: float = 0.95
    fuzzy_compare_chars: int = 200

    def __post_init__(self) -> None:
        if not 0 < self.fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold deve essere in (0, 1]")
        if self.max_total_chars < self.min_output_chars:
            raise ValueError("max_total_chars deve essere ≥ min_output_chars")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStats:
    """Statistiche complete di un dataset validato."""

    examples_in: int                            # input al validator
    examples_out: int                           # dopo filtri
    duplicates_removed_exact: int
    duplicates_removed_fuzzy: int
    too_short_filtered: int
    too_long_filtered: int
    empty_output_filtered: int

    # Per ContentType / strategy distribution
    strategy_counts: dict[str, int]

    # Lunghezza output (chars)
    output_chars_min: int
    output_chars_max: int
    output_chars_mean: float
    output_chars_median: float
    output_chars_p90: int
    output_chars_p99: int

    # Token stimati (chars / 4)
    estimated_tokens_total: int
    estimated_tokens_mean: float

    # Quality warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples_in": self.examples_in,
            "examples_out": self.examples_out,
            "duplicates_removed_exact": self.duplicates_removed_exact,
            "duplicates_removed_fuzzy": self.duplicates_removed_fuzzy,
            "too_short_filtered": self.too_short_filtered,
            "too_long_filtered": self.too_long_filtered,
            "empty_output_filtered": self.empty_output_filtered,
            "strategy_counts": dict(self.strategy_counts),
            "output_chars": {
                "min": self.output_chars_min,
                "max": self.output_chars_max,
                "mean": round(self.output_chars_mean, 1),
                "median": round(self.output_chars_median, 1),
                "p90": self.output_chars_p90,
                "p99": self.output_chars_p99,
            },
            "estimated_tokens": {
                "total": self.estimated_tokens_total,
                "mean": round(self.estimated_tokens_mean, 1),
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ValidatedDataset:
    """Dataset finale: esempi puliti + statistiche."""

    examples: list[InstructionExample]
    stats: DatasetStats

    def __len__(self) -> int:
        return len(self.examples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_hash(text: str) -> str:
    """Normalizza testo per hashing: lower + collapse whitespace."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _example_hash(ex: InstructionExample) -> str:
    """Hash MD5 stabile di un esempio (per dedup esatta)."""
    key = (
        _normalize_for_hash(ex.instruction)
        + "|"
        + _normalize_for_hash(ex.input)
        + "|"
        + _normalize_for_hash(ex.output)
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _percentile(values: list[int], p: float) -> int:
    """
    Percentile semplice senza dipendere da numpy.
    p è in [0, 1]. Per p=0.9 → 90° percentile.
    """
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = int(p * (len(sorted_v) - 1))
    return sorted_v[idx]


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Similarity Jaccard su bigrammi di parole, con normalizzazione
    aggressiva (lower + strip punteggiatura). Veloce ma rozza.
    Buona euristica per dedup fuzzy senza embedding.

    Bigrammi (n=2) invece di trigrammi: più tollerante alle differenze
    minime (1-2 parole diverse) tipiche di near-duplicates reali.
    """
    if not a or not b:
        return 0.0

    def normalize_words(text: str) -> list[str]:
        # Tokenizza e rimuove punteggiatura attaccata alle parole
        # (es. "pianeti." e "pianeti!!" → "pianeti")
        out: list[str] = []
        for w in text.lower().split():
            stripped = re.sub(r"[^\w]+", "", w, flags=re.UNICODE)
            if stripped:
                out.append(stripped)
        return out

    def bigrams(text: str) -> set[str]:
        words = normalize_words(text)
        if len(words) < 2:
            return {" ".join(words)} if words else set()
        return {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}

    sa = bigrams(a)
    sb = bigrams(b)
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Validation top-level
# ---------------------------------------------------------------------------


def validate_examples(
    examples: list[InstructionExample],
    config: ValidatorConfig | None = None,
) -> ValidatedDataset:
    """
    Pulisce e valida una lista di InstructionExample.

    Args:
        examples: output del Converter.
        config: parametri filtri (default sensati).

    Returns:
        ValidatedDataset con esempi puliti + statistiche.
    """
    config = config or ValidatorConfig()

    examples_in = len(examples)
    too_short = 0
    too_long = 0
    empty_output = 0
    dup_exact = 0
    dup_fuzzy = 0

    seen_hashes: set[str] = set()
    fuzzy_corpus: list[str] = []  # per dedup fuzzy
    cleaned: list[InstructionExample] = []

    for ex in examples:
        # 1. Empty output
        if not ex.output or not ex.output.strip():
            empty_output += 1
            continue

        # 2. Length filters
        output_len = len(ex.output)
        total_len = len(ex.instruction) + len(ex.input) + output_len

        if output_len < config.min_output_chars:
            too_short += 1
            continue
        if total_len > config.max_total_chars:
            too_long += 1
            continue
        if output_len > config.max_output_chars:
            too_long += 1
            continue

        # 3. Exact dedup
        h = _example_hash(ex)
        if h in seen_hashes:
            dup_exact += 1
            continue
        seen_hashes.add(h)

        # 4. Fuzzy dedup (opzionale)
        if config.enable_fuzzy_dedup:
            comparable = (
                ex.instruction[: config.fuzzy_compare_chars]
                + " "
                + ex.output[: config.fuzzy_compare_chars]
            )
            is_dup = any(
                _jaccard_similarity(comparable, prev) >= config.fuzzy_threshold
                for prev in fuzzy_corpus
            )
            if is_dup:
                dup_fuzzy += 1
                continue
            fuzzy_corpus.append(comparable)

        cleaned.append(ex)

    stats = _compute_stats(
        cleaned,
        examples_in=examples_in,
        too_short=too_short,
        too_long=too_long,
        empty_output=empty_output,
        dup_exact=dup_exact,
        dup_fuzzy=dup_fuzzy,
    )

    logger.info(
        "Validator: %d → %d esempi (filtrati: %d short, %d long, %d empty, "
        "%d dup_exact, %d dup_fuzzy)",
        examples_in, len(cleaned),
        too_short, too_long, empty_output, dup_exact, dup_fuzzy,
    )

    return ValidatedDataset(examples=cleaned, stats=stats)


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------


def _compute_stats(
    examples: list[InstructionExample],
    *,
    examples_in: int,
    too_short: int,
    too_long: int,
    empty_output: int,
    dup_exact: int,
    dup_fuzzy: int,
) -> DatasetStats:
    """Calcola tutte le statistiche del ValidatedDataset."""

    output_lens = [len(ex.output) for ex in examples]
    total_lens = [len(ex.instruction) + len(ex.input) + len(ex.output) for ex in examples]

    # Strategy distribution (dal metadata)
    strategy_counts: dict[str, int] = {}
    for ex in examples:
        strat = ex.metadata.get("strategy", "unknown")
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    # Token estimates
    total_tokens = sum(total_lens) // 4

    # Warnings di qualità
    warnings: list[str] = []
    if examples_in > 0 and len(examples) / examples_in < 0.3:
        warnings.append(
            f"Solo {len(examples)}/{examples_in} esempi superano i filtri "
            f"({len(examples) / examples_in:.0%}). Considera di rivedere la "
            "configurazione del Converter o del Chunker."
        )
    if examples and len(strategy_counts) == 1:
        only_strat = next(iter(strategy_counts))
        if only_strat in {"narrative_summary", "tabular_concat"}:
            warnings.append(
                f"Tutti gli esempi hanno la stessa strategia ({only_strat}). "
                "Il dataset potrebbe essere monotono per il fine-tuning."
            )
    if examples and dup_exact > examples_in * 0.3:
        warnings.append(
            f"Rimossi {dup_exact} duplicati esatti su {examples_in} esempi "
            f"({dup_exact / examples_in:.0%}). Verifica che il chunking non "
            "produca sovrapposizioni eccessive."
        )

    return DatasetStats(
        examples_in=examples_in,
        examples_out=len(examples),
        duplicates_removed_exact=dup_exact,
        duplicates_removed_fuzzy=dup_fuzzy,
        too_short_filtered=too_short,
        too_long_filtered=too_long,
        empty_output_filtered=empty_output,
        strategy_counts=strategy_counts,
        output_chars_min=min(output_lens) if output_lens else 0,
        output_chars_max=max(output_lens) if output_lens else 0,
        output_chars_mean=mean(output_lens) if output_lens else 0.0,
        output_chars_median=median(output_lens) if output_lens else 0.0,
        output_chars_p90=_percentile(output_lens, 0.90),
        output_chars_p99=_percentile(output_lens, 0.99),
        estimated_tokens_total=total_tokens,
        estimated_tokens_mean=(total_tokens / len(examples)) if examples else 0.0,
        warnings=warnings,
    )