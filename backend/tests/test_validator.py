"""
Test del Validator.
"""

from __future__ import annotations

import pytest

from core.dataset.converter import InstructionExample
from core.dataset.validator import (
    ValidatorConfig,
    validate_examples,
)


def make_example(
    instruction: str = "instr",
    input_: str = "",
    output: str = "out" * 20,  # ~60 chars default
    strategy: str = "test",
) -> InstructionExample:
    return InstructionExample(
        instruction=instruction,
        input=input_,
        output=output,
        metadata={"strategy": strategy},
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestValidatorConfig:
    def test_defaults(self):
        c = ValidatorConfig()
        assert c.min_output_chars == 20
        assert c.enable_fuzzy_dedup is False

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ValidatorConfig(fuzzy_threshold=1.5)

    def test_invalid_max_total(self):
        with pytest.raises(ValueError):
            ValidatorConfig(min_output_chars=100, max_total_chars=50)


# ---------------------------------------------------------------------------
# Filtri lunghezza
# ---------------------------------------------------------------------------


class TestLengthFilters:
    def test_too_short_output_filtered(self):
        ex = make_example(output="ciao")  # 4 chars < 20
        result = validate_examples([ex])
        assert len(result.examples) == 0
        assert result.stats.too_short_filtered == 1

    def test_too_long_total_filtered(self):
        # output + input + instruction > max_total_chars
        ex = make_example(output="x" * 20000)
        config = ValidatorConfig(max_total_chars=10000, max_output_chars=20000)
        result = validate_examples([ex], config)
        assert len(result.examples) == 0
        assert result.stats.too_long_filtered == 1

    def test_too_long_output_filtered(self):
        ex = make_example(output="x" * 5000)
        config = ValidatorConfig(max_output_chars=1000, max_total_chars=20000)
        result = validate_examples([ex], config)
        assert len(result.examples) == 0
        assert result.stats.too_long_filtered == 1

    def test_empty_output_filtered(self):
        ex = make_example(output="   \n  ")
        result = validate_examples([ex])
        assert len(result.examples) == 0
        assert result.stats.empty_output_filtered == 1

    def test_valid_example_passes(self):
        ex = make_example(output="Una risposta di lunghezza adeguata.")
        result = validate_examples([ex])
        assert len(result.examples) == 1


# ---------------------------------------------------------------------------
# Deduplica esatta
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_exact_duplicates_removed(self):
        ex1 = make_example(output="Risposta uguale di test.")
        ex2 = make_example(output="Risposta uguale di test.")
        ex3 = make_example(output="Risposta uguale di test.")
        result = validate_examples([ex1, ex2, ex3])
        assert len(result.examples) == 1
        assert result.stats.duplicates_removed_exact == 2

    def test_normalization_catches_whitespace_dups(self):
        ex1 = make_example(output="Risposta normale di test del sistema.")
        ex2 = make_example(output="  RISPOSTA   normale di TEST   del sistema.  ")
        result = validate_examples([ex1, ex2])
        assert len(result.examples) == 1
        assert result.stats.duplicates_removed_exact == 1

    def test_different_instructions_kept(self):
        ex1 = make_example(instruction="A", output="Risposta normale di test.")
        ex2 = make_example(instruction="B", output="Risposta normale di test.")
        result = validate_examples([ex1, ex2])
        # Output uguali ma istruzioni diverse → entrambi tenuti
        assert len(result.examples) == 2


# ---------------------------------------------------------------------------
# Deduplica fuzzy
# ---------------------------------------------------------------------------


class TestFuzzyDedup:
    def test_fuzzy_dedup_disabled_by_default(self):
        ex1 = make_example(output="Il sole splende alto nel cielo azzurro estivo.")
        ex2 = make_example(output="Il sole splende alto nel cielo azzurro estivo!")
        result = validate_examples([ex1, ex2])
        # Esatta diff (! vs .) → entrambi tenuti
        assert len(result.examples) == 2

    def test_fuzzy_dedup_catches_near_duplicates(self):
        ex1 = make_example(
            instruction="Spiega il sistema solare",
            output="Il sistema solare è composto dal Sole e otto pianeti.",
        )
        ex2 = make_example(
            instruction="Spiega il sistema solare",
            output="Il sistema solare è composto dal Sole e otto pianeti!!",
        )
        config = ValidatorConfig(enable_fuzzy_dedup=True, fuzzy_threshold=0.9)
        result = validate_examples([ex1, ex2], config)
        assert len(result.examples) == 1
        assert result.stats.duplicates_removed_fuzzy == 1


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------


class TestStats:
    def test_basic_counts(self):
        examples = [
            make_example(output="A" * 50, strategy="qa_extract"),
            make_example(output="B" * 100, strategy="narrative_summary"),
            make_example(output="C" * 200, strategy="qa_extract"),
        ]
        result = validate_examples(examples)
        s = result.stats

        assert s.examples_in == 3
        assert s.examples_out == 3
        assert s.strategy_counts == {"qa_extract": 2, "narrative_summary": 1}
        assert s.output_chars_min == 50
        assert s.output_chars_max == 200

    def test_stats_to_dict_serializable(self):
        import json

        examples = [make_example(output="x" * 50)]
        result = validate_examples(examples)
        d = result.stats.to_dict()
        # Deve essere JSON serializzabile
        json.dumps(d)
        assert "output_chars" in d
        assert "strategy_counts" in d

    def test_empty_input(self):
        result = validate_examples([])
        assert len(result.examples) == 0
        assert result.stats.examples_in == 0
        assert result.stats.examples_out == 0
        assert result.stats.output_chars_min == 0


# ---------------------------------------------------------------------------
# Quality warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_low_pass_rate_warning(self):
        # 10 esempi tutti vuoti → 0 passano → warning
        examples = [make_example(output="") for _ in range(10)]
        # Aggiungiamo 1 valido
        examples.append(make_example(output="Risposta valida normale."))
        result = validate_examples(examples)
        assert any("filtri" in w.lower() for w in result.stats.warnings)

    def test_monotone_strategy_warning(self):
        # Tutti narrative_summary → warning monotonia
        examples = [
            make_example(output=f"Output unico {i}." * 5, strategy="narrative_summary")
            for i in range(10)
        ]
        result = validate_examples(examples)
        assert any("monotono" in w.lower() for w in result.stats.warnings)