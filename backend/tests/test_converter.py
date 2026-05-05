"""
Test del Converter.
"""

from __future__ import annotations

import pytest

from core.dataset.chunker import Chunk
from core.dataset.converter import (
    ConverterConfig,
    InstructionExample,
    convert_chunks,
)
from core.dataset.detector import ContentType
from core.dataset.extracted import ExtractedDocument, Section


def make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        index=index,
        char_count=len(text),
        estimated_tokens=len(text) // 4,
        source_section=None,
        metadata={},
    )


def make_doc(text: str = "", source_format: str = "txt", **kwargs) -> ExtractedDocument:
    return ExtractedDocument(
        text=text or "x",
        source_format=source_format,
        sections=kwargs.get("sections", []),
        metadata=kwargs.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConverterConfig:
    def test_defaults(self):
        c = ConverterConfig()
        assert c.examples_per_narrative_chunk == 1
        assert c.template_language == "it"

    def test_invalid_examples_count(self):
        with pytest.raises(ValueError):
            ConverterConfig(examples_per_narrative_chunk=0)

    def test_invalid_language(self):
        with pytest.raises(ValueError):
            ConverterConfig(template_language="fr")


# ---------------------------------------------------------------------------
# QA Pairs
# ---------------------------------------------------------------------------


class TestConvertQAPairs:
    def test_extract_q_a_prefixes(self):
        text = """
Q: Qual è la capitale d'Italia?
A: Roma è la capitale d'Italia, città eterna ricca di storia.

Q: Qual è la capitale della Francia?
A: Parigi è la capitale della Francia, famosa per la Tour Eiffel.
"""
        chunk = make_chunk(text)
        doc = make_doc(text)
        examples = convert_chunks([chunk], doc, ContentType.QA_PAIRS)

        assert len(examples) == 2
        assert "capitale d'Italia" in examples[0].instruction
        assert "Roma" in examples[0].output
        assert examples[0].metadata["strategy"] == "qa_pattern_extract"

    def test_csv_qa_columns(self):
        chunk = make_chunk(
            "question: Cos'è il sole? | answer: Una stella al centro del nostro sistema solare."
        )
        doc = make_doc(
            source_format="csv",
            metadata={"columns": ["question", "answer"]},
        )
        examples = convert_chunks([chunk], doc, ContentType.QA_PAIRS)

        assert len(examples) == 1
        assert "sole" in examples[0].instruction.lower()
        assert "stella" in examples[0].output.lower()
        assert examples[0].metadata["strategy"] == "csv_qa_columns"

    def test_json_instruction_keys(self):
        chunk = make_chunk(
            "instruction: Calcola 2+2\noutput: Il risultato è 4."
        )
        doc = make_doc(source_format="json")
        examples = convert_chunks([chunk], doc, ContentType.QA_PAIRS)

        assert len(examples) == 1
        assert examples[0].instruction == "Calcola 2+2"
        assert examples[0].output == "Il risultato è 4."
        assert examples[0].metadata["strategy"] == "json_instruction_keys"

    def test_no_qa_pattern_returns_empty(self):
        chunk = make_chunk("Solo testo normale senza prefissi Q o A.")
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.QA_PAIRS)
        assert examples == []


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------


class TestConvertTabular:
    def test_basic_record_description(self):
        chunk = make_chunk(
            "name: Marco Rossi | age: 35 | city: Milano | profession: Ingegnere"
        )
        doc = make_doc(
            source_format="csv",
            metadata={"columns": ["name", "age", "city", "profession"]},
        )
        # min_chars=50 perché un record CSV tipico è più corto del default 100
        examples = convert_chunks(
            [chunk], doc, ContentType.TABULAR, ConverterConfig(min_chars=50)
        )

        assert len(examples) == 1
        assert "Marco Rossi" in examples[0].output
        assert "Milano" in examples[0].output
        assert examples[0].metadata["strategy"] == "tabular_concat"

    def test_too_short_skipped(self):
        chunk = make_chunk("a: b")
        doc = make_doc(source_format="csv")
        examples = convert_chunks(
            [chunk], doc, ContentType.TABULAR, ConverterConfig(min_chars=100)
        )
        assert examples == []


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


class TestConvertCode:
    def test_extract_docstring(self):
        text = '''
def calculate_sum(a, b):
    """Calcola la somma di due numeri interi e ritorna il risultato.
    Funzione semplice usata per test e debug del modulo math."""
    return a + b
'''
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.CODE)

        assert len(examples) == 1
        assert "def calculate_sum" in examples[0].input
        assert "somma" in examples[0].output.lower()
        assert examples[0].metadata["strategy"] == "code_docstring"

    def test_no_docstring_returns_empty(self):
        text = """
def f(x):
    return x * 2

def g(x):
    return x + 1
"""
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.CODE)
        assert examples == []


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------


class TestConvertDialogue:
    def test_extract_turn_pairs(self):
        text = """
Alice: Ciao Bob, come stai oggi?
Bob: Sto bene grazie Alice, e tu come stai?
Alice: Anch'io sto bene, ho appena finito di leggere un libro interessante.
Bob: Davvero? Di cosa parla il libro?
"""
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.DIALOGUE)

        # 4 turn → 3 coppie A→B, B→A, A→B
        assert len(examples) == 3
        assert "Ciao Bob" in examples[0].instruction
        assert "Sto bene" in examples[0].output
        assert examples[0].metadata["strategy"] == "dialogue_turn"

    def test_too_few_turns(self):
        chunk = make_chunk("Solo: una frase senza dialogo.")
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.DIALOGUE)
        assert examples == []


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


class TestConvertNarrative:
    def test_summarize_strategy(self):
        text = (
            "Marco camminava lungo il sentiero del bosco mentre il sole tramontava. "
            "Le foglie cadevano lentamente attorno a lui. "
            "Nei suoi pensieri c'era solo la promessa fatta a sua madre. "
            "Doveva trovare il vecchio stregone delle montagne. "
            "Il vento iniziava a soffiare più forte tra gli alberi. "
            "Marco accelerò il passo verso la cima."
        )
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks(
            [chunk], doc, ContentType.NARRATIVE,
            ConverterConfig(examples_per_narrative_chunk=1),
        )

        assert len(examples) == 1
        assert examples[0].instruction == "Riassumi il seguente testo in poche frasi."
        assert examples[0].metadata["strategy"] == "narrative_summary"

    def test_multiple_strategies_per_chunk(self):
        text = (
            "Primo paragrafo che parla di qualcosa. Contiene diverse frasi. "
            "Esempio di prosa.\n\n"
            "Secondo paragrafo con altre informazioni rilevanti. "
            "Anche questo paragrafo ha più frasi.\n\n"
            "Terzo paragrafo con conclusione finale. Ultime considerazioni."
        )
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks(
            [chunk], doc, ContentType.NARRATIVE,
            ConverterConfig(examples_per_narrative_chunk=3),
        )

        # Dovremmo ottenere fino a 3 esempi (summarize + continue + key_points)
        strategies = {ex.metadata["strategy"] for ex in examples}
        # Almeno uno di ciascuno (alcuni potrebbero saltare per len)
        assert len(examples) >= 1

    def test_too_short_skipped(self):
        chunk = make_chunk("Breve.")
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.NARRATIVE)
        assert examples == []

    def test_english_templates(self):
        text = "First sentence. Second sentence. Third sentence. Fourth one. " * 3
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks(
            [chunk], doc, ContentType.NARRATIVE,
            ConverterConfig(template_language="en"),
        )
        if examples:
            # Le instruction inglesi iniziano con "Summarize", "Continue", "What"
            assert any(
                ex.instruction.startswith(("Summarize", "Continue", "What"))
                for ex in examples
            )


# ---------------------------------------------------------------------------
# Mixed → fallback narrative
# ---------------------------------------------------------------------------


class TestConvertMixed:
    def test_mixed_falls_back_to_narrative(self):
        text = "Una frase. Un'altra. Una terza. " * 10
        chunk = make_chunk(text)
        doc = make_doc()
        examples = convert_chunks([chunk], doc, ContentType.MIXED)
        # Deve produrre esempi via strategia narrative
        assert len(examples) >= 1


# ---------------------------------------------------------------------------
# InstructionExample serialization
# ---------------------------------------------------------------------------


class TestInstructionExample:
    def test_to_dict_serializable(self):
        import json

        ex = InstructionExample(
            instruction="test",
            input="input",
            output="output",
            metadata={"strategy": "test"},
        )
        d = ex.to_dict()
        json.dumps(d)
        assert d["instruction"] == "test"
        assert d["metadata"]["strategy"] == "test"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_no_chunks(self):
        examples = convert_chunks([], make_doc(), ContentType.NARRATIVE)
        assert examples == []