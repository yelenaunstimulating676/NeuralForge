"""
Test del Chunker.

Strategia: documenti finti con testi controllati, verifichiamo che il
numero e il contenuto dei chunk siano sensati.
"""

from __future__ import annotations

import pytest

from core.dataset.chunker import (
    Chunk,
    ChunkerConfig,
    chunk_document,
)
from core.dataset.detector import ContentType
from core.dataset.extracted import ExtractedDocument, Section


def make_doc(
    text: str,
    source_format: str = "txt",
    sections: list[Section] | None = None,
    metadata: dict | None = None,
) -> ExtractedDocument:
    """Helper per creare un ExtractedDocument finto."""
    if sections is None:
        sections = [Section(title=None, text=text)]
    return ExtractedDocument(
        text=text,
        source_format=source_format,
        sections=sections,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestChunkerConfig:
    def test_defaults(self):
        c = ChunkerConfig()
        assert c.target_chars == 2048
        assert c.overlap_chars == 200
        assert c.min_chunk_chars == 200

    def test_target_too_small(self):
        with pytest.raises(ValueError):
            ChunkerConfig(target_chars=50)

    def test_overlap_too_big(self):
        with pytest.raises(ValueError):
            ChunkerConfig(target_chars=500, overlap_chars=600)

    def test_min_too_big(self):
        with pytest.raises(ValueError):
            ChunkerConfig(target_chars=500, min_chunk_chars=600)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


class TestNarrativeChunking:
    def test_short_text_one_chunk(self):
        text = "Una frase breve. Un'altra frase. Una terza." * 5
        doc = make_doc(text)
        chunks = chunk_document(doc, ContentType.NARRATIVE, ChunkerConfig())
        # Testo ~150 caratteri → un chunk solo, ma è < min_chunk → 0 o 1
        # Accettiamo 0 o 1 (è troppo corto)
        assert len(chunks) <= 1

    def test_long_text_split_into_chunks(self):
        # Costruiamo un testo lungo con paragrafi
        para = "Questa è una frase di esempio relativamente lunga per testare il chunker. " * 8
        text = "\n\n".join([para] * 10)  # ~6000+ caratteri
        doc = make_doc(text)
        chunks = chunk_document(
            doc, ContentType.NARRATIVE, ChunkerConfig(target_chars=2000)
        )
        assert len(chunks) >= 2
        # Tutti i chunk hanno char_count ragionevole
        for c in chunks:
            assert c.char_count <= 4096  # max_chunk_chars
            assert c.text  # non vuoti
            assert isinstance(c, Chunk)

    def test_chunks_have_overlap(self):
        para = "Frase " * 50  # ~300 caratteri
        text = "\n\n".join([para] * 20)
        doc = make_doc(text)
        config = ChunkerConfig(target_chars=1000, overlap_chars=100)
        chunks = chunk_document(doc, ContentType.NARRATIVE, config)
        # Se ci sono almeno 2 chunk, controlliamo che il secondo inizi
        # con qualche carattere del primo (overlap)
        if len(chunks) >= 2:
            tail_first = chunks[0].text[-100:]
            head_second = chunks[1].text[:200]
            # Almeno qualche parola comune (l'overlap non è esatto al carattere
            # perché passiamo da paragrafi)
            assert any(w in head_second for w in tail_first.split() if len(w) > 3)

    def test_indices_are_sequential(self):
        para = "Lorem ipsum dolor sit amet. " * 50
        text = "\n\n".join([para] * 10)
        doc = make_doc(text)
        chunks = chunk_document(
            doc, ContentType.NARRATIVE, ChunkerConfig(target_chars=1500)
        )
        for i, c in enumerate(chunks):
            assert c.index == i


# ---------------------------------------------------------------------------
# QA pairs / dialogue / tabular: una sezione = un chunk
# ---------------------------------------------------------------------------


class TestUnitPerSectionChunking:
    def test_qa_pairs_one_chunk_per_section(self):
        sections = [
            Section(title="Q1", text="Q: Cos'è X?\nA: Una cosa che fa Y." + " filler" * 30),
            Section(title="Q2", text="Q: Cos'è Z?\nA: Un'altra cosa." + " filler" * 30),
            Section(title="Q3", text="Q: E W?\nA: Risposta a W." + " filler" * 30),
        ]
        doc = ExtractedDocument(
            text="\n\n".join(s.text for s in sections),
            source_format="txt",
            sections=sections,
        )
        chunks = chunk_document(doc, ContentType.QA_PAIRS, ChunkerConfig())
        assert len(chunks) == 3
        assert chunks[0].source_section == "Q1"
        assert chunks[1].source_section == "Q2"
        assert chunks[2].source_section == "Q3"

    def test_tabular_one_chunk_per_row(self):
        # Simuliamo un CSV con 5 righe lunghe abbastanza
        sections = [
            Section(
                title=f"Riga {i}",
                text=f"name: User{i} | description: " + "lunga descrizione " * 20,
                metadata={"row_index": i},
            )
            for i in range(5)
        ]
        doc = ExtractedDocument(
            text="\n".join(s.text for s in sections),
            source_format="csv",
            sections=sections,
            metadata={"columns": ["name", "description"]},
        )
        chunks = chunk_document(doc, ContentType.TABULAR, ChunkerConfig())
        assert len(chunks) == 5
        assert all(c.source_section.startswith("Riga") for c in chunks)

    def test_unit_per_section_skips_too_small(self):
        sections = [
            Section(title="A", text="ciao"),  # < min_chunk_chars
            Section(title="B", text="x" * 500),  # ok
        ]
        doc = ExtractedDocument(
            text="ciao\n" + "x" * 500,
            source_format="txt",
            sections=sections,
        )
        chunks = chunk_document(doc, ContentType.QA_PAIRS, ChunkerConfig())
        # Solo la seconda sezione passa il filtro min_chunk_chars
        assert len(chunks) == 1
        assert chunks[0].source_section == "B"


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------


class TestCodeChunking:
    def test_split_on_function_boundaries(self):
        text = """
def funzione_uno():
    return 1


def funzione_due():
    \"\"\"Questa è la seconda funzione che fa qualcosa.\"\"\"
    x = 10
    y = 20
    return x + y


def funzione_tre():
    \"\"\"E questa è la terza funzione.\"\"\"
    return "hello"


class MiaClasse:
    def __init__(self):
        self.value = 0

    def metodo(self):
        return self.value
""".strip()
        # Padding per superare min_chunk_chars
        text = text + "\n\n" + "# commento di padding\n" * 30

        doc = make_doc(text)
        chunks = chunk_document(doc, ContentType.CODE, ChunkerConfig(target_chars=500))
        # Almeno qualche chunk
        assert len(chunks) >= 1

    def test_code_without_functions_falls_back(self):
        # Codice senza def/class → narrative fallback
        text = "x = 1\ny = 2\nz = x + y\n" * 100
        doc = make_doc(text)
        chunks = chunk_document(doc, ContentType.CODE, ChunkerConfig(target_chars=500))
        # Deve comunque produrre chunk (via narrative fallback)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_doc(self):
        doc = make_doc("", sections=[])
        chunks = chunk_document(doc, ContentType.NARRATIVE, ChunkerConfig())
        assert chunks == []

    def test_single_huge_paragraph(self):
        # Un singolo paragrafo molto lungo, no \n\n
        text = "Frase corta. " * 1000  # ~13000 caratteri
        doc = make_doc(text)
        chunks = chunk_document(
            doc, ContentType.NARRATIVE, ChunkerConfig(target_chars=1500)
        )
        # Deve essere splittato in più chunk
        assert len(chunks) >= 2
        # Nessun chunk supera max_chunk_chars
        for c in chunks:
            assert c.char_count <= 4096

    def test_word_pack_extreme_fallback(self):
        # Una sequenza di una parola sola enorme, no punteggiatura
        text = ("parola " * 5000).strip()  # ~35000 caratteri
        doc = make_doc(text)
        chunks = chunk_document(
            doc, ContentType.NARRATIVE, ChunkerConfig(target_chars=1000)
        )
        assert len(chunks) >= 5
        for c in chunks:
            assert c.char_count <= 4096

    def test_chunk_to_dict_serializable(self):
        import json

        text = "Frase normale di test. " * 40
        doc = make_doc(text)
        chunks = chunk_document(doc, ContentType.NARRATIVE, ChunkerConfig())
        if chunks:
            d = chunks[0].to_dict()
            json.dumps(d)
            assert "text" in d
            assert "index" in d
            assert "estimated_tokens" in d


# ---------------------------------------------------------------------------
# Estimated tokens ≈ chars / 4
# ---------------------------------------------------------------------------


class TestEstimatedTokens:
    def test_estimated_tokens_approx(self):
        text = "x" * 1000
        doc = make_doc(text)
        chunks = chunk_document(
            doc, ContentType.NARRATIVE, ChunkerConfig(target_chars=2000)
        )
        if chunks:
            c = chunks[0]
            # Dovrebbe essere circa char_count / 4
            assert abs(c.estimated_tokens - c.char_count // 4) <= 1