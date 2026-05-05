"""
Test del Content Detector.

Strategia: costruiamo ExtractedDocument finti (senza file su disco) con
testi controllati e verifichiamo che la detection ritorni il tipo atteso.
"""

from __future__ import annotations

import pytest

from core.dataset.detector import (
    ContentType,
    DetectionResult,
    detect_content_type,
)
from core.dataset.extracted import ExtractedDocument, Section


def make_doc(text: str, source_format: str = "txt", **kwargs) -> ExtractedDocument:
    """Helper per creare un ExtractedDocument finto."""
    return ExtractedDocument(
        text=text,
        source_format=source_format,
        sections=kwargs.get("sections", [Section(title=None, text=text)]),
        metadata=kwargs.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# QA pairs
# ---------------------------------------------------------------------------


class TestDetectQAPairs:
    def test_explicit_q_a_prefixes(self):
        text = """
Q: Qual è la capitale della Francia?
A: Parigi.

Q: Quanti pianeti ha il sistema solare?
A: Otto, dopo la riclassificazione di Plutone.

Q: Chi ha scritto la Divina Commedia?
A: Dante Alighieri.
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.QA_PAIRS

    def test_italian_prefixes(self):
        text = """
Domanda: Cos'è la fotosintesi?
Risposta: Il processo con cui le piante producono glucosio dalla luce solare.

Domanda: Cosa fa la mitocondri?
Risposta: Produce energia per la cellula.

Domanda: A cosa serve il DNA?
Risposta: Contiene le informazioni genetiche dell'organismo.
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.QA_PAIRS

    def test_csv_with_qa_columns(self):
        doc = ExtractedDocument(
            text="question: hello | answer: hi\nquestion: how are you | answer: fine",
            source_format="csv",
            metadata={"columns": ["question", "answer"]},
        )
        result = detect_content_type(doc)
        assert result.content_type == ContentType.QA_PAIRS

    def test_json_with_instruction_keys(self):
        doc = ExtractedDocument(
            text="instruction: do X\noutput: result",
            source_format="json",
            sections=[
                Section(
                    title="Item 1",
                    text="instruction: ...",
                    metadata={"raw_keys": ["instruction", "input", "output"]},
                ),
                Section(
                    title="Item 2",
                    text="instruction: ...",
                    metadata={"raw_keys": ["instruction", "output"]},
                ),
            ],
        )
        result = detect_content_type(doc)
        assert result.content_type == ContentType.QA_PAIRS


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


class TestDetectCode:
    def test_python_code_with_fences(self):
        text = """
Ecco un esempio:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self
```
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.CODE

    def test_raw_python_code(self):
        text = """
import os
from pathlib import Path

def list_files(directory):
    for path in Path(directory).iterdir():
        if path.is_file():
            yield path.name

class FileProcessor:
    def __init__(self, root):
        self.root = root
    
    def process(self):
        return list(list_files(self.root))
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.CODE


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------


class TestDetectDialogue:
    def test_chat_dialogue(self):
        text = """
Alice: Hey, hai visto il film ieri sera?
Bob: Sì, era fantastico! Ti è piaciuto?
Alice: Moltissimo. Specialmente la scena finale.
Bob: Concordo, il regista ha fatto un lavoro eccellente.
Alice: Andiamo al prossimo insieme?
Bob: Volentieri, quando esce il prossimo?
Alice: La settimana prossima.
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.DIALOGUE

    def test_user_assistant_pattern(self):
        text = """
User: How do I install Python?
Assistant: Download it from python.org and run the installer.
User: What about pip?
Assistant: Pip comes bundled with Python 3.4 and later.
User: Great, thanks!
Assistant: You're welcome.
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.DIALOGUE

    def test_single_speaker_is_not_dialogue(self):
        """Un solo speaker che parla non basta per essere dialogo."""
        text = """
Mario: Lorem ipsum dolor sit amet.
Mario: Consectetur adipiscing elit.
Mario: Sed do eiusmod tempor incididunt.
"""
        result = detect_content_type(make_doc(text))
        assert result.content_type != ContentType.DIALOGUE


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------


class TestDetectTabular:
    def test_csv_without_qa_columns_is_tabular(self):
        doc = ExtractedDocument(
            text="name: Alice | age: 30\nname: Bob | age: 25",
            source_format="csv",
            metadata={"columns": ["name", "age", "city"]},
        )
        result = detect_content_type(doc)
        assert result.content_type == ContentType.TABULAR


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


class TestDetectNarrative:
    def test_book_excerpt(self):
        text = (
            "Il sole stava tramontando dietro le colline mentre Marco "
            "camminava lungo il sentiero che portava al villaggio. "
            "Aveva camminato per ore quel giorno, e le gambe gli "
            "facevano male, ma la determinazione lo spingeva ancora "
            "avanti. Nei suoi pensieri c'era solo la promessa fatta a "
            "sua madre prima di partire. Doveva trovare il vecchio "
            "stregone delle montagne e portargli il libro che aveva "
            "ereditato. Solo lui poteva spiegargli cosa significasse "
            "il simbolo inciso sulla copertina. Il vento iniziò a "
            "soffiare più forte mentre le prime stelle apparivano nel "
            "cielo viola della sera. Marco strinse la cintura del "
            "mantello e accelerò il passo."
        )
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.NARRATIVE

    def test_article_text(self):
        text = (
            "L'intelligenza artificiale generativa ha rivoluzionato il "
            "modo in cui interagiamo con la tecnologia. Negli ultimi "
            "anni abbiamo assistito a un'esplosione di applicazioni "
            "che vanno dalla creazione di contenuti alla generazione "
            "di codice. Tuttavia, queste tecnologie sollevano "
            "importanti questioni etiche. La trasparenza dei modelli "
            "rimane una sfida aperta. Molti ricercatori stanno "
            "lavorando su tecniche di interpretabilità per rendere "
            "i sistemi più comprensibili."
        )
        result = detect_content_type(make_doc(text))
        assert result.content_type == ContentType.NARRATIVE


# ---------------------------------------------------------------------------
# Mixed (fallback)
# ---------------------------------------------------------------------------


class TestDetectMixed:
    def test_empty_text_is_mixed(self):
        result = detect_content_type(make_doc(""))
        assert result.content_type == ContentType.MIXED

    def test_random_short_text_is_mixed(self):
        # Testo troppo corto per essere classificato con confidenza
        result = detect_content_type(make_doc("Ciao."))
        # Può essere mixed o narrative-low, ma confidence deve essere bassa
        assert result.confidence < 0.5


# ---------------------------------------------------------------------------
# DetectionResult helpers
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_to_dict_serializable(self):
        import json

        result = detect_content_type(
            make_doc("Q: test\nA: test\nQ: test2\nA: test2\nQ: test3\nA: test3")
        )
        d = result.to_dict()
        # Deve essere JSON serializzabile
        assert json.dumps(d)
        assert "content_type" in d
        assert "confidence" in d
        assert "scores" in d
        assert "indicators" in d

    def test_scores_contains_all_types(self):
        result = detect_content_type(make_doc("test text"))
        # Tutti i 6 tipi devono essere presenti negli scores
        for ct in ContentType:
            assert ct in result.scores