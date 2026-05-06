"""
Test del data layer del Training Engine.

Strategia: mockiamo il tokenizer con un fake che simula HF (BOS/EOS,
chat_template opzionale, tokenize semplice). Niente download di modelli.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.training.data import (
    IGNORE_INDEX,
    DataCollatorWithPadding,
    DataConfig,
    InstructionTuningDataset,
    format_prompt_alpaca,
    has_chat_template,
    load_alpaca_examples,
    tokenize_example,
)


# ---------------------------------------------------------------------------
# Fake tokenizer: simula l'API minima di HF
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """
    Tokenizer fake "word-based": ogni parola = un token, ID = hash stabile.
    Gestisce i token speciali (<EOS>, <|user|>, ecc.) come unità atomiche
    anche se attaccati ad altri caratteri, simulando il comportamento dei
    tokenizer reali (BPE/SentencePiece) sui special tokens.
    """

    SPECIAL_TOKENS = ("<EOS>", "<|user|>", "<|assistant|>", "<|end|>", "<PAD>")

    def __init__(self, with_chat_template: bool = True):
        self.pad_token_id = 0
        self.eos_token = "<EOS>"
        self.eos_token_id = 1
        self._vocab: dict[str, int] = {"<PAD>": 0, "<EOS>": 1}
        self._next_id = 2
        if with_chat_template:
            # Template molto semplice tipo ChatML
            self.chat_template = (
                "{% for m in messages %}"
                "<|user|>\n{{ m.content }}<|end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
            )
        else:
            self.chat_template = None

    def _tokenize_word(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = self._next_id
            self._next_id += 1
        return self._vocab[w]

    def _split_with_specials(self, text: str) -> list[str]:
        """
        Split che riconosce i SPECIAL_TOKENS come unità atomiche, anche
        se attaccati ad altri caratteri. Replica il comportamento HF
        sui special tokens.
        """
        # Inseriamo spazi attorno ai special tokens, poi facciamo split normale
        for special in self.SPECIAL_TOKENS:
            text = text.replace(special, f" {special} ")
        return [w for w in text.split() if w]

    def _tokenize_text(self, text: str) -> list[int]:
        return [self._tokenize_word(w) for w in self._split_with_specials(text)]

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        ids = self._tokenize_text(text)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Implementazione minima senza Jinja
        parts = []
        for m in messages:
            parts.append(f"<|user|>\n{m['content']}<|end|>\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class TestDataConfig:
    def test_defaults(self):
        c = DataConfig()
        assert c.max_seq_length == 1024
        assert c.train_on_response_only is True
        assert c.add_eos is True

    def test_invalid_max_seq(self):
        with pytest.raises(ValueError):
            DataConfig(max_seq_length=10)


# ---------------------------------------------------------------------------
# Loader JSONL
# ---------------------------------------------------------------------------


class TestLoadAlpacaExamples:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "data.jsonl"
        rows = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "ctx", "output": "A2"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        examples = load_alpaca_examples(f)
        assert len(examples) == 2
        assert examples[0]["instruction"] == "Q1"
        assert examples[1]["input"] == "ctx"

    def test_skip_malformed(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"instruction": "Q", "output": "A"}\n'
            "NOT JSON\n"
            '{"instruction": "", "output": "A"}\n'  # instruction vuota → skip
            '{"instruction": "Q3", "output": "A3"}\n',
            encoding="utf-8",
        )
        examples = load_alpaca_examples(f)
        assert len(examples) == 2  # Q e Q3
        assert examples[0]["instruction"] == "Q"
        assert examples[1]["instruction"] == "Q3"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_alpaca_examples(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Format prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def test_alpaca_no_input(self):
        p = format_prompt_alpaca("Cos'è X?", "")
        assert "### Instruction:\nCos'è X?" in p
        assert "### Input:" not in p
        assert p.endswith("### Response:\n")

    def test_alpaca_with_input(self):
        p = format_prompt_alpaca("Riassumi", "testo lungo")
        assert "### Instruction:\nRiassumi" in p
        assert "### Input:\ntesto lungo" in p
        assert p.endswith("### Response:\n")

    def test_has_chat_template_true(self):
        tok = FakeTokenizer(with_chat_template=True)
        assert has_chat_template(tok) is True

    def test_has_chat_template_false(self):
        tok = FakeTokenizer(with_chat_template=False)
        assert has_chat_template(tok) is False


# ---------------------------------------------------------------------------
# Tokenize singolo esempio
# ---------------------------------------------------------------------------


class TestTokenizeExample:
    def test_basic_tokenize_with_chat_template(self):
        tok = FakeTokenizer(with_chat_template=True)
        ex = {"instruction": "ciao", "input": "", "output": "salve mondo"}
        out = tokenize_example(tok, ex, DataConfig())

        assert "input_ids" in out
        assert "labels" in out
        assert len(out["input_ids"]) == len(out["labels"])

        # I primi token (prompt) sono mascherati a -100
        # gli ultimi (response + eos) NO
        assert IGNORE_INDEX in out["labels"]
        assert out["labels"][-1] != IGNORE_INDEX  # ultimo token è EOS o response

    def test_alpaca_fallback_when_no_chat_template(self):
        tok = FakeTokenizer(with_chat_template=False)
        ex = {"instruction": "ciao", "input": "", "output": "salve"}
        out = tokenize_example(tok, ex, DataConfig())
        assert "input_ids" in out
        assert len(out["input_ids"]) > 0

    def test_loss_masking_disabled(self):
        tok = FakeTokenizer()
        ex = {"instruction": "x", "input": "", "output": "y z"}
        config = DataConfig(train_on_response_only=False)
        out = tokenize_example(tok, ex, config)
        # Tutti i labels sono uguali agli input_ids (no masking)
        assert out["labels"] == out["input_ids"]

    def test_eos_appended_when_enabled(self):
        tok = FakeTokenizer()
        ex = {"instruction": "x", "input": "", "output": "y"}
        out = tokenize_example(tok, ex, DataConfig(add_eos=True))
        # L'ultimo token deve essere l'EOS
        assert out["input_ids"][-1] == tok.eos_token_id

    def test_truncation_respects_max_seq(self):
        tok = FakeTokenizer()
        # Output molto lungo
        long_output = " ".join(f"word{i}" for i in range(500))
        ex = {"instruction": "x", "input": "", "output": long_output}
        out = tokenize_example(tok, ex, DataConfig(max_seq_length=64))
        assert len(out["input_ids"]) <= 64
        assert len(out["labels"]) == len(out["input_ids"])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class TestInstructionTuningDataset:
    @pytest.fixture
    def jsonl_path(self, tmp_path):
        f = tmp_path / "data.jsonl"
        rows = [
            {"instruction": f"Q{i}", "input": "", "output": f"A{i} risposta"}
            for i in range(5)
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return f

    def test_len(self, jsonl_path):
        tok = FakeTokenizer()
        ds = InstructionTuningDataset(jsonl_path, tok)
        assert len(ds) == 5

    def test_getitem_returns_tokenized(self, jsonl_path):
        tok = FakeTokenizer()
        ds = InstructionTuningDataset(jsonl_path, tok)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], list)

    def test_caching(self, jsonl_path):
        tok = FakeTokenizer()
        ds = InstructionTuningDataset(jsonl_path, tok)
        first = ds[0]
        second = ds[0]
        # Stessa istanza in cache
        assert first is second

    def test_stats(self, jsonl_path):
        tok = FakeTokenizer()
        ds = InstructionTuningDataset(jsonl_path, tok)
        stats = ds.stats()
        assert stats["num_examples"] == 5
        assert stats["tokens_min"] >= 1
        assert stats["tokens_max"] >= stats["tokens_min"]
        assert stats["tokens_total"] > 0

    def test_empty_dataset_raises(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        tok = FakeTokenizer()
        with pytest.raises(ValueError, match="vuoto"):
            InstructionTuningDataset(f, tok)


# ---------------------------------------------------------------------------
# DataCollator
# ---------------------------------------------------------------------------


class TestDataCollator:
    def test_pads_to_max_in_batch(self):
        tok = FakeTokenizer()
        collator = DataCollatorWithPadding(tokenizer=tok)
        batch = [
            {"input_ids": [10, 20, 30], "labels": [10, 20, 30]},
            {"input_ids": [40, 50], "labels": [40, 50]},
            {"input_ids": [60, 70, 80, 90], "labels": [60, 70, 80, 90]},
        ]
        out = collator(batch)

        assert out["input_ids"].shape == (3, 4)
        assert out["labels"].shape == (3, 4)
        assert out["attention_mask"].shape == (3, 4)

        # Pad token is 0, IGNORE_INDEX = -100
        # Riga 0: [10, 20, 30, 0]      | labels [10, 20, 30, -100] | mask [1,1,1,0]
        assert out["input_ids"][0, 3].item() == 0
        assert out["labels"][0, 3].item() == IGNORE_INDEX
        assert out["attention_mask"][0, 3].item() == 0

    def test_pad_to_multiple_of(self):
        tok = FakeTokenizer()
        collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
        batch = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        ]
        out = collator(batch)
        # max_len=3 → padded a 8
        assert out["input_ids"].shape == (1, 8)

    def test_returns_torch_tensors(self):
        tok = FakeTokenizer()
        collator = DataCollatorWithPadding(tokenizer=tok)
        batch = [
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5]},
        ]
        out = collator(batch)
        assert isinstance(out["input_ids"], torch.Tensor)
        assert out["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# End-to-end: dataset → collator → batch
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_dataset_to_dataloader_shape(self, tmp_path):
        from torch.utils.data import DataLoader

        f = tmp_path / "data.jsonl"
        rows = [
            {"instruction": f"Q{i}", "input": "", "output": f"A{i}"}
            for i in range(8)
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        tok = FakeTokenizer()
        ds = InstructionTuningDataset(f, tok)
        collator = DataCollatorWithPadding(tokenizer=tok)
        loader = DataLoader(ds, batch_size=4, collate_fn=collator)

        batches = list(loader)
        assert len(batches) == 2  # 8 / 4

        first = batches[0]
        assert first["input_ids"].shape[0] == 4
        # tutte le righe hanno la stessa lunghezza (paddata a max nel batch)
        assert first["input_ids"].shape == first["labels"].shape
        assert first["input_ids"].shape == first["attention_mask"].shape