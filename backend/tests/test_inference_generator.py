"""
Test del generator (mockati, no GPU).

Strategia: mock di `model.generate()` e tokenizer per verificare il flusso
senza torch.cuda. Verifichiamo: format_prompt, slicing del prompt, finish_reason,
calcolo throughput.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from core.inference.generator import (
    GenerationParams,
    GenerationResult,
    _determine_finish_reason,
    _format_prompt,
    _has_chat_template,
    generate_text,
)


# ---------------------------------------------------------------------------
# Helper: fake model + tokenizer
# ---------------------------------------------------------------------------


def make_fake_tokenizer(with_chat_template: bool = True):
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.pad_token_id = 0

    def fake_call(text, return_tensors=None, add_special_tokens=False):
        # Tokenizza splittando per spazi (simile a M4 fake tokenizer)
        ids = [10 + i for i, _ in enumerate(text.split())]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    tok.side_effect = fake_call

    def fake_decode(token_ids, skip_special_tokens=True):
        return f"<decoded {len(token_ids)} tokens>"

    tok.decode = fake_decode

    if with_chat_template:
        tok.chat_template = "<chat-template>"
        def fake_apply(messages, tokenize=False, add_generation_prompt=False):
            return f"<|user|>{messages[0]['content']}<|assistant|>"
        tok.apply_chat_template = fake_apply
    else:
        tok.chat_template = None
        tok.apply_chat_template = MagicMock(side_effect=AttributeError)

    return tok


def make_fake_model(generated_token_count: int = 5):
    """
    Modello fake: parameters() restituisce un Tensor su CPU.
    generate() restituisce input + N token nuovi finiti con eos.
    """
    model = MagicMock()
    fake_param = torch.zeros(1)
    model.parameters.return_value = iter([fake_param])

    def fake_generate(input_ids=None, attention_mask=None, **kwargs):
        # Prendi input + appendi N token nuovi (terminanti con eos=2)
        n = generated_token_count
        new_tokens = list(range(100, 100 + n - 1)) + [2]  # ultimo è eos
        full = torch.cat([input_ids[0], torch.tensor(new_tokens)])
        return full.unsqueeze(0)

    model.generate.side_effect = fake_generate
    return model


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


class TestGenerationParams:
    def test_defaults(self):
        p = GenerationParams()
        assert p.max_new_tokens == 256
        assert p.temperature == 0.7
        assert p.do_sample is True

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError):
            GenerationParams(max_new_tokens=0)
        with pytest.raises(ValueError):
            GenerationParams(max_new_tokens=10_000)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            GenerationParams(temperature=0)
        with pytest.raises(ValueError):
            GenerationParams(temperature=10)

    def test_invalid_top_p(self):
        with pytest.raises(ValueError):
            GenerationParams(top_p=0)
        with pytest.raises(ValueError):
            GenerationParams(top_p=1.1)

    def test_invalid_repetition_penalty(self):
        with pytest.raises(ValueError):
            GenerationParams(repetition_penalty=0.5)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def test_with_chat_template(self):
        tok = make_fake_tokenizer(with_chat_template=True)
        out = _format_prompt(tok, "ciao")
        assert "<|user|>ciao<|assistant|>" in out

    def test_without_chat_template(self):
        tok = make_fake_tokenizer(with_chat_template=False)
        out = _format_prompt(tok, "ciao")
        # Senza chat template ritorna il prompt grezzo
        assert out == "ciao"

    def test_has_chat_template_helper(self):
        assert _has_chat_template(make_fake_tokenizer(True)) is True
        assert _has_chat_template(make_fake_tokenizer(False)) is False


# ---------------------------------------------------------------------------
# Finish reason
# ---------------------------------------------------------------------------


class TestFinishReason:
    def test_eos(self):
        # output ends with eos_token_id=2
        out = torch.tensor([1, 2, 3, 4, 2])
        reason = _determine_finish_reason(out, input_length=2, eos_token_id=2, max_new_tokens=10)
        assert reason == "eos"

    def test_length(self):
        # max_new_tokens=3, generated 3
        out = torch.tensor([1, 2, 3, 4, 5])
        reason = _determine_finish_reason(out, input_length=2, eos_token_id=99, max_new_tokens=3)
        assert reason == "length"

    def test_unknown(self):
        out = torch.tensor([1, 2, 3])
        reason = _determine_finish_reason(out, input_length=2, eos_token_id=99, max_new_tokens=10)
        assert reason == "unknown"


# ---------------------------------------------------------------------------
# generate_text end-to-end
# ---------------------------------------------------------------------------


class TestGenerateText:
    def test_basic_generation(self):
        tok = make_fake_tokenizer(with_chat_template=True)
        model = make_fake_model(generated_token_count=5)

        result = generate_text(
            model=model,
            tokenizer=tok,
            prompt="ciao mondo",
            params=GenerationParams(max_new_tokens=10),
        )

        assert isinstance(result, GenerationResult)
        assert result.tokens_generated == 5
        assert result.text.startswith("<decoded")
        assert result.elapsed_seconds > 0
        assert result.throughput_tokens_per_sec > 0
        # Termina con eos → finish=eos
        assert result.finish_reason == "eos"

    def test_finish_length(self):
        tok = make_fake_tokenizer(with_chat_template=True)
        # Genera 5 token, ma max_new_tokens=5 → finish=length
        # NB: il fake termina sempre con eos, quindi questo test è approssimativo
        # Mockiamo manualmente
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])

        def gen(input_ids=None, attention_mask=None, **kwargs):
            n = 5
            new_tokens = list(range(100, 100 + n))  # NO eos
            full = torch.cat([input_ids[0], torch.tensor(new_tokens)])
            return full.unsqueeze(0)

        model.generate.side_effect = gen

        result = generate_text(
            model=model, tokenizer=tok,
            prompt="ciao",
            params=GenerationParams(max_new_tokens=5),
        )
        assert result.finish_reason == "length"

    def test_to_dict_serializable(self):
        import json
        tok = make_fake_tokenizer()
        model = make_fake_model()
        result = generate_text(model=model, tokenizer=tok, prompt="x")
        d = result.to_dict()
        json.dumps(d)


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_structure(self):
        r = GenerationResult(
            text="hello",
            formatted_prompt="<|user|>...",
            tokens_generated=10,
            elapsed_seconds=1.5,
            throughput_tokens_per_sec=6.66,
            finish_reason="eos",
        )
        d = r.to_dict()
        assert "text" in d
        assert d["tokens_generated"] == 10