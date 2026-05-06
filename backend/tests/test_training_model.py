"""
Test del model loader (training).

Strategia di test:
  - Tutti i test sono mockati: non scarichiamo né carichiamo modelli reali
  - Per `find_all_linear_names` usiamo un torch.nn.Module finto
  - Per le funzioni di alto livello mockiamo transformers e peft
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from core.training.model import (
    LoraConfigParams,
    ModelLoadError,
    QuantizationConfig,
    TARGET_MODULES_BY_FAMILY,
    count_trainable_parameters,
    find_all_linear_names,
    get_target_modules_for_family,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Configurazione validation
# ---------------------------------------------------------------------------


class TestLoraConfigParams:
    def test_defaults(self):
        c = LoraConfigParams()
        assert c.r == 16
        assert c.alpha == 32
        assert c.dropout == 0.05
        assert c.bias == "none"
        assert c.target_modules is None

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            LoraConfigParams(r=0)
        with pytest.raises(ValueError):
            LoraConfigParams(r=300)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            LoraConfigParams(alpha=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            LoraConfigParams(dropout=1.0)
        with pytest.raises(ValueError):
            LoraConfigParams(dropout=-0.1)

    def test_invalid_bias(self):
        with pytest.raises(ValueError):
            LoraConfigParams(bias="random")


class TestQuantizationConfig:
    def test_defaults(self):
        c = QuantizationConfig()
        assert c.load_in_4bit is True
        assert c.bnb_4bit_quant_type == "nf4"
        assert c.bnb_4bit_compute_dtype == "bfloat16"

    def test_invalid_quant_type(self):
        with pytest.raises(ValueError):
            QuantizationConfig(bnb_4bit_quant_type="int8")

    def test_invalid_compute_dtype(self):
        with pytest.raises(ValueError):
            QuantizationConfig(bnb_4bit_compute_dtype="float32")


# ---------------------------------------------------------------------------
# Whitelist target modules
# ---------------------------------------------------------------------------


class TestTargetModulesByFamily:
    def test_whitelist_has_all_expected_families(self):
        for family in (
            "qwen2.5", "phi3.5", "phi2", "smollm2", "smollm3", "mistral", "tinyllama"
        ):
            assert family in TARGET_MODULES_BY_FAMILY
            assert len(TARGET_MODULES_BY_FAMILY[family]) >= 2

    def test_get_target_modules_known(self):
        assert get_target_modules_for_family("qwen2.5") == [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]

    def test_get_target_modules_unknown(self):
        assert get_target_modules_for_family("unknown") is None
        assert get_target_modules_for_family(None) is None
        assert get_target_modules_for_family("") is None

    def test_case_insensitive(self):
        assert get_target_modules_for_family("QWEN2.5") is not None
        assert get_target_modules_for_family("Qwen2.5") is not None


# ---------------------------------------------------------------------------
# find_all_linear_names: usa un mini-modello PyTorch finto
# ---------------------------------------------------------------------------


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = nn.Linear(64, 256)


class FakeModel(nn.Module):
    """Modello finto con struttura simile a un transformer."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, 64)  # da escludere
        self.layers = nn.ModuleList([FakeBlock() for _ in range(n_layers)])
        self.lm_head = nn.Linear(64, 1000)  # da escludere


class TestFindAllLinearNames:
    def test_finds_attention_projs(self):
        model = FakeModel(n_layers=2)
        names = find_all_linear_names(model)
        # Si aspetta i 4 proj + il "mlp" generico
        assert "q_proj" in names
        assert "k_proj" in names
        assert "v_proj" in names
        assert "o_proj" in names

    def test_excludes_lm_head_and_embeddings(self):
        model = FakeModel(n_layers=2)
        names = find_all_linear_names(model)
        assert "lm_head" not in names
        assert "embed_tokens" not in names

    def test_returns_sorted_unique(self):
        model = FakeModel(n_layers=3)
        names = find_all_linear_names(model)
        assert names == sorted(names)
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# count_trainable_parameters
# ---------------------------------------------------------------------------


class TestCountTrainableParameters:
    def test_all_trainable(self):
        model = FakeModel(n_layers=1)
        info = count_trainable_parameters(model)
        # Tutti i parametri sono trainable di default
        assert info.trainable_params == info.total_params
        assert info.trainable_percent == 100.0

    def test_none_trainable(self):
        model = FakeModel(n_layers=1)
        for p in model.parameters():
            p.requires_grad = False
        info = count_trainable_parameters(model)
        assert info.trainable_params == 0
        assert info.trainable_percent == 0.0

    def test_partial_trainable(self):
        model = FakeModel(n_layers=2)
        # Freezo tutto tranne q_proj del primo layer
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layers[0].self_attn.q_proj.parameters():
            p.requires_grad = True

        info = count_trainable_parameters(model)
        assert info.trainable_params > 0
        assert info.trainable_params < info.total_params
        assert 0.0 < info.trainable_percent < 100.0

    def test_to_dict_serializable(self):
        import json

        model = FakeModel(n_layers=1)
        info = count_trainable_parameters(model)
        d = info.to_dict()
        json.dumps(d)  # deve essere JSON-safe
        assert "trainable_params" in d
        assert "total_params" in d


# ---------------------------------------------------------------------------
# load_tokenizer: mockiamo AutoTokenizer
# ---------------------------------------------------------------------------


class TestLoadTokenizer:
    def test_path_not_exists_raises(self, tmp_path):
        with pytest.raises(ModelLoadError, match="non esiste"):
            load_tokenizer(tmp_path / "missing")

    def test_sets_pad_token_when_missing(self, tmp_path, monkeypatch):
        # Path finto che esiste
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        fake_tok = MagicMock()
        fake_tok.pad_token_id = None
        fake_tok.eos_token = "<EOS>"
        fake_tok.eos_token_id = 1

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tok,
        ):
            result = load_tokenizer(model_dir)

        # Dopo load_tokenizer, pad_token deve essere settato a eos
        assert result.pad_token == "<EOS>"

    def test_keeps_pad_token_if_present(self, tmp_path):
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        fake_tok = MagicMock()
        fake_tok.pad_token_id = 0
        fake_tok.eos_token = "<EOS>"
        fake_tok.eos_token_id = 1

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tok,
        ):
            result = load_tokenizer(model_dir)

        # pad_token NON deve essere riscritto (era già settato)
        # MagicMock non triggera __setattr__ se non assegnamo
        assert result is fake_tok

    def test_no_pad_no_eos_raises(self, tmp_path):
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        fake_tok = MagicMock()
        fake_tok.pad_token_id = None
        fake_tok.eos_token_id = None

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tok,
        ):
            with pytest.raises(ModelLoadError, match="pad_token_id"):
                load_tokenizer(model_dir)

    def test_load_failure_wrapped(self, tmp_path):
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ModelLoadError, match="Impossibile caricare il tokenizer"):
                load_tokenizer(model_dir)