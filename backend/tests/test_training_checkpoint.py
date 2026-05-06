"""
Test del modulo checkpoint.

Strategia: usiamo modelli/optimizer/scheduler reali ma minimali (torch puro).
Per il "modello PEFT-like" facciamo un fake che ha `save_pretrained` come
method, sufficiente per il test del save_checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from core.training.checkpoint import (
    CheckpointError,
    TrainerState,
    delete_run_directory,
    find_latest_checkpoint,
    get_checkpoint_dir,
    get_final_dir,
    get_run_dir,
    keep_only_last_n_checkpoints,
    list_checkpoints,
    load_optimizer_state,
    load_scheduler_state,
    load_trainer_state,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixture: forza adapters_path a una tmp directory
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_adapters_path(tmp_path, monkeypatch):
    """Forza settings.adapters_path a tmp_path/adapters per i test."""
    from config import settings

    adapters_dir = tmp_path / "adapters"
    monkeypatch.setattr(settings, "adapters_dir", str(adapters_dir))
    return adapters_dir


# ---------------------------------------------------------------------------
# Fake "PEFT-like" model + tokenizer
# ---------------------------------------------------------------------------


class FakePeftModel:
    """Imita PeftModel.save_pretrained scrivendo file finti."""

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.safetensors").write_bytes(b"FAKE_ADAPTER_BYTES")
        (p / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")


class FakeTokenizer:
    """Imita tokenizer.save_pretrained."""

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_text('{"model": "fake"}', encoding="utf-8")
        (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)


def make_optimizer_and_scheduler():
    model = MiniModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    return model, opt, sched


def make_state(step: int = 100, run_id: str = "run-test") -> TrainerState:
    return TrainerState(
        run_id=run_id,
        step=step,
        epoch=0,
        final_loss=1.234,
        history=[{"step": 1, "loss": 5.0}, {"step": step, "loss": 1.234}],
        base_model_path="/path/to/base",
        family_tag="qwen2.5",
    )


# ---------------------------------------------------------------------------
# TrainerState
# ---------------------------------------------------------------------------


class TestTrainerState:
    def test_to_dict_from_dict_roundtrip(self):
        s = make_state(step=50)
        d = s.to_dict()
        s2 = TrainerState.from_dict(d)
        assert s2.run_id == s.run_id
        assert s2.step == s.step
        assert s2.history == s.history

    def test_serializable_to_json(self):
        s = make_state()
        json.dumps(s.to_dict())  # non deve sollevare


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_get_run_dir(self, patched_adapters_path):
        p = get_run_dir("my-run-123")
        assert p.parent == patched_adapters_path
        assert p.name == "my-run-123"

    def test_get_checkpoint_dir(self, patched_adapters_path):
        p = get_checkpoint_dir("my-run-123", 100)
        assert p.name == "checkpoint-100"
        assert p.parent.name == "my-run-123"

    def test_get_final_dir(self, patched_adapters_path):
        p = get_final_dir("my-run-123")
        assert p.name == "final"

    def test_sanitize_run_id_invalid(self, patched_adapters_path):
        # Caratteri rischiosi vengono rimpiazzati
        p = get_run_dir("../escape/path")
        # Non si esce da adapters_path
        assert "escape" in p.name
        assert ".." not in str(p.name)

    def test_sanitize_empty_raises(self, patched_adapters_path):
        with pytest.raises(CheckpointError):
            get_run_dir("@@@")  # tutto rimpiazzato → vuoto


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_basic_intermediate(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-1")

        path = save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state, is_final=False,
        )

        assert path.exists()
        assert path.name == "checkpoint-100"
        assert (path / "adapter_model.safetensors").exists()
        assert (path / "adapter_config.json").exists()
        assert (path / "tokenizer.json").exists()
        assert (path / "optimizer.pt").exists()
        assert (path / "scheduler.pt").exists()
        assert (path / "trainer_state.json").exists()

    def test_final_no_optimizer_scheduler(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=200, run_id="run-2")

        path = save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state, is_final=True,
        )

        assert path.name == "final"
        # Nel final non salviamo optimizer/scheduler
        assert not (path / "optimizer.pt").exists()
        assert not (path / "scheduler.pt").exists()
        # Ma trainer_state.json sì
        assert (path / "trainer_state.json").exists()

    def test_overwrites_existing_dir(self, patched_adapters_path):
        """Se una checkpoint dir esiste già, la rimpiazza."""
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-3")

        # Crea manualmente la dir con un file estraneo
        target = get_checkpoint_dir("run-3", 100)
        target.mkdir(parents=True, exist_ok=True)
        (target / "stale.txt").write_text("old")

        save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state,
        )

        # Il file stale è stato rimosso
        assert not (target / "stale.txt").exists()
        # I nuovi file ci sono
        assert (target / "adapter_model.safetensors").exists()

    def test_save_failure_rolls_back(self, patched_adapters_path):
        """Se save fallisce, la dir parziale viene rimossa."""
        # Modello finto che esplode su save_pretrained
        bad_model = MagicMock()
        bad_model.save_pretrained.side_effect = RuntimeError("disk full")
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-4")

        with pytest.raises(CheckpointError):
            save_checkpoint(
                model=bad_model, tokenizer=tokenizer,
                optimizer=opt, scheduler=sched,
                trainer_state=state,
            )

        # La dir è stata rimossa
        target = get_checkpoint_dir("run-4", 100)
        assert not target.exists()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_trainer_state(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-load")

        path = save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state,
        )

        loaded = load_trainer_state(path)
        assert loaded.step == 100
        assert loaded.run_id == "run-load"
        assert loaded.final_loss == 1.234

    def test_load_trainer_state_missing(self, tmp_path):
        with pytest.raises(CheckpointError, match="non trovato"):
            load_trainer_state(tmp_path / "nope")

    def test_load_optimizer_state(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-opt-load")

        path = save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state,
        )

        # Crea un nuovo optimizer e carica lo stato
        _, new_opt, _ = make_optimizer_and_scheduler()
        load_optimizer_state(path, new_opt)
        # Sanity: deve avere param_groups
        assert len(new_opt.param_groups) >= 1

    def test_load_scheduler_state(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        state = make_state(step=100, run_id="run-sched-load")

        path = save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=state,
        )

        _, _, new_sched = make_optimizer_and_scheduler()
        load_scheduler_state(path, new_sched)
        # sched è caricato senza eccezioni
        assert new_sched is not None

    def test_load_optimizer_missing(self, tmp_path):
        _, opt, _ = make_optimizer_and_scheduler()
        with pytest.raises(CheckpointError, match="optimizer"):
            load_optimizer_state(tmp_path, opt)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:
    def test_list_empty(self, patched_adapters_path):
        assert list_checkpoints("nope") == []

    def test_list_sorted_by_step(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()

        for step in [300, 100, 200]:
            save_checkpoint(
                model=model, tokenizer=tokenizer,
                optimizer=opt, scheduler=sched,
                trainer_state=make_state(step=step, run_id="run-list"),
            )

        checkpoints = list_checkpoints("run-list")
        assert len(checkpoints) == 3
        # Ordinati per step crescente
        assert checkpoints[0].name == "checkpoint-100"
        assert checkpoints[1].name == "checkpoint-200"
        assert checkpoints[2].name == "checkpoint-300"

    def test_excludes_final(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()

        save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=make_state(step=100, run_id="run-final"),
        )
        save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=make_state(step=200, run_id="run-final"),
            is_final=True,
        )

        checkpoints = list_checkpoints("run-final")
        # Il final/ NON è nella lista
        assert len(checkpoints) == 1
        assert checkpoints[0].name == "checkpoint-100"

    def test_find_latest(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()

        for step in [100, 200, 300]:
            save_checkpoint(
                model=model, tokenizer=tokenizer,
                optimizer=opt, scheduler=sched,
                trainer_state=make_state(step=step, run_id="run-latest"),
            )

        latest = find_latest_checkpoint("run-latest")
        assert latest is not None
        assert latest.name == "checkpoint-300"

    def test_find_latest_empty(self, patched_adapters_path):
        assert find_latest_checkpoint("nope") is None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_delete_run_directory(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()
        save_checkpoint(
            model=model, tokenizer=tokenizer,
            optimizer=opt, scheduler=sched,
            trainer_state=make_state(step=100, run_id="run-del"),
        )

        run_dir = get_run_dir("run-del")
        assert run_dir.exists()

        result = delete_run_directory("run-del")
        assert result is True
        assert not run_dir.exists()

    def test_delete_nonexistent(self, patched_adapters_path):
        assert delete_run_directory("nope") is False

    def test_keep_only_last_n(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()

        for step in [100, 200, 300, 400, 500]:
            save_checkpoint(
                model=model, tokenizer=tokenizer,
                optimizer=opt, scheduler=sched,
                trainer_state=make_state(step=step, run_id="run-keep"),
            )

        removed = keep_only_last_n_checkpoints("run-keep", n=2)
        assert removed == 3

        remaining = list_checkpoints("run-keep")
        assert len(remaining) == 2
        assert remaining[0].name == "checkpoint-400"
        assert remaining[1].name == "checkpoint-500"

    def test_keep_n_no_op_if_few(self, patched_adapters_path):
        model = FakePeftModel()
        tokenizer = FakeTokenizer()
        _, opt, sched = make_optimizer_and_scheduler()

        for step in [100, 200]:
            save_checkpoint(
                model=model, tokenizer=tokenizer,
                optimizer=opt, scheduler=sched,
                trainer_state=make_state(step=step, run_id="run-keep-noop"),
            )

        removed = keep_only_last_n_checkpoints("run-keep-noop", n=5)
        assert removed == 0
        assert len(list_checkpoints("run-keep-noop")) == 2

    def test_keep_n_invalid(self, patched_adapters_path):
        with pytest.raises(ValueError):
            keep_only_last_n_checkpoints("any", n=0)