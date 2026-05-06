"""
Test del runner orchestratore.

Strategia: mockiamo `prepare_model_for_training` e `train_loop` così
non serve GPU/modelli reali. Verifichiamo che:
  - Lo stato DB venga aggiornato correttamente (pending → running → completed)
  - In caso di cancellazione, lo stato rifletta il fatto
  - In caso di errore, si crei un record failed
  - Il FineTunedModel venga creato solo se completed
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.training.loop import StepLog, TrainingResult
from core.training.model import LoadedModel, TrainableParamsInfo
from core.training.runner import (
    TrainingConfig,
    TrainingOutcome,
    run_training,
)
from db.database import Base
from db.models import (
    BaseModel as BaseModelRow,
    Dataset as DatasetRow,
    FineTunedModel as FineTunedModelRow,
    TrainingRun,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    sess = SessionFactory()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Forza adapters/models/datasets paths a tmp_path."""
    from config import settings

    adapters = tmp_path / "adapters"
    models = tmp_path / "models"
    datasets = tmp_path / "datasets"
    monkeypatch.setattr(settings, "adapters_dir", str(adapters))
    monkeypatch.setattr(settings, "models_dir", str(models))
    monkeypatch.setattr(settings, "datasets_dir", str(datasets))
    return tmp_path


@pytest.fixture
def fake_model_dir(patched_paths):
    """Crea una dir 'modello' fittizia su disco."""
    from config import settings

    d = settings.models_path / "fake--model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    return d


@pytest.fixture
def fake_dataset_file(patched_paths):
    """Crea un dataset.jsonl finto."""
    from config import settings

    d = settings.datasets_path / "fake-dataset"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "dataset.jsonl"
    rows = [
        {"instruction": f"Q{i}", "input": "", "output": f"A{i} risposta"}
        for i in range(5)
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f


@pytest.fixture
def setup_db_records(session, fake_model_dir, fake_dataset_file):
    """Crea base_model + dataset records nel DB."""
    base = BaseModelRow(
        hf_repo="fake/model",
        display_name="Fake Model",
        tag="qwen2.5",
        local_path=str(fake_model_dir),
        size_bytes=1000,
        params_billions=0.1,
    )
    session.add(base)

    ds = DatasetRow(
        name="Fake Dataset",
        file_path=str(fake_dataset_file),
        num_examples=5,
        format="alpaca",
    )
    session.add(ds)
    session.commit()
    session.refresh(base)
    session.refresh(ds)
    return base, ds


# ---------------------------------------------------------------------------
# Mock del pipeline pesante
# ---------------------------------------------------------------------------


def make_fake_loaded_model():
    """Crea un LoadedModel finto."""
    fake_model = MagicMock()
    fake_model.parameters.return_value = []
    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token_id = 0
    fake_tokenizer.eos_token = "<EOS>"
    fake_tokenizer.eos_token_id = 1

    return LoadedModel(
        model=fake_model,
        tokenizer=fake_tokenizer,
        trainable_info=TrainableParamsInfo(
            trainable_params=1000,
            total_params=1000000,
            trainable_percent=0.1,
        ),
    )


def make_fake_training_result(
    completed: bool = True,
    cancelled: bool = False,
    final_loss: float = 1.234,
    total_steps: int = 5,
) -> TrainingResult:
    history = [
        StepLog(
            step=i + 1,
            epoch=0,
            loss=2.0 - i * 0.1,
            learning_rate=2e-4,
            grad_norm=0.8,
            vram_used_mb=100,
            throughput_tokens_per_sec=500,
            elapsed_seconds=i * 2.0,
        )
        for i in range(total_steps)
    ]
    return TrainingResult(
        completed=completed,
        cancelled=cancelled,
        total_steps=total_steps,
        final_loss=final_loss,
        elapsed_seconds=total_steps * 2.0,
        history=history,
    )


# ---------------------------------------------------------------------------
# Validation pre-flight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_unknown_base_model_raises(self, session, patched_paths):
        config = TrainingConfig(base_model_id=999, dataset_id=1)
        with pytest.raises(ValueError, match="Base model"):
            run_training(session=session, config=config)

    def test_unknown_dataset_raises(self, session, setup_db_records):
        base, _ = setup_db_records
        config = TrainingConfig(base_model_id=base.id, dataset_id=999)
        with pytest.raises(ValueError, match="Dataset"):
            run_training(session=session, config=config)

    def test_missing_base_model_files_raises(self, session, patched_paths, fake_dataset_file):
        # Inseriamo un BaseModel con local_path inesistente
        base = BaseModelRow(
            hf_repo="ghost/model", display_name="Ghost",
            local_path=str(patched_paths / "nope"), size_bytes=0,
        )
        ds = DatasetRow(
            name="Ds", file_path=str(fake_dataset_file),
            num_examples=5, format="alpaca",
        )
        session.add_all([base, ds])
        session.commit()

        config = TrainingConfig(base_model_id=base.id, dataset_id=ds.id)
        with pytest.raises(ValueError, match="non presente su disco"):
            run_training(session=session, config=config)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_completed_run(
        self, session, setup_db_records, patched_paths
    ):
        base, ds = setup_db_records
        config = TrainingConfig(
            base_model_id=base.id, dataset_id=ds.id,
            num_epochs=1, save_every_n_steps=0,
        )

        with patch(
            "core.training.runner.prepare_model_for_training",
            return_value=make_fake_loaded_model(),
        ), patch(
            "core.training.runner.build_optimizer",
            return_value=MagicMock(),
        ), patch(
            "core.training.runner.build_scheduler",
            return_value=MagicMock(),
        ), patch(
            "core.training.runner.train_loop",
            return_value=make_fake_training_result(),
        ), patch(
            "core.training.runner.save_checkpoint",
            return_value=patched_paths / "fake-checkpoint",
        ):
            outcome = run_training(session=session, config=config)

        assert isinstance(outcome, TrainingOutcome)
        assert outcome.status == "completed"
        assert outcome.total_steps == 5
        assert outcome.final_loss == 1.234

        # Verifica record DB
        run_db = session.get(TrainingRun, outcome.training_run_db_id)
        assert run_db.status == "completed"
        assert run_db.started_at is not None
        assert run_db.finished_at is not None

        # FineTunedModel creato
        assert outcome.finetuned_model_id is not None
        ft = session.get(FineTunedModelRow, outcome.finetuned_model_id)
        assert ft is not None
        assert ft.training_run_id == outcome.training_run_db_id

    def test_cancelled_run_no_finetuned_model(
        self, session, setup_db_records, patched_paths
    ):
        """Se cancellato, NON viene creato il FineTunedModel."""
        base, ds = setup_db_records
        config = TrainingConfig(base_model_id=base.id, dataset_id=ds.id)

        with patch(
            "core.training.runner.prepare_model_for_training",
            return_value=make_fake_loaded_model(),
        ), patch(
            "core.training.runner.build_optimizer", return_value=MagicMock(),
        ), patch(
            "core.training.runner.build_scheduler", return_value=MagicMock(),
        ), patch(
            "core.training.runner.train_loop",
            return_value=make_fake_training_result(
                completed=False, cancelled=True, total_steps=2,
            ),
        ), patch(
            "core.training.runner.save_checkpoint",
            return_value=patched_paths / "fake-checkpoint",
        ):
            outcome = run_training(session=session, config=config)

        assert outcome.status == "cancelled"
        assert outcome.finetuned_model_id is None

        run_db = session.get(TrainingRun, outcome.training_run_db_id)
        assert run_db.status == "cancelled"


# ---------------------------------------------------------------------------
# Errori durante training
# ---------------------------------------------------------------------------


class TestErrorPath:
    def test_load_model_failure_marks_failed(
        self, session, setup_db_records
    ):
        base, ds = setup_db_records
        config = TrainingConfig(base_model_id=base.id, dataset_id=ds.id)

        with patch(
            "core.training.runner.prepare_model_for_training",
            side_effect=RuntimeError("CUDA out of memory"),
        ):
            outcome = run_training(session=session, config=config)

        assert outcome.status == "failed"
        assert outcome.error is not None
        assert "CUDA out of memory" in outcome.error

        run_db = session.get(TrainingRun, outcome.training_run_db_id)
        assert run_db.status == "failed"
        assert run_db.error_message is not None

        # Nessun FineTunedModel creato
        assert outcome.finetuned_model_id is None


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults_and_to_dict(self):
        c = TrainingConfig(base_model_id=1, dataset_id=2)
        assert c.num_epochs == 3
        assert c.lora_r == 16
        d = c.to_dict()
        json.dumps(d)  # JSON-safe