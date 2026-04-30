"""
Test smoke per i modelli ORM SQLAlchemy.

Verifica che:
  - le tabelle si creino correttamente
  - i record si inseriscano
  - le relazioni funzionino (FK, cascade)
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.models import BaseModel, Dataset, FineTunedModel, TrainingRun


@pytest.fixture
def session():
    """Engine SQLite in-memory per ogni test (isolato)."""
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


# ---------------------------------------------------------------------------
# Tabelle create
# ---------------------------------------------------------------------------


def test_all_tables_present(session):
    """Tutte e 4 le tabelle previste devono esistere."""
    expected = {"base_models", "datasets", "finetuned_models", "training_runs"}
    assert expected.issubset(set(Base.metadata.tables.keys()))


# ---------------------------------------------------------------------------
# CRUD base
# ---------------------------------------------------------------------------


def test_insert_base_model(session):
    bm = BaseModel(
        hf_repo="Qwen/Qwen2.5-0.5B",
        display_name="Qwen 2.5 0.5B",
        local_path="/tmp/Qwen--Qwen2.5-0.5B",
        size_bytes=1_000_000_000,
        params_billions=0.5,
        tag="qwen2.5",
    )
    session.add(bm)
    session.commit()

    assert bm.id is not None
    assert bm.is_custom is False  # default
    assert isinstance(bm.downloaded_at, datetime)


def test_unique_hf_repo_constraint(session):
    """hf_repo è unique: non si può inserire due volte lo stesso repo."""
    bm1 = BaseModel(
        hf_repo="Qwen/Qwen2.5-0.5B",
        display_name="Q1",
        local_path="/tmp/a",
    )
    bm2 = BaseModel(
        hf_repo="Qwen/Qwen2.5-0.5B",
        display_name="Q2",
        local_path="/tmp/b",
    )
    session.add(bm1)
    session.commit()
    session.add(bm2)
    with pytest.raises(Exception):  # IntegrityError
        session.commit()
    session.rollback()


def test_training_run_with_relationships(session):
    """Un TrainingRun referenzia BaseModel (required) e Dataset (optional)."""
    bm = BaseModel(
        hf_repo="microsoft/phi-2",
        display_name="Phi-2",
        local_path="/tmp/phi-2",
    )
    ds = Dataset(
        name="test_dataset",
        file_path="/tmp/ds.json",
        num_examples=100,
    )
    session.add_all([bm, ds])
    session.commit()

    run = TrainingRun(
        base_model_id=bm.id,
        dataset_id=ds.id,
        status="pending",
        config_json='{"batch_size": 4}',
    )
    session.add(run)
    session.commit()

    assert run.base_model.hf_repo == "microsoft/phi-2"
    assert run.dataset.name == "test_dataset"
    assert run.status == "pending"


def test_cascade_delete_base_model(session):
    """Cancellando un BaseModel, i suoi TrainingRun spariscono (cascade)."""
    bm = BaseModel(
        hf_repo="microsoft/phi-2",
        display_name="Phi-2",
        local_path="/tmp/phi-2",
    )
    session.add(bm)
    session.commit()

    run = TrainingRun(
        base_model_id=bm.id,
        status="pending",
        config_json="{}",
    )
    session.add(run)
    session.commit()
    run_id = run.id

    session.delete(bm)
    session.commit()

    # Run deve essere stato cancellato a cascata
    assert session.get(TrainingRun, run_id) is None


def test_finetuned_model_unique_per_run(session):
    """Non si possono avere 2 FineTunedModel per lo stesso TrainingRun."""
    bm = BaseModel(
        hf_repo="microsoft/phi-2",
        display_name="Phi-2",
        local_path="/tmp/phi-2",
    )
    session.add(bm)
    session.commit()

    run = TrainingRun(
        base_model_id=bm.id,
        status="completed",
        config_json="{}",
    )
    session.add(run)
    session.commit()

    ft1 = FineTunedModel(
        base_model_id=bm.id,
        training_run_id=run.id,
        name="ft1",
        adapter_path="/tmp/ft1",
    )
    session.add(ft1)
    session.commit()

    ft2 = FineTunedModel(
        base_model_id=bm.id,
        training_run_id=run.id,
        name="ft2",
        adapter_path="/tmp/ft2",
    )
    session.add(ft2)
    with pytest.raises(Exception):  # IntegrityError per uq_finetuned_per_run
        session.commit()
    session.rollback()