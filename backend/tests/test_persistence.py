"""
Test della persistenza dataset (DB + filesystem).
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.dataset.converter import InstructionExample
from core.dataset.persistence import (
    DatasetNameConflictError,
    DatasetNotFoundError,
    delete_dataset,
    get_dataset_by_id,
    list_datasets,
    load_dataset_examples,
    sanitize_name_for_dirname,
    save_dataset,
)
from core.dataset.validator import validate_examples
from db.database import Base


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
def patched_datasets_path(tmp_path, monkeypatch):
    """Forza datasets_path a tmp_path per i test."""
    from config import settings

    datasets_dir = tmp_path / "datasets"
    monkeypatch.setattr(settings, "datasets_dir", str(datasets_dir))
    return datasets_dir


@pytest.fixture
def sample_validated():
    """ValidatedDataset finto con 3 esempi."""
    examples = [
        InstructionExample(
            instruction=f"Domanda {i}",
            input="",
            output=f"Risposta lunga abbastanza per il filtro min {i}.",
            metadata={"strategy": "test"},
        )
        for i in range(3)
    ]
    return validate_examples(examples)


# ---------------------------------------------------------------------------
# Sanitize
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_basic(self):
        assert sanitize_name_for_dirname("Mio Dataset") == "mio-dataset"

    def test_special_chars(self):
        assert sanitize_name_for_dirname("Test! @#$ 123") == "test-123"

    def test_only_special(self):
        assert sanitize_name_for_dirname("@#$%") == "dataset"

    def test_strips_dashes(self):
        assert sanitize_name_for_dirname("---hello---") == "hello"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSaveDataset:
    def test_basic_save(self, session, patched_datasets_path, sample_validated):
        row = save_dataset(
            session,
            name="Mio test",
            validated=sample_validated,
        )

        assert row.id is not None
        assert row.name == "Mio test"
        assert row.num_examples == 3
        assert row.format == "alpaca"

        # Verifica file su disco
        from pathlib import Path

        jsonl_path = Path(row.file_path)
        assert jsonl_path.exists()
        meta_path = jsonl_path.parent / "metadata.json"
        assert meta_path.exists()

        # JSONL deve avere 3 righe
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        # Ogni riga è un JSON valido con i campi giusti
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "input" in obj
            assert "output" in obj

    def test_metadata_json_contents(
        self, session, patched_datasets_path, sample_validated
    ):
        row = save_dataset(
            session, name="Test", validated=sample_validated,
            extra_metadata={"converter_config": "default"},
        )
        from pathlib import Path

        meta_path = Path(row.file_path).parent / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["name"] == "Test"
        assert meta["num_examples"] == 3
        assert meta["format"] == "alpaca"
        assert "stats" in meta
        assert meta["extra"]["converter_config"] == "default"

    def test_duplicate_name_raises(
        self, session, patched_datasets_path, sample_validated
    ):
        save_dataset(session, name="Unico", validated=sample_validated)
        with pytest.raises(DatasetNameConflictError):
            save_dataset(session, name="Unico", validated=sample_validated)

    def test_empty_name_raises(
        self, session, patched_datasets_path, sample_validated
    ):
        with pytest.raises(Exception):  # DatasetPersistenceError
            save_dataset(session, name="", validated=sample_validated)


# ---------------------------------------------------------------------------
# List / Get
# ---------------------------------------------------------------------------


class TestListAndGet:
    def test_list_empty(self, session):
        assert list_datasets(session) == []

    def test_list_returns_recent_first(
        self, session, patched_datasets_path, sample_validated
    ):
        from datetime import datetime, timedelta, timezone

        from db.models import Dataset as DatasetRow
        from sqlalchemy import select

        save_dataset(session, name="Vecchio", validated=sample_validated)

        # Backdate
        old = session.scalar(
            select(DatasetRow).where(DatasetRow.name == "Vecchio")
        )
        old.created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        session.commit()

        save_dataset(session, name="Nuovo", validated=sample_validated)

        rows = list_datasets(session)
        assert len(rows) == 2
        assert rows[0].name == "Nuovo"
        assert rows[1].name == "Vecchio"

    def test_get_by_id_not_found(self, session):
        with pytest.raises(DatasetNotFoundError):
            get_dataset_by_id(session, 99999)


# ---------------------------------------------------------------------------
# Load examples
# ---------------------------------------------------------------------------


class TestLoadExamples:
    def test_load_all(self, session, patched_datasets_path, sample_validated):
        row = save_dataset(
            session, name="Test load", validated=sample_validated
        )
        examples = load_dataset_examples(session, row.id)
        assert len(examples) == 3
        assert all("instruction" in e for e in examples)

    def test_load_with_limit(
        self, session, patched_datasets_path, sample_validated
    ):
        row = save_dataset(
            session, name="Test limit", validated=sample_validated
        )
        examples = load_dataset_examples(session, row.id, limit=2)
        assert len(examples) == 2


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_record_and_files(
        self, session, patched_datasets_path, sample_validated
    ):
        from pathlib import Path

        row = save_dataset(
            session, name="Da cancellare", validated=sample_validated
        )
        dataset_dir = Path(row.file_path).parent
        assert dataset_dir.exists()

        delete_dataset(session, row.id, remove_files=True)

        assert not dataset_dir.exists()
        with pytest.raises(DatasetNotFoundError):
            get_dataset_by_id(session, row.id)

    def test_delete_record_only(
        self, session, patched_datasets_path, sample_validated
    ):
        from pathlib import Path

        row = save_dataset(
            session, name="Solo record", validated=sample_validated
        )
        dataset_dir = Path(row.file_path).parent

        delete_dataset(session, row.id, remove_files=False)

        # File preservati
        assert dataset_dir.exists()
        # Record sparito
        with pytest.raises(DatasetNotFoundError):
            get_dataset_by_id(session, row.id)

    def test_delete_not_found(self, session):
        with pytest.raises(DatasetNotFoundError):
            delete_dataset(session, 99999)