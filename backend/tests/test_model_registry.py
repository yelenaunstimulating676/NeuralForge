"""
Test per `core/model_registry.py`.

Copertura:
  - Validazione formato hf_repo
  - Sanitizzazione path
  - get_local_path
  - compute_directory_size (su tmp_path)
  - Whitelist queries
  - CRUD: register_model, list, get, delete
  - Cascade delete files
  - Errori: ModelNotFoundError, ModelAlreadyExistsError
  - validate_hf_repo_exists con HfApi mockato (no network)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from core.model_registry import (
    HFRepoNotAccessibleError,
    InvalidRepoFormatError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    WhitelistEntry,
    compute_directory_size,
    delete_model,
    find_in_whitelist,
    get_local_path,
    get_model_by_id,
    get_model_by_repo,
    get_whitelist,
    is_model_present,
    list_local_models,
    register_model,
    sanitize_repo_to_dirname,
    validate_repo_format,
)
from db.database import Base
from db.models import BaseModel as BaseModelRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    """SQLite in-memory isolato per ogni test."""
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
def fake_model_dir(tmp_path: Path) -> Path:
    """Crea una directory finta che simula un modello scaricato."""
    model_dir = tmp_path / "Qwen--Qwen2.5-0.5B"
    model_dir.mkdir()
    # File di varia dimensione per testare anche compute_directory_size
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    (model_dir / "tokenizer.json").write_bytes(b"x" * 1024)        # 1 KB
    (model_dir / "model.safetensors").write_bytes(b"y" * 2048)     # 2 KB
    sub = model_dir / "subdir"
    sub.mkdir()
    (sub / "extra.bin").write_bytes(b"z" * 512)                    # 0.5 KB
    return model_dir


# ---------------------------------------------------------------------------
# Validazione formato hf_repo
# ---------------------------------------------------------------------------


class TestValidateRepoFormat:
    @pytest.mark.parametrize(
        "repo",
        [
            "Qwen/Qwen2.5-0.5B",
            "microsoft/Phi-3.5-mini-instruct",
            "HuggingFaceTB/SmolLM2-135M",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "user/repo_v1.2.3",
            "a/b",
        ],
    )
    def test_valid(self, repo):
        assert validate_repo_format(repo) == repo

    @pytest.mark.parametrize(
        "repo",
        [
            "",
            "no-slash",
            "/leading-slash",
            "trailing-slash/",
            "double//slash",
            "two/slashes/here",
            "spaces in/repo",
            "user/-leading-dash",  # name non può iniziare con -
            "-user/repo",          # org non può iniziare con -
        ],
    )
    def test_invalid(self, repo):
        with pytest.raises(InvalidRepoFormatError):
            validate_repo_format(repo)

    def test_too_long(self):
        long = "a" * 200 + "/" + "b" * 200
        with pytest.raises(InvalidRepoFormatError):
            validate_repo_format(long)


# ---------------------------------------------------------------------------
# Sanitizzazione e path
# ---------------------------------------------------------------------------


def test_sanitize_repo_to_dirname():
    assert sanitize_repo_to_dirname("Qwen/Qwen2.5-3B") == "Qwen--Qwen2.5-3B"
    assert sanitize_repo_to_dirname("microsoft/phi-2") == "microsoft--phi-2"


def test_sanitize_invalid_raises():
    with pytest.raises(InvalidRepoFormatError):
        sanitize_repo_to_dirname("bad repo")


def test_get_local_path_is_under_models_dir():
    """get_local_path deve restituire un path sotto settings.models_path."""
    from config import settings

    p = get_local_path("Qwen/Qwen2.5-0.5B")
    assert p.is_absolute()
    assert p.parent == settings.models_path
    assert p.name == "Qwen--Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# compute_directory_size
# ---------------------------------------------------------------------------


class TestComputeDirectorySize:
    def test_nonexistent_path_returns_zero(self, tmp_path):
        assert compute_directory_size(tmp_path / "missing") == 0

    def test_single_file(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"x" * 100)
        assert compute_directory_size(f) == 100

    def test_recursive(self, fake_model_dir):
        # 1024 + 2048 + 512 = 3584. Più config.json (~22 byte).
        size = compute_directory_size(fake_model_dir)
        assert size >= 3584
        assert size < 4096  # ragionevole upper bound


# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------


class TestWhitelist:
    def test_get_whitelist_not_empty(self):
        wl = get_whitelist()
        assert len(wl) > 0
        assert all(isinstance(e, WhitelistEntry) for e in wl)

    def test_find_in_whitelist_hit(self):
        e = find_in_whitelist("Qwen/Qwen2.5-0.5B")
        assert e is not None
        assert e.tag == "qwen2.5"

    def test_find_in_whitelist_miss(self):
        assert find_in_whitelist("nope/not-real") is None

    def test_find_in_whitelist_case_sensitive(self):
        # HF repo sono case-sensitive
        assert find_in_whitelist("qwen/qwen2.5-0.5b") is None

    def test_no_duplicate_repos_in_whitelist(self):
        wl = get_whitelist()
        repos = [e.hf_repo for e in wl]
        assert len(repos) == len(set(repos)), "Duplicato in whitelist!"

    def test_all_whitelist_repos_pass_validation(self):
        for entry in get_whitelist():
            # Non deve sollevare
            validate_repo_format(entry.hf_repo)


# ---------------------------------------------------------------------------
# CRUD via DB
# ---------------------------------------------------------------------------


class TestRegisterModel:
    def test_basic_register(self, session, fake_model_dir):
        row = register_model(
            session,
            hf_repo="Qwen/Qwen2.5-0.5B",
            local_path=fake_model_dir,
        )
        assert row.id is not None
        assert row.hf_repo == "Qwen/Qwen2.5-0.5B"
        assert row.size_bytes >= 3584
        # Auto-popolato da whitelist
        assert row.tag == "qwen2.5"
        assert row.display_name == "Qwen 2.5 0.5B"
        assert row.params_billions == 0.5
        assert row.is_custom is False

    def test_custom_repo_no_whitelist(self, session, fake_model_dir):
        row = register_model(
            session,
            hf_repo="someuser/some-model",
            local_path=fake_model_dir,
            is_custom=True,
        )
        assert row.is_custom is True
        # Display name fallback al nome dopo lo slash
        assert row.display_name == "some-model"
        assert row.tag is None

    def test_invalid_repo_raises(self, session, fake_model_dir):
        with pytest.raises(InvalidRepoFormatError):
            register_model(
                session,
                hf_repo="bad repo!",
                local_path=fake_model_dir,
            )

    def test_duplicate_raises(self, session, fake_model_dir):
        register_model(
            session,
            hf_repo="Qwen/Qwen2.5-0.5B",
            local_path=fake_model_dir,
        )
        with pytest.raises(ModelAlreadyExistsError):
            register_model(
                session,
                hf_repo="Qwen/Qwen2.5-0.5B",
                local_path=fake_model_dir,
            )


class TestListAndGet:
    def test_list_empty(self, session):
        assert list_local_models(session) == []

    def test_list_returns_recent_first(self, session, fake_model_dir):
        register_model(session, hf_repo="microsoft/phi-2", local_path=fake_model_dir)

        # Forziamo timestamp diversi: SQLite func.now() ha precisione al
        # secondo, due insert consecutivi possono avere lo stesso timestamp
        # rendendo l'ORDER BY non deterministico. Backdate del primo insert.
        first = session.scalar(
            select(BaseModelRow).where(BaseModelRow.hf_repo == "microsoft/phi-2")
        )
        first.downloaded_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        session.commit()

        register_model(session, hf_repo="Qwen/Qwen2.5-0.5B", local_path=fake_model_dir)

        rows = list_local_models(session)
        assert len(rows) == 2
        # Ordinato per downloaded_at desc → ultimo inserito è primo
        assert rows[0].hf_repo == "Qwen/Qwen2.5-0.5B"
        assert rows[1].hf_repo == "microsoft/phi-2"

    def test_get_by_id_ok(self, session, fake_model_dir):
        row = register_model(
            session, hf_repo="microsoft/phi-2", local_path=fake_model_dir
        )
        got = get_model_by_id(session, row.id)
        assert got.id == row.id

    def test_get_by_id_not_found(self, session):
        with pytest.raises(ModelNotFoundError):
            get_model_by_id(session, 99999)

    def test_get_by_repo(self, session, fake_model_dir):
        register_model(
            session, hf_repo="microsoft/phi-2", local_path=fake_model_dir
        )
        assert get_model_by_repo(session, "microsoft/phi-2") is not None
        assert get_model_by_repo(session, "nope/nope") is None

    def test_is_model_present(self, session, fake_model_dir):
        assert is_model_present(session, "microsoft/phi-2") is False
        register_model(
            session, hf_repo="microsoft/phi-2", local_path=fake_model_dir
        )
        assert is_model_present(session, "microsoft/phi-2") is True


class TestDeleteModel:
    def test_delete_record_only(self, session, fake_model_dir):
        row = register_model(
            session, hf_repo="microsoft/phi-2", local_path=fake_model_dir
        )
        assert fake_model_dir.exists()

        delete_model(session, row.id, remove_files=False)

        # Cartella conservata
        assert fake_model_dir.exists()
        # Record sparito
        with pytest.raises(ModelNotFoundError):
            get_model_by_id(session, row.id)

    def test_delete_record_and_files(self, session, tmp_path, monkeypatch):
        """delete_model con remove_files=True deve cancellare il dir."""
        from config import settings as cfg

        # Forziamo settings.models_path a tmp_path così la safety check passa
        models_root = tmp_path / "models"
        models_root.mkdir()
        monkeypatch.setattr(cfg, "models_dir", str(models_root))

        model_dir = models_root / "microsoft--phi-2"
        model_dir.mkdir()
        (model_dir / "f.bin").write_bytes(b"x" * 100)

        row = register_model(
            session, hf_repo="microsoft/phi-2", local_path=model_dir
        )
        assert model_dir.exists()

        delete_model(session, row.id, remove_files=True)

        assert not model_dir.exists()

    def test_delete_not_found(self, session):
        with pytest.raises(ModelNotFoundError):
            delete_model(session, 99999)


# ---------------------------------------------------------------------------
# validate_hf_repo_exists (mockato — niente network)
# ---------------------------------------------------------------------------


class TestValidateHfRepoExists:
    def test_invalid_format_raises_before_network(self):
        from core.model_registry import validate_hf_repo_exists

        # Nemmeno tenta di chiamare HF
        with pytest.raises(InvalidRepoFormatError):
            validate_hf_repo_exists("malformato")

    def test_repo_not_found_raises(self, monkeypatch):
        """Mocka HfApi.model_info per simulare repo inesistente."""
        from huggingface_hub.errors import RepositoryNotFoundError

        from core import model_registry

        def fake_init(self, token=None):
            self.token = token

        def fake_model_info(self, repo):
            raise RepositoryNotFoundError("not found")

        monkeypatch.setattr("huggingface_hub.HfApi.__init__", fake_init)
        monkeypatch.setattr("huggingface_hub.HfApi.model_info", fake_model_info)

        with pytest.raises(HFRepoNotAccessibleError, match="non trovato"):
            model_registry.validate_hf_repo_exists("user/nope")

    def test_gated_repo_raises(self, monkeypatch):
        """Mocka HfApi.model_info per simulare repo gated.

        IMPORTANTE: GatedRepoError eredita da RepositoryNotFoundError in
        huggingface_hub. L'ordine dei `except` nel codice di produzione
        DEVE quindi catturare GatedRepoError prima.
        """
        from huggingface_hub.errors import GatedRepoError

        from core import model_registry

        def fake_init(self, token=None):
            self.token = token

        def fake_model_info(self, repo):
            raise GatedRepoError("gated")

        monkeypatch.setattr("huggingface_hub.HfApi.__init__", fake_init)
        monkeypatch.setattr("huggingface_hub.HfApi.model_info", fake_model_info)

        with pytest.raises(HFRepoNotAccessibleError, match="gated"):
            model_registry.validate_hf_repo_exists("google/gemma-2-2b")