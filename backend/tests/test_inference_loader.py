"""
Test del ModelLoader.

Strategia: mockiamo `prepare_base_for_inference` e `prepare_ft_for_inference`
così non carichiamo modelli reali. Verifichiamo solo logic di cache, lookup
DB, eviction LRU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core.inference.loader import (
    CachedModel,
    InferenceLoaderError,
    ModelLoader,
)
from core.training.model import LoadedModel, TrainableParamsInfo
from db.database import Base
from db.models import (
    BaseModel as BaseModelRow,
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


def make_fake_loaded():
    """Crea un LoadedModel fake."""
    fake_model = MagicMock()
    fake_model.parameters.return_value = []
    fake_tokenizer = MagicMock()
    return LoadedModel(
        model=fake_model,
        tokenizer=fake_tokenizer,
        trainable_info=TrainableParamsInfo(0, 1000, 0.0),
    )


@pytest.fixture
def setup_db_with_models(session, tmp_path, monkeypatch):
    """Crea base + ft + path su disco."""
    from config import settings
    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "adapters_dir", str(tmp_path / "adapters"))

    base_path = tmp_path / "models" / "fake_base"
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "config.json").write_text("{}")

    adapter_path = tmp_path / "adapters" / "run-1" / "final"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_model.safetensors").write_bytes(b"FAKE")

    base = BaseModelRow(
        hf_repo="fake/base",
        display_name="Fake Base",
        local_path=str(base_path),
        size_bytes=1000,
        params_billions=0.1,
    )
    session.add(base)
    session.commit()

    # TrainingRun per FK del FineTunedModel
    run = TrainingRun(
        run_id="train-fake-001",
        base_model_id=base.id,
        status="completed",
        config_json="{}",
    )
    session.add(run)
    session.commit()

    ft = FineTunedModelRow(
        base_model_id=base.id,
        training_run_id=run.id,
        name="Fake FT",
        adapter_path=str(adapter_path),
        size_bytes=500,
    )
    session.add(ft)
    session.commit()

    return base, ft


# ---------------------------------------------------------------------------
# Key utilities
# ---------------------------------------------------------------------------


class TestKeys:
    def test_make_key_base(self):
        assert ModelLoader.make_key("base", 5) == "base:5"

    def test_make_key_ft(self):
        assert ModelLoader.make_key("ft", 12) == "ft:12"

    def test_make_key_invalid_kind(self):
        with pytest.raises(ValueError):
            ModelLoader.make_key("xxx", 1)

    def test_parse_key(self):
        assert ModelLoader.parse_key("base:5") == ("base", 5)
        assert ModelLoader.parse_key("ft:42") == ("ft", 42)

    def test_parse_key_invalid(self):
        with pytest.raises(ValueError):
            ModelLoader.parse_key("not-a-key")


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    def test_cache_miss_initially(self):
        loader = ModelLoader()
        assert loader.get("base:1") is None

    def test_load_base_caches(self, session, setup_db_with_models):
        base, _ = setup_db_with_models
        loader = ModelLoader()

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ):
            cached = loader.load_base(session, base.id)

        assert cached.key == f"base:{base.id}"
        assert cached.kind == "base"
        assert cached.has_adapter is False
        assert loader.get(cached.key) is cached

    def test_second_load_uses_cache(self, session, setup_db_with_models):
        """La seconda load NON deve chiamare prepare_*."""
        base, _ = setup_db_with_models
        loader = ModelLoader()

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ) as mock_prepare:
            loader.load_base(session, base.id)
            loader.load_base(session, base.id)
            loader.load_base(session, base.id)

        # Chiamato una sola volta nonostante 3 load
        assert mock_prepare.call_count == 1

    def test_load_ft_caches(self, session, setup_db_with_models):
        _, ft = setup_db_with_models
        loader = ModelLoader()

        with patch(
            "core.inference.loader.prepare_ft_for_inference",
            return_value=make_fake_loaded(),
        ):
            cached = loader.load_ft(session, ft.id)

        assert cached.key == f"ft:{ft.id}"
        assert cached.kind == "ft"
        assert cached.has_adapter is True


# ---------------------------------------------------------------------------
# Errori
# ---------------------------------------------------------------------------


class TestErrors:
    def test_load_base_missing_db(self, session):
        loader = ModelLoader()
        with pytest.raises(InferenceLoaderError, match="non trovato"):
            loader.load_base(session, 999)

    def test_load_ft_missing_db(self, session):
        loader = ModelLoader()
        with pytest.raises(InferenceLoaderError, match="non trovato"):
            loader.load_ft(session, 999)

    def test_load_base_missing_disk(self, session, tmp_path):
        # BaseModel registrato nel DB ma path inesistente
        base = BaseModelRow(
            hf_repo="ghost/model",
            display_name="Ghost",
            local_path=str(tmp_path / "nope"),
            size_bytes=0,
        )
        session.add(base)
        session.commit()

        loader = ModelLoader()
        with pytest.raises(InferenceLoaderError, match="non presente su disco"):
            loader.load_base(session, base.id)

    def test_load_ft_missing_adapter_disk(self, session, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))

        base_path = tmp_path / "models" / "fake_base"
        base_path.mkdir(parents=True)
        (base_path / "config.json").write_text("{}")

        base = BaseModelRow(
            hf_repo="fake/base", display_name="Fake",
            local_path=str(base_path), size_bytes=0,
        )
        session.add(base)
        session.commit()

        run = TrainingRun(
            run_id="train-ghost", base_model_id=base.id,
            status="completed", config_json="{}",
        )
        session.add(run)
        session.commit()

        ft = FineTunedModelRow(
            base_model_id=base.id, training_run_id=run.id,
            name="Ghost FT", adapter_path=str(tmp_path / "ghost"),
            size_bytes=0,
        )
        session.add(ft)
        session.commit()

        loader = ModelLoader()
        with pytest.raises(InferenceLoaderError, match="Adapter"):
            loader.load_ft(session, ft.id)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_evicts_when_full(self, session, tmp_path, monkeypatch):
        """Con cache_limit=2, il terzo caricamento deve evict il primo."""
        from config import settings
        monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))

        # Crea 3 base models
        bases = []
        for i in range(3):
            d = tmp_path / "models" / f"m{i}"
            d.mkdir(parents=True)
            (d / "config.json").write_text("{}")
            b = BaseModelRow(
                hf_repo=f"fake/m{i}",
                display_name=f"M{i}",
                local_path=str(d),
                size_bytes=0,
            )
            session.add(b)
            bases.append(b)
        session.commit()

        loader = ModelLoader(cache_limit=2)

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ):
            loader.load_base(session, bases[0].id)
            loader.load_base(session, bases[1].id)
            loader.load_base(session, bases[2].id)

        # Cache contiene solo gli ultimi 2
        assert len(loader.list_loaded()) == 2
        assert loader.get(f"base:{bases[0].id}") is None  # evicted
        assert loader.get(f"base:{bases[1].id}") is not None
        assert loader.get(f"base:{bases[2].id}") is not None

    def test_get_touches_lru(self, session, tmp_path, monkeypatch):
        """get() sposta il modello in fondo (touch LRU)."""
        from config import settings
        monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))

        bases = []
        for i in range(3):
            d = tmp_path / "models" / f"m{i}"
            d.mkdir(parents=True)
            (d / "config.json").write_text("{}")
            b = BaseModelRow(
                hf_repo=f"fake/m{i}", display_name=f"M{i}",
                local_path=str(d), size_bytes=0,
            )
            session.add(b)
            bases.append(b)
        session.commit()

        loader = ModelLoader(cache_limit=2)

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ):
            loader.load_base(session, bases[0].id)
            loader.load_base(session, bases[1].id)
            # Touch del primo: ora è il più recente
            loader.get(f"base:{bases[0].id}")
            # Carica il terzo: dovrebbe evict il SECONDO (non il primo)
            loader.load_base(session, bases[2].id)

        assert loader.get(f"base:{bases[0].id}") is not None  # tenuto (touched)
        assert loader.get(f"base:{bases[1].id}") is None      # evicted
        assert loader.get(f"base:{bases[2].id}") is not None


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------


class TestUnload:
    def test_unload_existing(self, session, setup_db_with_models):
        base, _ = setup_db_with_models
        loader = ModelLoader()

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ):
            loader.load_base(session, base.id)

        key = f"base:{base.id}"
        assert loader.get(key) is not None
        assert loader.unload(key) is True
        assert loader.get(key) is None

    def test_unload_missing(self):
        loader = ModelLoader()
        assert loader.unload("base:999") is False

    def test_unload_all(self, session, setup_db_with_models):
        base, ft = setup_db_with_models
        loader = ModelLoader()

        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ), patch(
            "core.inference.loader.prepare_ft_for_inference",
            return_value=make_fake_loaded(),
        ):
            loader.load_base(session, base.id)
            loader.load_ft(session, ft.id)

        assert len(loader.list_loaded()) == 2
        removed = loader.unload_all()
        assert removed == 2
        assert len(loader.list_loaded()) == 0


# ---------------------------------------------------------------------------
# load() dispatcher
# ---------------------------------------------------------------------------


class TestLoadDispatcher:
    def test_load_base(self, session, setup_db_with_models):
        base, _ = setup_db_with_models
        loader = ModelLoader()
        with patch(
            "core.inference.loader.prepare_base_for_inference",
            return_value=make_fake_loaded(),
        ):
            cached = loader.load(session, "base", base.id)
        assert cached.kind == "base"

    def test_load_ft(self, session, setup_db_with_models):
        _, ft = setup_db_with_models
        loader = ModelLoader()
        with patch(
            "core.inference.loader.prepare_ft_for_inference",
            return_value=make_fake_loaded(),
        ):
            cached = loader.load(session, "ft", ft.id)
        assert cached.kind == "ft"

    def test_load_invalid_kind(self, session, setup_db_with_models):
        base, _ = setup_db_with_models
        loader = ModelLoader()
        with pytest.raises(ValueError):
            loader.load(session, "weird", base.id)