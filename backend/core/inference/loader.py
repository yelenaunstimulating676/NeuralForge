"""
ModelLoader: cache singleton per modelli inference.

Strategia:
  - LRU cache con limite hardcoded di 2 modelli simultanei
  - Lock per model_key per evitare doppi caricamenti concorrenti
  - Unload esplicito + torch.cuda.empty_cache() per liberare VRAM
  - Riuso di prepare_*_for_inference da core/training/model.py

I modelli vengono identificati da una `model_key`:
  - "base:<id>" → BaseModel di id N
  - "ft:<id>"   → FineTunedModel di id N (carica base + adapter)
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from core.training.model import (
    LoadedModel,
    QuantizationConfig,
    prepare_base_for_inference,
    prepare_ft_for_inference,
)
from db.models import BaseModel as BaseModelRow, FineTunedModel as FineTunedModelRow

logger = logging.getLogger(__name__)


# Limite massimo di modelli simultanei in cache. Oltre, evicting LRU.
DEFAULT_CACHE_LIMIT = 2


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class InferenceLoaderError(Exception):
    """Errore durante caricamento di un modello per inference."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CachedModel:
    """Modello cached con metadata per UI/lookup."""

    key: str                     # "base:N" o "ft:N"
    kind: str                    # "base" | "ft"
    model_id: int
    display_name: str
    base_model_id: int           # per i ft, l'id del base sottostante
    loaded: LoadedModel          # bundle del modello con tokenizer
    has_adapter: bool

    def to_summary(self) -> dict[str, Any]:
        """Summary serializzabile per API list."""
        return {
            "key": self.key,
            "kind": self.kind,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "base_model_id": self.base_model_id,
            "has_adapter": self.has_adapter,
        }


# ---------------------------------------------------------------------------
# ModelLoader: singleton
# ---------------------------------------------------------------------------


class ModelLoader:
    """
    Gestisce caricamento, cache e unload di modelli per inference.

    Thread-safe: protegge il dict cache con un lock globale e crea
    lock per-key per i caricamenti lunghi.
    """

    def __init__(self, cache_limit: int = DEFAULT_CACHE_LIMIT) -> None:
        self.cache_limit = cache_limit
        # OrderedDict per LRU: front = vecchio, end = recente
        self._cache: OrderedDict[str, CachedModel] = OrderedDict()
        # Lock globale per il dict
        self._cache_lock = threading.Lock()
        # Lock per key per evitare doppi caricamenti concorrenti
        self._key_locks: dict[str, threading.Lock] = {}

    # ---------------------------------------------------------------------
    # Key utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def make_key(kind: str, model_id: int) -> str:
        if kind not in {"base", "ft"}:
            raise ValueError(f"kind invalido: {kind!r} (deve essere 'base' o 'ft')")
        return f"{kind}:{model_id}"

    @staticmethod
    def parse_key(key: str) -> tuple[str, int]:
        try:
            kind, id_str = key.split(":", 1)
            return kind, int(id_str)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"key invalida: {key!r}") from exc

    # ---------------------------------------------------------------------
    # Cache access primitives
    # ---------------------------------------------------------------------

    def _get_or_create_key_lock(self, key: str) -> threading.Lock:
        with self._cache_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def _evict_oldest_if_full(self) -> None:
        """
        Se il cache è pieno (>= cache_limit), evict il modello più vecchio.
        Da chiamare con _cache_lock acquisito.
        """
        while len(self._cache) >= self.cache_limit:
            oldest_key, oldest_cached = self._cache.popitem(last=False)
            logger.info("Evicting modello LRU dalla cache: %s", oldest_key)
            self._free_cached_model(oldest_cached)

    def _free_cached_model(self, cached: CachedModel) -> None:
        """Libera la memoria di un modello cached."""
        try:
            del cached.loaded.model
            del cached.loaded.tokenizer
        except Exception:  # noqa: BLE001
            pass
        # GC + cuda cache
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get(self, key: str) -> CachedModel | None:
        """Ritorna il modello cached se presente, None altrimenti."""
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                # Sposta in fondo (LRU touch)
                self._cache.move_to_end(key)
            return cached

    def list_loaded(self) -> list[dict[str, Any]]:
        """Lista summary dei modelli attualmente in cache."""
        with self._cache_lock:
            return [c.to_summary() for c in self._cache.values()]

    def unload(self, key: str) -> bool:
        """
        Rimuove un modello dalla cache e libera VRAM.

        Returns:
            True se il modello era cached, False altrimenti.
        """
        with self._cache_lock:
            cached = self._cache.pop(key, None)
            if cached is None:
                return False
        self._free_cached_model(cached)
        logger.info("Modello %s scaricato dalla VRAM.", key)
        return True

    def unload_all(self) -> int:
        """Scarica tutti i modelli dalla cache. Ritorna il numero rimossi."""
        with self._cache_lock:
            keys = list(self._cache.keys())
            cached_list = list(self._cache.values())
            self._cache.clear()
        for cached in cached_list:
            self._free_cached_model(cached)
        if keys:
            logger.info("Scaricati %d modelli dalla cache.", len(keys))
        return len(keys)

    def load_base(
        self,
        session: Session,
        base_model_id: int,
        quant_config: QuantizationConfig | None = None,
    ) -> CachedModel:
        """
        Carica un BaseModel per inference. Usa cache se disponibile.

        Args:
            session: SQLAlchemy session.
            base_model_id: id della tabella base_models.
            quant_config: parametri quantizzazione (default 4-bit nf4 + bf16).

        Returns:
            CachedModel pronto per generate().

        Raises:
            InferenceLoaderError: se il modello non esiste o caricamento fallisce.
        """
        key = self.make_key("base", base_model_id)

        # Fast path: cache hit
        cached = self.get(key)
        if cached is not None:
            logger.debug("Cache hit per %s", key)
            return cached

        # Slow path: lock per-key per evitare race
        lock = self._get_or_create_key_lock(key)
        with lock:
            # Re-check dopo lock acquisito
            cached = self.get(key)
            if cached is not None:
                return cached

            # Risolvi DB
            base_row = session.get(BaseModelRow, base_model_id)
            if base_row is None:
                raise InferenceLoaderError(
                    f"BaseModel id={base_model_id} non trovato nel DB."
                )
            if not Path(base_row.local_path).exists():
                raise InferenceLoaderError(
                    f"BaseModel {base_row.hf_repo!r} non presente su disco: "
                    f"{base_row.local_path}"
                )

            logger.info("Caricamento base model %s per inference…", base_row.hf_repo)
            try:
                loaded = prepare_base_for_inference(
                    Path(base_row.local_path), quant_config=quant_config
                )
            except Exception as exc:
                raise InferenceLoaderError(
                    f"Errore caricamento base model: {exc}"
                ) from exc

            cached_model = CachedModel(
                key=key,
                kind="base",
                model_id=base_model_id,
                display_name=base_row.display_name,
                base_model_id=base_model_id,
                loaded=loaded,
                has_adapter=False,
            )

            with self._cache_lock:
                self._evict_oldest_if_full()
                self._cache[key] = cached_model
                # Aggiunto, sposta in fondo
                self._cache.move_to_end(key)

            logger.info("Base model %s pronto per inference.", base_row.hf_repo)
            return cached_model

    def load_ft(
        self,
        session: Session,
        ft_model_id: int,
        quant_config: QuantizationConfig | None = None,
    ) -> CachedModel:
        """
        Carica un FineTunedModel (base + adapter LoRA) per inference.

        Args:
            session: SQLAlchemy session.
            ft_model_id: id della tabella finetuned_models.
            quant_config: parametri quantizzazione.

        Returns:
            CachedModel pronto per generate().
        """
        key = self.make_key("ft", ft_model_id)

        cached = self.get(key)
        if cached is not None:
            return cached

        lock = self._get_or_create_key_lock(key)
        with lock:
            cached = self.get(key)
            if cached is not None:
                return cached

            # Risolvi FT model + base associato
            ft_row = session.get(FineTunedModelRow, ft_model_id)
            if ft_row is None:
                raise InferenceLoaderError(
                    f"FineTunedModel id={ft_model_id} non trovato nel DB."
                )
            if not Path(ft_row.adapter_path).exists():
                raise InferenceLoaderError(
                    f"Adapter di FT model {ft_row.name!r} non trovato su disco: "
                    f"{ft_row.adapter_path}"
                )

            base_row = session.get(BaseModelRow, ft_row.base_model_id)
            if base_row is None:
                raise InferenceLoaderError(
                    f"BaseModel id={ft_row.base_model_id} (parent del FT) "
                    f"non trovato."
                )
            if not Path(base_row.local_path).exists():
                raise InferenceLoaderError(
                    f"Base model {base_row.hf_repo!r} non presente su disco: "
                    f"{base_row.local_path}"
                )

            logger.info(
                "Caricamento FT model %s (base=%s, adapter=%s)…",
                ft_row.name, base_row.hf_repo, ft_row.adapter_path,
            )
            try:
                loaded = prepare_ft_for_inference(
                    base_model_path=Path(base_row.local_path),
                    adapter_path=Path(ft_row.adapter_path),
                    quant_config=quant_config,
                )
            except Exception as exc:
                raise InferenceLoaderError(
                    f"Errore caricamento FT model: {exc}"
                ) from exc

            cached_model = CachedModel(
                key=key,
                kind="ft",
                model_id=ft_model_id,
                display_name=ft_row.name,
                base_model_id=ft_row.base_model_id,
                loaded=loaded,
                has_adapter=True,
            )

            with self._cache_lock:
                self._evict_oldest_if_full()
                self._cache[key] = cached_model
                self._cache.move_to_end(key)

            logger.info("FT model %s pronto per inference.", ft_row.name)
            return cached_model

    def load(
        self,
        session: Session,
        kind: str,
        model_id: int,
        quant_config: QuantizationConfig | None = None,
    ) -> CachedModel:
        """Convenience: dispatch su load_base / load_ft."""
        if kind == "base":
            return self.load_base(session, model_id, quant_config)
        elif kind == "ft":
            return self.load_ft(session, model_id, quant_config)
        else:
            raise ValueError(f"kind invalido: {kind!r}")


# Singleton globale
model_loader = ModelLoader()