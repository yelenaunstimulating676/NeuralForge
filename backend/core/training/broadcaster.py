"""
EventBroadcaster — pub/sub di eventi di training cross-thread.

Concetti:
  - Channel per `run_id`: ogni training run ha il suo canale.
  - Thread-safe publish: dal thread del training si pubblica con
    `publish_from_thread`, che fa il bridging via run_coroutine_threadsafe.
  - Multi-subscriber: più WebSocket client possono leggere lo stesso channel.
  - History buffer: gli ultimi N eventi vengono mantenuti per il replay
    quando un client si connette mid-training.

Lifecycle channel:
  - Creato implicitamente al primo publish o subscribe
  - Restano vivi finché non vengono esplicitamente eliminati o per cleanup TTL
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


# Stringhe convenzionali per il campo `type` di un evento.
EVENT_STEP_LOG = "step_log"
EVENT_STATUS = "status_change"
EVENT_ERROR = "error"
EVENT_FINISHED = "finished"   # marker speciale per terminare lo streaming


def make_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Helper: costruisce un evento standardizzato.

    Tutti gli eventi hanno: type, timestamp_iso, e i campi specifici dentro `data`.
    """
    return {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    """
    Canale pub/sub per un singolo run_id.

    Mantiene una lista di subscriber (asyncio.Queue) e un history buffer
    degli ultimi `history_size` eventi pubblicati.
    """

    run_id: str
    history_size: int = 1000
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    finished: bool = False  # se True, niente nuovi eventi attesi

    def __post_init__(self) -> None:
        # Riallineamento maxlen della deque al parametro
        self.history = deque(self.history, maxlen=self.history_size)

    async def publish(self, event: dict[str, Any]) -> None:
        """
        Pubblica un evento sul channel.
        Lo aggiunge alla history e lo distribuisce a tutti i subscriber attivi.

        DEVE essere chiamato dall'event loop. Per chiamare dal training thread,
        usare `EventBroadcaster.publish_from_thread`.
        """
        self.history.append(event)

        # Marca canale finito se è un evento di terminazione
        if event.get("type") in {EVENT_FINISHED, EVENT_ERROR}:
            self.finished = True

        # Distribuisci a tutti i subscriber. Iteriamo su una copia perché
        # un subscriber potrebbe essere rimosso durante l'iterazione.
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Se la coda è piena, droppiamo l'evento per quel subscriber.
                # Meglio perdere un evento che bloccare gli altri.
                logger.warning(
                    "Coda subscriber piena per run %s, evento droppato.",
                    self.run_id,
                )

    @asynccontextmanager
    async def subscribe(self, replay: bool = True):
        """
        Context manager per sottoscriversi al channel.

        Args:
            replay: se True (default), invia subito tutta la history al
                subscriber prima dei nuovi eventi.

        Yields:
            asyncio.Queue dalla quale leggere gli eventi.

        Esempio uso:
            async with channel.subscribe() as queue:
                while True:
                    event = await queue.get()
                    await ws.send_json(event)
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

        # Replay history (in ordine cronologico)
        if replay:
            for past_event in list(self.history):
                queue.put_nowait(past_event)

        self.subscribers.append(queue)
        try:
            yield queue
        finally:
            try:
                self.subscribers.remove(queue)
            except ValueError:
                pass

    @property
    def num_subscribers(self) -> int:
        return len(self.subscribers)

    @property
    def num_events(self) -> int:
        return len(self.history)


# ---------------------------------------------------------------------------
# Broadcaster (singleton)
# ---------------------------------------------------------------------------


class EventBroadcaster:
    """
    Manager centrale dei channel. Singleton condiviso da tutta l'app.

    Mantiene una mappa `run_id → Channel`. I channel vengono creati
    on-demand al primo accesso.
    """

    def __init__(self, default_history_size: int = 1000) -> None:
        self._channels: dict[str, Channel] = {}
        self._lock = asyncio.Lock()
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._default_history_size = default_history_size

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Registra l'event loop principale. Necessario per `publish_from_thread`.
        Da chiamare nel `lifespan` di FastAPI all'avvio.
        """
        self._main_loop = loop

    async def get_or_create_channel(self, run_id: str) -> Channel:
        """Restituisce il channel per `run_id`, creandolo se non esiste."""
        async with self._lock:
            channel = self._channels.get(run_id)
            if channel is None:
                channel = Channel(
                    run_id=run_id,
                    history_size=self._default_history_size,
                )
                self._channels[run_id] = channel
                logger.debug("Channel creato per run %s", run_id)
            return channel

    async def get_channel(self, run_id: str) -> Channel | None:
        """Restituisce il channel per `run_id`, None se non esiste."""
        async with self._lock:
            return self._channels.get(run_id)

    async def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """
        Pubblica un evento sul channel del `run_id` (lo crea se non esiste).
        DEVE essere chiamato dall'event loop principale.
        """
        channel = await self.get_or_create_channel(run_id)
        await channel.publish(event)

    def publish_from_thread(self, run_id: str, event: dict[str, Any]) -> None:
        """
        Pubblica un evento da un thread NON-async (es. il thread del training).

        Usa `run_coroutine_threadsafe` per schedulare la publish sul main loop.
        Non blocca il thread chiamante (fire-and-forget).
        """
        if self._main_loop is None:
            logger.warning(
                "publish_from_thread chiamato senza bind_loop: evento droppato."
            )
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.publish(run_id, event), self._main_loop
            )
        except RuntimeError as exc:
            # Il loop potrebbe essere già chiuso (shutdown in corso)
            logger.warning("publish_from_thread fallito: %s", exc)

    async def remove_channel(self, run_id: str) -> bool:
        """
        Rimuove un channel. Non chiude i subscriber attivi (continueranno
        a leggere finché non escono dal context manager).

        Returns:
            True se rimosso, False se non esisteva.
        """
        async with self._lock:
            return self._channels.pop(run_id, None) is not None

    async def cleanup_finished(self, max_age_seconds: float = 3600) -> int:
        """
        Rimuove channel terminati e con history più vecchia di `max_age_seconds`.

        Returns:
            Numero di channel rimossi.
        """
        now = datetime.now(timezone.utc)
        removed = 0
        async with self._lock:
            to_remove: list[str] = []
            for run_id, channel in self._channels.items():
                if not channel.finished:
                    continue
                if not channel.history:
                    to_remove.append(run_id)
                    continue
                last = channel.history[-1]
                try:
                    last_ts = datetime.fromisoformat(last["timestamp"])
                except (KeyError, ValueError):
                    continue
                if (now - last_ts).total_seconds() > max_age_seconds:
                    to_remove.append(run_id)

            for run_id in to_remove:
                del self._channels[run_id]
                removed += 1

        if removed:
            logger.info("Cleanup: rimossi %d channel terminati.", removed)
        return removed

    @property
    def num_channels(self) -> int:
        return len(self._channels)


# Singleton globale, da importare con `from core.training.broadcaster import broadcaster`
broadcaster = EventBroadcaster()