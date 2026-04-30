"""
Job Manager — gestione asincrona di task lunghi (download, training, ecc.).

Caratteristiche:
  - Job identificati da UUID
  - Status lifecycle: pending → running → completed | failed | cancelled
  - Progress tracking 0.0-1.0 + messaggio testuale
  - Cancellazione cooperativa via asyncio.Event
  - Storage in-memory (singleton thread-safe via asyncio.Lock)
  - Cleanup automatico di job vecchi (configurabile)

Uso tipico:

    async def my_long_task(progress_cb, cancel_event):
        for i in range(100):
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            await asyncio.sleep(0.1)
            progress_cb(i / 100, f"Step {i}")
        return {"result": "done"}

    job = await job_manager.submit("custom", my_long_task)
    # ... poll job.status / job.progress ...
    await job_manager.cancel(job.id)

I job NON sono persistiti: muoiono col processo uvicorn. Per uso locale
single-user è accettabile; in M5+ valuteremo persistenza su DB se serve.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Status terminali (job non più modificabili)
TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


# ---------------------------------------------------------------------------
# Tipo della coroutine factory
# ---------------------------------------------------------------------------

# Una "coroutine factory" è una funzione che, data un progress callback e
# un cancel event, ritorna una coroutine eseguibile. Permette al JobManager
# di iniettare i due hook senza che il chiamante li gestisca a mano.
ProgressCallback = Callable[[float, str], None]
JobCoroutineFactory = Callable[[ProgressCallback, asyncio.Event], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Stato di un singolo job asincrono."""

    id: str
    kind: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    progress_message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Internal: riferimenti runtime (non serializzabili)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def is_terminal(self) -> bool:
        """True se il job è in uno stato definitivo."""
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Serializzazione JSON-safe per le response API."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "progress_message": self.progress_message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


# ---------------------------------------------------------------------------
# JobManager singleton
# ---------------------------------------------------------------------------


class JobManager:
    """
    Manager in-memory dei job asincroni. Singleton condiviso da tutta l'app.

    Thread-safety: tutti gli accessi al `_jobs` dict passano da `_lock`.
    Le coroutine eseguite dai job girano nello stesso event loop di uvicorn.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    # ---- submit / cancel / lookup ----

    async def submit(
        self,
        kind: str,
        coro_factory: JobCoroutineFactory,
    ) -> Job:
        """
        Crea un nuovo job e ne lancia l'esecuzione asincrona.

        Args:
            kind: stringa che descrive il tipo (es. "download", "training").
            coro_factory: funzione che, dati progress_cb e cancel_event,
                ritorna la coroutine da eseguire.

        Returns:
            Il Job creato (subito in stato PENDING, presto RUNNING).
        """
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, kind=kind)

        async with self._lock:
            self._jobs[job_id] = job

        # Progress callback chiuso sull'istanza del job
        def progress_cb(value: float, message: str = "") -> None:
            # Accesso non protetto da lock: non serve, sono campi del singolo
            # Job a cui solo questa coroutine scrive.
            job.progress = max(0.0, min(1.0, value))
            if message:
                job.progress_message = message

        # Avvia il task e tienine traccia
        task = asyncio.create_task(
            self._run_job(job, coro_factory, progress_cb),
            name=f"job-{job_id}-{kind}",
        )
        job._task = task

        logger.info("Job submitted: id=%s kind=%s", job_id, kind)
        return job

    async def _run_job(
        self,
        job: Job,
        coro_factory: JobCoroutineFactory,
        progress_cb: ProgressCallback,
    ) -> None:
        """Wrapper che gestisce status lifecycle ed eccezioni."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        try:
            result = await coro_factory(progress_cb, job._cancel_event)
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result if isinstance(result, dict) else {"value": result}
            logger.info("Job completed: id=%s kind=%s", job.id, job.kind)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.error = "Job cancellato dall'utente."
            logger.info("Job cancelled: id=%s kind=%s", job.id, job.kind)
            # Non rilanciamo CancelledError: il task termina puliti.
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Job failed: id=%s kind=%s", job.id, job.kind)
        finally:
            job.finished_at = datetime.now(timezone.utc)

    async def cancel(self, job_id: str) -> bool:
        """
        Richiede la cancellazione di un job.

        Returns:
            True se la richiesta è stata registrata, False se il job
            non esiste o è già in stato terminale.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.is_terminal:
            return False

        # Setta l'event cooperativo: la coroutine deve controllarlo.
        job._cancel_event.set()
        # Cancella anche il Task come backup (per task che non controllano
        # l'event entro un tempo ragionevole).
        if job._task is not None and not job._task.done():
            job._task.cancel()

        logger.info("Job cancel requested: id=%s", job_id)
        return True

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(self, kind: str | None = None) -> list[Job]:
        """
        Lista i job, opzionalmente filtrati per kind.
        Ordinati per created_at decrescente (più recenti prima).
        """
        async with self._lock:
            jobs = list(self._jobs.values())
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    # ---- maintenance ----

    async def cleanup_old(self, max_age_seconds: float = 3600) -> int:
        """
        Rimuove job terminali più vecchi di `max_age_seconds`.

        Returns:
            Numero di job rimossi.
        """
        now = datetime.now(timezone.utc)
        removed = 0
        async with self._lock:
            ids_to_remove = [
                jid
                for jid, j in self._jobs.items()
                if j.is_terminal
                and j.finished_at
                and (now - j.finished_at).total_seconds() > max_age_seconds
            ]
            for jid in ids_to_remove:
                del self._jobs[jid]
                removed += 1
        if removed:
            logger.info("Cleanup: rimossi %d job vecchi.", removed)
        return removed

    async def shutdown(self) -> None:
        """
        Cancella tutti i job non-terminali. Da chiamare nello shutdown
        del lifespan FastAPI per terminare puliti.
        """
        async with self._lock:
            jobs = list(self._jobs.values())
        active = [j for j in jobs if not j.is_terminal]
        if not active:
            return
        logger.info("Shutdown: cancello %d job attivi.", len(active))
        for j in active:
            j._cancel_event.set()
            if j._task is not None and not j._task.done():
                j._task.cancel()
        # Aspettiamo brevemente che i task terminino
        await asyncio.gather(
            *(j._task for j in active if j._task is not None),
            return_exceptions=True,
        )


# Singleton esportato. Importare come `from core.jobs import job_manager`.
job_manager = JobManager()