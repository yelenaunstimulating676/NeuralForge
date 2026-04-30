"""
Test del JobManager.

Copre lifecycle, progress, cancel, list/get, cleanup.
Usa coroutine fittizie controllabili (no network, no I/O reale).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


from core.jobs import JobManager, JobStatus, TERMINAL_STATUSES


# pytest-asyncio non è in requirements, lo aggiungiamo. Per ora usiamo
# una fixture che gestisce il loop a mano per non aggiungere dipendenze.


def run_async(coro):
    """Esegue una coroutine in un nuovo event loop e ritorna il risultato."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers: coroutine factory di test
# ---------------------------------------------------------------------------


def make_factory_quick_success():
    """Factory che termina subito con un risultato fisso."""
    async def factory(progress_cb, cancel_event):
        progress_cb(0.5, "halfway")
        await asyncio.sleep(0)  # cede il loop
        progress_cb(1.0, "done")
        return {"output": 42}
    return factory


def make_factory_raises():
    """Factory che solleva un'eccezione."""
    async def factory(progress_cb, cancel_event):
        await asyncio.sleep(0)
        raise ValueError("boom")
    return factory


def make_factory_long(steps: int = 10, step_sleep: float = 0.05):
    """Factory che fa N step rispettando cancel_event."""
    async def factory(progress_cb, cancel_event):
        for i in range(steps):
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            await asyncio.sleep(step_sleep)
            progress_cb((i + 1) / steps, f"step {i+1}/{steps}")
        return {"steps_done": steps}
    return factory


# ---------------------------------------------------------------------------
# Lifecycle base
# ---------------------------------------------------------------------------


def test_submit_returns_pending_or_running():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_quick_success())
        # Subito dopo submit può essere PENDING o già RUNNING:
        # è asincrono e dipende dallo scheduler.
        assert job.status in {JobStatus.PENDING, JobStatus.RUNNING}
        assert job.id
        assert job.kind == "test"
        # Aspetta che termini
        await job._task
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 1.0
        assert job.result == {"output": 42}

    run_async(scenario())


def test_failed_job_records_error():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_raises())
        await job._task
        assert job.status == JobStatus.FAILED
        assert "ValueError" in (job.error or "")
        assert "boom" in (job.error or "")
        assert job.is_terminal

    run_async(scenario())


def test_cancel_stops_long_job():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_long(steps=100, step_sleep=0.05))
        await asyncio.sleep(0.1)  # lascia partire
        cancelled = await mgr.cancel(job.id)
        assert cancelled is True
        await asyncio.sleep(0.1)  # lascia che finisca
        assert job.status == JobStatus.CANCELLED
        assert job.is_terminal

    run_async(scenario())


def test_cancel_unknown_job_returns_false():
    async def scenario():
        mgr = JobManager()
        assert await mgr.cancel("nope") is False

    run_async(scenario())


def test_cancel_terminal_job_returns_false():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_quick_success())
        await job._task
        # ora è terminato
        assert await mgr.cancel(job.id) is False

    run_async(scenario())


# ---------------------------------------------------------------------------
# get / list
# ---------------------------------------------------------------------------


def test_get_existing_and_missing():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_quick_success())
        got = await mgr.get(job.id)
        assert got is not None
        assert got.id == job.id
        assert await mgr.get("nonexistent") is None

    run_async(scenario())


def test_list_filters_by_kind():
    async def scenario():
        mgr = JobManager()
        a = await mgr.submit("download", make_factory_quick_success())
        b = await mgr.submit("training", make_factory_quick_success())
        # aspetta che finiscano per stabilità
        await asyncio.gather(a._task, b._task)
        assert len(await mgr.list()) == 2
        assert len(await mgr.list(kind="download")) == 1
        assert len(await mgr.list(kind="training")) == 1
        assert len(await mgr.list(kind="other")) == 0

    run_async(scenario())


def test_list_sorted_recent_first():
    async def scenario():
        mgr = JobManager()
        a = await mgr.submit("test", make_factory_quick_success())
        await asyncio.sleep(0.02)
        b = await mgr.submit("test", make_factory_quick_success())
        await asyncio.gather(a._task, b._task)
        jobs = await mgr.list()
        assert jobs[0].id == b.id
        assert jobs[1].id == a.id

    run_async(scenario())


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_old_terminal_jobs():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_quick_success())
        await job._task
        # forziamo finished_at indietro nel tempo
        job.finished_at = datetime.now(timezone.utc) - timedelta(hours=2)
        removed = await mgr.cleanup_old(max_age_seconds=3600)
        assert removed == 1
        assert await mgr.get(job.id) is None

    run_async(scenario())


def test_cleanup_keeps_running_jobs():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_long(steps=100, step_sleep=0.05))
        # ancora in corso
        removed = await mgr.cleanup_old(max_age_seconds=0)
        assert removed == 0
        # cleanup
        await mgr.cancel(job.id)
        await asyncio.sleep(0.1)

    run_async(scenario())


# ---------------------------------------------------------------------------
# to_dict serializzazione
# ---------------------------------------------------------------------------


def test_to_dict_is_json_safe():
    async def scenario():
        mgr = JobManager()
        job = await mgr.submit("test", make_factory_quick_success())
        await job._task
        d = job.to_dict()
        assert d["status"] == "completed"
        assert isinstance(d["progress"], float)
        assert isinstance(d["created_at"], str)
        # Testa che è effettivamente serializzabile in JSON
        import json
        json.dumps(d)

    run_async(scenario())


def test_terminal_statuses_set():
    assert JobStatus.COMPLETED in TERMINAL_STATUSES
    assert JobStatus.FAILED in TERMINAL_STATUSES
    assert JobStatus.CANCELLED in TERMINAL_STATUSES
    assert JobStatus.PENDING not in TERMINAL_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_STATUSES