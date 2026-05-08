"""
Test dell'EventBroadcaster.

Strategia: testiamo publish/subscribe, history replay, multi-subscriber,
cleanup, e la chiamata cross-thread con run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

import pytest

from core.training.broadcaster import (
    EVENT_FINISHED,
    EVENT_STATUS,
    EVENT_STEP_LOG,
    Channel,
    EventBroadcaster,
    make_event,
)


def run_async(coro):
    """Esegue una coroutine in un nuovo event loop e ritorna il risultato."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# make_event
# ---------------------------------------------------------------------------


class TestMakeEvent:
    def test_basic_structure(self):
        ev = make_event(EVENT_STEP_LOG, {"loss": 1.5})
        assert ev["type"] == EVENT_STEP_LOG
        assert ev["data"] == {"loss": 1.5}
        assert "timestamp" in ev
        # ISO 8601 parseable
        datetime.fromisoformat(ev["timestamp"])


# ---------------------------------------------------------------------------
# Channel: publish + subscribe
# ---------------------------------------------------------------------------


class TestChannelPubSub:
    def test_publish_appends_to_history(self):
        async def scenario():
            ch = Channel(run_id="r1")
            ev = make_event(EVENT_STEP_LOG, {"x": 1})
            await ch.publish(ev)
            assert ch.num_events == 1
            assert list(ch.history)[0] == ev

        run_async(scenario())

    def test_subscribe_receives_published_events(self):
        async def scenario():
            ch = Channel(run_id="r2")
            received = []

            async def consumer():
                async with ch.subscribe(replay=False) as queue:
                    for _ in range(3):
                        ev = await asyncio.wait_for(queue.get(), timeout=1.0)
                        received.append(ev)

            consumer_task = asyncio.create_task(consumer())
            await asyncio.sleep(0.05)  # lascia partire il consumer

            for i in range(3):
                await ch.publish(make_event(EVENT_STEP_LOG, {"step": i}))

            await consumer_task
            assert len(received) == 3
            assert received[0]["data"]["step"] == 0

        run_async(scenario())

    def test_replay_history_on_subscribe(self):
        async def scenario():
            ch = Channel(run_id="r3")
            # Publish 3 eventi PRIMA di subscriversi
            for i in range(3):
                await ch.publish(make_event(EVENT_STEP_LOG, {"step": i}))

            received = []
            async with ch.subscribe(replay=True) as queue:
                for _ in range(3):
                    ev = await asyncio.wait_for(queue.get(), timeout=1.0)
                    received.append(ev)

            assert [e["data"]["step"] for e in received] == [0, 1, 2]

        run_async(scenario())

    def test_replay_disabled(self):
        async def scenario():
            ch = Channel(run_id="r4")
            await ch.publish(make_event(EVENT_STEP_LOG, {"step": 0}))

            async with ch.subscribe(replay=False) as queue:
                # Nessun evento in coda
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(queue.get(), timeout=0.1)

        run_async(scenario())

    def test_multi_subscriber_all_receive(self):
        async def scenario():
            ch = Channel(run_id="r5")
            results_a = []
            results_b = []

            async def consumer(label, results):
                async with ch.subscribe(replay=False) as queue:
                    while True:
                        try:
                            ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                        except asyncio.TimeoutError:
                            return
                        results.append(ev["data"]["step"])

            task_a = asyncio.create_task(consumer("A", results_a))
            task_b = asyncio.create_task(consumer("B", results_b))
            await asyncio.sleep(0.05)

            for i in range(5):
                await ch.publish(make_event(EVENT_STEP_LOG, {"step": i}))
                await asyncio.sleep(0.01)

            await asyncio.gather(task_a, task_b)
            assert results_a == [0, 1, 2, 3, 4]
            assert results_b == [0, 1, 2, 3, 4]

        run_async(scenario())

    def test_subscriber_cleanup_on_exit(self):
        async def scenario():
            ch = Channel(run_id="r6")
            assert ch.num_subscribers == 0
            async with ch.subscribe(replay=False):
                assert ch.num_subscribers == 1
            assert ch.num_subscribers == 0

        run_async(scenario())

    def test_history_size_capped(self):
        async def scenario():
            ch = Channel(run_id="r7", history_size=3)
            for i in range(10):
                await ch.publish(make_event(EVENT_STEP_LOG, {"step": i}))
            assert ch.num_events == 3
            steps = [e["data"]["step"] for e in ch.history]
            assert steps == [7, 8, 9]

        run_async(scenario())

    def test_finished_marker(self):
        async def scenario():
            ch = Channel(run_id="r8")
            assert ch.finished is False
            await ch.publish(make_event(EVENT_FINISHED, {}))
            assert ch.finished is True

        run_async(scenario())


# ---------------------------------------------------------------------------
# EventBroadcaster: API top-level
# ---------------------------------------------------------------------------


class TestEventBroadcaster:
    def test_get_or_create_channel(self):
        async def scenario():
            b = EventBroadcaster()
            assert b.num_channels == 0
            ch = await b.get_or_create_channel("run-x")
            assert b.num_channels == 1
            assert ch.run_id == "run-x"

            # Idempotenza: stessa istanza
            ch2 = await b.get_or_create_channel("run-x")
            assert ch is ch2
            assert b.num_channels == 1

        run_async(scenario())

    def test_get_channel_missing(self):
        async def scenario():
            b = EventBroadcaster()
            assert await b.get_channel("nope") is None

        run_async(scenario())

    def test_publish_creates_channel(self):
        async def scenario():
            b = EventBroadcaster()
            ev = make_event(EVENT_STEP_LOG, {"loss": 0.5})
            await b.publish("run-y", ev)
            ch = await b.get_channel("run-y")
            assert ch is not None
            assert ch.num_events == 1

        run_async(scenario())

    def test_remove_channel(self):
        async def scenario():
            b = EventBroadcaster()
            await b.get_or_create_channel("run-r")
            assert b.num_channels == 1
            assert await b.remove_channel("run-r") is True
            assert b.num_channels == 0
            assert await b.remove_channel("run-r") is False

        run_async(scenario())


# ---------------------------------------------------------------------------
# publish_from_thread: cross-thread bridging
# ---------------------------------------------------------------------------


class TestPublishFromThread:
    def test_no_bind_loop_drops_silently(self):
        """Senza bind_loop, publish_from_thread fa solo log e ritorna."""
        b = EventBroadcaster()
        # Nessun bind_loop
        b.publish_from_thread("run-z", make_event(EVENT_STEP_LOG, {"x": 1}))
        # Nessuna eccezione, nessun channel creato
        assert b.num_channels == 0

    def test_publish_from_thread_delivers_to_main_loop(self):
        """
        Test cross-thread: simuliamo un thread "training" che chiama
        publish_from_thread mentre il main loop è in attesa.
        """
        async def scenario():
            b = EventBroadcaster()
            b.bind_loop(asyncio.get_running_loop())

            received: list[dict] = []

            async def consumer():
                ch = await b.get_or_create_channel("run-t")
                async with ch.subscribe(replay=False) as queue:
                    for _ in range(3):
                        ev = await asyncio.wait_for(queue.get(), timeout=2.0)
                        received.append(ev)

            consumer_task = asyncio.create_task(consumer())
            await asyncio.sleep(0.05)  # consumer in attesa

            # Lancia 3 publish da un thread separato
            def thread_publisher():
                for i in range(3):
                    b.publish_from_thread(
                        "run-t", make_event(EVENT_STEP_LOG, {"step": i})
                    )
                    time.sleep(0.02)

            t = threading.Thread(target=thread_publisher)
            t.start()
            t.join()

            await consumer_task
            assert len(received) == 3
            assert [e["data"]["step"] for e in received] == [0, 1, 2]

        run_async(scenario())


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_finished_old_channels(self):
        async def scenario():
            b = EventBroadcaster()
            await b.publish("run-old", make_event(EVENT_FINISHED, {}))
            ch = await b.get_channel("run-old")
            # Forziamo timestamp vecchio
            old_event = ch.history[-1]
            ch.history.clear()
            ch.history.append(
                {**old_event, "timestamp": "2020-01-01T00:00:00+00:00"}
            )

            removed = await b.cleanup_finished(max_age_seconds=3600)
            assert removed == 1
            assert b.num_channels == 0

        run_async(scenario())

    def test_cleanup_keeps_active_channels(self):
        async def scenario():
            b = EventBroadcaster()
            await b.publish(
                "run-active", make_event(EVENT_STEP_LOG, {"step": 1})
            )
            removed = await b.cleanup_finished(max_age_seconds=0)
            assert removed == 0
            assert b.num_channels == 1

        run_async(scenario())