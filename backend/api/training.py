"""
Endpoint REST `/api/training/*`.

Workflow:
    POST /api/training/estimate         → stima pre-flight (per la UI form)
    POST /api/training/start            → avvia training, ritorna run_id+job_id
    GET  /api/training/runs             → storico TrainingRun
    GET  /api/training/runs/{id}        → dettaglio
    POST /api/training/runs/{id}/cancel → cancellazione cooperativa
    DELETE /api/training/runs/{id}      → cancella run + adapter
    GET  /api/training/jobs             → job attivi (per polling generico)
    GET  /api/training/jobs/{id}        → status singolo job
"""

from __future__ import annotations

import json
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import WebSocket, WebSocketDisconnect

from api.schemas import (
    TrainingStartRequestSchema,
    TrainingEstimateRequestSchema,
    TrainingEstimateResponseSchema,
    TrainingJobSchema,
    TrainingRunSchema,
    TrainingStartResponseSchema,
)
from core.jobs import job_manager
from core.training.broadcaster import (
    EVENT_FINISHED,
    EVENT_STATUS,
    EVENT_STEP_LOG,
    broadcaster,
    make_event,
)
from core.training.checkpoint import delete_run_directory
from core.training.estimator import estimate_training
from core.training.runner import TrainingConfig, run_training
from db import SessionLocal, get_session
from db.models import (
    BaseModel as BaseModelRow,
    Dataset as DatasetRow,
    FineTunedModel as FineTunedModelRow,
    TrainingRun,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


# Mapping job_id → run_id + db_id, mantenuto a runtime per lookup veloci
# dei WS handler. Cleanup automatico col job_manager.
_job_metadata: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_base_model(session: Session, base_model_id: int) -> BaseModelRow:
    row = session.get(BaseModelRow, base_model_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Base model id={base_model_id} non trovato.")
    return row


def _resolve_dataset(session: Session, dataset_id: int) -> DatasetRow:
    row = session.get(DatasetRow, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Dataset id={dataset_id} non trovato.")
    return row


def _row_to_run_schema(
    run: TrainingRun,
    base_name: str | None = None,
    dataset_name: str | None = None,
) -> TrainingRunSchema:
    """Converte un record TrainingRun in TrainingRunSchema."""
    config = None
    metrics = None
    if run.config_json:
        try:
            config = json.loads(run.config_json)
        except json.JSONDecodeError:
            pass
    if run.metrics_json:
        try:
            metrics = json.loads(run.metrics_json)
        except json.JSONDecodeError:
            pass

    return TrainingRunSchema(
        id=run.id,
        run_id=run.run_id,
        base_model_id=run.base_model_id,
        base_model_name=base_name,
        dataset_id=run.dataset_id,
        dataset_name=dataset_name,
        status=run.status,
        config=config,
        metrics=metrics,
        error_message=run.error_message,
        started_at=run.started_at.isoformat() if run.started_at else None,
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        created_at=run.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


@router.post(
    "/estimate",
    response_model=TrainingEstimateResponseSchema,
    summary="Stima euristica VRAM/tempo per la UI",
)
def estimate_endpoint(
    body: TrainingEstimateRequestSchema,
    session: Session = Depends(get_session),
) -> TrainingEstimateResponseSchema:
    """
    Calcola una stima di VRAM e tempo basata su modello + dataset + config.
    NON è un benchmark: è una formula euristica, margine ±30%.
    """
    base = _resolve_base_model(session, body.base_model_id)
    dataset = _resolve_dataset(session, body.dataset_id)

    estimate = estimate_training(
        params_billions=base.params_billions or 0.1,
        num_examples=dataset.num_examples,
        num_epochs=body.num_epochs,
        per_device_batch_size=body.per_device_batch_size,
        grad_accum_steps=body.grad_accum_steps,
        max_seq_length=body.max_seq_length,
        lora_r=body.lora_r,
        use_4bit=body.use_4bit,
    )

    return TrainingEstimateResponseSchema(
        estimated_vram_mb=estimate.estimated_vram_mb,
        estimated_time_seconds=estimate.estimated_time_seconds,
        total_steps=estimate.total_steps,
        steps_per_epoch=estimate.steps_per_epoch,
        trainable_params_estimated=estimate.trainable_params_estimated,
        notes=estimate.notes,
    )


# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------


def _build_training_coroutine_factory(
    config: TrainingConfig,
    finetuned_name: str | None,
):
    """
    Costruisce una coroutine factory compatibile col JobManager.
    All'interno, esegue `run_training` in un thread separato e bridge
    gli eventi al broadcaster.
    """

    async def factory(progress_cb, cancel_event):
        # cancel_event qui è un asyncio.Event, ma run_training si aspetta
        # un threading.Event. Lo convertiamo.
        thread_cancel = threading.Event()

        # Watch task: setta thread_cancel quando asyncio cancel_event scatta
        async def cancel_watcher():
            await cancel_event.wait()
            thread_cancel.set()

        watcher = __import__("asyncio").create_task(cancel_watcher())

        # run_id non lo conosciamo ancora — verrà generato da run_training.
        # Lo recupereremo via TrainingOutcome.
        # Per pubblicare eventi DURANTE il training, ci serve subito il
        # run_id. Lo otteniamo modificando leggermente il flusso: il
        # callback step contiene già il run_id grazie al binding del
        # TrainerState dentro runner.py.
        #
        # Tuttavia run_id viene generato in run_training(); per averlo
        # PRIMA degli step, lo passiamo come dict mutabile condiviso.
        run_id_holder: dict = {"run_id": None}

        def step_callback(step_log):
            """Callback chiamato dal thread training per ogni step."""
            run_id = run_id_holder["run_id"]
            if run_id is None:
                # Recuperiamo run_id dal log se c'è (in step_log non c'è
                # direttamente, dovremmo passarlo via wrap).
                # Soluzione: il run_id viene fornito a wrap_run_training.
                return

            # Pubblica evento step_log al broadcaster
            broadcaster.publish_from_thread(
                run_id,
                make_event(EVENT_STEP_LOG, step_log.to_dict()),
            )

            # Aggiorna progress JobManager
            # Non sappiamo total_steps qui, calcoliamo approssimativamente
            # dalla loop config (sarà raffinato dopo).
            try:
                progress_cb(
                    min(0.99, step_log.step / max(1, config.num_epochs * 100)),
                    f"Step {step_log.step} loss={step_log.loss:.4f}",
                )
            except Exception:  # noqa: BLE001
                pass

        # Wrap che salva il run_id appena viene generato e fa il bridging
        from core.training.runner import _generate_run_id
        run_id = _generate_run_id()
        run_id_holder["run_id"] = run_id

        # Pubblichiamo subito uno status_change "started"
        broadcaster.publish_from_thread(
            run_id,
            make_event(EVENT_STATUS, {"status": "running", "run_id": run_id}),
        )

        import asyncio as _asyncio

        # Esegui il training in un thread (run_training è bloccante)
        def thread_target():
            with SessionLocal() as session:
                # Hack: forziamo il run_id nel runner via monkeypatch della
                # funzione di generazione (per questa singola call)
                import core.training.runner as runner_mod
                _original = runner_mod._generate_run_id
                runner_mod._generate_run_id = lambda: run_id
                try:
                    return run_training(
                        session=session,
                        config=config,
                        cancel_event=thread_cancel,
                        on_step=step_callback,
                        finetuned_name=finetuned_name,
                    )
                finally:
                    runner_mod._generate_run_id = _original

        outcome = await _asyncio.to_thread(thread_target)

        # Cancella il watcher
        watcher.cancel()
        try:
            await watcher
        except _asyncio.CancelledError:
            pass

        # Pubblica evento finale
        broadcaster.publish_from_thread(
            run_id,
            make_event(
                EVENT_FINISHED,
                {
                    "status": outcome.status,
                    "final_loss": outcome.final_loss,
                    "total_steps": outcome.total_steps,
                    "elapsed_seconds": outcome.elapsed_seconds,
                    "error": outcome.error,
                    "finetuned_model_id": outcome.finetuned_model_id,
                    "training_run_db_id": outcome.training_run_db_id,
                },
            ),
        )

        progress_cb(1.0, f"Training {outcome.status}: loss={outcome.final_loss:.4f}")

        return {
            "run_id": run_id,
            "training_run_db_id": outcome.training_run_db_id,
            "status": outcome.status,
            "final_loss": outcome.final_loss,
            "total_steps": outcome.total_steps,
            "finetuned_model_id": outcome.finetuned_model_id,
        }

    return factory


@router.post(
    "/start",
    response_model=TrainingStartResponseSchema,
    summary="Avvia un training asincrono",
)
async def start_training(
    body: TrainingStartRequestSchema,
    session: Session = Depends(get_session),
) -> TrainingStartRequestSchema:
    """
    Avvia un nuovo training in background tramite il JobManager.
    Ritorna subito il run_id + job_id, il client deve poi:
      - fare polling su /api/training/jobs/{job_id} per lo status grezzo
      - oppure connettersi via WebSocket /ws/training/{run_id} per gli eventi
    """
    base = _resolve_base_model(session, body.base_model_id)
    dataset = _resolve_dataset(session, body.dataset_id)

    # Costruisci TrainingConfig dal request
    config = TrainingConfig(
        base_model_id=body.base_model_id,
        dataset_id=body.dataset_id,
        num_epochs=body.num_epochs,
        per_device_batch_size=body.per_device_batch_size,
        grad_accum_steps=body.grad_accum_steps,
        max_grad_norm=body.max_grad_norm,
        log_every_n_steps=body.log_every_n_steps,
        max_steps=body.max_steps,
        learning_rate=body.learning_rate,
        weight_decay=body.weight_decay,
        use_8bit_optimizer=body.use_8bit_optimizer,
        warmup_ratio=body.warmup_ratio,
        min_lr_ratio=body.min_lr_ratio,
        max_seq_length=body.max_seq_length,
        train_on_response_only=body.train_on_response_only,
        lora_r=body.lora_r,
        lora_alpha=body.lora_alpha,
        lora_dropout=body.lora_dropout,
        use_4bit=body.use_4bit,
        compute_dtype=body.compute_dtype,
        save_every_n_steps=body.save_every_n_steps,
        keep_last_n=body.keep_last_n,
    )

    # Pre-genera il run_id così possiamo collegarlo subito al job
    from core.training.runner import _generate_run_id
    run_id = _generate_run_id()

    # Costruisci la factory wrappando il run_id pre-generato
    def _factory_with_fixed_run_id():
        async def factory(progress_cb, cancel_event):
            thread_cancel = threading.Event()
            import asyncio as _asyncio

            async def cancel_watcher():
                await cancel_event.wait()
                thread_cancel.set()

            watcher = _asyncio.create_task(cancel_watcher())

            broadcaster.publish_from_thread(
                run_id,
                make_event(EVENT_STATUS, {"status": "running", "run_id": run_id}),
            )

            def step_callback(step_log):
                broadcaster.publish_from_thread(
                    run_id,
                    make_event(EVENT_STEP_LOG, step_log.to_dict()),
                )
                # Progress: usa lo step relativo a una stima totale euristica
                try:
                    progress_cb(
                        min(0.99, step_log.step / max(1, body.num_epochs * 50)),
                        f"Step {step_log.step} loss={step_log.loss:.4f}",
                    )
                except Exception:  # noqa: BLE001
                    pass

            def thread_target():
                with SessionLocal() as new_session:
                    import core.training.runner as runner_mod
                    _original = runner_mod._generate_run_id
                    runner_mod._generate_run_id = lambda: run_id
                    try:
                        return run_training(
                            session=new_session,
                            config=config,
                            cancel_event=thread_cancel,
                            on_step=step_callback,
                            finetuned_name=body.finetuned_name,
                        )
                    finally:
                        runner_mod._generate_run_id = _original

            outcome = await _asyncio.to_thread(thread_target)

            watcher.cancel()
            try:
                await watcher
            except _asyncio.CancelledError:
                pass

            broadcaster.publish_from_thread(
                run_id,
                make_event(
                    EVENT_FINISHED,
                    {
                        "status": outcome.status,
                        "final_loss": outcome.final_loss,
                        "total_steps": outcome.total_steps,
                        "elapsed_seconds": outcome.elapsed_seconds,
                        "error": outcome.error,
                        "finetuned_model_id": outcome.finetuned_model_id,
                        "training_run_db_id": outcome.training_run_db_id,
                    },
                ),
            )

            progress_cb(
                1.0, f"{outcome.status}: loss={outcome.final_loss:.4f}"
            )

            # Salva il db_id nel metadata del job
            _job_metadata.setdefault(run_id, {})["training_run_db_id"] = (
                outcome.training_run_db_id
            )

            return {
                "run_id": run_id,
                "training_run_db_id": outcome.training_run_db_id,
                "status": outcome.status,
                "final_loss": outcome.final_loss,
                "total_steps": outcome.total_steps,
                "finetuned_model_id": outcome.finetuned_model_id,
            }

        return factory

    job = await job_manager.submit("training", _factory_with_fixed_run_id())

    # Crea il TrainingRun in DB SUBITO con status "pending" così appare in lista
    # Il runner lo aggiornerà a "running" appena parte.
    # NOTA: il runner di M4.6 crea già il record. Per evitare duplicati,
    # NON lo creiamo qui — lasciamo che lo crei lui. Ritorniamo db_id=None
    # per ora, sarà aggiornato dalla UI tramite polling.
    # Tuttavia per la response dobbiamo dare un db_id... soluzione:
    # creiamo il record qui, poi il runner lo riusa.
    # Per semplicità di M5.2: creiamo qui un placeholder, lo aggiorneremo
    # dopo il primo step real-world.

    # Track metadata
    _job_metadata[run_id] = {
        "job_id": job.id,
        "run_id": run_id,
        "training_run_db_id": None,  # verrà popolato dal runner
    }

    logger.info(
        "Training submitted: run_id=%s job_id=%s base_model=%s dataset=%s",
        run_id, job.id, base.hf_repo, dataset.name,
    )

    return TrainingStartResponseSchema(
        run_id=run_id,
        job_id=job.id,
        training_run_db_id=0,  # placeholder, il client può fare polling
    )


# ---------------------------------------------------------------------------
# Runs (DB)
# ---------------------------------------------------------------------------


@router.get(
    "/runs",
    response_model=list[TrainingRunSchema],
    summary="Lista TrainingRun (storico DB)",
)
def list_runs(
    session: Session = Depends(get_session),
) -> list[TrainingRunSchema]:
    """Lista tutti i training run, ordinati per data creazione (recenti prima)."""
    stmt = select(TrainingRun).order_by(TrainingRun.created_at.desc())
    runs = list(session.scalars(stmt).all())
    out: list[TrainingRunSchema] = []
    for r in runs:
        base = session.get(BaseModelRow, r.base_model_id)
        ds = session.get(DatasetRow, r.dataset_id) if r.dataset_id else None
        out.append(_row_to_run_schema(
            r,
            base_name=base.display_name if base else None,
            dataset_name=ds.name if ds else None,
        ))
    return out


@router.get(
    "/runs/{run_db_id}",
    response_model=TrainingRunSchema,
    summary="Dettaglio TrainingRun",
)
def get_run(
    run_db_id: int,
    session: Session = Depends(get_session),
) -> TrainingRunSchema:
    run = session.get(TrainingRun, run_db_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"TrainingRun id={run_db_id} non trovato.")

    base = session.get(BaseModelRow, run.base_model_id)
    ds = session.get(DatasetRow, run.dataset_id) if run.dataset_id else None
    return _row_to_run_schema(
        run,
        base_name=base.display_name if base else None,
        dataset_name=ds.name if ds else None,
    )


@router.post(
    "/runs/{run_db_id}/cancel",
    summary="Cancella un training in corso",
)
async def cancel_run(
    run_db_id: int,
    session: Session = Depends(get_session),
) -> dict:
    """
    Richiede la cancellazione cooperativa di un training.
    """
    run = session.get(TrainingRun, run_db_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"TrainingRun id={run_db_id} non trovato.")
    if run.status not in {"pending", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"TrainingRun id={run_db_id} è già in stato '{run.status}'.",
        )

    # Trova il job_id dal metadata cache
    target_job_id: str | None = None
    for run_id, meta in _job_metadata.items():
        if meta.get("training_run_db_id") == run_db_id:
            target_job_id = meta.get("job_id")
            break

    if target_job_id is None:
        # Fallback: cerca il job più recente di kind=training in pending/running
        all_jobs = await job_manager.list(kind="training")
        for j in all_jobs:
            if j.status.value in {"pending", "running"}:
                target_job_id = j.id
                break

    if target_job_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Nessun job attivo trovato per TrainingRun id={run_db_id}.",
        )

    cancelled = await job_manager.cancel(target_job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Cancellazione del job {target_job_id} fallita o già terminato.",
        )

    return {"cancelled": True, "job_id": target_job_id}


@router.delete(
    "/runs/{run_db_id}",
    summary="Cancella un TrainingRun (DB + adapter)",
)
def delete_run(
    run_db_id: int,
    remove_files: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> dict:
    """
    Cancella un TrainingRun dal DB. Se `remove_files=True`, rimuove anche
    la directory dell'adapter su disco. Cascade su FineTunedModel.
    """
    run = session.get(TrainingRun, run_db_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"TrainingRun id={run_db_id} non trovato.")

    # Trova run_id dal metadata cache
    run_id_str: str | None = None
    for r_id, meta in _job_metadata.items():
        if meta.get("training_run_db_id") == run_db_id:
            run_id_str = r_id
            break

    session.delete(run)
    session.commit()

    if remove_files and run_id_str:
        try:
            delete_run_directory(run_id_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Errore rimozione adapter %s: %s", run_id_str, exc)

    return {"deleted": True, "id": run_db_id}


# ---------------------------------------------------------------------------
# Jobs (JobManager)
# ---------------------------------------------------------------------------


@router.get(
    "/jobs",
    response_model=list[TrainingJobSchema],
    summary="Lista job di training attivi e recenti",
)
async def list_training_jobs() -> list[TrainingJobSchema]:
    """Lista i job di kind=training (attivi e recenti) dal JobManager."""
    jobs = await job_manager.list(kind="training")
    out = []
    for j in jobs:
        # run_id e db_id dal metadata se disponibili
        run_id = None
        db_id = None
        for r_id, meta in _job_metadata.items():
            if meta.get("job_id") == j.id:
                run_id = r_id
                db_id = meta.get("training_run_db_id")
                break

        # Se db_id non c'è in metadata, prova a ricavarlo dal result del job
        if db_id is None and j.result and isinstance(j.result, dict):
            db_id = j.result.get("training_run_db_id")
        if run_id is None and j.result and isinstance(j.result, dict):
            run_id = j.result.get("run_id")

        out.append(TrainingJobSchema(
            job_id=j.id,
            run_id=run_id,
            training_run_db_id=db_id,
            status=j.status.value,
            progress=j.progress,
            progress_message=j.progress_message,
            error=j.error,
            created_at=j.created_at.isoformat(),
            started_at=j.started_at.isoformat() if j.started_at else None,
            finished_at=j.finished_at.isoformat() if j.finished_at else None,
        ))
    return out


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobSchema,
    summary="Status di un singolo job",
)
async def get_training_job(job_id: str) -> TrainingJobSchema:
    j = await job_manager.get(job_id)
    if j is None or j.kind != "training":
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato.")

    run_id = None
    db_id = None
    for r_id, meta in _job_metadata.items():
        if meta.get("job_id") == job_id:
            run_id = r_id
            db_id = meta.get("training_run_db_id")
            break
    if db_id is None and j.result and isinstance(j.result, dict):
        db_id = j.result.get("training_run_db_id")
    if run_id is None and j.result and isinstance(j.result, dict):
        run_id = j.result.get("run_id")

    return TrainingJobSchema(
        job_id=j.id,
        run_id=run_id,
        training_run_db_id=db_id,
        status=j.status.value,
        progress=j.progress,
        progress_message=j.progress_message,
        error=j.error,
        created_at=j.created_at.isoformat(),
        started_at=j.started_at.isoformat() if j.started_at else None,
        finished_at=j.finished_at.isoformat() if j.finished_at else None,
    )
    
    
    # ---------------------------------------------------------------------------
# WebSocket: stream eventi live di un run
# ---------------------------------------------------------------------------


@router.websocket("/ws/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str) -> None:
    """
    Stream live degli eventi di un training run.

    Comportamento:
      1. Accetta la connessione
      2. Sottoscrive al Channel del `run_id` (history replay incluso)
      3. Streama eventi al client finché:
         - arriva un evento di tipo `finished` (training terminato)
         - il client disconnette
         - errore di rete
      4. Cleanup automatico: il channel resta vivo per altri subscriber.

    Formato eventi: JSON con struttura `{type, timestamp, data}`.
    """
    await websocket.accept()
    logger.info("WebSocket aperto per run_id=%s", run_id)

    # Crea il channel se non esiste (es. se il client si connette PRIMA
    # del primo evento). Subscriber riceverà gli eventi appena pubblicati.
    channel = await broadcaster.get_or_create_channel(run_id)

    try:
        async with channel.subscribe(replay=True) as queue:
            while True:
                # Aspetta evento dal channel
                event = await queue.get()

                # Invia al client
                try:
                    await websocket.send_json(event)
                except (WebSocketDisconnect, RuntimeError):
                    # Client chiuso, esci
                    break

                # Se è un evento di terminazione, chiudi pulito
                if event.get("type") in {"finished", "error"}:
                    logger.info(
                        "Run %s terminato (type=%s), chiudo WS.",
                        run_id, event.get("type"),
                    )
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnesso da run %s", run_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore WebSocket run %s: %s", run_id, exc)
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass