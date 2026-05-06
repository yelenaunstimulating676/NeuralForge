"""
Orchestratore del Training Engine.

Mette insieme tutti i moduli M4.1-M4.5:
  - data.py        → InstructionTuningDataset + DataCollator
  - model.py       → load + LoRA
  - optimizer.py   → AdamW8bit + cosine warmup
  - loop.py        → training loop
  - checkpoint.py  → save periodico + final

Espone una funzione top-level `run_training(config)` che esegue tutto
in un colpo, gestisce lo stato nel DB (TrainingRun), e ritorna
un TrainingOutcome.

Cancellazione: il caller passa un threading.Event come `cancel_event`,
il loop lo controlla a ogni step.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from config import settings
from core.training.checkpoint import (
    TrainerState,
    delete_run_directory,
    keep_only_last_n_checkpoints,
    save_checkpoint,
)
from core.training.data import (
    DataCollatorWithPadding,
    DataConfig,
    InstructionTuningDataset,
)
from core.training.loop import (
    CancellationToken,
    LoopConfig,
    StepLog,
    TrainingResult,
    train_loop,
)
from core.training.model import (
    LoraConfigParams,
    QuantizationConfig,
    prepare_model_for_training,
)
from core.training.optimizer import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)
from db.models import (
    BaseModel as BaseModelRow,
    Dataset as DatasetRow,
    FineTunedModel as FineTunedModelRow,
    TrainingRun,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione completa training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Configurazione completa di un training run."""

    base_model_id: int
    dataset_id: int
    # Loop
    num_epochs: int = 3
    per_device_batch_size: int = 2
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    log_every_n_steps: int = 1
    max_steps: int = 0
    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    use_8bit_optimizer: bool = True
    # Scheduler
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.0
    # Data
    max_seq_length: int = 1024
    train_on_response_only: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Quantization
    use_4bit: bool = True
    compute_dtype: str = "bfloat16"
    # Checkpoint
    save_every_n_steps: int = 100   # 0 = solo final
    keep_last_n: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "dataset_id": self.dataset_id,
            "num_epochs": self.num_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "use_8bit_optimizer": self.use_8bit_optimizer,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "compute_dtype": self.compute_dtype,
            "save_every_n_steps": self.save_every_n_steps,
            "max_steps": self.max_steps,
            "max_grad_norm": self.max_grad_norm,
            "train_on_response_only": self.train_on_response_only,
        }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingOutcome:
    """Risultato finale di un training."""

    run_id: str
    training_run_db_id: int
    status: str                 # 'completed' | 'failed' | 'cancelled'
    final_loss: float
    total_steps: int
    elapsed_seconds: float
    final_checkpoint_path: str | None
    finetuned_model_id: int | None
    history: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "training_run_db_id": self.training_run_db_id,
            "status": self.status,
            "final_loss": round(self.final_loss, 6),
            "total_steps": self.total_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "final_checkpoint_path": self.final_checkpoint_path,
            "finetuned_model_id": self.finetuned_model_id,
            "history_count": len(self.history),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_run_id() -> str:
    """Genera un run_id leggibile: train-YYYYMMDD-XXXX."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    short = uuid.uuid4().hex[:6]
    return f"train-{today}-{short}"


def _resolve_base_model(session: Session, base_model_id: int) -> BaseModelRow:
    row = session.get(BaseModelRow, base_model_id)
    if row is None:
        raise ValueError(f"Base model id={base_model_id} non trovato.")
    if not Path(row.local_path).exists():
        raise ValueError(
            f"Base model {row.hf_repo!r} non presente su disco: {row.local_path}"
        )
    return row


def _resolve_dataset(session: Session, dataset_id: int) -> DatasetRow:
    row = session.get(DatasetRow, dataset_id)
    if row is None:
        raise ValueError(f"Dataset id={dataset_id} non trovato.")
    if not Path(row.file_path).exists():
        raise ValueError(
            f"Dataset {row.name!r} non presente su disco: {row.file_path}"
        )
    return row


def _create_training_run_row(
    session: Session,
    base_model_id: int,
    dataset_id: int,
    config: TrainingConfig,
) -> TrainingRun:
    row = TrainingRun(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        status="pending",
        config_json=json.dumps(config.to_dict(), ensure_ascii=False),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _update_training_status(
    session: Session,
    run_db: TrainingRun,
    status: str,
    *,
    error: str | None = None,
    metrics: dict | None = None,
) -> None:
    run_db.status = status
    if error is not None:
        run_db.error_message = error
    if metrics is not None:
        run_db.metrics_json = json.dumps(metrics, ensure_ascii=False)
    if status == "running" and run_db.started_at is None:
        run_db.started_at = datetime.now(timezone.utc)
    if status in {"completed", "failed", "cancelled"}:
        run_db.finished_at = datetime.now(timezone.utc)
    session.commit()


def _create_finetuned_model_row(
    session: Session,
    *,
    base_model_id: int,
    training_run_id: int,
    name: str,
    adapter_path: Path,
    metrics: dict,
) -> FineTunedModelRow:
    """Registra il FineTunedModel risultante dal training."""
    # Calcola size della cartella adapter
    size = sum(
        f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()
    )
    row = FineTunedModelRow(
        base_model_id=base_model_id,
        training_run_id=training_run_id,
        name=name,
        adapter_path=str(adapter_path.resolve()),
        size_bytes=size,
        metrics_json=json.dumps(metrics, ensure_ascii=False),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# ---------------------------------------------------------------------------
# Top-level: run_training
# ---------------------------------------------------------------------------


def run_training(
    *,
    session: Session,
    config: TrainingConfig,
    cancel_event: threading.Event | None = None,
    on_step: callable | None = None,
    finetuned_name: str | None = None,
) -> TrainingOutcome:
    """
    Esegue un training completo end-to-end.

    Args:
        session: SQLAlchemy session per CRUD su TrainingRun.
        config: configurazione completa.
        cancel_event: Event per cancellazione cooperativa. Se None, no cancel.
        on_step: callback opzionale chiamato a ogni log step (passa StepLog).
        finetuned_name: nome user-facing del FineTunedModel risultante.
            Se None, derivato da run_id.

    Returns:
        TrainingOutcome.
    """
    run_id = _generate_run_id()
    logger.info("=== Inizio training run %s ===", run_id)

    # 1. Risolvi base model + dataset (può sollevare se mancanti)
    try:
        base_model_row = _resolve_base_model(session, config.base_model_id)
        dataset_row = _resolve_dataset(session, config.dataset_id)
    except ValueError as exc:
        # Non possiamo creare il TrainingRun se le FK non risolvono
        logger.error("Risoluzione FK fallita: %s", exc)
        raise

    # 2. Crea record TrainingRun in DB (status=pending)
    run_db = _create_training_run_row(
        session, config.base_model_id, config.dataset_id, config
    )

    # 3. Configura cancellazione
    cancel_token = (
        CancellationToken.from_event(cancel_event)
        if cancel_event is not None
        else CancellationToken()
    )

    final_path: Path | None = None
    finetuned_id: int | None = None

    try:
        _update_training_status(session, run_db, "running")

        # 4. Carica modello + LoRA
        logger.info("Caricamento modello base: %s", base_model_row.hf_repo)
        loaded = prepare_model_for_training(
            model_path=Path(base_model_row.local_path),
            family_tag=base_model_row.tag,
            quant_config=QuantizationConfig(
                load_in_4bit=config.use_4bit,
                bnb_4bit_compute_dtype=config.compute_dtype,
            ),
            lora_params=LoraConfigParams(
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            ),
        )
        logger.info(
            "Modello pronto. Trainable: %d / %d (%.4f%%)",
            loaded.trainable_info.trainable_params,
            loaded.trainable_info.total_params,
            loaded.trainable_info.trainable_percent,
        )

        # 5. Carica dataset
        logger.info("Caricamento dataset: %s", dataset_row.name)
        ds = InstructionTuningDataset(
            jsonl_path=Path(dataset_row.file_path),
            tokenizer=loaded.tokenizer,
            config=DataConfig(
                max_seq_length=config.max_seq_length,
                train_on_response_only=config.train_on_response_only,
            ),
        )
        collator = DataCollatorWithPadding(tokenizer=loaded.tokenizer)
        loader = DataLoader(
            ds,
            batch_size=config.per_device_batch_size,
            collate_fn=collator,
            shuffle=True,
        )

        # 6. Calcola total_steps per scheduler
        steps_per_epoch = max(
            1,
            len(ds) // (config.per_device_batch_size * config.grad_accum_steps),
        )
        total_steps = steps_per_epoch * config.num_epochs
        if config.max_steps > 0:
            total_steps = min(total_steps, config.max_steps)
        logger.info(
            "Steps stimati: %d (per_epoch=%d × epochs=%d)",
            total_steps, steps_per_epoch, config.num_epochs,
        )

        # 7. Optimizer + Scheduler
        optimizer = build_optimizer(
            loaded.model,
            OptimizerConfig(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                use_8bit=config.use_8bit_optimizer,
            ),
        )
        scheduler = build_scheduler(
            optimizer,
            SchedulerConfig(
                total_steps=total_steps,
                warmup_ratio=config.warmup_ratio,
                min_lr_ratio=config.min_lr_ratio,
            ),
        )

        # 8. Setup callback per save periodico
        history: list[StepLog] = []

        def step_callback(log_entry: StepLog) -> None:
            history.append(log_entry)

            # Save periodico
            if (
                config.save_every_n_steps > 0
                and log_entry.step % config.save_every_n_steps == 0
            ):
                state = TrainerState(
                    run_id=run_id,
                    step=log_entry.step,
                    epoch=log_entry.epoch,
                    final_loss=log_entry.loss,
                    history=[h.to_dict() for h in history],
                    base_model_path=base_model_row.local_path,
                    family_tag=base_model_row.tag,
                )
                try:
                    save_checkpoint(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        trainer_state=state,
                    )
                    keep_only_last_n_checkpoints(run_id, n=config.keep_last_n)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Errore save checkpoint: %s", exc)

            # Forward al callback esterno
            if on_step is not None:
                try:
                    on_step(log_entry)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Errore in on_step esterno: %s", exc)

        # 9. Train loop
        loop_result = train_loop(
            model=loaded.model,
            train_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=LoopConfig(
                num_epochs=config.num_epochs,
                grad_accum_steps=config.grad_accum_steps,
                max_grad_norm=config.max_grad_norm,
                log_every_n_steps=config.log_every_n_steps,
                max_steps=config.max_steps,
            ),
            cancel_token=cancel_token,
            on_step=step_callback,
        )

        # 10. Save final checkpoint
        final_state = TrainerState(
            run_id=run_id,
            step=loop_result.total_steps,
            epoch=config.num_epochs - 1,
            final_loss=loop_result.final_loss,
            history=[h.to_dict() for h in loop_result.history],
            base_model_path=base_model_row.local_path,
            family_tag=base_model_row.tag,
        )
        final_path = save_checkpoint(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=final_state,
            is_final=True,
        )

        # 11. Determina status finale
        if loop_result.cancelled:
            status = "cancelled"
        elif loop_result.completed:
            status = "completed"
        else:
            status = "failed"

        metrics = {
            "final_loss": loop_result.final_loss,
            "total_steps": loop_result.total_steps,
            "elapsed_seconds": loop_result.elapsed_seconds,
            "history": [h.to_dict() for h in loop_result.history],
        }
        _update_training_status(session, run_db, status, metrics=metrics)

        # 12. Crea FineTunedModel se completato
        if status == "completed":
            ft_name = finetuned_name or f"{base_model_row.display_name} ({run_id})"
            ft_row = _create_finetuned_model_row(
                session,
                base_model_id=base_model_row.id,
                training_run_id=run_db.id,
                name=ft_name,
                adapter_path=final_path,
                metrics=metrics,
            )
            finetuned_id = ft_row.id
            logger.info("FineTunedModel creato: id=%d", finetuned_id)

        logger.info(
            "=== Training run %s terminato: status=%s loss=%.4f steps=%d ===",
            run_id, status, loop_result.final_loss, loop_result.total_steps,
        )

        return TrainingOutcome(
            run_id=run_id,
            training_run_db_id=run_db.id,
            status=status,
            final_loss=loop_result.final_loss,
            total_steps=loop_result.total_steps,
            elapsed_seconds=loop_result.elapsed_seconds,
            final_checkpoint_path=str(final_path),
            finetuned_model_id=finetuned_id,
            history=[h.to_dict() for h in loop_result.history],
        )

    except Exception as exc:
        # Failure: aggiorna status e re-raise
        logger.exception("Training fallito per run %s", run_id)
        try:
            _update_training_status(
                session, run_db, "failed", error=f"{type(exc).__name__}: {exc}"
            )
        except Exception:  # noqa: BLE001
            pass
        # Cleanup directory parziale (best-effort)
        try:
            delete_run_directory(run_id)
        except Exception:  # noqa: BLE001
            pass

        return TrainingOutcome(
            run_id=run_id,
            training_run_db_id=run_db.id,
            status="failed",
            final_loss=0.0,
            total_steps=0,
            elapsed_seconds=0.0,
            final_checkpoint_path=None,
            finetuned_model_id=None,
            error=f"{type(exc).__name__}: {exc}",
        )