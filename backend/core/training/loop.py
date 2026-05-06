"""
Training loop del Training Engine.

Compito: prendere un PeftModel + DataLoader + Optimizer + Scheduler già
costruiti e farci girare sopra il training, gestendo:
  - Forward / backward
  - Gradient accumulation (batch effettivo > batch fisico)
  - Gradient clipping
  - Mixed precision via bf16 compute dtype del modello
  - Logging strutturato per step
  - Cancellazione cooperativa

Output: TrainingResult con history dei log + stato finale.

Note di design:
  - NON gestiamo qui il save dei checkpoint (responsabilità di M4.5)
  - NON gestiamo VRAM monitoring complesso (basta torch.cuda.memory_allocated)
  - Cancel cooperativo: si interrompe ALLA FINE di uno step, niente kill brutali
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopConfig:
    """Parametri del training loop."""

    num_epochs: int = 1
    grad_accum_steps: int = 1            # batch effettivo = batch_size * grad_accum
    max_grad_norm: float = 1.0           # gradient clipping
    log_every_n_steps: int = 10          # frequenza log
    # Stop dopo N step totali (logici, non micro-batch). 0 = no limit.
    max_steps: int = 0

    def __post_init__(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("num_epochs deve essere ≥ 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps deve essere ≥ 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm deve essere > 0")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps deve essere ≥ 1")
        if self.max_steps < 0:
            raise ValueError("max_steps deve essere ≥ 0")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepLog:
    """Snapshot di un singolo step di training (loggable in M5)."""

    step: int                       # step LOGICO (non micro-batch)
    epoch: int
    loss: float                     # loss media del batch logico
    learning_rate: float
    grad_norm: float                # norma gradienti pre-clip
    vram_used_mb: float             # VRAM allocata al momento del log
    throughput_tokens_per_sec: float
    elapsed_seconds: float          # tempo da inizio training

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": round(self.loss, 6),
            "learning_rate": round(self.learning_rate, 8),
            "grad_norm": round(self.grad_norm, 4),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


@dataclass(frozen=True)
class TrainingResult:
    """Output del training loop."""

    completed: bool                  # True se finito normalmente, False se cancel
    cancelled: bool                  # True se interrotto da cancel_event
    total_steps: int
    final_loss: float
    elapsed_seconds: float
    history: list[StepLog] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "cancelled": self.cancelled,
            "total_steps": self.total_steps,
            "final_loss": round(self.final_loss, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "history_count": len(self.history),
        }


# ---------------------------------------------------------------------------
# Helper VRAM
# ---------------------------------------------------------------------------


def _vram_used_mb() -> float:
    """VRAM allocata in MB. Ritorna 0 se CUDA non disponibile."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _count_real_tokens(attention_mask: torch.Tensor) -> int:
    """
    Conta i token "veri" (non-padding) di un batch, dalla attention_mask.
    Usato per il calcolo throughput.
    """
    return int(attention_mask.sum().item())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class CancellationToken:
    """
    Wrapper minimale attorno a un asyncio.Event o threading.Event.
    Permette al loop di girare in un thread/process senza dipendere
    da asyncio direttamente.
    """

    def __init__(self):
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    @classmethod
    def from_event(cls, event) -> "CancellationToken":
        """Crea un token che riflette lo stato di un threading/asyncio Event."""
        token = cls()
        # Hack semplice: il token CHIEDE all'event. Funziona sia per asyncio
        # che threading: entrambi hanno is_set().
        original_is_cancelled = token.is_cancelled

        def is_cancelled_via_event():
            return original_is_cancelled() or event.is_set()

        token.is_cancelled = is_cancelled_via_event  # type: ignore
        return token


def train_loop(
    *,
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    config: LoopConfig | None = None,
    cancel_token: CancellationToken | None = None,
    on_step: Callable[[StepLog], None] | None = None,
    device: str | None = None,
) -> TrainingResult:
    """
    Esegue il training loop completo.

    Args:
        model: PeftModel (o nn.Module) già preparato.
        train_loader: DataLoader che produce batch di input_ids/labels/attention_mask.
        optimizer: torch optimizer.
        scheduler: LR scheduler (LambdaLR).
        config: parametri loop.
        cancel_token: token per cancellazione cooperativa. None = no cancel.
        on_step: callback chiamato a ogni log step. Riceve uno StepLog.
        device: 'cuda', 'cpu', o None per auto.

    Returns:
        TrainingResult con history e stato finale.
    """
    config = config or LoopConfig()
    cancel = cancel_token or CancellationToken()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    history: list[StepLog] = []

    start_time = time.time()
    micro_batch_count = 0   # micro-batch totali processati
    logical_step = 0        # step LOGICI (dopo accumulazione)
    accumulated_loss = 0.0  # somma loss dei micro-batch correnti
    accumulated_tokens = 0  # token "veri" dei micro-batch correnti
    last_log_time = start_time
    last_log_tokens = 0

    final_loss = 0.0
    completed = False
    cancelled = False

    optimizer.zero_grad()

    try:
        for epoch in range(config.num_epochs):
            for batch in train_loader:
                # Cancel check PRIMA del lavoro pesante
                if cancel.is_cancelled():
                    cancelled = True
                    break

                # Sposta batch su device
                batch = {k: v.to(device) for k, v in batch.items()}
                attention_mask = batch.get("attention_mask")

                # Forward
                outputs = model(**batch)
                loss = outputs.loss

                # Scala per grad_accum (così quando sommiamo è una media)
                loss = loss / config.grad_accum_steps

                # Backward (accumula gradienti)
                loss.backward()

                # Tracciamento per log
                accumulated_loss += loss.item() * config.grad_accum_steps
                if attention_mask is not None:
                    accumulated_tokens += _count_real_tokens(attention_mask)

                micro_batch_count += 1

                # Step LOGICO ogni grad_accum_steps micro-batch
                if micro_batch_count % config.grad_accum_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=config.max_grad_norm,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    logical_step += 1

                    # Log periodico
                    if (
                        logical_step % config.log_every_n_steps == 0
                        or logical_step == 1
                    ):
                        now = time.time()
                        delta_t = max(0.001, now - last_log_time)
                        delta_tokens = accumulated_tokens - last_log_tokens
                        throughput = delta_tokens / delta_t

                        avg_loss = accumulated_loss / config.grad_accum_steps

                        log_entry = StepLog(
                            step=logical_step,
                            epoch=epoch,
                            loss=avg_loss,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            grad_norm=float(grad_norm),
                            vram_used_mb=_vram_used_mb(),
                            throughput_tokens_per_sec=throughput,
                            elapsed_seconds=now - start_time,
                        )
                        history.append(log_entry)
                        if on_step is not None:
                            try:
                                on_step(log_entry)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "Errore in on_step callback: %s", exc
                                )

                        logger.info(
                            "step=%d epoch=%d loss=%.4f lr=%.2e grad=%.2f "
                            "vram=%.0fMB tok/s=%.0f",
                            log_entry.step, log_entry.epoch,
                            log_entry.loss, log_entry.learning_rate,
                            log_entry.grad_norm, log_entry.vram_used_mb,
                            log_entry.throughput_tokens_per_sec,
                        )

                        last_log_time = now
                        last_log_tokens = accumulated_tokens

                    final_loss = accumulated_loss / config.grad_accum_steps
                    accumulated_loss = 0.0

                    # Stop su max_steps
                    if (
                        config.max_steps > 0
                        and logical_step >= config.max_steps
                    ):
                        logger.info("Raggiunto max_steps=%d", config.max_steps)
                        completed = True
                        break

            if cancelled or completed:
                break

        if not cancelled and not completed:
            completed = True

    except Exception as exc:
        # Logghiamo ma rilanciamo: l'errore va gestito dall'orchestrator
        logger.exception("Errore durante training loop: %s", exc)
        raise

    elapsed = time.time() - start_time
    return TrainingResult(
        completed=completed,
        cancelled=cancelled,
        total_steps=logical_step,
        final_loss=final_loss,
        elapsed_seconds=elapsed,
        history=history,
    )