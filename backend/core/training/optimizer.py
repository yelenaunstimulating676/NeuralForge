"""
Optimizer e learning rate scheduler del Training Engine.

Optimizer:
  - AdamW8bit (bitsandbytes) — quantizza optimizer states a 8 bit,
    riducendo VRAM degli stati da 4x a 1x. Fallback ad AdamW standard
    di torch se bnb non disponibile.

Scheduler:
  - Cosine schedule con warmup lineare. Standard per LLM.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfig:
    """Parametri dell'optimizer."""

    learning_rate: float = 2e-4         # peak LR (sweet spot per LoRA)
    weight_decay: float = 0.01          # regolarizzazione
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    use_8bit: bool = True               # se True, prova AdamW8bit, fallback ad AdamW

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.learning_rate > 1.0:
            raise ValueError("learning_rate deve essere in (0, 1]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay deve essere ≥ 0")
        if not (0.0 < self.betas[0] < 1.0 and 0.0 < self.betas[1] < 1.0):
            raise ValueError("betas devono essere in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps deve essere > 0")


@dataclass(frozen=True)
class SchedulerConfig:
    """Parametri del learning rate scheduler."""

    total_steps: int                    # totale step di training (no epoch-based)
    warmup_ratio: float = 0.03          # frazione step di warmup (3%)
    min_lr_ratio: float = 0.0           # LR finale = peak_lr * min_lr_ratio

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps deve essere ≥ 1")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio deve essere in [0, 1)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio deve essere in [0, 1]")

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.total_steps * self.warmup_ratio))


# ---------------------------------------------------------------------------
# Optimizer factory con fallback bnb → torch
# ---------------------------------------------------------------------------


def _try_import_adamw8bit():
    """
    Importa AdamW8bit di bitsandbytes con fallback robusto.

    Returns:
        La classe AdamW8bit se disponibile, None altrimenti.
    """
    try:
        import bitsandbytes as bnb  # type: ignore
        return bnb.optim.AdamW8bit
    except ImportError:
        logger.info("bitsandbytes non installato, fallback ad AdamW torch.")
        return None
    except Exception as exc:  # noqa: BLE001
        # Su Windows può capitare: DLL non trovate, CUDA mismatch, ecc.
        logger.warning(
            "bitsandbytes presente ma non utilizzabile (%s). "
            "Fallback ad AdamW torch.",
            type(exc).__name__,
        )
        return None


def get_trainable_parameters(model) -> list:
    """
    Filtra solo i parametri con `requires_grad=True`. Da passare all'optimizer
    per non sprecare VRAM su stati di parametri congelati.
    """
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(
    model,
    config: OptimizerConfig | None = None,
) -> Optimizer:
    """
    Costruisce l'optimizer (AdamW8bit se disponibile, altrimenti AdamW torch).

    Args:
        model: il modello (PeftModel o torch.nn.Module). Vengono usati solo
            i parametri trainable (requires_grad=True).
        config: parametri optimizer.

    Returns:
        Un torch.optim.Optimizer pronto per il training loop.
    """
    config = config or OptimizerConfig()
    trainable_params = get_trainable_parameters(model)

    if not trainable_params:
        raise ValueError(
            "Nessun parametro trainable trovato nel modello. "
            "Verifica che LoRA sia stato applicato correttamente."
        )

    # Tenta AdamW8bit
    if config.use_8bit:
        AdamW8bit = _try_import_adamw8bit()
        if AdamW8bit is not None:
            logger.info(
                "Optimizer: AdamW8bit | lr=%.2e wd=%.4f params=%d",
                config.learning_rate, config.weight_decay, len(trainable_params),
            )
            return AdamW8bit(
                trainable_params,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )

    # Fallback torch.AdamW
    logger.info(
        "Optimizer: AdamW (torch) | lr=%.2e wd=%.4f params=%d",
        config.learning_rate, config.weight_decay, len(trainable_params),
    )
    return AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


# ---------------------------------------------------------------------------
# Cosine schedule con warmup
# ---------------------------------------------------------------------------


def _cosine_warmup_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    Funzione lambda passata a `LambdaLR`. Ritorna un MOLTIPLICATORE del
    learning rate base (NON il LR finale).

    Forma:
      - [0, warmup_steps):     lineare 0 → 1
      - [warmup_steps, total]: cosine 1 → min_lr_ratio
    """
    if current_step < warmup_steps:
        # Warmup lineare
        return float(current_step) / float(max(1, warmup_steps))

    # Cosine decay
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LambdaLR:
    """
    Costruisce il LR scheduler cosine con warmup.

    Args:
        optimizer: optimizer torch.
        config: parametri scheduler.

    Returns:
        LambdaLR pronto. Va chiamato `scheduler.step()` dopo ogni
        `optimizer.step()` (NON ogni mini-batch se grad_accum > 1).
    """
    warmup_steps = config.warmup_steps
    total_steps = config.total_steps
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        return _cosine_warmup_lambda(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

    logger.info(
        "Scheduler: cosine warmup | total=%d warmup=%d min_lr_ratio=%.3f",
        total_steps, warmup_steps, min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)