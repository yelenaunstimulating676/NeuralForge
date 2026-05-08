"""
Estimator euristico pre-training.

Calcola stime grossolane di VRAM richiesta e tempo totale, basandosi
su parametri del modello e config training. Senza benchmark hardware
(vedi parking lot).

Numeri tarati su misurazioni empiriche di test M4 su RTX 4070 12GB:
  - SmolLM2-135M (~80M params totali con head): ~200 MB VRAM in 4-bit + LoRA
  - Tempo per step: ~0.3-0.5s su batch piccoli
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Coefficienti euristici, tarati sui test M4 reali
# Modello in 4-bit ≈ 0.6 byte per parametro (peso + overhead quantizzazione)
BYTES_PER_PARAM_4BIT = 0.6
BYTES_PER_PARAM_FP16 = 2.0

# Activation overhead: dipende da batch e seq_len, fattore moltiplicativo
# sui pesi del modello
ACTIVATION_FACTOR_PER_BATCH_PER_TOK = 0.0001  # MB

# Tempo per step baseline (in secondi). Tarato su un modello ~80M params.
BASE_STEP_SECONDS = 0.4
# Scaling con dimensione modello (lineare, conservativo)
STEP_TIME_SCALING_PER_BILLION = 1.5  # un modello da 1B → 0.4 + 1.5 = 1.9s/step

# LoRA overhead (parametri trainable per layer × num target_modules)
# Stima conservativa: 4 target modules × r × hidden_size × 2 (A+B matrices)
DEFAULT_HIDDEN_SIZE = 2048   # tipico per modelli 1-3B


@dataclass(frozen=True)
class TrainingEstimate:
    """Risultato della stima."""

    estimated_vram_mb: int
    estimated_time_seconds: int
    total_steps: int
    steps_per_epoch: int
    trainable_params_estimated: int
    notes: list[str]


def estimate_training(
    *,
    params_billions: float,           # dimensione modello in miliardi
    num_examples: int,                # esempi nel dataset
    num_epochs: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    max_seq_length: int,
    lora_r: int,
    use_4bit: bool = True,
) -> TrainingEstimate:
    """
    Stima VRAM, tempo e step di un training.

    Tutti i calcoli sono **euristici e conservativi**. Margine d'errore: ±30%.
    """
    notes: list[str] = []

    # Validazioni
    if params_billions <= 0:
        params_billions = 0.1
        notes.append("Dimensione modello non specificata, assumo 100M.")

    # === VRAM ===
    n_params = params_billions * 1e9

    # Pesi del modello
    if use_4bit:
        weights_mb = (n_params * BYTES_PER_PARAM_4BIT) / (1024 * 1024)
    else:
        weights_mb = (n_params * BYTES_PER_PARAM_FP16) / (1024 * 1024)

    # Activations (dipende da batch × seq_len)
    activation_mb = (
        per_device_batch_size
        * max_seq_length
        * ACTIVATION_FACTOR_PER_BATCH_PER_TOK
        * params_billions
        * 1000   # scala con dim modello
    )

    # LoRA + optimizer states (8bit o 32bit)
    # Stima trainable params per LoRA: 4 modules × r × hidden × 2
    trainable_estimate = int(4 * lora_r * DEFAULT_HIDDEN_SIZE * 2)
    # Optimizer states 8bit ≈ 1 byte/param × 2 (m + v) = 2 bytes/param
    optimizer_mb = (trainable_estimate * 2) / (1024 * 1024)

    # Buffer + overhead generico (gradient checkpointing aiuta ma non azzera)
    buffer_mb = 200

    total_vram_mb = int(weights_mb + activation_mb + optimizer_mb + buffer_mb)

    # === Tempo ===
    # Step logici totali
    steps_per_epoch = max(
        1, num_examples // (per_device_batch_size * grad_accum_steps)
    )
    total_steps = steps_per_epoch * num_epochs

    # Tempo per step
    step_time = BASE_STEP_SECONDS + (
        params_billions * STEP_TIME_SCALING_PER_BILLION
    )
    # Scala leggermente con seq_len (sequenze più lunghe = più computo)
    step_time *= 1.0 + (max_seq_length / 8192) * 0.5

    estimated_time = int(total_steps * step_time)

    # === Notes ===
    if total_vram_mb > 12000:
        notes.append(
            f"VRAM stimata ({total_vram_mb} MB) supera 12 GB. "
            "Riduci batch_size o usa un modello più piccolo."
        )
    if total_steps < 5:
        notes.append(
            f"Solo {total_steps} step totali: training molto breve, "
            "potrebbe non convergere. Aumenta epochs."
        )
    if estimated_time > 7200:
        notes.append(
            f"Tempo stimato {estimated_time // 60} minuti: training lungo."
        )

    return TrainingEstimate(
        estimated_vram_mb=total_vram_mb,
        estimated_time_seconds=estimated_time,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        trainable_params_estimated=trainable_estimate,
        notes=notes,
    )