"""
Checkpoint save/load del Training Engine.

Strategia:
  - Salviamo SOLO l'adapter LoRA + tokenizer (PEFT.save_pretrained)
  - Per resume: salviamo anche optimizer/scheduler/trainer_state
  - Il base model NON viene salvato (è già su disco da M2)

Layout:
    data/adapters/<run_id>/
        ├── checkpoint-100/        ← intermedio
        │   ├── adapter_model.safetensors
        │   ├── adapter_config.json
        │   ├── tokenizer.json
        │   ├── optimizer.pt
        │   ├── scheduler.pt
        │   └── trainer_state.json
        └── final/                 ← finale (no optimizer/scheduler)
            ├── adapter_model.safetensors
            ├── adapter_config.json
            └── tokenizer.json
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Errore durante save/load di un checkpoint."""


# ---------------------------------------------------------------------------
# Trainer state
# ---------------------------------------------------------------------------


@dataclass
class TrainerState:
    """
    Stato del trainer da serializzare con il checkpoint.
    Permette di riprendere il training esattamente dove è stato interrotto.
    """

    run_id: str
    step: int                       # ultimo step logico completato
    epoch: int
    final_loss: float
    history: list[dict] = field(default_factory=list)
    base_model_path: str = ""       # per ricostruire il modello base
    family_tag: str | None = None   # per ricostruire i target_modules

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        return cls(**data)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_run_dir(run_id: str) -> Path:
    """
    Ritorna la directory base per un run di training.
    Sempre dentro `settings.adapters_path` (no path arbitrari).
    """
    return settings.adapters_path / _sanitize_run_id(run_id)


def get_checkpoint_dir(run_id: str, step: int) -> Path:
    """Path di un checkpoint intermedio."""
    return get_run_dir(run_id) / f"checkpoint-{step}"


def get_final_dir(run_id: str) -> Path:
    """Path del checkpoint finale."""
    return get_run_dir(run_id) / "final"


def _sanitize_run_id(run_id: str) -> str:
    """
    Sanifica il run_id per uso come dirname.
    Rimuove caratteri rischiosi, collassi multipli, mantiene UUID-like.
    """
    import re

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id.strip())
    safe = safe.strip("-_")
    if not safe:
        raise CheckpointError(f"run_id non valido: {run_id!r}")
    return safe


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    trainer_state: TrainerState,
    is_final: bool = False,
) -> Path:
    """
    Salva un checkpoint del training.

    Args:
        model: PeftModel da cui prelevare l'adapter LoRA.
        tokenizer: tokenizer da salvare insieme.
        optimizer: torch.optim.Optimizer (None se is_final).
        scheduler: torch LR scheduler (None se is_final).
        trainer_state: stato corrente del trainer.
        is_final: se True, salva in /final/ e omette optimizer/scheduler.

    Returns:
        Path della directory di checkpoint creata.
    """
    if is_final:
        checkpoint_dir = get_final_dir(trainer_state.run_id)
    else:
        checkpoint_dir = get_checkpoint_dir(trainer_state.run_id, trainer_state.step)

    settings.ensure_directories()

    # Se la directory esiste già, la cancelliamo (es. retry stesso step)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    try:
        # 1. Adapter LoRA (PEFT)
        try:
            model.save_pretrained(str(checkpoint_dir))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(
                f"Errore salvataggio adapter LoRA: {exc}"
            ) from exc

        # 2. Tokenizer
        try:
            tokenizer.save_pretrained(str(checkpoint_dir))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(
                f"Errore salvataggio tokenizer: {exc}"
            ) from exc

        # 3. Optimizer + Scheduler (solo se non è final)
        if not is_final:
            if optimizer is not None:
                opt_path = checkpoint_dir / "optimizer.pt"
                torch.save(optimizer.state_dict(), opt_path)
            if scheduler is not None:
                sched_path = checkpoint_dir / "scheduler.pt"
                torch.save(scheduler.state_dict(), sched_path)

        # 4. Trainer state (sempre)
        state_path = checkpoint_dir / "trainer_state.json"
        state_path.write_text(
            json.dumps(trainer_state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    except CheckpointError:
        # Rollback: rimuovi cartella incompleta
        try:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        raise

    logger.info(
        "Checkpoint salvato: run=%s step=%d final=%s path=%s",
        trainer_state.run_id, trainer_state.step, is_final, checkpoint_dir,
    )
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_trainer_state(checkpoint_dir: Path) -> TrainerState:
    """
    Carica solo il TrainerState da un checkpoint (no modello, no optimizer).
    Utile per ispezionare un checkpoint senza materializzarlo.
    """
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise CheckpointError(
            f"trainer_state.json non trovato in {checkpoint_dir}"
        )
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(
            f"trainer_state.json malformato: {exc}"
        ) from exc
    return TrainerState.from_dict(data)


def load_optimizer_state(checkpoint_dir: Path, optimizer) -> None:
    """
    Carica lo state_dict dell'optimizer da un checkpoint.

    Args:
        checkpoint_dir: directory contenente optimizer.pt.
        optimizer: optimizer torch già costruito (DEVE avere stessa struttura
            dei param_groups del checkpoint).

    Raises:
        CheckpointError: se il file manca o non caricabile.
    """
    opt_path = checkpoint_dir / "optimizer.pt"
    if not opt_path.exists():
        raise CheckpointError(f"optimizer.pt non trovato in {checkpoint_dir}")
    try:
        state = torch.load(opt_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(
            f"Errore caricamento optimizer: {exc}"
        ) from exc


def load_scheduler_state(checkpoint_dir: Path, scheduler) -> None:
    """
    Carica lo state_dict dello scheduler da un checkpoint.
    """
    sched_path = checkpoint_dir / "scheduler.pt"
    if not sched_path.exists():
        raise CheckpointError(f"scheduler.pt non trovato in {checkpoint_dir}")
    try:
        state = torch.load(sched_path, map_location="cpu", weights_only=True)
        scheduler.load_state_dict(state)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(
            f"Errore caricamento scheduler: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Listing checkpoints
# ---------------------------------------------------------------------------


def list_checkpoints(run_id: str) -> list[Path]:
    """
    Lista i checkpoint intermedi di un run, ordinati per step crescente.
    Esclude la cartella final/.
    """
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.startswith("checkpoint-"):
            continue
        try:
            step = int(entry.name.split("-")[1])
            checkpoints.append((step, entry))
        except (IndexError, ValueError):
            continue
    checkpoints.sort(key=lambda x: x[0])
    return [p for _, p in checkpoints]


def find_latest_checkpoint(run_id: str) -> Path | None:
    """
    Ritorna il checkpoint più recente di un run, o None se non ne esistono.
    """
    checkpoints = list_checkpoints(run_id)
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def delete_run_directory(run_id: str) -> bool:
    """
    Cancella INTERA la directory di un run (tutti i checkpoint).
    Safety check: deve essere dentro adapters_path.

    Returns:
        True se cancellato, False se non esisteva.
    """
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        return False

    # Safety: verifica che sia dentro adapters_path
    try:
        run_dir.resolve().relative_to(settings.adapters_path.resolve())
    except ValueError:
        logger.error(
            "RIFIUTO di cancellare %s: fuori da adapters_path %s",
            run_dir, settings.adapters_path,
        )
        return False

    shutil.rmtree(run_dir)
    logger.info("Run directory rimossa: %s", run_dir)
    return True


def keep_only_last_n_checkpoints(run_id: str, n: int = 3) -> int:
    """
    Mantiene solo gli N checkpoint più recenti di un run, cancellando i
    più vecchi. Utile durante training per non saturare il disco.

    Args:
        run_id: identificativo del run.
        n: numero di checkpoint da mantenere (default 3).

    Returns:
        Numero di checkpoint rimossi.
    """
    if n < 1:
        raise ValueError("n deve essere ≥ 1")

    checkpoints = list_checkpoints(run_id)
    if len(checkpoints) <= n:
        return 0

    to_remove = checkpoints[:-n]
    removed = 0
    for path in to_remove:
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            logger.warning("Errore rimozione %s: %s", path, exc)
    if removed:
        logger.info("Rimossi %d checkpoint vecchi del run %s", removed, run_id)
    return removed