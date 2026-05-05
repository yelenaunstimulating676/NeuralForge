"""
Persistenza Dataset: salva ValidatedDataset come JSONL su disco
e crea il record corrispondente nella tabella `datasets`.

Layout su disco:
    data/datasets/<dataset_id>/
        ├── dataset.jsonl       (un esempio per riga, formato Alpaca)
        └── metadata.json       (config + stats + info)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from core.dataset.validator import ValidatedDataset
from db.models import Dataset as DatasetRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class DatasetPersistenceError(Exception):
    """Errore durante salvataggio o caricamento del dataset."""


class DatasetNotFoundError(DatasetPersistenceError):
    """Dataset richiesto non trovato nel DB."""


class DatasetNameConflictError(DatasetPersistenceError):
    """Esiste già un dataset con quel nome."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAFE_NAME_RE = re.compile(r"[^a-z0-9_-]+")


def sanitize_name_for_dirname(name: str) -> str:
    """
    Trasforma un nome utente in un dirname sicuro.
    "Mio Dataset!" → "mio-dataset"
    """
    s = name.strip().lower()
    s = _SAFE_NAME_RE.sub("-", s)
    s = s.strip("-_")
    return s or "dataset"


def _name_exists(session: Session, name: str) -> bool:
    """True se esiste già un Dataset con quel nome (case-sensitive)."""
    stmt = select(DatasetRow).where(DatasetRow.name == name)
    return session.scalars(stmt).first() is not None


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_dataset(
    session: Session,
    *,
    name: str,
    validated: ValidatedDataset,
    source_file: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DatasetRow:
    """
    Salva un ValidatedDataset:
      1. Crea record DB con name + path
      2. Scrive dataset.jsonl + metadata.json su disco

    Args:
        session: SQLAlchemy session.
        name: nome user-facing (deve essere univoco).
        validated: output del Validator.
        source_file: path del file sorgente uploadato (opzionale, per audit).
        extra_metadata: dati aggiuntivi da serializzare (es. config Converter).

    Returns:
        Il record DatasetRow creato.

    Raises:
        DatasetNameConflictError: se esiste già un dataset con quel nome.
        DatasetPersistenceError: errori di I/O.
    """
    if not name or not name.strip():
        raise DatasetPersistenceError("Il nome del dataset non può essere vuoto.")

    if _name_exists(session, name):
        raise DatasetNameConflictError(
            f"Esiste già un dataset con il nome {name!r}."
        )

    # Path su disco
    settings.ensure_directories()
    safe_name = sanitize_name_for_dirname(name)
    dataset_dir = settings.datasets_path / safe_name

    # Se la cartella esiste già (per safe_name collidente), aggiungi suffisso numerico
    counter = 1
    while dataset_dir.exists():
        dataset_dir = settings.datasets_path / f"{safe_name}-{counter}"
        counter += 1

    try:
        dataset_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise DatasetPersistenceError(
            f"Impossibile creare la cartella {dataset_dir}: {exc}"
        ) from exc

    jsonl_path = dataset_dir / "dataset.jsonl"
    meta_path = dataset_dir / "metadata.json"

    try:
        # 1. Scrivi JSONL
        with jsonl_path.open("w", encoding="utf-8") as f:
            for ex in validated.examples:
                row = {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 2. Scrivi metadata.json
        metadata = {
            "name": name,
            "format": "alpaca",
            "num_examples": len(validated),
            "source_file": str(source_file) if source_file else None,
            "stats": validated.stats.to_dict(),
            "extra": extra_metadata or {},
        }
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    except OSError as exc:
        # Rollback: rimuovi cartella se errore
        try:
            import shutil

            shutil.rmtree(dataset_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        raise DatasetPersistenceError(
            f"Errore scrittura dataset su disco: {exc}"
        ) from exc

    # 3. Inserisci record DB
    row = DatasetRow(
        name=name,
        source_file=str(source_file) if source_file else None,
        file_path=str(jsonl_path.resolve()),
        num_examples=len(validated),
        format="alpaca",
        stats_json=json.dumps(validated.stats.to_dict(), ensure_ascii=False),
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    logger.info(
        "Dataset salvato: id=%d name=%r examples=%d path=%s",
        row.id, name, len(validated), jsonl_path,
    )
    return row


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def list_datasets(session: Session) -> list[DatasetRow]:
    """Lista tutti i dataset, ordinati per data creazione (recenti prima)."""
    stmt = select(DatasetRow).order_by(DatasetRow.created_at.desc())
    return list(session.scalars(stmt).all())


def get_dataset_by_id(session: Session, dataset_id: int) -> DatasetRow:
    """
    Ritorna il dataset con l'ID dato.

    Raises:
        DatasetNotFoundError: se non esiste.
    """
    row = session.get(DatasetRow, dataset_id)
    if row is None:
        raise DatasetNotFoundError(f"Dataset con id={dataset_id} non trovato.")
    return row


def load_dataset_examples(
    session: Session, dataset_id: int, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """
    Carica gli esempi (riga per riga dal JSONL) di un dataset.

    Args:
        dataset_id: id del dataset.
        limit: se fornito, ritorna max N esempi (utile per preview UI).

    Returns:
        Lista di dict con campi instruction/input/output.
    """
    row = get_dataset_by_id(session, dataset_id)
    path = Path(row.file_path)
    if not path.exists():
        raise DatasetPersistenceError(
            f"File dataset mancante su disco: {path}"
        )

    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Riga %d corrotta in %s: %s", i + 1, path, exc
                )

    return examples


def delete_dataset(
    session: Session, dataset_id: int, *, remove_files: bool = True
) -> None:
    """
    Cancella un dataset dal DB e (opzionale) dal disco.

    Args:
        session: SQLAlchemy session.
        dataset_id: id del dataset.
        remove_files: se True, rimuove anche la cartella locale.

    Raises:
        DatasetNotFoundError: se non esiste.
    """
    row = get_dataset_by_id(session, dataset_id)
    file_path = Path(row.file_path) if row.file_path else None

    session.delete(row)
    session.commit()

    if remove_files and file_path and file_path.exists():
        # Cancelliamo l'INTERA cartella del dataset (jsonl + metadata)
        dataset_dir = file_path.parent
        # Safety: deve essere dentro datasets_path
        try:
            dataset_dir.resolve().relative_to(settings.datasets_path.resolve())
        except ValueError:
            logger.error(
                "RIFIUTO di cancellare %s: fuori da datasets_path %s",
                dataset_dir, settings.datasets_path,
            )
            return

        import shutil

        shutil.rmtree(dataset_dir, ignore_errors=False)
        logger.info("Cartella dataset rimossa: %s", dataset_dir)
    else:
        logger.info("Dataset %d cancellato dal DB (files preservati).", dataset_id)