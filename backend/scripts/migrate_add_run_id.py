"""
Migration script: aggiunge la colonna run_id alla tabella training_runs.

Per i record esistenti (TrainingRun creati prima di M5.6), genera un
run_id placeholder unico basato sull'id del record.

Idempotente: se la colonna esiste già, esce senza errori.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text

from db import engine, SessionLocal
from db.models import TrainingRun


def column_exists(table_name: str, column_name: str) -> bool:
    """Verifica se una colonna esiste nella tabella."""
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def main():
    table = "training_runs"
    column = "run_id"

    if column_exists(table, column):
        print(f"✓ Colonna '{column}' già presente in '{table}'. Niente da fare.")
        return

    print(f"→ Aggiungo colonna '{column}' a '{table}'...")
    with engine.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE {table} "
                f"ADD COLUMN {column} VARCHAR(64) "
                f"NOT NULL DEFAULT ''"
            )
        )

    # Popola i record esistenti con un run_id placeholder unico
    print("→ Popolo i record esistenti con run_id placeholder...")
    with SessionLocal() as session:
        runs = session.query(TrainingRun).filter(TrainingRun.run_id == "").all()
        for run in runs:
            run.run_id = f"legacy-{run.id:06d}"
        session.commit()
        print(f"   {len(runs)} record aggiornati.")

    # Crea l'indice unique (SQLite non supporta UNIQUE in ALTER, lo creiamo a parte)
    print("→ Creo indice unique...")
    with engine.begin() as conn:
        try:
            conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS "
                    f"ix_training_runs_run_id ON {table}({column})"
                )
            )
        except Exception as exc:
            print(f"⚠ Indice non creato: {exc}")

    print("✓ Migration completata.")


if __name__ == "__main__":
    main()