"""
Modelli ORM SQLAlchemy per NeuralForge.

Tabelle:
    - base_models       → modelli base scaricati da HuggingFace
    - training_runs     → esecuzioni di training (popolate in M5)
    - finetuned_models  → adapter LoRA risultanti dal training (M5)
    - datasets          → dataset di instruction tuning (M3)

Le tabelle di M3-M5 sono definite già qui (FK e schema corretti) ma
verranno usate davvero solo nelle milestone successive.

Convenzioni:
    - PK/FK: BigInteger su DB seri, Integer su SQLite (autoincrement).
      SQLite non supporta autoincrement su BIGINT — usiamo with_variant.
    - Timestamps: DateTime con default server-side `func.now()`
    - JSON content: TEXT colonne (config_json, metrics_json) parsate a runtime
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base

if TYPE_CHECKING:
    # Per evitare circular imports in futuro
    pass


# ---------------------------------------------------------------------------
# Type alias per primary key e foreign key.
#
# SQLite ha una stranezza storica: solo le colonne dichiarate esattamente
# `INTEGER PRIMARY KEY` (non `BIGINT PRIMARY KEY`) sono trattate come alias
# di ROWID e quindi auto-incrementate. Se usi BIGINT, l'INSERT lascia id=NULL
# e fallisce il NOT NULL constraint.
#
# `BigInteger().with_variant(Integer, "sqlite")` dice a SQLAlchemy:
#   - su SQLite emetti INTEGER (autoincrement OK)
#   - su Postgres/MySQL emetti BIGINT (range esteso)
# ---------------------------------------------------------------------------

PKType = BigInteger().with_variant(Integer, "sqlite")
FKType = BigInteger().with_variant(Integer, "sqlite")


# ---------------------------------------------------------------------------
# BaseModel (modello scaricato da HuggingFace)
# ---------------------------------------------------------------------------


class BaseModel(Base):
    """
    Modello base scaricato da HuggingFace ed installato localmente.

    Esempio: Qwen/Qwen2.5-3B-Instruct → un record qui dopo download.
    """

    __tablename__ = "base_models"

    id: Mapped[int] = mapped_column(PKType, primary_key=True, autoincrement=True)

    # Identificazione
    hf_repo: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tag: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Storage
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Metadata
    params_billions: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Timestamps
    downloaded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relazioni
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="base_model",
        cascade="all, delete-orphan",
    )
    finetuned_models: Mapped[list["FineTunedModel"]] = relationship(
        back_populates="base_model",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<BaseModel id={self.id} hf_repo={self.hf_repo!r}>"


# ---------------------------------------------------------------------------
# Dataset (M3) — definito qui perché TrainingRun lo referenzia
# ---------------------------------------------------------------------------


class Dataset(Base):
    """
    Dataset di instruction tuning generato dal Dataset Engine.
    Popolato davvero in M3.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(PKType, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    num_examples: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    format: Mapped[str] = mapped_column(String(32), nullable=False, default="instruction")
    stats_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="dataset",
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} examples={self.num_examples}>"


# ---------------------------------------------------------------------------
# TrainingRun (M5)
# ---------------------------------------------------------------------------


class TrainingRun(Base):
    """
    Esecuzione di training (placeholder per M5).

    Stati:
        - 'pending'    → creato ma non ancora avviato
        - 'running'    → in corso
        - 'completed'  → finito con successo
        - 'failed'     → errore durante training
        - 'cancelled'  → interrotto manualmente
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(PKType, primary_key=True, autoincrement=True)

    # FK
    base_model_id: Mapped[int] = mapped_column(
        FKType,
        ForeignKey("base_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[int | None] = mapped_column(
        FKType,
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Stato
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relazioni
    base_model: Mapped["BaseModel"] = relationship(back_populates="training_runs")
    dataset: Mapped["Dataset | None"] = relationship(back_populates="training_runs")
    finetuned_model: Mapped["FineTunedModel | None"] = relationship(
        back_populates="training_run",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} status={self.status!r}>"


# ---------------------------------------------------------------------------
# FineTunedModel (M5)
# ---------------------------------------------------------------------------


class FineTunedModel(Base):
    """
    Modello fine-tunato risultato di un TrainingRun.
    In pratica: il base model + un adapter LoRA salvato su disco.
    """

    __tablename__ = "finetuned_models"
    __table_args__ = (
        UniqueConstraint("training_run_id", name="uq_finetuned_per_run"),
    )

    id: Mapped[int] = mapped_column(PKType, primary_key=True, autoincrement=True)

    base_model_id: Mapped[int] = mapped_column(
        FKType,
        ForeignKey("base_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    training_run_id: Mapped[int] = mapped_column(
        FKType,
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    adapter_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    base_model: Mapped["BaseModel"] = relationship(back_populates="finetuned_models")
    training_run: Mapped["TrainingRun"] = relationship(back_populates="finetuned_model")

    def __repr__(self) -> str:
        return f"<FineTunedModel id={self.id} name={self.name!r}>"