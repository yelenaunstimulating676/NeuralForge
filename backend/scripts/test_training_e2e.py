"""
Test E2E del Training Engine: scarica training reale su SmolLM2-135M
con il dataset 'Capitali Europee' (5 esempi).

Scopo: validare che TUTTA la pipeline funzioni con modelli veri:
  - bitsandbytes carica 4-bit OK
  - PEFT applica LoRA OK
  - Train loop produce loss decrescente
  - Checkpoint si salva correttamente
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Aggiungi backend/ a sys.path se lanciato come script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import SessionLocal
from db.models import BaseModel, Dataset
from core.training.runner import TrainingConfig, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


def main():
    with SessionLocal() as session:
        # Trova SmolLM2-135M nel DB
        from sqlalchemy import select
        bm = session.scalar(
            select(BaseModel).where(BaseModel.hf_repo == "HuggingFaceTB/SmolLM2-135M")
        )
        if bm is None:
            print("ERRORE: SmolLM2-135M non trovato nel DB.")
            print("Vai su /models e scaricalo prima di lanciare questo test.")
            sys.exit(1)

        # Trova un dataset
        ds = session.scalars(select(Dataset).limit(1)).first()
        if ds is None:
            print("ERRORE: nessun dataset nel DB.")
            print("Vai su /dataset e crea il dataset 'Capitali Europee Test' prima.")
            sys.exit(1)

        print(f"Base model: {bm.display_name} (id={bm.id})")
        print(f"Dataset: {ds.name} (id={ds.id}, esempi={ds.num_examples})")
        print()

        config = TrainingConfig(
            base_model_id=bm.id,
            dataset_id=ds.id,
            num_epochs=20,
            per_device_batch_size=2,
            grad_accum_steps=1,
            learning_rate=5e-4,
            lora_r=16,
            log_every_n_steps=1,
            save_every_n_steps=2,  # solo final
        )

        print("Avvio training...")
        print()
        outcome = run_training(session=session, config=config)

        print()
        print("=" * 60)
        print(f"STATUS:        {outcome.status}")
        print(f"RUN_ID:        {outcome.run_id}")
        print(f"TOTAL_STEPS:   {outcome.total_steps}")
        print(f"FINAL_LOSS:    {outcome.final_loss:.4f}")
        print(f"ELAPSED:       {outcome.elapsed_seconds:.1f}s")
        print(f"CHECKPOINT:    {outcome.final_checkpoint_path}")
        print(f"FINETUNED_ID:  {outcome.finetuned_model_id}")
        if outcome.error:
            print(f"ERROR:         {outcome.error}")
        print("=" * 60)

        if outcome.history:
            print("\nLoss history:")
            for entry in outcome.history:
                print(f"  step {entry['step']:3d}  loss={entry['loss']:.4f}  lr={entry['learning_rate']:.2e}")


if __name__ == "__main__":
    main()