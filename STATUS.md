# NeuralForge — Stato sviluppo

## Completato
- M0: Backend FastAPI + frontend Vite/React/Tailwind4 + healthcheck
- M1: System Detector (GPU/VRAM live, training config suggestion)
- M2: Model Manager (whitelist 18 modelli, download async, custom HF, gated detection)
- M3: Dataset Engine (PDF/CSV/TXT/JSON/JSONL/DOCX → Alpaca instruction tuning, wizard 3-step)

## In corso
- M4: Training Engine (QLoRA + AdamW8bit + custom PyTorch loop)

## Stato test
- 193 test passing

## Comandi rapidi
- Backend: `cd backend && .\venv\Scripts\activate && python -m uvicorn main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Tests: `cd backend && pytest tests/ -v`
- URL: http://127.0.0.1:5173 (frontend) — http://127.0.0.1:8000/docs (API)

---

M0 ✅ Bootstrap full-stack
M1 ✅ System Detector
M2 ✅ Model Manager  
M3 ✅ Dataset Engine            ← test: 193 passed
M4 ⏭️  Training Engine            ← riprenderemo qui
M5 ⏳ Training API + Live Monitor
M6 ⏳ Inference & comparison
M7 ⏳ Export GGUF
M8 ⏳ Polish & onboarding

Parking lot:
  • GPU Benchmark on-demand
  • Estimation card pre-training
  • ETA dinamica training
  • Smart converter mode con LLM (per narrative dataset più ricchi)
  • OCR per PDF scansionati
  • Confidence detector più furba (top score - second != margine sempre)

  ---
  🏆 M4 — TRAINING ENGINE COMPLETATO
M4.1 ✅ Data layer (215 → 216 test)
M4.2 ✅ Model loader QLoRA (216 → 240 test)
M4.3 ✅ Optimizer + Scheduler (240 → 271 test)
M4.4 ✅ Training loop (271 → 290 test)
M4.5 ✅ Checkpoint save/load (290 → 316 test)
M4.6 ✅ Orchestratore + DB integration (316 → 323 test)
       ✅ Test E2E reale: SmolLM2-135M, 60 step, loss 3.43→0.77
Statistiche finali M4:

+130 test rispetto a fine M3 (193 → 323)
~2200 righe di codice nuovo
2 training reali completati (4 step + 60 step)
2 FineTunedModel registrati in DB
Stack ML completo: PyTorch + transformers + PEFT + bitsandbytes + accelerate, tutto orchestrato manualmente senza HF Trainer

Hardware utilizzato (RTX 4070 12GB): VRAM stabile ~200 MB durante training (modello in 4bit + adapter LoRA), 200-350 tok/s.

Cosa abbiamo dimostrato
✅ Forward + backward + step funzionano correttamente (la loss scende)
✅ Mixed precision bf16 è stabile (no NaN, no esplosioni)
✅ AdamW8bit ottimizza correttamente (non c'è degrado vs torch AdamW)
✅ Gradient clipping efficace (grad_norm sempre tra 1.0 e 2.8)
✅ Cosine schedule traccia perfettamente (5e-4 → 0 in 40 step)
✅ Loss masking è corretto (se fosse rotto, la loss sarebbe instabile o flat)
✅ Checkpoint rotation funziona (10 checkpoint salvati, sempre solo gli ultimi 3 mantenuti)
✅ Gradient accumulation (batch_eff = 2, step a ogni batch fisico — corretto)
✅ Bitsandbytes 4-bit + 8-bit optimizer su Windows funzionante
E tutto questo in 19.4 secondi totali per 60 step di update. Su hardware reale, robusto, ripetibile.


---
🎯 Stato roadmap
M0 ✅ Bootstrap full-stack
M1 ✅ System Detector
M2 ✅ Model Manager
M3 ✅ Dataset Engine
M4 ✅ Training Engine            ← test: 323 passed + E2E reale verificato
M5 ⏭️  Training API + Live Monitor   ← riprenderemo qui
M6 ⏳ Inference & comparison
M7 ⏳ Export GGUF
M8 ⏳ Polish & onboarding

Parking lot:
  • GPU Benchmark on-demand
  • Estimation card pre-training (VRAM/tempo)
  • ETA dinamica training (in M5 magari)
  • Smart converter mode con LLM (per narrative dataset più ricchi)
  • OCR per PDF scansionati
  • Confidence detector più furba

Cose da considerare in M5:
  • WebSocket per loss live
  • JobManager (riuso quello di M2) per start/cancel training
  • Estimation card che usa il system detector di M1
  • Charts via recharts (già installato per il dashboard)

---

# NeuralForge — Stato sviluppo

## Completato
- M0: Backend FastAPI + frontend Vite/React/Tailwind4 + healthcheck
- M1: System Detector (GPU/VRAM live, training config suggestion)
- M2: Model Manager (whitelist 18 modelli, download async, custom HF, gated detection)
- M3: Dataset Engine (PDF/CSV/TXT/JSON/JSONL/DOCX → Alpaca instruction tuning, wizard 3-step)
- M4: Training Engine (QLoRA + AdamW8bit + custom PyTorch loop + checkpoint, E2E verificato)

## In corso
- M5: Training API + Live Monitor (WebSocket + chart live)

## Stato test
- 323 test passing
- 1 training E2E reale verificato (SmolLM2-135M, loss 3.43→0.77 in 19s)

## Comandi rapidi
- Backend: `cd backend && .\venv\Scripts\activate && python -m uvicorn main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Tests: `cd backend && pytest tests/ -v`
- Test E2E training: `python scripts/test_training_e2e.py`
- URL: http://127.0.0.1:5173 (frontend) — http://127.0.0.1:8000/docs (API)

🏆 M5 COMPLETATO
M5.1 ✅ EventBroadcaster (340 test)
M5.2 ✅ API REST /api/training/* (345 test)  
M5.3 ✅ WebSocket /ws/{run_id}
M5.4 ✅ Frontend Training page (config + estimation + start)
M5.5 ✅ Frontend Live Monitor (chart live + WebSocket + cancel)
Statistiche M5:

+5 test mockati (estimator)
~1500 righe di codice nuovo (backend training API + frontend Training/TrainingLive)
WebSocket end-to-end funzionante con thread bridging via run_coroutine_threadsafe
Stream di eventi live dal training thread → broadcaster → WebSocket → React state → recharts


🎯 Stato roadmap
M0 ✅ Bootstrap full-stack
M1 ✅ System Detector
M2 ✅ Model Manager
M3 ✅ Dataset Engine
M4 ✅ Training Engine
M5 ✅ Training API + Live Monitor   ← appena fatto!
M6 ⏭️  Inference & comparison
M7 ⏳ Export GGUF
M8 ⏳ Polish & onboarding
6/9 milestone fatte. Da qui in poi:

M6 — Inference & Comparison: scegli un FT model, scrivi un prompt, vedi la risposta del modello base vs fine-tunato side-by-side. Pezzo "WOW" perché vedi finalmente il risultato del fine-tuning. ~3h.
M7 — Export GGUF: per chi vuole esportare e usare il modello su llama.cpp/Ollama. Conversione one-shot. ~1.5h.
M8 — Polish: onboarding wizard al primo avvio, README, MIT license, error boundaries, refinement UX. ~2h.


M0 ✅ Bootstrap full-stack
M1 ✅ System Detector
M2 ✅ Model Manager
M3 ✅ Dataset Engine
M4 ✅ Training Engine
M5 ✅ Training API + Live Monitor + Storico       ← appena completato!
M6 ⏭️  Inference & Comparison                       ← riprenderemo qui
M7 ⏳ Export GGUF
M8 ⏳ Polish & onboarding

Risorse pronte per M6:
  • SmolLM2-135M base model in data/models/
  • Dataset "Capitali Europee Test" (5 esempi)
  • 6 FineTunedModel disponibili (loss da 0.11 a 0.47)
  • Schema DB con run_id persistito
  • Tutto end-to-end funzionante e testato

Stato test backend: 345 passing