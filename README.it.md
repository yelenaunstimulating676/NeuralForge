<div align="center">

# ⚡ NeuralForge

**Fine-tuning di LLM in locale, reso semplice.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuna i large language models su GPU consumer con un'interfaccia web.
Training QLoRA, confronto inference side-by-side, export GGUF in un click — tutto in locale.

[🇬🇧 English](README.md) · **🇮🇹 Italiano**

</div>

---

## 📸 Anteprima

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="NeuralForge Dashboard" width="900" />
</p>

<details>
<summary><strong>Altri screenshot</strong></summary>

### Model Manager
<p align="center">
  <img src="docs/screenshots/models.png" alt="Model Manager" width="850" />
</p>

### Wizard Dataset
<p align="center">
  <img src="docs/screenshots/dataset.png" alt="Wizard Dataset" width="850" />
</p>

### In arrivo
- Live Monitor Training (chart in tempo reale) — `docs/screenshots/training-live.png`
- Confronto Inference (base vs fine-tunato) — `docs/screenshots/inference.png`
- Export GGUF — `docs/screenshots/export.png`

</details>

### 🎥 Video demo

<!-- TODO: Sostituire con il link YouTube reale una volta registrato -->
> Un walkthrough di 2-3 minuti del workflow completo (scarica modello → carica dataset → training → confronto → export).
>
> **In arrivo:** `https://youtu.be/...`

---

## Cos'è NeuralForge?

NeuralForge è una piattaforma di fine-tuning local-first per LLM. Avvolge lo stack PyTorch + PEFT + bitsandbytes con un'interfaccia web moderna, in modo che fare fine-tuning di un modello sui propri dati non richieda notebook, gymnastica da CLI o account cloud.

**L'intera pipeline gira sulla tua macchina.** Nessuna telemetria, nessuna API key, nessun dato che esce dall'hardware. Tu porti un base model da HuggingFace e un dataset; NeuralForge gestisce tokenizzazione, training QLoRA con monitoring live, confronto inference, ed export al formato GGUF usato da Ollama, LM Studio e llama.cpp.

Pensato per il sweet spot delle GPU consumer: una singola scheda da 12 GB (classe RTX 4070) è sufficiente per fare fine-tuning di modelli classe 3B.

## Funzionalità

- **Model Manager** — Whitelist di modelli HuggingFace curati + supporto a repo custom con detection dei modelli gated.
- **Dataset Engine** — Import da PDF, DOCX, CSV, TXT, JSON, JSONL. Chunking intelligente, deduplica, conversione in formato alpaca in un wizard a 3 step.
- **Training Engine** — Loop di training PyTorch custom con QLoRA (base a 4-bit), AdamW8bit, cosine warmup, gradient checkpointing. Cancel-safe.
- **Live Monitor** — Chart in tempo reale (loss, learning rate, VRAM, throughput) streammati via WebSocket durante il training.
- **Inference Playground** — Confronto side-by-side tra modello base e fine-tunato. Cache LRU con eviction a 2 slot. Preset di sampling (Preciso / Bilanciato / Creativo).
- **Export GGUF** — Conversione one-click a GGUF (Q4_K_M, Q5_K_M, Q8_0, Q3_K_M, F16). Scarica automaticamente llama.cpp al primo export. ~8s end-to-end per un modello da 135M.
- **System Awareness** — Auto-detection della GPU con configurazione di training suggerita in base alla VRAM disponibile.

## Stack

**Backend**
- Python 3.11+, FastAPI 0.115, SQLAlchemy 2.0, SQLite
- PyTorch 2.11 (CUDA 12.8), `transformers`, `peft`, `bitsandbytes`, `accelerate`
- llama.cpp (auto-installato dai release binaries) per l'export GGUF
- 408+ unit test che coprono ogni layer

**Frontend**
- React 19 + Vite + Tailwind 4
- `recharts` (chart training), `lucide-react` (icone), `react-router-dom`
- Consumer WebSocket per eventi training live

**Hardware target**
- GPU NVIDIA con CUDA 12.x, 8 GB+ di VRAM (consigliato 12 GB)
- Verificato su Windows 11; Linux supportato in linea di principio (non testato)

## Quick start

### Prerequisiti
- Python 3.11 o superiore
- Node.js 18+ e npm
- GPU NVIDIA con driver CUDA 12.x

### Setup

```powershell
git clone https://github.com/isilderrr1/NeuralForge.git
cd NeuralForge

# Backend
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### Avvio

Apri due terminali.

**Terminale 1 (backend):**
```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn main:app --reload
```

**Terminale 2 (frontend):**
```powershell
cd frontend
npm run dev
```

Poi apri <http://127.0.0.1:5173> nel browser. Le API docs sono su <http://127.0.0.1:8000/docs>.

## Architettura

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (React + Vite, porta 5173)                          │
│   ├─ Pagine: Dashboard, Dataset, Training, TrainingLive,    │
│   │          Inference, Monitor, Export, Models             │
│   └─ Componenti: ConfigSuggestion, GPUCard, QuickStartGuide │
└─────────────┬───────────────────────────────────────────────┘
              │ REST + WebSocket
┌─────────────▼───────────────────────────────────────────────┐
│ Backend FastAPI (uvicorn, porta 8000)                       │
│   ├─ /api/system   — detection GPU/CPU/RAM                  │
│   ├─ /api/models   — registry base model + download         │
│   ├─ /api/dataset  — ingestion multi-formato + alpaca conv. │
│   ├─ /api/training — start/cancel/list + eventi WS live     │
│   ├─ /api/inference— generazione side-by-side, cache modelli│
│   └─ /api/export   — pipeline GGUF                          │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│ Core (Python)                                               │
│   ├─ Training: loop QLoRA custom (PyTorch)                  │
│   ├─ Inference: model loader con cache (LRU, 2 slot)        │
│   ├─ Export: merge → convert → quantize (subproc llama.cpp) │
│   └─ Job: job manager async con cancel events               │
└─────────────┬───────────────────────────────────────────────┘
              │
       ┌──────┴──────┐
       │             │
   ┌───▼───┐   ┌─────▼──────┐
   │SQLite │   │ Filesystem │
   └───────┘   │  - models  │
               │  - datasets│
               │  - adapters│
               │  - exports │
               └────────────┘
```

## Roadmap

### v0.1.0 (attuale)
- ✅ M0: bootstrap full-stack
- ✅ M1: system detector + suggestion config training
- ✅ M2: model manager (download, gated detection, custom HF)
- ✅ M3: dataset engine (6 extractor, chunker intelligente, alpaca conv.)
- ✅ M4: training engine (QLoRA, AdamW8bit, cosine warmup)
- ✅ M5: training API + WebSocket live monitor + storico
- ✅ M6: inference + confronto side-by-side base/FT
- ✅ M7: pipeline export GGUF con llama.cpp auto-scaricato
- ✅ M8: polish, onboarding, documentazione

### v0.2.0 (pianificato)
- Modulo Dataset Importer (MITRE ATT&CK, NIST CSF, CVE database, plugin custom)
- Base model più grandi (Gemma 4, Phi-3.5, Llama-3.2)
- Inference streaming via WebSocket (token-by-token)
- Confronto a 3 modelli
- Packaging desktop con Tauri (app nativa, installer MSI)
- i18n (Inglese + Italiano)

### Parking lot (un giorno)
- Pubblicazione Microsoft Store (con code signing)
- Modulo di adversarial testing (jailbreak detection, prompt injection probe)
- Dashboard di interpretability (attention head, layer activation)
- Modalità training con differential privacy

## Contribuire

Issue e PR benvenuti. Vedi [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.

## Licenza

MIT — vedi [LICENSE](LICENSE).

---

<div align="center">

Costruito da **Antonio Ruocco**
Cybersecurity Engineer che impara AI engineering partendo dalle fondamenta.

[GitHub](https://github.com/isilderrr1) · [LinkedIn](https://www.linkedin.com/in/antonio-ruocco)

</div>
