<div align="center">

# ⚡ NeuralForge

**Local LLM fine-tuning, made simple.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tune large language models on consumer GPUs with a web interface.
QLoRA training, side-by-side inference comparison, one-click GGUF export — all running locally.

**🇬🇧 English** · [🇮🇹 Italiano](README.it.md)

</div>

---

## 📸 Preview

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="NeuralForge Dashboard" width="900" />
</p>

### 🎥 Video demo

[![Watch the demo](https://img.shields.io/badge/▶_Watch_on_YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/sem7k0spFh4)

End-to-end walkthrough: custom HuggingFace download → dataset import → QLoRA training with live monitor → base vs fine-tuned comparison → GGUF export.

<details>
<summary><strong>More screenshots</strong></summary>

### Model Manager
Whitelist of curated HuggingFace models, custom HF repo download with gated detection, and a local registry of downloaded models.

<p align="center">
  <img src="docs/screenshots/models.png" alt="Model Manager" width="850" />
</p>

### Dataset wizard
3-step wizard for importing datasets from PDF, DOCX, CSV, TXT, JSON, or JSONL into alpaca format.

<p align="center">
  <img src="docs/screenshots/dataset.png" alt="Dataset wizard" width="850" />
</p>

### Training configuration
Auto-suggested config based on detected GPU/VRAM, plus full manual control over QLoRA hyperparameters.

<p align="center">
  <img src="docs/screenshots/training.png" alt="Training configuration" width="850" />
</p>

### Live Monitor
Real-time charts (loss, learning rate, VRAM, throughput) streamed via WebSocket while training runs.

<p align="center">
  <img src="docs/screenshots/live-training.png" alt="Live Monitor" width="850" />
</p>

### Inference comparison
Side-by-side comparison of base vs fine-tuned models on the same prompt.

<p align="center">
  <img src="docs/screenshots/inference.png" alt="Inference comparison" width="850" />
</p>

### GGUF Export
One-click conversion to GGUF (Q4_K_M, Q5_K_M, Q8_0, Q3_K_M, F16). Auto-downloads llama.cpp on first export.

<p align="center">
  <img src="docs/screenshots/export.png" alt="GGUF Export" width="850" />
</p>

</details>

---

## What is NeuralForge?

NeuralForge is a local-first fine-tuning platform for LLMs. It wraps the PyTorch + PEFT + bitsandbytes stack with a modern web UI so that fine-tuning a model on your data does not require notebooks, CLI gymnastics, or cloud accounts.

**The whole pipeline runs on your machine.** No telemetry, no API keys, no data leaving your hardware. You bring a base model from HuggingFace and a dataset; NeuralForge handles tokenization, QLoRA training with live monitoring, inference comparison, and export to the GGUF format used by Ollama, LM Studio, and llama.cpp.

Built for the consumer-GPU sweet spot: a single 12 GB card (RTX 4070-class) is enough to fine-tune 1-3B class models.

## Real-world validation

NeuralForge has been used to fine-tune a model on its own API documentation as a proof-of-concept. The full run is captured in the demo video above.

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4070 (12 GB VRAM) |
| Base model | Qwen2.5-1.5B-Instruct |
| Dataset | 123 hand-written Q&A pairs (NeuralForge API docs, no AI generation) |
| Strategy | QLoRA, LoRA rank 32, 3 epochs |
| Training time | **1 min 38 sec** |
| Steps | 93 |
| Final loss | 2.09 |
| Peak VRAM during training | ~3.9 GB |

### Sample output: base vs fine-tuned

Same prompt to both models: *"What is the purpose of the /api/training endpoint?"*

**Base model (Qwen2.5-1.5B-Instruct, no fine-tuning):**

> The /api/training endpoint is used to submit training requests and receive responses from an assistant. It allows users to request specific tasks or information from the assistant, such as completing a task, finding relevant information, or answering questions.

→ Generic and off-topic. The base model has no idea what NeuralForge is and treats "training" as some kind of corporate training service.

**Fine-tuned model (same base + 1m 38s of QLoRA training on the dataset):**

> The /api/training endpoint runs fine-tuning jobs (...). The frontend POSTs a JSON with model name, dataset path (optional), base config (e.g., LlamaConfig), and optional overrides (custom learning rate, temperature). The backend creates an AsyncJob object in state.db, which tracks progress through the run() method. Once complete, the frontend polls for status updates until 'finished' or 'failed', then displays the result page with download links to the final weights and logs files.

→ Same prompt, completely different answer: now in the NeuralForge domain. The model has learned the request/response structure (POST + JSON payload), the async job pattern, and the polling flow. There are minor inaccuracies (the actual backend doesn't run on AWS), but the domain shift is visible and the structural pattern matches the documentation.

This is a single-run proof-of-concept on a tiny dataset. With a larger dataset (1000+ examples), bigger models (Phi-3.5-mini, Llama-3.2-3B), and more epochs, much sharper specialization is achievable — that's the v0.2.0 roadmap.

## Features

- **Model Manager** — Whitelist of curated HuggingFace models + custom repo support with gated-model detection.
- **Dataset Engine** — Import PDF, DOCX, CSV, TXT, JSON, JSONL. Smart chunking, deduplication, alpaca-format conversion in a 3-step wizard.
- **Training Engine** — Custom PyTorch training loop with QLoRA (4-bit base), AdamW8bit, cosine warmup, gradient checkpointing. Cancel-safe.
- **Live Monitor** — Real-time charts (loss, learning rate, VRAM, throughput) streamed via WebSocket while training runs.
- **Inference Playground** — Side-by-side comparison of base vs fine-tuned models. LRU model cache with 2-slot eviction. Sampling presets (Precise / Balanced / Creative).
- **GGUF Export** — One-click conversion to GGUF (Q4_K_M, Q5_K_M, Q8_0, Q3_K_M, F16). Auto-downloads llama.cpp on first export. ~8s end-to-end for a 135M model.
- **System Awareness** — Auto-detects GPU and suggests a training configuration that fits the available VRAM.

## Stack

**Backend**
- Python 3.11+, FastAPI 0.115, SQLAlchemy 2.0, SQLite
- PyTorch 2.11 (CUDA 12.8), `transformers`, `peft`, `bitsandbytes`, `accelerate`
- llama.cpp (auto-installed via release binaries) for GGUF export
- 408+ unit tests covering every layer

**Frontend**
- React 19 + Vite + Tailwind 4
- `recharts` (training charts), `lucide-react` (icons), `react-router-dom`
- WebSocket consumer for live training events

**Hardware target**
- NVIDIA GPU with CUDA 12.x, 8 GB+ VRAM (12 GB recommended)
- Windows 11 verified; Linux supported in principle (untested)

## Quick start

### Prerequisites
- Python 3.11 or newer
- Node.js 18+ and npm
- NVIDIA GPU with CUDA 12.x drivers

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

### Run

Open two terminals.

**Terminal 1 (backend):**
```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn main:app --reload
```

**Terminal 2 (frontend):**
```powershell
cd frontend
npm run dev
```

Then open <http://127.0.0.1:5173> in your browser. API docs are at <http://127.0.0.1:8000/docs>.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (React + Vite, port 5173)                           │
│   ├─ Pages: Dashboard, Dataset, Training, TrainingLive,     │
│   │         Inference, Monitor, Export, Models              │
│   └─ Components: ConfigSuggestion, GPUCard, QuickStartGuide │
└─────────────┬───────────────────────────────────────────────┘
              │ REST + WebSocket
┌─────────────▼───────────────────────────────────────────────┐
│ FastAPI backend (uvicorn, port 8000)                        │
│   ├─ /api/system   — GPU/CPU/RAM detection                  │
│   ├─ /api/models   — base model registry + downloads        │
│   ├─ /api/dataset  — multi-format ingestion + alpaca conv.  │
│   ├─ /api/training — start/cancel/list + live WS events     │
│   ├─ /api/inference— side-by-side generation, model cache   │
│   └─ /api/export   — GGUF pipeline                          │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│ Core (Python)                                               │
│   ├─ Training: custom QLoRA loop (PyTorch)                  │
│   ├─ Inference: cached model loader (LRU, 2 slots)          │
│   ├─ Export: merge → convert → quantize (llama.cpp subproc) │
│   └─ Jobs: async job manager with cancel events             │
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

### v0.1.0 (current)
- ✅ M0: full-stack bootstrap
- ✅ M1: system detector + training config suggestion
- ✅ M2: model manager (downloads, gated detection, custom HF)
- ✅ M3: dataset engine (6 extractors, smart chunker, alpaca conv.)
- ✅ M4: training engine (QLoRA, AdamW8bit, cosine warmup)
- ✅ M5: training API + WebSocket live monitor + history
- ✅ M6: inference + base/FT side-by-side comparison
- ✅ M7: GGUF export pipeline with auto-downloaded llama.cpp
- ✅ M8: polish, onboarding, documentation, real-world validation

### v0.2.0 (planned)
- Dataset Importer module (MITRE ATT&CK, NIST CSF, CVE database, custom plugins)
- Bigger base models (Gemma 4, Phi-3.5, Llama-3.2)
- Streaming WebSocket inference (token-by-token)
- 3-way model comparison
- Tauri desktop packaging (native app, MSI installer)
- i18n (English + Italian)

### Parking lot (someday)
- Microsoft Store submission (with code signing)
- Adversarial testing module (jailbreak detection, prompt injection probe)
- Interpretability dashboard (attention heads, layer activations)
- Differential privacy training mode

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by **Antonio Ruocco**
A Cybersecurity Engineer learning AI engineering from the ground up.

[GitHub](https://github.com/isilderrr1) · [LinkedIn](https://www.linkedin.com/in/antonio-ruocco)

</div>
