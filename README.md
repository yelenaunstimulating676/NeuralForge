# NeuralForge — Stato sviluppo

## Completato
- M0: Backend FastAPI + frontend Vite/React/Tailwind4 + healthcheck
- M1: System Detector (GPU/VRAM live, training config suggestion)
- M2: Model Manager (whitelist 18 modelli, download async, custom HF, gated detection)

## In corso
- M3: Dataset Engine (PDF/CSV/TXT/JSON/DOCX → instruction tuning)

## Comandi rapidi
- Backend: `cd backend && .\venv\Scripts\activate && python -m uvicorn main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Tests: `cd backend && pytest tests/ -v`
- URL: http://127.0.0.1:5173 (frontend) — http://127.0.0.1:8000/docs (API)