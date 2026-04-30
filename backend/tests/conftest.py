"""
Configurazione globale pytest per NeuralForge.

Aggiunge la cartella `backend/` al sys.path così i test possono fare
`from core.memory import ...` come fa l'app stessa.
"""

import sys
from pathlib import Path

# backend/ root → un livello sopra a tests/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))