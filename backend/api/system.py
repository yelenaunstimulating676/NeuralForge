"""
Endpoint FastAPI per il System Detector.

Espone:
    GET /api/system/info          → snapshot OS + GPU + CUDA
    GET /api/system/gpus          → lista GPU rilevate
    GET /api/system/vram          → VRAM live (per polling frequente)
    GET /api/system/suggest       → configurazione training suggerita
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from core.memory import (
    detect_gpus,
    get_system_info,
    suggest_training_config,
    system_info_to_dict,
    training_config_to_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/info")
def read_system_info() -> dict[str, Any]:
    """
    Snapshot completo del sistema: OS, Python, PyTorch, CUDA, GPU.
    Da chiamare al boot del frontend.
    """
    try:
        info = get_system_info()
        return system_info_to_dict(info)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante get_system_info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/gpus")
def read_gpus() -> list[dict[str, Any]]:
    """Elenco GPU rilevate, con dati VRAM letti al momento della call."""
    try:
        return [asdict(g) for g in detect_gpus()]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante detect_gpus")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/vram")
def read_vram(index: int = Query(0, ge=0, description="Indice GPU")) -> dict[str, int]:
    """
    Lettura VRAM live della GPU. Pensato per polling leggero (1-2s)
    quando il WebSocket di training non è attivo.
    """
    gpus = detect_gpus()
    if not gpus:
        raise HTTPException(status_code=404, detail="Nessuna GPU NVIDIA rilevata.")
    if index >= len(gpus):
        raise HTTPException(
            status_code=404,
            detail=f"GPU index {index} non valido (disponibili: {len(gpus)}).",
        )
    g = gpus[index]
    return {
        "total_mb": g.vram_total_mb,
        "used_mb": g.vram_used_mb,
        "free_mb": g.vram_free_mb,
    }


@router.get("/suggest")
def read_training_suggestion(
    index: int = Query(0, ge=0, description="Indice GPU"),
    target_effective_batch: int = Query(
        16, ge=1, le=256,
        description="Batch effettivo desiderato (batch * grad_accum)",
    ),
) -> dict[str, Any]:
    """
    Configurazione di training suggerita in base alla VRAM.
    Il frontend la usa come default editabile.
    """
    gpus = detect_gpus()
    if not gpus:
        raise HTTPException(status_code=404, detail="Nessuna GPU NVIDIA rilevata.")
    if index >= len(gpus):
        raise HTTPException(
            status_code=404,
            detail=f"GPU index {index} non valido (disponibili: {len(gpus)}).",
        )

    try:
        cfg = suggest_training_config(
            gpus[index], target_effective_batch=target_effective_batch
        )
        return training_config_to_dict(cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante suggest_training_config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc