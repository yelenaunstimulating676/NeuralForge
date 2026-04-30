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

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    GPUInfoSchema,
    SystemInfoSchema,
    TrainingConfigSchema,
    VRAMReadingSchema,
)
from core.memory import (
    detect_gpus,
    get_system_info,
    suggest_training_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get(
    "/info",
    response_model=SystemInfoSchema,
    summary="System info snapshot",
)
def read_system_info() -> SystemInfoSchema:
    """
    Snapshot completo del sistema: OS, Python, PyTorch, CUDA, GPU.
    Da chiamare al boot del frontend.
    """
    try:
        info = get_system_info()
        return SystemInfoSchema(
            os=info.os,
            python_version=info.python_version,
            torch_version=info.torch_version,
            cuda_available=info.cuda_available,
            gpu_count=info.gpu_count,
            gpus=[GPUInfoSchema(**asdict(g)) for g in info.gpus],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante get_system_info")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/gpus",
    response_model=list[GPUInfoSchema],
    summary="GPU detection",
)
def read_gpus() -> list[GPUInfoSchema]:
    """Elenco GPU rilevate, con dati VRAM letti al momento della call."""
    try:
        return [GPUInfoSchema(**asdict(g)) for g in detect_gpus()]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante detect_gpus")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/vram",
    response_model=VRAMReadingSchema,
    summary="VRAM live reading",
)
def read_vram(
    index: int = Query(0, ge=0, description="Indice GPU"),
) -> VRAMReadingSchema:
    """
    Lettura VRAM live della GPU. Pensato per polling leggero (1-2s).
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
    return VRAMReadingSchema(
        total_mb=g.vram_total_mb,
        used_mb=g.vram_used_mb,
        free_mb=g.vram_free_mb,
    )


@router.get(
    "/suggest",
    response_model=TrainingConfigSchema,
    summary="Suggested training configuration",
)
def read_training_suggestion(
    index: int = Query(0, ge=0, description="Indice GPU"),
    target_effective_batch: int = Query(
        16, ge=1, le=256,
        description="Batch effettivo desiderato (batch * grad_accum).",
    ),
) -> TrainingConfigSchema:
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
        return TrainingConfigSchema(**asdict(cfg))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante suggest_training_config")
        raise HTTPException(status_code=500, detail=str(exc)) from exc