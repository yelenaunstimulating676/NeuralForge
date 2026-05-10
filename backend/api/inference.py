"""
Endpoint REST `/api/inference/*`.

Workflow tipico:
    1. Client chiama GET /models/available → vede base + ft disponibili
    2. Client chiama POST /generate con model_kind+model_id e prompt
    3. Backend carica (cached) e genera
    4. Client può chiamare DELETE /models/{key} per liberare VRAM
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import (
    AvailableModelSchema,
    InferenceGenerateRequestSchema,
    InferenceGenerateResponseSchema,
    LoadedModelSchema,
)
from core.inference.generator import GenerationParams, generate_text
from core.inference.loader import (
    InferenceLoaderError,
    ModelLoader,
    model_loader,
)
from db import get_session
from db.models import (
    BaseModel as BaseModelRow,
    FineTunedModel as FineTunedModelRow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


# ---------------------------------------------------------------------------
# Models available (base + ft)
# ---------------------------------------------------------------------------


@router.get(
    "/models/available",
    response_model=list[AvailableModelSchema],
    summary="Lista modelli disponibili per inference (base + fine-tuned)",
)
def list_available_models(
    session: Session = Depends(get_session),
) -> list[AvailableModelSchema]:
    """
    Combina BaseModel e FineTunedModel in un'unica lista.
    Indica quali sono già in cache (per UI 'caricamento istantaneo' vs 'caricherò').
    """
    out: list[AvailableModelSchema] = []

    loaded_keys = {m["key"] for m in model_loader.list_loaded()}

    # Base models
    bases = list(session.scalars(select(BaseModelRow)).all())
    for b in bases:
        key = ModelLoader.make_key("base", b.id)
        out.append(AvailableModelSchema(
            key=key,
            kind="base",
            model_id=b.id,
            display_name=b.display_name,
            base_model_id=b.id,
            base_model_name=b.display_name,
            is_loaded=key in loaded_keys,
            metadata={
                "params_billions": b.params_billions,
                "size_bytes": b.size_bytes,
            },
        ))

    # Fine-tuned models
    fts = list(
        session.scalars(
            select(FineTunedModelRow).order_by(FineTunedModelRow.created_at.desc())
        ).all()
    )
    for ft in fts:
        key = ModelLoader.make_key("ft", ft.id)
        base = session.get(BaseModelRow, ft.base_model_id)
        # Estrai metriche dal JSON se presente
        import json
        metrics = None
        if ft.metrics_json:
            try:
                metrics = json.loads(ft.metrics_json)
            except json.JSONDecodeError:
                pass

        out.append(AvailableModelSchema(
            key=key,
            kind="ft",
            model_id=ft.id,
            display_name=ft.name,
            base_model_id=ft.base_model_id,
            base_model_name=base.display_name if base else None,
            is_loaded=key in loaded_keys,
            metadata={
                "final_loss": metrics.get("final_loss") if metrics else None,
                "total_steps": metrics.get("total_steps") if metrics else None,
                "size_bytes": ft.size_bytes,
            },
        ))

    return out


# ---------------------------------------------------------------------------
# Models loaded (cache state)
# ---------------------------------------------------------------------------


@router.get(
    "/models/loaded",
    response_model=list[LoadedModelSchema],
    summary="Lista modelli attualmente caricati in VRAM",
)
def list_loaded_models() -> list[LoadedModelSchema]:
    return [LoadedModelSchema(**m) for m in model_loader.list_loaded()]


@router.delete(
    "/models/{key}",
    summary="Scarica un modello dalla VRAM",
)
def unload_model(key: str) -> dict:
    """Unload manuale di un modello. `key` è del tipo 'base:5' o 'ft:12'."""
    removed = model_loader.unload(key)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Modello {key} non in cache.")
    return {"unloaded": True, "key": key}


@router.delete(
    "/models",
    summary="Scarica TUTTI i modelli dalla VRAM",
)
def unload_all_models() -> dict:
    n = model_loader.unload_all()
    return {"unloaded_count": n}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=InferenceGenerateResponseSchema,
    summary="Genera testo con un modello (base o fine-tuned)",
)
async def generate_endpoint(
    body: InferenceGenerateRequestSchema,
    session: Session = Depends(get_session),
) -> InferenceGenerateResponseSchema:
    """
    Genera testo. Carica il modello in cache se non lo è già.
    L'inference gira in un thread (modello GPU-bound, blocking).
    """
    # 1. Carica (cached) il modello
    try:
        cached = model_loader.load(
            session, body.model_kind, body.model_id
        )
    except InferenceLoaderError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 2. Costruisci GenerationParams
    try:
        params = GenerationParams(
            max_new_tokens=body.params.max_new_tokens,
            temperature=body.params.temperature,
            top_p=body.params.top_p,
            top_k=body.params.top_k,
            repetition_penalty=body.params.repetition_penalty,
            do_sample=body.params.do_sample,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Parametri invalidi: {exc}")

    # 3. Genera (in thread per non bloccare event loop)
    try:
        result = await asyncio.to_thread(
            generate_text,
            model=cached.loaded.model,
            tokenizer=cached.loaded.tokenizer,
            prompt=body.prompt,
            params=params,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Errore durante generate")
        raise HTTPException(status_code=500, detail=f"Errore generazione: {exc}")

    return InferenceGenerateResponseSchema(
        text=result.text,
        tokens_generated=result.tokens_generated,
        elapsed_seconds=result.elapsed_seconds,
        throughput_tokens_per_sec=result.throughput_tokens_per_sec,
        finish_reason=result.finish_reason,
        model_key=cached.key,
        model_display_name=cached.display_name,
    )