"""
Endpoint REST `/api/export/*`.

Pattern (copiato da api/training.py):
  - Factory async che riceve (progress_cb, cancel_event)
  - export_ft_to_gguf è SYNC → asyncio.to_thread
  - Watcher async ponte: cancel_event (asyncio) → threading.Event
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.schemas import (
    ExportFileSchema,
    ExportJobSchema,
    ExportStartRequestSchema,
    ExportStartResponseSchema,
    QuantizationOptionSchema,
)
from config import settings
from core.export.exporter import (
    VALID_QUANTIZATIONS,
    ExportError,
    export_ft_to_gguf,
    validate_quantization,
)
from core.jobs import Job, job_manager
from db import get_session
from db.models import (
    BaseModel as BaseModelRow,
    FineTunedModel as FineTunedModelRow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])


# ---------------------------------------------------------------------------
# Quantization options (statico, per UI)
# ---------------------------------------------------------------------------


QUANT_OPTIONS = [
    QuantizationOptionSchema(
        value="Q4_K_M",
        label="Q4_K_M — bilanciato",
        description="Default consigliato. ~50% size, ~95% qualità. Usato da Ollama.",
        is_default=True,
    ),
    QuantizationOptionSchema(
        value="Q5_K_M",
        label="Q5_K_M — qualità",
        description="Migliore qualità di Q4, file più grande (~60% size).",
        is_default=False,
    ),
    QuantizationOptionSchema(
        value="Q8_0",
        label="Q8_0 — alta qualità",
        description="Quasi identico al F16. File grande (~80% size).",
        is_default=False,
    ),
    QuantizationOptionSchema(
        value="Q3_K_M",
        label="Q3_K_M — compatto",
        description="Massima compressione (~40% size), qualità ridotta.",
        is_default=False,
    ),
    QuantizationOptionSchema(
        value="F16",
        label="F16 — senza quantizzazione",
        description="Modello originale, nessuna perdita. File 5-10x più grande.",
        is_default=False,
    ),
]


@router.get(
    "/quantizations",
    response_model=list[QuantizationOptionSchema],
    summary="Lista quantizzazioni disponibili",
)
async def list_quantizations() -> list[QuantizationOptionSchema]:
    return QUANT_OPTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str) -> str:
    """Rimuove caratteri non validi per nomi file Windows."""
    s = _INVALID_NAME_CHARS.sub("_", name).strip(" .")
    s = re.sub(r"\s+", "_", s)
    return s or "export"


def build_output_filename(
    custom_name: str | None,
    ft_name: str,
    quantization: str,
) -> str:
    """Costruisce il nome file .gguf."""
    base = sanitize_filename(custom_name) if custom_name else sanitize_filename(ft_name)
    return f"{base}__{quantization}.gguf"


def parse_filename_metadata(filename: str) -> dict[str, str | None]:
    """Estrae ft_name + quantization dal nome file."""
    name_without_ext = filename.removesuffix(".gguf")
    if "__" in name_without_ext:
        ft_name, _, quant = name_without_ext.rpartition("__")
        if quant.upper() in VALID_QUANTIZATIONS:
            return {"ft_name": ft_name, "quantization": quant.upper()}
    return {"ft_name": None, "quantization": "unknown"}


def _job_to_schema(job: Job) -> ExportJobSchema:
    return ExportJobSchema(
        job_id=job.id,
        kind=job.kind,
        status=job.status.value,
        progress=job.progress,
        message=job.progress_message,
        created_at=job.created_at.isoformat() if job.created_at else "",
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        error=job.error,
        result=job.result,
    )


# ---------------------------------------------------------------------------
# Start export
# ---------------------------------------------------------------------------


@router.post(
    "/start",
    response_model=ExportStartResponseSchema,
    summary="Avvia export GGUF di un FT model",
)
async def start_export(
    body: ExportStartRequestSchema,
    session: Session = Depends(get_session),
) -> ExportStartResponseSchema:
    # 1. Valida quantization
    try:
        quantization = validate_quantization(body.quantization)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 2. Verifica FT model + base model
    ft = session.get(FineTunedModelRow, body.ft_model_id)
    if ft is None:
        raise HTTPException(
            status_code=404,
            detail=f"FineTunedModel id={body.ft_model_id} non trovato.",
        )
    if not Path(ft.adapter_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Adapter non trovato su disco: {ft.adapter_path}",
        )

    base = session.get(BaseModelRow, ft.base_model_id)
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"BaseModel parent (id={ft.base_model_id}) non trovato.",
        )
    if not Path(base.local_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Base model non presente su disco: {base.local_path}",
        )

    # 3. Path output + check duplicato
    output_filename = build_output_filename(body.output_name, ft.name, quantization)
    output_path = settings.exports_path / output_filename

    if output_path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"File già esistente: {output_filename}. "
                "Scegli un altro nome o elimina il file esistente."
            ),
        )

    # Capture per closure (no DB session dentro al thread!)
    base_path = Path(base.local_path)
    adapter_path = Path(ft.adapter_path)
    ft_model_id = ft.id
    ft_model_name = ft.name

    # 4. Factory async che lancia export_ft_to_gguf in thread
    async def factory(progress_cb, cancel_event):
        # Map stage → segmento [0.0, 1.0] (JobManager usa 0-1, NON 0-100)
        STAGE_RANGES = {
            "llamacpp:downloading_binaries": (0.00, 0.10),
            "llamacpp:extracting_binaries": (0.10, 0.12),
            "llamacpp:downloading_script": (0.12, 0.15),
            "preparing_llamacpp": (0.00, 0.15),
            "merging_loading_base": (0.15, 0.20),
            "merging_loading_adapter": (0.20, 0.25),
            "merging_running": (0.25, 0.35),
            "merging_saving": (0.35, 0.40),
            "merging_done": (0.40, 0.40),
            "converting": (0.40, 0.70),
            "quantizing": (0.70, 1.00),
            "done": (1.00, 1.00),
        }
        STAGE_LABELS = {
            "llamacpp:downloading_binaries": "Download binari llama.cpp…",
            "llamacpp:downloading_script": "Download script conversione…",
            "preparing_llamacpp": "Preparazione llama.cpp…",
            "merging_loading_base": "Caricamento base model…",
            "merging_loading_adapter": "Caricamento adapter LoRA…",
            "merging_running": "Merge LoRA + base in corso…",
            "merging_saving": "Salvataggio modello fuso…",
            "merging_done": "Merge completato.",
            "converting": "Conversione safetensors → GGUF F16…",
            "quantizing": f"Quantizzazione → {quantization}…",
            "done": "Export completato.",
        }

        # Bridge cancel: asyncio.Event → threading.Event (export è sync)
        thread_cancel = threading.Event()

        async def cancel_watcher():
            await cancel_event.wait()
            thread_cancel.set()

        watcher = asyncio.create_task(cancel_watcher())

        # Callback che gira nel thread di export (sync)
        # Deve essere thread-safe. progress_cb del JobManager va bene a basso
        # rate (qualche update al secondo, non blocking).
        def stage_cb(stage: str, stage_pct: float) -> None:
            if thread_cancel.is_set():
                # Segnala interruzione al merge/convert/quantize (best effort)
                raise RuntimeError("Export cancellato dall'utente")
            lo, hi = STAGE_RANGES.get(stage, (0.0, 1.0))
            overall_pct = lo + (hi - lo) * stage_pct
            label = STAGE_LABELS.get(stage, stage)
            progress_cb(overall_pct, label)

        def thread_target():
            return export_ft_to_gguf(
                base_model_path=base_path,
                adapter_path=adapter_path,
                output_path=output_path,
                quantization=quantization,
                progress_callback=stage_cb,
            )

        try:
            result = await asyncio.to_thread(thread_target)
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        return {
            "output_filename": output_filename,
            "output_path": str(result.output_path),
            "quantization": result.quantization,
            "size_bytes": result.size_bytes,
            "elapsed_seconds": result.elapsed_seconds,
            "ft_model_id": ft_model_id,
            "ft_model_name": ft_model_name,
        }

    job = await job_manager.submit("export", factory)

    logger.info(
        "Export submitted: job_id=%s ft_id=%s quant=%s filename=%s",
        job.id, ft_model_id, quantization, output_filename,
    )

    return ExportStartResponseSchema(
        job_id=job.id,
        ft_model_id=ft_model_id,
        quantization=quantization,
        expected_filename=output_filename,
    )


# ---------------------------------------------------------------------------
# Jobs list + detail + cancel
# ---------------------------------------------------------------------------


@router.get(
    "/jobs",
    response_model=list[ExportJobSchema],
    summary="Lista job di export",
)
async def list_jobs() -> list[ExportJobSchema]:
    jobs = await job_manager.list(kind="export")
    return [_job_to_schema(j) for j in jobs]


@router.get(
    "/jobs/{job_id}",
    response_model=ExportJobSchema,
    summary="Stato di un job export",
)
async def get_job(job_id: str) -> ExportJobSchema:
    job = await job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato.")
    return _job_to_schema(job)


@router.delete(
    "/jobs/{job_id}",
    summary="Cancella un job export (se ancora in corso)",
)
async def cancel_job(job_id: str) -> dict:
    success = await job_manager.cancel(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} non cancellabile (forse già finito).",
        )
    return {"cancelled": True, "job_id": job_id}


# ---------------------------------------------------------------------------
# Files: list / delete / download
# ---------------------------------------------------------------------------


@router.get(
    "/files",
    response_model=list[ExportFileSchema],
    summary="Lista file .gguf esportati",
)
async def list_files() -> list[ExportFileSchema]:
    exports_dir = settings.exports_path
    exports_dir.mkdir(parents=True, exist_ok=True)

    files: list[ExportFileSchema] = []
    for f in sorted(
        exports_dir.glob("*.gguf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        meta = parse_filename_metadata(f.name)
        stat = f.stat()
        files.append(ExportFileSchema(
            filename=f.name,
            path=str(f),
            size_bytes=stat.st_size,
            quantization=meta["quantization"] or "unknown",
            ft_name=meta["ft_name"],
            created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))
    return files


@router.delete(
    "/files/{filename}",
    summary="Elimina un export GGUF",
)
async def delete_file(filename: str) -> dict:
    # Path traversal protection
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nome file non valido.")
    if not filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Solo file .gguf possono essere eliminati.")

    target = settings.exports_path / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} non trovato.")

    try:
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Errore eliminazione: {exc}")

    return {"deleted": True, "filename": filename}


@router.get(
    "/files/{filename}/download",
    summary="Scarica un file GGUF",
)
async def download_file(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nome file non valido.")
    if not filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Solo file .gguf scaricabili.")

    target = settings.exports_path / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} non trovato.")

    return FileResponse(
        path=str(target),
        filename=filename,
        media_type="application/octet-stream",
    )