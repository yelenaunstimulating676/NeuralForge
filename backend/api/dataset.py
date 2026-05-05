"""
Endpoint REST per il Dataset Engine.

Workflow:
    1. POST /api/dataset/upload                          → ottieni upload_id
    2. POST /api/dataset/upload/{id}/analyze             → vedi tipo + metadata
    3. POST /api/dataset/upload/{id}/preview             → vedi esempi generati
    4. POST /api/dataset/upload/{id}/save                → salva dataset definitivo

Operazioni di gestione:
    GET    /api/dataset                                  → lista
    GET    /api/dataset/{id}                             → dettaglio
    GET    /api/dataset/{id}/examples                    → esempi
    DELETE /api/dataset/{id}                             → rimuovi
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from api.schemas import (
    AnalyzeResponseSchema,
    DatasetSchema,
    DetectionResultSchema,
    ExtractedDocumentSchema,
    InstructionExampleSchema,
    PreviewRequestSchema,
    PreviewResponseSchema,
    SaveDatasetRequestSchema,
    SaveDatasetResponseSchema,
    UploadResponseSchema,
)
from core.dataset.chunker import ChunkerConfig, chunk_document
from core.dataset.converter import ConverterConfig, convert_chunks
from core.dataset.detector import ContentType, detect_content_type
from core.dataset.extractors import (
    UnsupportedFormatError,
    extract_file,
    is_supported_extension,
)
from core.dataset.extractors.base import ExtractorError
from core.dataset.persistence import (
    DatasetNameConflictError,
    DatasetNotFoundError,
    delete_dataset,
    get_dataset_by_id,
    list_datasets,
    load_dataset_examples,
    save_dataset,
)
from core.dataset.uploads import upload_manager
from core.dataset.validator import ValidatorConfig, validate_examples
from db import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# Limite dimensione upload: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_chunker_config(payload) -> ChunkerConfig:
    """Costruisce ChunkerConfig dai campi non-None del payload."""
    kwargs = {}
    if payload.target_chars is not None:
        kwargs["target_chars"] = payload.target_chars
    if payload.overlap_chars is not None:
        kwargs["overlap_chars"] = payload.overlap_chars
    if payload.min_chunk_chars is not None:
        kwargs["min_chunk_chars"] = payload.min_chunk_chars
    if payload.max_chunk_chars is not None:
        kwargs["max_chunk_chars"] = payload.max_chunk_chars
    return ChunkerConfig(**kwargs)


def _build_converter_config(payload) -> ConverterConfig:
    kwargs = {}
    if payload.examples_per_narrative_chunk is not None:
        kwargs["examples_per_narrative_chunk"] = payload.examples_per_narrative_chunk
    if payload.template_language is not None:
        kwargs["template_language"] = payload.template_language
    if payload.min_chars is not None:
        kwargs["min_chars"] = payload.min_chars
    if payload.min_output_chars is not None:
        kwargs["min_output_chars"] = payload.min_output_chars
    return ConverterConfig(**kwargs)


def _build_validator_config(payload) -> ValidatorConfig:
    kwargs = {}
    if payload.min_output_chars is not None:
        kwargs["min_output_chars"] = payload.min_output_chars
    if payload.max_output_chars is not None:
        kwargs["max_output_chars"] = payload.max_output_chars
    if payload.max_total_chars is not None:
        kwargs["max_total_chars"] = payload.max_total_chars
    if payload.enable_fuzzy_dedup is not None:
        kwargs["enable_fuzzy_dedup"] = payload.enable_fuzzy_dedup
    if payload.fuzzy_threshold is not None:
        kwargs["fuzzy_threshold"] = payload.fuzzy_threshold
    return ValidatorConfig(**kwargs)


def _resolve_content_type(override: str | None, detected: ContentType) -> ContentType:
    """Se override è valido, usalo. Altrimenti detected."""
    if override is None:
        return detected
    try:
        return ContentType(override)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"content_type_override non valido: {override!r}. "
                   f"Valori validi: {[c.value for c in ContentType]}",
        ) from exc


def _row_to_schema(row, parse_stats: bool = True) -> DatasetSchema:
    """Converte un Dataset ORM in DatasetSchema."""
    stats = None
    if parse_stats and row.stats_json:
        try:
            stats = json.loads(row.stats_json)
        except json.JSONDecodeError:
            stats = None
    return DatasetSchema(
        id=row.id,
        name=row.name,
        source_file=row.source_file,
        file_path=row.file_path,
        num_examples=row.num_examples,
        format=row.format,
        stats=stats,
        created_at=row.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=UploadResponseSchema,
    summary="Carica un file da convertire in dataset",
)
async def upload_file(file: UploadFile = File(...)) -> UploadResponseSchema:
    """
    Carica un file (PDF/CSV/TXT/JSON/JSONL/DOCX/MD).
    Il file vive in `data/uploads/<upload_id>/` finché non viene processato.
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="Nome file mancante.")

    ext = Path(file.filename).suffix.lower()
    if not is_supported_extension(Path(file.filename)):
        raise HTTPException(
            status_code=422,
            detail=f"Estensione {ext!r} non supportata. "
                   f"Formati ammessi: PDF, CSV, TSV, TXT, MD, JSON, JSONL, DOCX.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File vuoto.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File troppo grande: {len(content) / 1024 / 1024:.1f} MB "
                   f"(max {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB).",
        )

    info = upload_manager.register(file.filename, content)
    return UploadResponseSchema(**info.to_dict())


# ---------------------------------------------------------------------------
# Analyze (extract + detect)
# ---------------------------------------------------------------------------


@router.post(
    "/upload/{upload_id}/analyze",
    response_model=AnalyzeResponseSchema,
    summary="Estrai e classifica il contenuto del file",
)
def analyze_upload(upload_id: str) -> AnalyzeResponseSchema:
    """
    Esegue Extractor + Detector. Ritorna:
      - riassunto del documento estratto (formato, char count, sezioni)
      - tipo di contenuto rilevato + confidence
    """
    info = upload_manager.get(upload_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} non trovato.")

    try:
        doc = extract_file(info.file_path)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ExtractorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    detection = detect_content_type(doc)

    return AnalyzeResponseSchema(
        upload_id=upload_id,
        extracted=ExtractedDocumentSchema(**doc.to_dict()),
        detection=DetectionResultSchema(**detection.to_dict()),
    )


# ---------------------------------------------------------------------------
# Preview (chunk + convert)
# ---------------------------------------------------------------------------


@router.post(
    "/upload/{upload_id}/preview",
    response_model=PreviewResponseSchema,
    summary="Genera una preview di esempi con i parametri dati",
)
def preview_upload(
    upload_id: str,
    body: PreviewRequestSchema,
) -> PreviewResponseSchema:
    """
    Estrae, classifica, chunka e converte in esempi. Ritorna i primi N
    esempi (max_examples) per anteprima.

    NB: per file grossi processa l'intero documento ma ritorna solo N
    esempi. Se serve più velocità in futuro, possiamo introdurre uno
    short-circuit dopo X chunk.
    """
    info = upload_manager.get(upload_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} non trovato.")

    try:
        doc = extract_file(info.file_path)
    except (UnsupportedFormatError, ExtractorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    detection = detect_content_type(doc)
    content_type = _resolve_content_type(
        body.content_type_override, detection.content_type
    )

    chunker_cfg = _build_chunker_config(body.chunker_config)
    converter_cfg = _build_converter_config(body.converter_config)

    chunks = chunk_document(doc, content_type, chunker_cfg)
    examples = convert_chunks(chunks, doc, content_type, converter_cfg)

    preview = examples[: body.max_examples]
    return PreviewResponseSchema(
        upload_id=upload_id,
        content_type=content_type.value,
        examples=[InstructionExampleSchema(**e.to_dict()) for e in preview],
        total_chunks=len(chunks),
        total_examples_estimated=len(examples),
    )


# ---------------------------------------------------------------------------
# Save (full pipeline + persist)
# ---------------------------------------------------------------------------


@router.post(
    "/upload/{upload_id}/save",
    response_model=SaveDatasetResponseSchema,
    summary="Esegui la pipeline completa e salva il dataset",
)
def save_upload_as_dataset(
    upload_id: str,
    body: SaveDatasetRequestSchema,
    session: Session = Depends(get_session),
) -> SaveDatasetResponseSchema:
    """
    Pipeline completa: extract → detect → chunk → convert → validate → save.
    Cancella l'upload temporaneo dopo il salvataggio.
    """
    info = upload_manager.get(upload_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} non trovato.")

    try:
        doc = extract_file(info.file_path)
    except (UnsupportedFormatError, ExtractorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    detection = detect_content_type(doc)
    content_type = _resolve_content_type(
        body.content_type_override, detection.content_type
    )

    chunker_cfg = _build_chunker_config(body.chunker_config)
    converter_cfg = _build_converter_config(body.converter_config)
    validator_cfg = _build_validator_config(body.validator_config)

    chunks = chunk_document(doc, content_type, chunker_cfg)
    examples = convert_chunks(chunks, doc, content_type, converter_cfg)
    validated = validate_examples(examples, validator_cfg)

    if len(validated) == 0:
        raise HTTPException(
            status_code=422,
            detail="Nessun esempio supera i filtri. Rivedi la configurazione "
                   "(es. min_output_chars più basso) o usa un file più grande.",
        )

    extra_meta = {
        "content_type": content_type.value,
        "detection_confidence": detection.confidence,
        "chunker_config": {
            "target_chars": chunker_cfg.target_chars,
            "overlap_chars": chunker_cfg.overlap_chars,
            "min_chunk_chars": chunker_cfg.min_chunk_chars,
            "max_chunk_chars": chunker_cfg.max_chunk_chars,
        },
        "converter_config": {
            "examples_per_narrative_chunk": converter_cfg.examples_per_narrative_chunk,
            "template_language": converter_cfg.template_language,
            "min_chars": converter_cfg.min_chars,
        },
    }

    try:
        row = save_dataset(
            session,
            name=body.name,
            validated=validated,
            source_file=info.filename,
            extra_metadata=extra_meta,
        )
    except DatasetNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Cleanup upload temporaneo
    upload_manager.delete(upload_id)

    return SaveDatasetResponseSchema(dataset=_row_to_schema(row))


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[DatasetSchema],
    summary="Lista dei dataset salvati",
)
def read_datasets(
    session: Session = Depends(get_session),
) -> list[DatasetSchema]:
    return [_row_to_schema(r, parse_stats=False) for r in list_datasets(session)]


@router.get(
    "/{dataset_id}",
    response_model=DatasetSchema,
    summary="Dettaglio di un dataset",
)
def read_dataset(
    dataset_id: int,
    session: Session = Depends(get_session),
) -> DatasetSchema:
    try:
        row = get_dataset_by_id(session, dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _row_to_schema(row)


@router.get(
    "/{dataset_id}/examples",
    response_model=list[InstructionExampleSchema],
    summary="Esempi di un dataset (con limit)",
)
def read_dataset_examples(
    dataset_id: int,
    limit: int = Query(default=10, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[InstructionExampleSchema]:
    try:
        examples = load_dataset_examples(session, dataset_id, limit=limit)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Le entry da JSONL non hanno metadata, normalizziamo a dict vuoto
    return [
        InstructionExampleSchema(
            instruction=e.get("instruction", ""),
            input=e.get("input", ""),
            output=e.get("output", ""),
            metadata=e.get("metadata", {}),
        )
        for e in examples
    ]


@router.delete(
    "/{dataset_id}",
    summary="Cancella un dataset",
)
def delete_dataset_endpoint(
    dataset_id: int,
    remove_files: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> dict:
    try:
        delete_dataset(session, dataset_id, remove_files=remove_files)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "id": dataset_id}