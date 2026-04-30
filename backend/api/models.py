"""
Endpoint FastAPI per la gestione dei modelli base.

Espone:
    GET    /api/models/whitelist         → modelli raccomandati
    GET    /api/models/base              → modelli scaricati localmente
    GET    /api/models/base/{id}         → dettaglio modello
    DELETE /api/models/base/{id}         → cancella (DB + opz. file)
    POST   /api/models/base/download     → avvia download (async)
    GET    /api/models/jobs              → lista job
    GET    /api/models/jobs/{job_id}     → status job
    DELETE /api/models/jobs/{job_id}     → cancella job
    POST   /api/models/validate-repo     → verifica esistenza repo HF
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas import (
    BaseModelSchema,
    DeleteResponseSchema,
    DownloadRequestSchema,
    JobCreatedSchema,
    JobSchema,
    ValidateRepoRequestSchema,
    ValidateRepoResponseSchema,
    WhitelistEntrySchema,
)
from core.downloader import download_model_job
from core.jobs import job_manager
from core.model_registry import (
    HFRepoNotAccessibleError,
    InvalidRepoFormatError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    delete_model,
    get_local_path,
    get_model_by_id,
    get_model_by_repo,
    get_whitelist,
    list_local_models,
    register_model,
    validate_hf_repo_exists,
    validate_repo_format,
)
from db import get_session
from db.models import BaseModel as BaseModelRow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


# ---------------------------------------------------------------------------
# Helpers di serializzazione
# ---------------------------------------------------------------------------


def _row_to_schema(row: BaseModelRow) -> BaseModelSchema:
    """Converte un BaseModel ORM in BaseModelSchema."""
    return BaseModelSchema(
        id=row.id,
        hf_repo=row.hf_repo,
        display_name=row.display_name,
        tag=row.tag,
        local_path=row.local_path,
        size_bytes=row.size_bytes,
        params_billions=row.params_billions,
        is_custom=row.is_custom,
        downloaded_at=row.downloaded_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------


@router.get(
    "/whitelist",
    response_model=list[WhitelistEntrySchema],
    summary="Modelli raccomandati (whitelist curata, ungated)",
)
def read_whitelist() -> list[WhitelistEntrySchema]:
    """
    Lista dei modelli base raccomandati. Tutti Apache 2.0 / MIT, scaricabili
    senza login HuggingFace.
    """
    return [WhitelistEntrySchema(**asdict(e)) for e in get_whitelist()]


# ---------------------------------------------------------------------------
# Base models (CRUD locale)
# ---------------------------------------------------------------------------


@router.get(
    "/base",
    response_model=list[BaseModelSchema],
    summary="Modelli base scaricati localmente",
)
def read_base_models(
    session: Session = Depends(get_session),
) -> list[BaseModelSchema]:
    """Lista dei modelli base scaricati, ordinati per data download (recenti prima)."""
    return [_row_to_schema(r) for r in list_local_models(session)]


@router.get(
    "/base/{model_id}",
    response_model=BaseModelSchema,
    summary="Dettaglio modello base",
)
def read_base_model(
    model_id: int,
    session: Session = Depends(get_session),
) -> BaseModelSchema:
    """Ritorna il singolo modello con l'ID dato. 404 se non esiste."""
    try:
        row = get_model_by_id(session, model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _row_to_schema(row)


@router.delete(
    "/base/{model_id}",
    response_model=DeleteResponseSchema,
    summary="Cancella modello base",
)
def delete_base_model(
    model_id: int,
    remove_files: bool = Query(
        default=True,
        description="Se True, rimuove anche la cartella su disco (default: True).",
    ),
    session: Session = Depends(get_session),
) -> DeleteResponseSchema:
    """
    Cancella un modello base dal DB e (opzionale) dal disco.
    Cancellerà a cascata anche eventuali TrainingRun e FineTunedModel.
    """
    try:
        delete_model(session, model_id, remove_files=remove_files)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DeleteResponseSchema(
        deleted=True,
        id=model_id,
        message=f"Modello {model_id} cancellato (files rimossi: {remove_files}).",
    )


# ---------------------------------------------------------------------------
# Download asincrono
# ---------------------------------------------------------------------------


@router.post(
    "/base/download",
    response_model=JobCreatedSchema,
    summary="Avvia il download asincrono di un modello da HuggingFace",
)
async def start_download(
    body: DownloadRequestSchema,
    session: Session = Depends(get_session),
) -> JobCreatedSchema:
    """
    Avvia il download di un modello HF. L'operazione è asincrona:
    il backend ritorna subito il `job_id`, il frontend deve fare polling
    su `GET /api/models/jobs/{job_id}` per il progress.

    Comportamento:
      - Valida il formato `hf_repo`
      - Rifiuta se il modello è già nel DB (idempotenza)
      - Crea un Job nel JobManager
      - Quando il Job completa con successo, registra il modello in DB
        (la registrazione avviene dentro la coroutine via callback).
    """
    # Validazione fast-fail
    try:
        validate_repo_format(body.hf_repo)
    except InvalidRepoFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Idempotenza: rifiuta se già registrato
    if get_model_by_repo(session, body.hf_repo) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Modello {body.hf_repo!r} già scaricato.",
        )

    # Costruiamo la factory che il JobManager userà.
    # Wrappamo la factory di download per registrare il modello su DB
    # alla fine, IN-PROCESS (stessa app, stessa SessionLocal).
    base_factory = download_model_job(body.hf_repo, token=body.token)

    async def factory_with_registration(progress_cb, cancel_event):
        """Esegue il download e poi registra il modello nel DB."""
        result = await base_factory(progress_cb, cancel_event)

        # Registrazione DB: usiamo una nuova session perché la `session`
        # del request handler è già chiusa quando il job termina.
        from db import SessionLocal
        from pathlib import Path

        with SessionLocal() as new_session:
            try:
                row = register_model(
                    new_session,
                    hf_repo=body.hf_repo,
                    local_path=Path(result["local_path"]),
                    is_custom=False,  # se passa per qui, è whitelist o custom?
                    # Nota: il flag is_custom non è attualmente passato
                    # dalla request body. Lo dedurremo: se il repo NON è
                    # in whitelist, è custom.
                )
                # Aggiorniamo il flag dopo: refresh per leggere dalla riga.
                from core.model_registry import find_in_whitelist
                if find_in_whitelist(body.hf_repo) is None:
                    row.is_custom = True
                    new_session.commit()
                    new_session.refresh(row)
                result["model_id"] = row.id
            except ModelAlreadyExistsError:
                # Race: qualcun altro l'ha registrato nel frattempo.
                existing = get_model_by_repo(new_session, body.hf_repo)
                result["model_id"] = existing.id if existing else None

        return result

    job = await job_manager.submit("download", factory_with_registration)
    logger.info(
        "Avviato job di download: id=%s repo=%s target=%s",
        job.id, body.hf_repo, get_local_path(body.hf_repo),
    )
    return JobCreatedSchema(job_id=job.id, status=job.status.value)


# ---------------------------------------------------------------------------
# Jobs (status / cancel / list)
# ---------------------------------------------------------------------------


@router.get(
    "/jobs",
    response_model=list[JobSchema],
    summary="Lista job asincroni",
)
async def list_jobs(
    kind: str | None = Query(default=None, description="Filtra per kind (es. 'download')"),
) -> list[JobSchema]:
    """Lista dei job, opzionalmente filtrata per kind."""
    jobs = await job_manager.list(kind=kind)
    return [JobSchema(**j.to_dict()) for j in jobs]


@router.get(
    "/jobs/{job_id}",
    response_model=JobSchema,
    summary="Status di un job",
)
async def get_job(job_id: str) -> JobSchema:
    """Ritorna lo stato corrente di un job. 404 se non esiste."""
    job = await job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato.")
    return JobSchema(**job.to_dict())


@router.delete(
    "/jobs/{job_id}",
    response_model=DeleteResponseSchema,
    summary="Cancella un job in corso",
)
async def cancel_job(job_id: str) -> DeleteResponseSchema:
    """
    Richiede la cancellazione di un job. La cancellazione è cooperativa:
    può richiedere qualche secondo prima che lo stato diventi 'cancelled'.
    """
    cancelled = await job_manager.cancel(job_id)
    if not cancelled:
        # Job non esiste o è già terminale
        job = await job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato.")
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} è già in stato '{job.status.value}'.",
        )
    return DeleteResponseSchema(
        deleted=True,
        message=f"Cancellazione richiesta per job {job_id}.",
    )


# ---------------------------------------------------------------------------
# Repo validation (per la UI custom repo)
# ---------------------------------------------------------------------------


@router.post(
    "/validate-repo",
    response_model=ValidateRepoResponseSchema,
    summary="Verifica esistenza e accessibilità di un repo HF",
)
def validate_repo(body: ValidateRepoRequestSchema) -> ValidateRepoResponseSchema:
    """
    Chiama HF API per verificare che un repo esista e sia accessibile.
    Distingue tra:
      - non esiste / malformato → accessible=False con messaggio
      - gated (serve token) → accessible=True, gated=True, requires_token=True
      - ok pubblico → accessible=True, gated=False
    """
    try:
        info = validate_hf_repo_exists(body.hf_repo, token=body.token)
    except InvalidRepoFormatError as exc:
        return ValidateRepoResponseSchema(
            hf_repo=body.hf_repo,
            accessible=False,
            message=str(exc),
        )
    except HFRepoNotAccessibleError as exc:
        return ValidateRepoResponseSchema(
            hf_repo=body.hf_repo,
            accessible=False,
            message=str(exc),
        )

    # Repo è "esistente". Costruiamo un message coerente con i flag.
    requires_token = info.get("requires_token", False)
    gated = info.get("gated", False)

    if requires_token:
        message = (
            "Questo repo è gated e richiede un token HuggingFace. "
            "Genera un token su huggingface.co/settings/tokens, accetta "
            "la licenza sulla pagina del modello, poi inserisci il token "
            "qui sotto."
        )
    elif gated:
        message = "Repo accessibile (con token fornito)."
    else:
        message = None

    return ValidateRepoResponseSchema(
        hf_repo=info["id"],
        accessible=True,
        tags=info["tags"],
        siblings_count=info["siblings_count"],
        gated=gated,
        requires_token=requires_token,
        message=message,
    )