"""
Model Registry — gestione modelli base scaricati da HuggingFace.

Responsabilità:
  - Whitelist di modelli supportati (curata, ungated)
  - Sanitizzazione path (es. "Qwen/Qwen2.5-3B" → "Qwen--Qwen2.5-3B")
  - Validazione formato hf_repo
  - Calcolo size on-disk
  - Query CRUD verso DB SQLAlchemy
  - Validazione esistenza repo su HuggingFace (HEAD request)

Il download asincrono vero e proprio è implementato in `core/jobs.py`
(Job Manager) + `core/downloader.py` (helper attorno a snapshot_download).
Questo modulo invece è puro: nessuna network call eccetto quelle
esplicite (validate_hf_repo).

Convenzioni:
  - Le funzioni che parlano col DB accettano una `Session` come primo arg
  - I path sono sempre `pathlib.Path` assoluti
  - Le eccezioni del modulo ereditano da `ModelRegistryError`
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from db.models import BaseModel as BaseModelRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class ModelRegistryError(Exception):
    """Errore base del Model Registry."""


class InvalidRepoFormatError(ModelRegistryError):
    """Repo HF malformato (non rispetta lo schema 'org/name')."""


class ModelNotFoundError(ModelRegistryError):
    """Modello richiesto non presente nel DB."""


class ModelAlreadyExistsError(ModelRegistryError):
    """Tentativo di registrare un modello già presente."""


class HFRepoNotAccessibleError(ModelRegistryError):
    """Repo HF non esiste, gated, o non raggiungibile."""


# ---------------------------------------------------------------------------
# Whitelist modelli supportati
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhitelistEntry:
    """
    Voce della whitelist: un modello base "raccomandato" che NeuralForge
    propone all'utente già preconfigurato.
    """

    hf_repo: str
    display_name: str
    size_gb: float          # dimensione approssimativa in GB
    params_billions: float  # parametri in miliardi
    tag: str                # famiglia (qwen2.5, phi3.5, smollm2, ecc.)
    description: str = ""


WHITELIST: list[WhitelistEntry] = [
    # === Qwen 2.5 (Alibaba, Apache 2.0, ungated) ===
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-0.5B",
        display_name="Qwen 2.5 0.5B",
        size_gb=1.0,
        params_billions=0.5,
        tag="qwen2.5",
        description="Modello molto piccolo, ideale per test rapidi della pipeline.",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen 2.5 0.5B Instruct",
        size_gb=1.0,
        params_billions=0.5,
        tag="qwen2.5",
        description="Variante instruction-tuned del Qwen 0.5B.",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-1.5B",
        display_name="Qwen 2.5 1.5B",
        size_gb=3.1,
        params_billions=1.5,
        tag="qwen2.5",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen 2.5 1.5B Instruct",
        size_gb=3.1,
        params_billions=1.5,
        tag="qwen2.5",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-3B",
        display_name="Qwen 2.5 3B",
        size_gb=6.2,
        params_billions=3.0,
        tag="qwen2.5",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen 2.5 3B Instruct",
        size_gb=6.2,
        params_billions=3.0,
        tag="qwen2.5",
        description="Sweet spot per fine-tuning su RTX 4070 12GB.",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-7B",
        display_name="Qwen 2.5 7B",
        size_gb=15.0,
        params_billions=7.6,
        tag="qwen2.5",
    ),
    WhitelistEntry(
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen 2.5 7B Instruct",
        size_gb=15.0,
        params_billions=7.6,
        tag="qwen2.5",
    ),
    # === Microsoft Phi (MIT, ungated) ===
    WhitelistEntry(
        hf_repo="microsoft/Phi-3.5-mini-instruct",
        display_name="Phi 3.5 Mini Instruct",
        size_gb=7.6,
        params_billions=3.8,
        tag="phi3.5",
        description="Eccellente reasoning per la sua dimensione.",
    ),
    WhitelistEntry(
        hf_repo="microsoft/phi-2",
        display_name="Phi 2",
        size_gb=5.5,
        params_billions=2.7,
        tag="phi2",
    ),
    # === HuggingFace SmolLM2 / SmolLM3 (Apache 2.0, ungated) ===
    WhitelistEntry(
        hf_repo="HuggingFaceTB/SmolLM2-135M",
        display_name="SmolLM2 135M",
        size_gb=0.3,
        params_billions=0.135,
        tag="smollm2",
        description="Velocissimo, perfetto per debug del training loop.",
    ),
    WhitelistEntry(
        hf_repo="HuggingFaceTB/SmolLM2-360M",
        display_name="SmolLM2 360M",
        size_gb=0.7,
        params_billions=0.36,
        tag="smollm2",
    ),
    WhitelistEntry(
        hf_repo="HuggingFaceTB/SmolLM2-1.7B",
        display_name="SmolLM2 1.7B",
        size_gb=3.4,
        params_billions=1.7,
        tag="smollm2",
    ),
    WhitelistEntry(
        hf_repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        display_name="SmolLM2 1.7B Instruct",
        size_gb=3.4,
        params_billions=1.7,
        tag="smollm2",
    ),
    WhitelistEntry(
        hf_repo="HuggingFaceTB/SmolLM3-3B",
        display_name="SmolLM3 3B",
        size_gb=6.0,
        params_billions=3.0,
        tag="smollm3",
        description="Best-in-class 3B nei benchmark recenti.",
    ),
    # === Mistral (Apache 2.0) ===
    WhitelistEntry(
        hf_repo="mistralai/Mistral-7B-v0.3",
        display_name="Mistral 7B v0.3",
        size_gb=14.5,
        params_billions=7.2,
        tag="mistral",
    ),
    WhitelistEntry(
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral 7B Instruct v0.3",
        size_gb=14.5,
        params_billions=7.2,
        tag="mistral",
    ),
    # === TinyLlama (Apache 2.0) ===
    WhitelistEntry(
        hf_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama 1.1B Chat",
        size_gb=2.2,
        params_billions=1.1,
        tag="tinyllama",
        description="Storico modello compatto, ottimo per primi test.",
    ),
]


def get_whitelist() -> list[WhitelistEntry]:
    """Ritorna la whitelist completa (read-only, tuple-like)."""
    return list(WHITELIST)


def find_in_whitelist(hf_repo: str) -> WhitelistEntry | None:
    """Cerca un repo nella whitelist. Case-sensitive (HF repos sono case-sensitive)."""
    for entry in WHITELIST:
        if entry.hf_repo == hf_repo:
            return entry
    return None


# ---------------------------------------------------------------------------
# Validazione e sanitizzazione
# ---------------------------------------------------------------------------


# HuggingFace repo: <org_or_user>/<repo_name>
# Caratteri ammessi nei segmenti: lettere, numeri, "-", "_", "."
_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_repo_format(hf_repo: str) -> str:
    """
    Valida il formato di un repo HuggingFace: 'org/name'.

    Args:
        hf_repo: stringa da validare.

    Returns:
        La stringa originale (se valida).

    Raises:
        InvalidRepoFormatError: se il formato non è valido.
    """
    if not hf_repo or not isinstance(hf_repo, str):
        raise InvalidRepoFormatError("Repo HF vuoto o non stringa.")
    if not _HF_REPO_PATTERN.match(hf_repo):
        raise InvalidRepoFormatError(
            f"Formato repo HF non valido: {hf_repo!r}. "
            "Atteso: 'org/name' (caratteri ammessi: A-Z a-z 0-9 . - _)."
        )
    if len(hf_repo) > 255:
        raise InvalidRepoFormatError("Repo HF troppo lungo (max 255 caratteri).")
    return hf_repo


def sanitize_repo_to_dirname(hf_repo: str) -> str:
    """
    Trasforma 'Qwen/Qwen2.5-3B' → 'Qwen--Qwen2.5-3B' (nome cartella safe).

    Usiamo '--' come separatore (convenzione HuggingFace cache).
    """
    validate_repo_format(hf_repo)
    return hf_repo.replace("/", "--")


def get_local_path(hf_repo: str) -> Path:
    """
    Calcola il path locale dove un modello vivrebbe (esista o meno).
    Sempre dentro `settings.models_path`.
    """
    return settings.models_path / sanitize_repo_to_dirname(hf_repo)


# ---------------------------------------------------------------------------
# Disk size helpers
# ---------------------------------------------------------------------------


def compute_directory_size(path: Path) -> int:
    """
    Calcola la dimensione totale di una directory ricorsivamente, in byte.
    Ritorna 0 se la directory non esiste.

    Note:
        Segue i symlink? No, per evitare loop o di contare cache HF
        condivisa due volte.
    """
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            try:
                total += entry.stat().st_size
            except OSError as exc:
                logger.warning("Impossibile leggere size di %s: %s", entry, exc)
    return total


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------


def list_local_models(session: Session) -> list[BaseModelRow]:
    """
    Lista tutti i modelli base registrati nel DB, ordinati per data download
    (più recenti prima).
    """
    stmt = select(BaseModelRow).order_by(BaseModelRow.downloaded_at.desc())
    return list(session.scalars(stmt).all())


def get_model_by_id(session: Session, model_id: int) -> BaseModelRow:
    """
    Ritorna il modello con l'ID dato.

    Raises:
        ModelNotFoundError: se non esiste.
    """
    model = session.get(BaseModelRow, model_id)
    if model is None:
        raise ModelNotFoundError(f"Modello con id={model_id} non trovato.")
    return model


def get_model_by_repo(session: Session, hf_repo: str) -> BaseModelRow | None:
    """Ritorna il modello con il dato hf_repo, o None se non esiste."""
    stmt = select(BaseModelRow).where(BaseModelRow.hf_repo == hf_repo)
    return session.scalars(stmt).first()


def is_model_present(session: Session, hf_repo: str) -> bool:
    """True se il modello è già nel DB."""
    return get_model_by_repo(session, hf_repo) is not None


def register_model(
    session: Session,
    *,
    hf_repo: str,
    local_path: Path,
    display_name: str | None = None,
    tag: str | None = None,
    params_billions: float | None = None,
    is_custom: bool = False,
) -> BaseModelRow:
    """
    Inserisce un modello nel DB dopo che è stato scaricato sul disco.

    Args:
        session: SQLAlchemy session.
        hf_repo: identificativo HuggingFace (validato).
        local_path: directory dove sta il modello (deve esistere).
        display_name: nome mostrato in UI. Se None, derivato da whitelist o hf_repo.
        tag: famiglia (autopopolato dalla whitelist se non fornito).
        params_billions: parametri in miliardi (autopopolato dalla whitelist).
        is_custom: True se viene da repo custom (non whitelist).

    Returns:
        Il record BaseModel inserito.

    Raises:
        InvalidRepoFormatError: se hf_repo malformato.
        ModelAlreadyExistsError: se già presente nel DB.
    """
    validate_repo_format(hf_repo)
    if is_model_present(session, hf_repo):
        raise ModelAlreadyExistsError(f"Modello {hf_repo!r} già registrato.")

    # Auto-completion da whitelist se possibile
    wl_entry = find_in_whitelist(hf_repo)
    if wl_entry:
        display_name = display_name or wl_entry.display_name
        tag = tag or wl_entry.tag
        params_billions = params_billions or wl_entry.params_billions

    display_name = display_name or hf_repo.split("/")[-1]
    size_bytes = compute_directory_size(local_path)

    row = BaseModelRow(
        hf_repo=hf_repo,
        display_name=display_name,
        tag=tag,
        local_path=str(local_path.resolve()),
        size_bytes=size_bytes,
        params_billions=params_billions,
        is_custom=is_custom,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    logger.info(
        "Modello registrato: id=%d hf_repo=%r size=%.2f GB path=%s",
        row.id, row.hf_repo, size_bytes / 1024 / 1024 / 1024, row.local_path,
    )
    return row


def delete_model(session: Session, model_id: int, *, remove_files: bool = True) -> None:
    """
    Cancella un modello dal DB e (opzionalmente) dal disco.

    Args:
        session: SQLAlchemy session.
        model_id: id del modello.
        remove_files: se True, cancella anche la cartella locale.

    Raises:
        ModelNotFoundError: se non esiste.

    Note:
        Cancellare un BaseModel cascadea su TrainingRun e FineTunedModel
        (vedi `db/models.py`).
    """
    model = get_model_by_id(session, model_id)
    local_path = Path(model.local_path)

    session.delete(model)
    session.commit()

    if remove_files and local_path.exists() and local_path.is_dir():
        # Safety: verifichiamo che il path sia dentro models_path
        try:
            local_path.resolve().relative_to(settings.models_path.resolve())
        except ValueError:
            logger.error(
                "RIFIUTO di cancellare %s: fuori da models_path %s",
                local_path, settings.models_path,
            )
            return
        shutil.rmtree(local_path, ignore_errors=False)
        logger.info("Cartella modello rimossa: %s", local_path)
    else:
        logger.info("Modello %d cancellato dal DB (files preservati).", model_id)


def refresh_size_on_disk(session: Session, model_id: int) -> int:
    """
    Ricomputa size_bytes leggendo dal disco e aggiorna il DB.
    Utile dopo una migrazione o se la cartella è cambiata.

    Returns:
        La nuova dimensione in byte.
    """
    model = get_model_by_id(session, model_id)
    new_size = compute_directory_size(Path(model.local_path))
    model.size_bytes = new_size
    session.commit()
    logger.info(
        "Size aggiornato per modello %d: %.2f GB",
        model_id, new_size / 1024 / 1024 / 1024,
    )
    return new_size


# ---------------------------------------------------------------------------
# HF repo validation (network call)
# ---------------------------------------------------------------------------


def validate_hf_repo_exists(hf_repo: str, *, token: str | None = None) -> dict:
    """
    Verifica che un repo HuggingFace esista e sia accessibile.
    Fa una chiamata HEAD/GET leggera senza scaricare nulla di pesante.

    Args:
        hf_repo: identificativo da verificare.
        token: opzionale, token HF per repo gated.

    Returns:
        Dict con metadata base del repo:
            {"id", "tags", "siblings_count", "gated", "requires_token"}.
        - `gated` (bool): True se il repo richiede l'accettazione di una licenza.
        - `requires_token` (bool): True se servirà un token per scaricare
          (gated E non abbiamo token disponibile).

    Raises:
        InvalidRepoFormatError: se hf_repo malformato.
        HFRepoNotAccessibleError: se non esiste, o c'è un errore network.
    """
    validate_repo_format(hf_repo)

    # Import locale per non rallentare il boot del modulo se HF hub non
    # viene mai usato.
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    api = HfApi(token=token)
    try:
        info = api.model_info(hf_repo)
    except GatedRepoError as exc:
        # Tornati 401/403 → repo gated, ma esiste. Non lanciamo eccezione:
        # informiamo l'utente che serve un token e lasciamo che decida.
        return {
            "id": hf_repo,
            "tags": [],
            "siblings_count": 0,
            "gated": True,
            "requires_token": True,
        }
    except RepositoryNotFoundError as exc:
        raise HFRepoNotAccessibleError(
            f"Repo {hf_repo!r} non trovato su HuggingFace."
        ) from exc
    except HfHubHTTPError as exc:
        raise HFRepoNotAccessibleError(
            f"Errore HF accedendo {hf_repo!r}: {exc}"
        ) from exc

    # Repo accessibile. Ma può comunque essere "gated con auto-approve",
    # nel qual caso `info.gated` è valorizzato. Significa che servirà un
    # token in fase di download anche se la metadata è pubblica.
    gated_flag = getattr(info, "gated", None)
    is_gated = bool(gated_flag) and gated_flag != "false"

    return {
        "id": info.id,
        "tags": list(info.tags or []),
        "siblings_count": len(info.siblings or []),
        "gated": is_gated,
        "requires_token": is_gated and token is None,
    }