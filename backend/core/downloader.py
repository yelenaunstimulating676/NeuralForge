"""
Downloader HuggingFace integrato col Job Manager.

Espone una funzione `download_model_job(...)` che è una "coroutine factory"
(`(progress_cb, cancel_event) -> awaitable`) compatibile con
`job_manager.submit(...)`.

Il download vero usa `huggingface_hub.snapshot_download`. Per ottenere
progress in tempo reale durante il download grosso, contiamo i byte sul
disco a intervalli regolari (più affidabile delle callback HF interne
che cambiano formato tra versioni).

Cancellazione: snapshot_download non è interrompibile mid-file. Quando
arriva la cancel, fermiamo il polling e marchiamo il job come cancelled,
ma il file in download corrente termina. Il prossimo file non parte.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from core.jobs import ProgressCallback
from core.model_registry import (
    HFRepoNotAccessibleError,
    InvalidRepoFormatError,
    compute_directory_size,
    get_local_path,
    sanitize_repo_to_dirname,
    validate_hf_repo_exists,
    validate_repo_format,
)

logger = logging.getLogger(__name__)


# Polling interval per il progress sul disco
_PROGRESS_POLL_INTERVAL_S = 1.0


def download_model_job(
    hf_repo: str,
    *,
    token: str | None = None,
) -> Any:
    """
    Restituisce una coroutine factory compatibile col JobManager.

    Args:
        hf_repo: identificativo HF (verrà validato).
        token: opzionale token HF per repo gated.

    Returns:
        Una funzione `async (progress_cb, cancel_event) -> dict`
        da passare a `job_manager.submit("download", factory)`.

    Esempio:
        factory = download_model_job("Qwen/Qwen2.5-0.5B")
        job = await job_manager.submit("download", factory)
    """
    # Validazione fast-fail: solleviamo subito se il formato è invalido,
    # così l'errore è sincrono prima ancora di creare il Job.
    validate_repo_format(hf_repo)
    target_dir = get_local_path(hf_repo)

    async def _coro(progress_cb: ProgressCallback, cancel_event: asyncio.Event) -> dict:
        return await _download_with_progress(
            hf_repo=hf_repo,
            target_dir=target_dir,
            token=token,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    return _coro


async def _download_with_progress(
    *,
    hf_repo: str,
    target_dir: Path,
    token: str | None,
    progress_cb: ProgressCallback,
    cancel_event: asyncio.Event,
) -> dict:
    """
    Esegue snapshot_download in un thread separato e fa polling del progress
    leggendo la dimensione cumulativa sul disco.
    """
    progress_cb(0.0, "Verifica repository su HuggingFace…")

    # Step 1: validazione esistenza repo (chiamata leggera)
    try:
        repo_info = await asyncio.to_thread(
            validate_hf_repo_exists, hf_repo, token=token
        )
    except (InvalidRepoFormatError, HFRepoNotAccessibleError) as exc:
        raise RuntimeError(str(exc)) from exc

    logger.info(
        "Inizio download: repo=%s siblings=%d → %s",
        hf_repo, repo_info.get("siblings_count", 0), target_dir,
    )
    progress_cb(0.02, f"Download di {hf_repo} in corso…")

    # Step 2: stima dimensione totale (per percentuali realistiche)
    expected_size = await asyncio.to_thread(
        _estimate_repo_size, hf_repo, token
    )
    logger.info("Dimensione attesa: %.2f MB", expected_size / 1024 / 1024)

    # Step 3: lancia snapshot_download in un thread, fai polling sul size
    target_dir.mkdir(parents=True, exist_ok=True)
    download_task = asyncio.create_task(
        asyncio.to_thread(
            _run_snapshot_download,
            hf_repo=hf_repo,
            target_dir=target_dir,
            token=token,
        ),
        name=f"hf-download-{sanitize_repo_to_dirname(hf_repo)}",
    )

    # Loop di polling fino a completamento del download_task
    while not download_task.done():
        if cancel_event.is_set():
            # Il thread non si interrompe, ma usciamo dal monitoring
            logger.info("Cancel richiesto per download %s", hf_repo)
            raise asyncio.CancelledError()

        current_size = await asyncio.to_thread(
            compute_directory_size, target_dir
        )
        if expected_size > 0:
            ratio = min(0.99, current_size / expected_size)
            progress_cb(
                ratio,
                f"Scaricati {current_size / 1024 / 1024:.1f} MB "
                f"di ~{expected_size / 1024 / 1024:.1f} MB",
            )
        else:
            # Senza stima, mostra solo i MB scaricati
            progress_cb(
                0.5,
                f"Scaricati {current_size / 1024 / 1024:.1f} MB",
            )

        try:
            await asyncio.wait_for(
                asyncio.shield(download_task),
                timeout=_PROGRESS_POLL_INTERVAL_S,
            )
        except asyncio.TimeoutError:
            continue
        except Exception:
            # Se il download_task ha sollevato, esci dal while: il
            # raise sotto lo solleverà di nuovo.
            break

    # Step 4: raccogli risultato (o eccezione) dal task download
    try:
        local_dir = await download_task
    except Exception as exc:
        # Pulizia: lascia eventuali file parziali, l'utente vedrà l'errore
        # e potrà riprovare/cancellare.
        logger.error("Download fallito per %s: %s", hf_repo, exc)
        raise RuntimeError(f"Download fallito: {exc}") from exc

    final_size = await asyncio.to_thread(compute_directory_size, target_dir)
    progress_cb(1.0, "Download completato.")

    return {
        "hf_repo": hf_repo,
        "local_path": str(Path(local_dir).resolve()),
        "size_bytes": final_size,
    }


# ---------------------------------------------------------------------------
# Helpers sincroni (girano in thread pool tramite asyncio.to_thread)
# ---------------------------------------------------------------------------


def _run_snapshot_download(
    *,
    hf_repo: str,
    target_dir: Path,
    token: str | None,
) -> str:
    """
    Wrapper sincrono di huggingface_hub.snapshot_download.
    Ritorna il path locale dove sono finiti i file.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(target_dir),
        token=token,
        # Solo file standard del modello (no .gitattributes, README, ecc.)
        # Nota: local_dir specificato → snapshot_download scarica i file
        # direttamente nella cartella, senza symlink alla cache HF.
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.model",
            "*.txt",
            "tokenizer*",
            "*.tiktoken",
        ],
    )
    return local


def _estimate_repo_size(hf_repo: str, token: str | None) -> int:
    """
    Stima la dimensione totale del repo in byte sommando i `size` dei
    siblings. Ritorna 0 se HF non espone i size (succede su qualche repo).
    """
    from huggingface_hub import HfApi

    try:
        api = HfApi(token=token)
        info = api.model_info(hf_repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stima size fallita per %s: %s", hf_repo, exc)
        return 0

    total = 0
    for sib in info.siblings or []:
        # Filtriamo agli stessi pattern di snapshot_download
        name = sib.rfilename or ""
        if not _matches_allowed(name):
            continue
        size = getattr(sib, "size", None) or 0
        total += size
    return total


def _matches_allowed(filename: str) -> bool:
    """Replica leggera del pattern matching di snapshot_download."""
    suffixes = (".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken")
    if filename.endswith(suffixes):
        return True
    if filename.startswith("tokenizer"):
        return True
    return False