"""
Gestione upload temporanei in `data/uploads/`.

Gli upload NON sono persisti in DB. Vivono solo in memoria + filesystem,
e vengono cancellati quando il dataset finale viene salvato (oppure al
boot dell'app come cleanup safety).

Identificativi: UUID4. Path: `data/uploads/<upload_id>/<filename>`.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class UploadInfo:
    """Metadata di un upload temporaneo."""

    upload_id: str
    filename: str
    size_bytes: int
    extension: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def file_path(self) -> Path:
        return settings.uploads_path / self.upload_id / self.filename

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "created_at": self.created_at.isoformat(),
        }


class UploadManager:
    """
    Manager in-memory degli upload. Thread-safe via Lock.
    Singleton: usa `upload_manager` dal modulo.
    """

    def __init__(self) -> None:
        self._uploads: dict[str, UploadInfo] = {}
        self._lock = Lock()

    def register(self, filename: str, content: bytes) -> UploadInfo:
        """
        Registra un nuovo upload, salva il file su disco, ritorna l'info.

        Args:
            filename: nome originale del file.
            content: contenuto binario.

        Returns:
            UploadInfo con upload_id univoco.
        """
        upload_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        # Sanitize filename: rimpiazza caratteri rischiosi
        safe_filename = filename.replace("/", "_").replace("\\", "_")

        upload_dir = settings.uploads_path / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        file_path = upload_dir / safe_filename
        file_path.write_bytes(content)

        info = UploadInfo(
            upload_id=upload_id,
            filename=safe_filename,
            size_bytes=len(content),
            extension=ext,
        )
        with self._lock:
            self._uploads[upload_id] = info

        logger.info(
            "Upload registrato: id=%s file=%s size=%d",
            upload_id, safe_filename, len(content),
        )
        return info

    def get(self, upload_id: str) -> UploadInfo | None:
        with self._lock:
            return self._uploads.get(upload_id)

    def delete(self, upload_id: str) -> bool:
        """
        Rimuove un upload (DB + disco).

        Returns:
            True se rimosso, False se non esiste.
        """
        with self._lock:
            info = self._uploads.pop(upload_id, None)
        if info is None:
            return False

        upload_dir = settings.uploads_path / upload_id
        if upload_dir.exists():
            try:
                shutil.rmtree(upload_dir)
            except OSError as exc:
                logger.warning("Errore rimozione %s: %s", upload_dir, exc)

        return True

    def cleanup_all(self) -> int:
        """
        Rimuove TUTTI gli upload e svuota `data/uploads/`. Da chiamare
        al boot dell'app: gli upload non sono persistiti, quindi qualunque
        cosa sia rimasta su disco è "spazzatura" da run precedenti.

        Returns:
            Numero di item cancellati.
        """
        settings.ensure_directories()
        count = 0
        if settings.uploads_path.exists():
            for entry in settings.uploads_path.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    count += 1
                except OSError as exc:
                    logger.warning("Cleanup: errore su %s: %s", entry, exc)
        with self._lock:
            self._uploads.clear()
        if count > 0:
            logger.info("Cleanup uploads al boot: rimossi %d item.", count)
        return count


# Singleton
upload_manager = UploadManager()