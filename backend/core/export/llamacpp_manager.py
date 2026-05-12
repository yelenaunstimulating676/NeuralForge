"""
LlamaCppManager: gestisce binari e script di llama.cpp.

Strategia auto-download:
  - Cache permanente in ~/.neuralforge/llamacpp/<version>/
  - Verifica al primo uso, scarica se mancano
  - Espone path per `convert_hf_to_gguf.py` e `llama-quantize.exe`
  - Version pinning per stabilità

I file scaricati:
  - llama-bXXXX-bin-win-cpu-x64.zip (binari, ~40 MB) → llama-quantize.exe
  - convert_hf_to_gguf.py (script Python, ~80 KB) → conversione safetensors→GGUF
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


# Versione pinned di llama.cpp (build number da releases GitHub).
# NB: l'organizzazione GitHub è stata rinominata da "ggerganov" a "ggml-org".
LLAMACPP_VERSION = "b3447"
LLAMACPP_ORG = "ggml-org"

# URL release asset Windows. Usiamo la build AVX2 (CPU x64, leggera ~7MB).
# AVX2 è supportata da tutte le CPU Intel/AMD dal 2013 in poi.
LLAMACPP_BIN_URL = (
    f"https://github.com/{LLAMACPP_ORG}/llama.cpp/releases/download/"
    f"{LLAMACPP_VERSION}/llama-{LLAMACPP_VERSION}-bin-win-avx2-x64.zip"
)

# convert_hf_to_gguf.py: lo prendiamo dal tag stesso della release
LLAMACPP_CONVERT_SCRIPT_URL = (
    f"https://raw.githubusercontent.com/{LLAMACPP_ORG}/llama.cpp/"
    f"{LLAMACPP_VERSION}/convert_hf_to_gguf.py"
)


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class LlamaCppError(Exception):
    """Errore relativo a LlamaCppManager (download fallito, binari mancanti, etc.)."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaCppPaths:
    """Path concreti dei binari/script una volta installati."""

    root_dir: Path                     # ~/.neuralforge/llamacpp/<version>/
    quantize_bin: Path                 # llama-quantize.exe (Win) o llama-quantize (Linux)
    convert_script: Path               # convert_hf_to_gguf.py

    def all_exist(self) -> bool:
        return self.quantize_bin.exists() and self.convert_script.exists()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LlamaCppManager:
    """
    Gestisce installazione e accesso ai binari di llama.cpp.

    Thread-safe: lock globale per evitare download concorrenti.
    """

    def __init__(self, version: str = LLAMACPP_VERSION) -> None:
        self.version = version
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Path resolution
    # ---------------------------------------------------------------------

    @property
    def root_dir(self) -> Path:
        """Directory dove vivono i binari, fuori dal progetto."""
        return Path.home() / ".neuralforge" / "llamacpp" / self.version

    def get_paths(self) -> LlamaCppPaths:
        """Ritorna i path concreti (non verifica esistenza)."""
        is_windows = platform.system() == "Windows"
        quantize_name = "llama-quantize.exe" if is_windows else "llama-quantize"
        return LlamaCppPaths(
            root_dir=self.root_dir,
            quantize_bin=self.root_dir / quantize_name,
            convert_script=self.root_dir / "convert_hf_to_gguf.py",
        )

    def is_installed(self) -> bool:
        """True se tutti i file richiesti sono presenti."""
        return self.get_paths().all_exist()

    # ---------------------------------------------------------------------
    # Install / download
    # ---------------------------------------------------------------------

    def ensure_installed(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> LlamaCppPaths:
        """
        Verifica che llama.cpp sia installato. Se manca qualcosa, scarica.

        Args:
            progress_callback: callback (stage, percent_0_1) per progress UI.

        Returns:
            LlamaCppPaths con tutti i path verificati.

        Raises:
            LlamaCppError: download o estrazione fallita.
        """
        with self._lock:
            paths = self.get_paths()
            if paths.all_exist():
                logger.debug("llama.cpp %s già installato.", self.version)
                return paths

            logger.info("Installazione llama.cpp %s in %s…", self.version, self.root_dir)
            self.root_dir.mkdir(parents=True, exist_ok=True)

            # 1. Scarica binari ZIP se manca llama-quantize
            if not paths.quantize_bin.exists():
                self._download_binaries(progress_callback)

            # 2. Scarica script Python se manca
            if not paths.convert_script.exists():
                self._download_convert_script(progress_callback)

            # 3. Verifica finale
            paths = self.get_paths()
            if not paths.all_exist():
                raise LlamaCppError(
                    f"Installazione llama.cpp fallita: file mancanti dopo download. "
                    f"Verifica {self.root_dir}"
                )

            logger.info("llama.cpp %s installato correttamente.", self.version)
            return paths

    def _download_binaries(
        self,
        progress_callback: Callable[[str, float], None] | None,
    ) -> None:
        """Scarica e estrae il ZIP dei binari Windows."""
        zip_path = self.root_dir / "llamacpp.zip"

        logger.info("Download binari da %s", LLAMACPP_BIN_URL)
        if progress_callback:
            progress_callback("downloading_binaries", 0.0)

        try:
            self._download_with_progress(
                LLAMACPP_BIN_URL,
                zip_path,
                lambda pct: progress_callback("downloading_binaries", pct) if progress_callback else None,
            )
        except Exception as exc:
            raise LlamaCppError(
                f"Download binari fallito da {LLAMACPP_BIN_URL}: {exc}"
            ) from exc

        if progress_callback:
            progress_callback("extracting_binaries", 0.5)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Estraiamo TUTTI i file .exe e .dll (in path "flat", senza
                # subdirectory). llama-quantize.exe ha dipendenze runtime
                # (ggml.dll, llama.dll, ...) che devono stare nella stessa
                # cartella del binario.
                quantize_name = self.get_paths().quantize_bin.name
                quantize_found = False

                for member in zf.namelist():
                    member_path = zipfile.Path(zf, member)
                    if member_path.is_dir():
                        continue
                    name_lower = member.lower()
                    # Estrai .exe e .dll. Ignora altro (sample, README, ...).
                    if not (name_lower.endswith(".exe") or name_lower.endswith(".dll")):
                        continue

                    # Flat extraction: solo il basename, no subdir
                    basename = Path(member).name
                    target = self.root_dir / basename

                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    if basename == quantize_name:
                        quantize_found = True

                if not quantize_found:
                    raise LlamaCppError(
                        f"{quantize_name} non trovato nel ZIP {zip_path}"
                    )
        except zipfile.BadZipFile as exc:
            raise LlamaCppError(f"ZIP corrotto: {exc}") from exc
        finally:
            try:
                zip_path.unlink()
            except FileNotFoundError:
                pass

        # Rendi eseguibile su Linux/macOS
        if platform.system() != "Windows":
            os.chmod(self.get_paths().quantize_bin, 0o755)

        logger.info("Binari estratti.")

    def _download_convert_script(
        self,
        progress_callback: Callable[[str, float], None] | None,
    ) -> None:
        """Scarica convert_hf_to_gguf.py."""
        logger.info("Download script convert_hf_to_gguf.py")
        if progress_callback:
            progress_callback("downloading_script", 0.0)

        target = self.get_paths().convert_script
        try:
            self._download_with_progress(
                LLAMACPP_CONVERT_SCRIPT_URL,
                target,
                lambda pct: progress_callback("downloading_script", pct) if progress_callback else None,
            )
        except Exception as exc:
            raise LlamaCppError(
                f"Download convert_hf_to_gguf.py fallito: {exc}"
            ) from exc

    @staticmethod
    def _download_with_progress(
        url: str,
        target: Path,
        progress_callback: Callable[[float], None] | None,
    ) -> None:
        """Download con progress 0.0 → 1.0."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(target.suffix + ".tmp")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 64 * 1024  # 64 KB

                with open(tmp_target, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and progress_callback:
                            progress_callback(downloaded / total)

            # Rename atomico solo se tutto è andato bene
            tmp_target.replace(target)

            if progress_callback:
                progress_callback(1.0)

        except Exception:
            if tmp_target.exists():
                tmp_target.unlink()
            raise

    # ---------------------------------------------------------------------
    # Uninstall
    # ---------------------------------------------------------------------

    def uninstall(self) -> bool:
        """Rimuove l'installazione locale. Ritorna True se qualcosa è stato rimosso."""
        if not self.root_dir.exists():
            return False
        shutil.rmtree(self.root_dir)
        logger.info("llama.cpp rimosso da %s", self.root_dir)
        return True


# Singleton globale
llamacpp_manager = LlamaCppManager()