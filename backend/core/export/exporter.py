"""
Pipeline di export FT model → GGUF quantizzato.

Workflow:
  1. Merge LoRA adapter + base model → directory safetensors
  2. Convert safetensors → GGUF F16 (via convert_hf_to_gguf.py)
  3. Quantize GGUF F16 → GGUF Q4_K_M (via llama-quantize.exe)
  4. Cleanup intermediate files

Funziona SINCRONO. Va chiamato in un thread da JobManager.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from core.export.llamacpp_manager import LlamaCppPaths, llamacpp_manager
from core.training.model import load_quantized_model, load_tokenizer

logger = logging.getLogger(__name__)


# Quantization formats accettati da llama-quantize
VALID_QUANTIZATIONS = {
    "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S",
    "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S",
    "Q6_K", "Q8_0",
    "F16", "F32",
}

DEFAULT_QUANTIZATION = "Q4_K_M"


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class ExportError(Exception):
    """Errore durante export GGUF."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    """Risultato di un export completato."""

    output_path: Path
    quantization: str
    size_bytes: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_quantization(quant: str) -> str:
    """Normalizza e valida il nome della quantizzazione."""
    quant = quant.upper().strip()
    if quant not in VALID_QUANTIZATIONS:
        raise ExportError(
            f"Quantizzazione invalida: {quant!r}. "
            f"Accettati: {sorted(VALID_QUANTIZATIONS)}"
        )
    return quant


def check_disk_space(
    needed_bytes: int, target_dir: Path, safety_factor: float = 1.2
) -> None:
    """
    Verifica spazio libero su disco. Solleva ExportError se insufficiente.

    safety_factor: per export servono 2-3x la size del FT.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target_dir).free
    needed = int(needed_bytes * safety_factor)
    if free < needed:
        raise ExportError(
            f"Spazio insufficiente: servono almeno {needed / 1e6:.0f} MB, "
            f"liberi solo {free / 1e6:.0f} MB in {target_dir}"
        )


# ---------------------------------------------------------------------------
# Steps individuali
# ---------------------------------------------------------------------------


def merge_lora(
    base_model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """
    Merge LoRA adapter su base model e salva su disco.

    Args:
        base_model_path: path al base model (cartella safetensors).
        adapter_path: path all'adapter LoRA (cartella con adapter_model.*).
        output_dir: dove salvare il modello fuso.

    Returns:
        Path della cartella che contiene il modello fuso.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if progress_callback:
        progress_callback("merging_loading_base", 0.0)

    logger.info("Merge LoRA: caricamento base da %s", base_model_path)
    # IMPORTANTE: NO quantization qui. Per il merge serve full precision.
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if progress_callback:
        progress_callback("merging_loading_adapter", 0.3)

    logger.info("Merge LoRA: caricamento adapter da %s", adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    if progress_callback:
        progress_callback("merging_running", 0.5)

    logger.info("Merge LoRA: esecuzione merge_and_unload()")
    merged = peft_model.merge_and_unload()

    if progress_callback:
        progress_callback("merging_saving", 0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    # Salva anche il tokenizer per il convert script
    tokenizer = load_tokenizer(base_model_path)
    tokenizer.save_pretrained(str(output_dir))

    # Free memory
    del peft_model
    del merged
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    if progress_callback:
        progress_callback("merging_done", 1.0)

    logger.info("Merge completato in %s", output_dir)
    return output_dir


def convert_to_gguf(
    merged_dir: Path,
    output_gguf: Path,
    llamacpp_paths: LlamaCppPaths,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """
    Converte un modello safetensors merged in GGUF F16.

    Args:
        merged_dir: cartella prodotta da merge_lora().
        output_gguf: path del file .gguf da creare (F16).
        llamacpp_paths: path verificati al script convert.

    Returns:
        Path del .gguf F16 creato.
    """
    if not llamacpp_paths.convert_script.exists():
        raise ExportError(
            f"Script convert non trovato: {llamacpp_paths.convert_script}"
        )

    if progress_callback:
        progress_callback("converting", 0.0)

    cmd = [
        sys.executable,
        str(llamacpp_paths.convert_script),
        str(merged_dir),
        "--outfile", str(output_gguf),
        "--outtype", "f16",
    ]

    logger.info("Convert GGUF: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
    except subprocess.TimeoutExpired as exc:
        raise ExportError(f"Conversione GGUF timeout dopo 10 min") from exc

    if result.returncode != 0:
        logger.error("Convert stderr:\n%s", result.stderr[-2000:])
        raise ExportError(
            f"Conversione GGUF fallita (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )

    if not output_gguf.exists():
        raise ExportError(
            f"Conversione finita ma output non trovato: {output_gguf}"
        )

    if progress_callback:
        progress_callback("converting", 1.0)

    logger.info(
        "GGUF F16 creato: %s (%.1f MB)",
        output_gguf, output_gguf.stat().st_size / 1e6,
    )
    return output_gguf


def quantize_gguf(
    input_gguf: Path,
    output_gguf: Path,
    quantization: str,
    llamacpp_paths: LlamaCppPaths,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    """
    Quantizza un GGUF F16 nel formato richiesto (es. Q4_K_M).
    """
    quantization = validate_quantization(quantization)

    if not llamacpp_paths.quantize_bin.exists():
        raise ExportError(
            f"Binario quantize non trovato: {llamacpp_paths.quantize_bin}"
        )

    if progress_callback:
        progress_callback("quantizing", 0.0)

    cmd = [
        str(llamacpp_paths.quantize_bin),
        str(input_gguf),
        str(output_gguf),
        quantization,
    ]

    logger.info("Quantize: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min max
        )
    except subprocess.TimeoutExpired as exc:
        raise ExportError("Quantizzazione timeout dopo 15 min") from exc

    if result.returncode != 0:
        logger.error("Quantize stderr:\n%s", result.stderr[-2000:])
        raise ExportError(
            f"Quantizzazione fallita (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )

    if not output_gguf.exists():
        raise ExportError(
            f"Quantize finito ma output non trovato: {output_gguf}"
        )

    if progress_callback:
        progress_callback("quantizing", 1.0)

    logger.info(
        "GGUF %s creato: %s (%.1f MB)",
        quantization, output_gguf, output_gguf.stat().st_size / 1e6,
    )
    return output_gguf


# ---------------------------------------------------------------------------
# Pipeline completa
# ---------------------------------------------------------------------------


def export_ft_to_gguf(
    base_model_path: Path,
    adapter_path: Path,
    output_path: Path,
    quantization: str = DEFAULT_QUANTIZATION,
    workdir: Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> ExportResult:
    """
    Pipeline completa: FT model → GGUF quantizzato.

    Args:
        base_model_path: path al base model.
        adapter_path: path all'adapter LoRA.
        output_path: path finale del .gguf (es. data/exports/foo__Q4_K_M.gguf).
        quantization: formato target (Q4_K_M, Q5_K_M, ...).
        workdir: cartella temporanea per file intermedi. Auto-generata se None.
        progress_callback: callback(stage: str, pct_0_1: float).

    Returns:
        ExportResult con path, size, elapsed.

    Raises:
        ExportError: qualsiasi step fallisce.
    """
    quantization = validate_quantization(quantization)
    start = time.time()

    # 1. Verifica llama.cpp installato
    if progress_callback:
        progress_callback("preparing_llamacpp", 0.0)
    llamacpp_paths = llamacpp_manager.ensure_installed(
        progress_callback=lambda stage, pct: (
            progress_callback(f"llamacpp:{stage}", pct) if progress_callback else None
        )
    )

    # 2. Workdir setup
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workdir is None:
        workdir = output_path.parent / f"_tmp_export_{uuid.uuid4().hex[:8]}"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    merged_dir = workdir / "merged"
    intermediate_gguf = workdir / "model_f16.gguf"

    try:
        # 3. Merge LoRA
        merge_lora(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_dir=merged_dir,
            progress_callback=progress_callback,
        )

        # 4. Convert to GGUF F16
        convert_to_gguf(
            merged_dir=merged_dir,
            output_gguf=intermediate_gguf,
            llamacpp_paths=llamacpp_paths,
            progress_callback=progress_callback,
        )

        # 5. Quantize
        quantize_gguf(
            input_gguf=intermediate_gguf,
            output_gguf=output_path,
            quantization=quantization,
            llamacpp_paths=llamacpp_paths,
            progress_callback=progress_callback,
        )

        elapsed = time.time() - start
        size_bytes = output_path.stat().st_size

        logger.info(
            "Export completato: %s (%.1f MB) in %.1fs",
            output_path, size_bytes / 1e6, elapsed,
        )

        if progress_callback:
            progress_callback("done", 1.0)

        return ExportResult(
            output_path=output_path,
            quantization=quantization,
            size_bytes=size_bytes,
            elapsed_seconds=elapsed,
        )

    finally:
        # Cleanup workdir
        try:
            if workdir.exists():
                shutil.rmtree(workdir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup workdir fallito: %s", exc)