"""
System Detector — rilevamento hardware GPU NVIDIA e calcolo della
configurazione ottimale di training per NeuralForge.

Questo modulo è la fonte di verità per:
  - presenza/assenza di GPU NVIDIA compatibili CUDA
  - quantità di VRAM totale, usata e disponibile (in MB)
  - versione driver NVIDIA e versione CUDA runtime
  - configurazione di training suggerita (batch_size, grad_accum,
    max_seq_length, QLoRA vs full fine-tuning, dtype)

Le API REST in `backend/api/system.py` consumano queste funzioni.
Tutte le funzioni sono read-only sull'hardware (nessun side-effect).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# pynvml è opzionale: se non disponibile, fallback su torch.cuda.
try:
    import pynvml  # type: ignore

    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore
    _PYNVML_AVAILABLE = False
    logger.warning(
        "pynvml non disponibile: i dati VRAM saranno meno precisi. "
        "Installa con `pip install pynvml`."
    )

# Stato di inizializzazione NVML gestito a livello applicazione.
# Si inizializza UNA volta nel lifespan di FastAPI (vedi main.py),
# si chiude allo shutdown. Evita init/shutdown ad ogni call.
_nvml_initialized: bool = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUInfo:
    """Informazioni statiche e dinamiche su una singola GPU NVIDIA."""

    index: int
    name: str
    compute_capability: str         # es. "8.9" per RTX 4070
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    driver_version: str | None
    cuda_runtime_version: str | None
    bf16_supported: bool
    fp16_supported: bool


@dataclass(frozen=True)
class SystemInfo:
    """Snapshot completo del sistema rilevante per il training."""

    os: str
    python_version: str
    torch_version: str
    cuda_available: bool
    gpu_count: int
    gpus: list[GPUInfo]


@dataclass(frozen=True)
class TrainingConfigSuggestion:
    """
    Configurazione di training suggerita in base alla VRAM rilevata.
    Pensata per un custom training loop PyTorch con QLoRA + AdamW8bit.
    """

    strategy: str                   # "qlora" | "lora" | "full"
    use_4bit: bool                  # quantizzazione base model 4-bit
    use_8bit_optimizer: bool        # AdamW8bit
    gradient_checkpointing: bool
    mixed_precision_dtype: str      # "bf16" | "fp16" | "fp32"
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    max_seq_length: int
    lora_rank: int
    lora_alpha: int
    notes: list[str]


# ---------------------------------------------------------------------------
# Helpers NVML
# ---------------------------------------------------------------------------


def _bytes_to_mb(value: int) -> int:
    """Converte byte in megabyte (interi, troncati)."""
    return int(value // (1024 * 1024))


def init_nvml() -> bool:
    """
    Inizializza pynvml a livello applicazione. Idempotente.
    Da chiamare UNA volta nel lifespan startup di FastAPI.

    Returns:
        True se NVML è stato inizializzato (o lo era già), False altrimenti.
    """
    global _nvml_initialized
    if _nvml_initialized:
        return True
    if not _PYNVML_AVAILABLE:
        logger.info("pynvml non disponibile: salto init.")
        return False
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        logger.info("NVML inizializzato.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("NVML init fallita: %s", exc)
        return False


def shutdown_nvml() -> None:
    """
    Chiude pynvml. Da chiamare UNA volta nel lifespan shutdown di FastAPI.
    Idempotente: chiamarla più volte è sicuro.
    """
    global _nvml_initialized
    if not _nvml_initialized:
        return
    try:
        pynvml.nvmlShutdown()
        logger.info("NVML chiuso.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("NVML shutdown ignorata: %s", exc)
    finally:
        _nvml_initialized = False


def _is_nvml_ready() -> bool:
    """Predicato interno: NVML disponibile E già inizializzato."""
    return _PYNVML_AVAILABLE and _nvml_initialized


def _get_driver_and_cuda_version() -> tuple[str | None, str | None]:
    """Ritorna (driver_version, cuda_runtime_version)."""
    driver_version: str | None = None
    if _is_nvml_ready():
        try:
            raw = pynvml.nvmlSystemGetDriverVersion()
            driver_version = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lettura driver NVIDIA fallita: %s", exc)

    cuda_runtime = torch.version.cuda
    return driver_version, cuda_runtime


def _read_vram_via_nvml(index: int) -> tuple[int, int, int] | None:
    """Legge (total, used, free) in MB tramite NVML. None se non disponibile."""
    if not _is_nvml_ready():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return _bytes_to_mb(mem.total), _bytes_to_mb(mem.used), _bytes_to_mb(mem.free)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lettura VRAM NVML fallita per GPU %d: %s", index, exc)
        return None


def _read_vram_via_torch(index: int) -> tuple[int, int, int]:
    """Fallback: legge VRAM via torch.cuda (solo memoria allocata dal processo)."""
    props = torch.cuda.get_device_properties(index)
    total_mb = _bytes_to_mb(props.total_memory)
    used_mb = _bytes_to_mb(torch.cuda.memory_allocated(index))
    free_mb = max(total_mb - used_mb, 0)
    return total_mb, used_mb, free_mb


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpus() -> list[GPUInfo]:
    """
    Rileva tutte le GPU NVIDIA visibili a PyTorch.

    Returns:
        Lista di GPUInfo. Vuota se nessuna GPU CUDA è disponibile.
    """
    if not torch.cuda.is_available():
        logger.info("torch.cuda.is_available() = False, nessuna GPU rilevata.")
        return []

    driver_version, cuda_runtime = _get_driver_and_cuda_version()
    gpus: list[GPUInfo] = []

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        cc = f"{props.major}.{props.minor}"

        vram = _read_vram_via_nvml(idx) or _read_vram_via_torch(idx)
        total_mb, used_mb, free_mb = vram

        # bf16 supportato nativamente da Ampere+ (cc >= 8.0). RTX 4070 = 8.9.
        bf16 = (props.major, props.minor) >= (8, 0)
        fp16 = (props.major, props.minor) >= (5, 3)

        gpu = GPUInfo(
            index=idx,
            name=props.name,
            compute_capability=cc,
            vram_total_mb=total_mb,
            vram_used_mb=used_mb,
            vram_free_mb=free_mb,
            driver_version=driver_version,
            cuda_runtime_version=cuda_runtime,
            bf16_supported=bf16,
            fp16_supported=fp16,
        )
        gpus.append(gpu)
        logger.info(
            "GPU %d: %s | VRAM %d/%d MB liberi | cc=%s | bf16=%s",
            idx, gpu.name, gpu.vram_free_mb, gpu.vram_total_mb, cc, bf16,
        )

    return gpus


def get_system_info() -> SystemInfo:
    """Snapshot completo del sistema. Sicuro anche senza GPU."""
    gpus = detect_gpus()
    return SystemInfo(
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_count=len(gpus),
        gpus=gpus,
    )


# ---------------------------------------------------------------------------
# Training config suggestion
# ---------------------------------------------------------------------------


def suggest_training_config(
    gpu: GPUInfo | None = None,
    *,
    target_effective_batch: int = 16,
) -> TrainingConfigSuggestion:
    """
    Calcola una configurazione di training conservativa ma efficace
    in base alla VRAM disponibile sulla GPU.

    Strategia:
        - VRAM < 6 GB   → non supportato
        - 6-8 GB        → QLoRA aggressiva, batch=1, gc on, seq=1024
        - 8-12 GB       → QLoRA, batch=2, seq=2048
        - 12-16 GB      → QLoRA, batch=4, seq=2048 (target RTX 4070 12GB)
        - > 16 GB       → LoRA fp16/bf16, batch=4-8, seq=4096

    Args:
        gpu: GPUInfo della scheda. Se None, usa la prima rilevata.
        target_effective_batch: batch effettivo desiderato (batch * grad_accum).

    Returns:
        TrainingConfigSuggestion pronta per il trainer.

    Raises:
        RuntimeError: se non c'è alcuna GPU o VRAM insufficiente.
    """
    if gpu is None:
        gpus = detect_gpus()
        if not gpus:
            raise RuntimeError(
                "Nessuna GPU NVIDIA rilevata. NeuralForge richiede una GPU CUDA."
            )
        gpu = gpus[0]

    vram_mb = gpu.vram_total_mb
    notes: list[str] = []

    # Mixed precision dtype
    if gpu.bf16_supported:
        dtype = "bf16"
    elif gpu.fp16_supported:
        dtype = "fp16"
        notes.append("bf16 non supportato dalla GPU, uso fp16.")
    else:
        dtype = "fp32"
        notes.append("Né bf16 né fp16 supportati: training in fp32 (lento).")

# Profilo VRAM
    if vram_mb < 6 * 1024:
        raise RuntimeError(
            f"VRAM insufficiente: {vram_mb} MB. Richiesti almeno 6 GB."
        )

    if vram_mb < 7 * 1024:
        # 6-7 GB → consumer entry-level
        strategy = "qlora"
        batch_size = 1
        max_seq_length = 1024
        gradient_checkpointing = True
        lora_rank = 8
        notes.append("VRAM bassa: gradient checkpointing forzato, seq ridotta.")
    elif vram_mb < 11500:
        # 7-11.5 GB → mid range (RTX 3060/3070/3080 8-10GB)
        strategy = "qlora"
        batch_size = 2
        max_seq_length = 2048
        gradient_checkpointing = True
        lora_rank = 16
    elif vram_mb < 15500:
        # 11.5-15.5 GB → 12 GB class (RTX 4070, 3060 12GB, 3080 Ti)
        strategy = "qlora"
        batch_size = 4
        max_seq_length = 2048
        gradient_checkpointing = False
        lora_rank = 16
        notes.append("Configurazione ottimizzata per GPU classe 12 GB.")
    else:
        # ≥ 15.5 GB → high end (RTX 4080/4090, A6000+)
        strategy = "lora"
        batch_size = 4
        max_seq_length = 4096
        gradient_checkpointing = False
        lora_rank = 32
        notes.append("VRAM abbondante: LoRA fp16/bf16 senza quantizzazione.")

    use_4bit = strategy == "qlora"
    grad_accum = max(1, target_effective_batch // batch_size)
    effective_batch = batch_size * grad_accum

    config = TrainingConfigSuggestion(
        strategy=strategy,
        use_4bit=use_4bit,
        use_8bit_optimizer=True,
        gradient_checkpointing=gradient_checkpointing,
        mixed_precision_dtype=dtype,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        effective_batch_size=effective_batch,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        notes=notes,
    )

    logger.info(
        "Config suggerita per %s (%d MB): strategy=%s batch=%d grad_accum=%d seq=%d dtype=%s",
        gpu.name, vram_mb, strategy, batch_size, grad_accum, max_seq_length, dtype,
    )
    return config


# ---------------------------------------------------------------------------
# Serializers per FastAPI
# ---------------------------------------------------------------------------


def system_info_to_dict(info: SystemInfo) -> dict[str, Any]:
    """Converte SystemInfo in dict JSON-serializzabile."""
    return {
        "os": info.os,
        "python_version": info.python_version,
        "torch_version": info.torch_version,
        "cuda_available": info.cuda_available,
        "gpu_count": info.gpu_count,
        "gpus": [asdict(g) for g in info.gpus],
    }


def training_config_to_dict(cfg: TrainingConfigSuggestion) -> dict[str, Any]:
    """Converte TrainingConfigSuggestion in dict JSON-serializzabile."""
    return asdict(cfg)