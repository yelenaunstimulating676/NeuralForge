"""
Schemi Pydantic per le response delle API NeuralForge.

I modelli core (GPUInfo, SystemInfo, TrainingConfigSuggestion) vivono
in `core/memory.py` come dataclass perché usati anche internamente
senza dipendenza da FastAPI/Pydantic.

Gli schemi qui sono la "view" esposta sul wire: stessi campi, ma con
metadata Pydantic (descrizioni, esempi) per la documentazione OpenAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GPUInfoSchema(BaseModel):
    """Informazioni statiche e dinamiche su una singola GPU NVIDIA."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": "8.9",
                "vram_total_mb": 12282,
                "vram_used_mb": 1700,
                "vram_free_mb": 10582,
                "driver_version": "596.36",
                "cuda_runtime_version": "12.8",
                "bf16_supported": True,
                "fp16_supported": True,
            }
        }
    )

    index: int = Field(description="Indice GPU (0-based) come visto da CUDA.")
    name: str = Field(description="Nome commerciale della GPU.")
    compute_capability: str = Field(
        description="CUDA compute capability (es. '8.9' = Ada Lovelace)."
    )
    vram_total_mb: int = Field(ge=0, description="VRAM totale in megabyte.")
    vram_used_mb: int = Field(ge=0, description="VRAM utilizzata in megabyte.")
    vram_free_mb: int = Field(ge=0, description="VRAM libera in megabyte.")
    driver_version: str | None = Field(
        default=None, description="Versione driver NVIDIA (None se NVML non disponibile)."
    )
    cuda_runtime_version: str | None = Field(
        default=None, description="Versione CUDA runtime (None se CUDA non disponibile)."
    )
    bf16_supported: bool = Field(description="Supporto nativo bfloat16 (Ampere+).")
    fp16_supported: bool = Field(description="Supporto nativo float16 (Maxwell+).")


class SystemInfoSchema(BaseModel):
    """Snapshot completo del sistema rilevante per il training."""

    os: str = Field(description="Sistema operativo + versione.")
    python_version: str = Field(description="Versione Python interprete.")
    torch_version: str = Field(description="Versione PyTorch installata.")
    cuda_available: bool = Field(description="True se torch.cuda.is_available().")
    gpu_count: int = Field(ge=0, description="Numero di GPU NVIDIA rilevate.")
    gpus: list[GPUInfoSchema] = Field(
        default_factory=list, description="Lista delle GPU rilevate."
    )


class VRAMReadingSchema(BaseModel):
    """Lettura puntuale della VRAM. Usata per polling frequente."""

    total_mb: int = Field(ge=0)
    used_mb: int = Field(ge=0)
    free_mb: int = Field(ge=0)


class TrainingConfigSchema(BaseModel):
    """Configurazione di training suggerita in base alla VRAM."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy": "qlora",
                "use_4bit": True,
                "use_8bit_optimizer": True,
                "gradient_checkpointing": False,
                "mixed_precision_dtype": "bf16",
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 16,
                "max_seq_length": 2048,
                "lora_rank": 16,
                "lora_alpha": 32,
                "notes": ["Configurazione ottimizzata per GPU classe 12 GB."],
            }
        }
    )

    strategy: str = Field(description="'qlora' | 'lora' | 'full'.")
    use_4bit: bool = Field(description="Quantizzazione 4-bit del base model.")
    use_8bit_optimizer: bool = Field(description="AdamW8bit per stati optimizer.")
    gradient_checkpointing: bool = Field(
        description="Gradient checkpointing (riduce VRAM, rallenta).",
    )
    mixed_precision_dtype: str = Field(description="'bf16' | 'fp16' | 'fp32'.")
    batch_size: int = Field(ge=1, description="Batch size per step.")
    gradient_accumulation_steps: int = Field(
        ge=1, description="Passi di accumulazione gradient prima dello step optimizer."
    )
    effective_batch_size: int = Field(
        ge=1, description="batch_size * gradient_accumulation_steps."
    )
    max_seq_length: int = Field(ge=128, description="Lunghezza massima sequenza.")
    lora_rank: int = Field(ge=1, description="Rank delle matrici LoRA.")
    lora_alpha: int = Field(ge=1, description="Alpha LoRA (di solito 2 * rank).")
    notes: list[str] = Field(
        default_factory=list, description="Note esplicative sulla configurazione."
    )