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
    
# ===========================================================================
# Models domain (M2)
# ===========================================================================


class WhitelistEntrySchema(BaseModel):
    """Voce della whitelist di modelli supportati."""

    hf_repo: str = Field(description="Identificativo HuggingFace 'org/name'.")
    display_name: str = Field(description="Nome visualizzato in UI.")
    size_gb: float = Field(ge=0, description="Dimensione approssimativa in GB.")
    params_billions: float = Field(ge=0, description="Parametri in miliardi.")
    tag: str = Field(description="Famiglia (qwen2.5, phi3.5, smollm2, ...).")
    description: str = Field(default="", description="Descrizione opzionale.")


class BaseModelSchema(BaseModel):
    """Modello base scaricato e registrato in DB."""

    id: int
    hf_repo: str
    display_name: str
    tag: str | None
    local_path: str
    size_bytes: int = Field(ge=0)
    params_billions: float | None
    is_custom: bool
    downloaded_at: str = Field(description="ISO 8601 timestamp.")


class DownloadRequestSchema(BaseModel):
    """Body POST /api/models/base/download."""

    hf_repo: str = Field(
        description="Repository HuggingFace 'org/name'.",
        examples=["Qwen/Qwen2.5-0.5B"],
    )
    token: str | None = Field(
        default=None,
        description="Token HF opzionale per repo gated.",
    )


class JobSchema(BaseModel):
    """Stato di un Job asincrono."""

    id: str
    kind: str
    status: str = Field(description="pending | running | completed | failed | cancelled.")
    progress: float = Field(ge=0, le=1)
    progress_message: str
    result: dict | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class JobCreatedSchema(BaseModel):
    """Response a POST /api/models/base/download."""

    job_id: str
    status: str


class ValidateRepoRequestSchema(BaseModel):
    """Body POST /api/models/validate-repo."""

    hf_repo: str
    token: str | None = None


class ValidateRepoResponseSchema(BaseModel):
    """Response a POST /api/models/validate-repo."""

    hf_repo: str
    accessible: bool
    tags: list[str] = Field(default_factory=list)
    siblings_count: int = 0
    gated: bool = Field(
        default=False,
        description="True se il repo richiede l'accettazione di una licenza HF.",
    )
    requires_token: bool = Field(
        default=False,
        description="True se serve un token HF per scaricarlo (gated + nessun token fornito).",
    )
    message: str | None = None


class DeleteResponseSchema(BaseModel):
    """Response generica per le DELETE."""

    deleted: bool
    id: int | None = None
    message: str | None = None
    

# ===========================================================================
# Dataset domain (M3)
# ===========================================================================


class UploadResponseSchema(BaseModel):
    """Response a POST /api/dataset/upload."""

    upload_id: str
    filename: str
    size_bytes: int
    extension: str


class SectionPreviewSchema(BaseModel):
    """Preview di una Section (max 5 mostrate)."""

    title: str | None
    text_preview: str = Field(description="Primi 200 chars del testo della sezione.")
    metadata: dict


class ExtractedDocumentSchema(BaseModel):
    """Riassunto di un ExtractedDocument per la UI."""

    source_format: str
    char_count: int
    section_count: int
    metadata: dict
    sections: list[SectionPreviewSchema]


class DetectionResultSchema(BaseModel):
    """Risultato del Content Detector."""

    content_type: str
    confidence: float
    scores: dict[str, float]
    indicators: list[str]


class AnalyzeResponseSchema(BaseModel):
    """Response a POST /api/dataset/upload/{id}/analyze."""

    upload_id: str
    extracted: ExtractedDocumentSchema
    detection: DetectionResultSchema


class ChunkerConfigSchema(BaseModel):
    """Override dei parametri Chunker (tutti opzionali)."""

    target_chars: int | None = Field(default=None, ge=100, le=8192)
    overlap_chars: int | None = Field(default=None, ge=0, le=2048)
    min_chunk_chars: int | None = Field(default=None, ge=50, le=4096)
    max_chunk_chars: int | None = Field(default=None, ge=500, le=16384)


class ConverterConfigSchema(BaseModel):
    """Override dei parametri Converter."""

    examples_per_narrative_chunk: int | None = Field(default=None, ge=1, le=5)
    template_language: str | None = Field(default=None, description="'it' o 'en'")
    min_chars: int | None = Field(default=None, ge=20, le=2048)
    min_output_chars: int | None = Field(default=None, ge=10, le=512)


class ValidatorConfigSchema(BaseModel):
    """Override dei parametri Validator."""

    min_output_chars: int | None = Field(default=None, ge=10, le=512)
    max_output_chars: int | None = Field(default=None, ge=100, le=32768)
    max_total_chars: int | None = Field(default=None, ge=200, le=65536)
    enable_fuzzy_dedup: bool | None = None
    fuzzy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)


class PreviewRequestSchema(BaseModel):
    """Body POST /api/dataset/upload/{id}/preview."""

    content_type_override: str | None = Field(
        default=None,
        description="Forza un ContentType (es. 'narrative'). Default: usa detection.",
    )
    chunker_config: ChunkerConfigSchema = Field(default_factory=ChunkerConfigSchema)
    converter_config: ConverterConfigSchema = Field(default_factory=ConverterConfigSchema)
    max_examples: int = Field(default=10, ge=1, le=50)


class InstructionExampleSchema(BaseModel):
    """Esempio di instruction tuning."""

    instruction: str
    input: str
    output: str
    metadata: dict


class PreviewResponseSchema(BaseModel):
    """Response a POST /api/dataset/upload/{id}/preview."""

    upload_id: str
    content_type: str
    examples: list[InstructionExampleSchema]
    total_chunks: int
    total_examples_estimated: int = Field(
        description="Stima totale esempi se si processasse l'intero documento."
    )


class SaveDatasetRequestSchema(BaseModel):
    """Body POST /api/dataset/upload/{id}/save."""

    name: str = Field(min_length=1, max_length=255)
    content_type_override: str | None = None
    chunker_config: ChunkerConfigSchema = Field(default_factory=ChunkerConfigSchema)
    converter_config: ConverterConfigSchema = Field(default_factory=ConverterConfigSchema)
    validator_config: ValidatorConfigSchema = Field(default_factory=ValidatorConfigSchema)


class DatasetSchema(BaseModel):
    """Dataset salvato."""

    id: int
    name: str
    source_file: str | None
    file_path: str
    num_examples: int
    format: str
    stats: dict | None = Field(default=None, description="stats_json parsato.")
    created_at: str


class SaveDatasetResponseSchema(BaseModel):
    """Response a POST /api/dataset/upload/{id}/save."""

    dataset: DatasetSchema
    
    # ===========================================================================
# Training domain (M5)
# ===========================================================================


class TrainingStartRequestSchema(BaseModel):
    """Configurazione di un training (input lato API)."""

    base_model_id: int
    dataset_id: int

    num_epochs: int = Field(default=3, ge=1, le=100)
    per_device_batch_size: int = Field(default=2, ge=1, le=32)
    grad_accum_steps: int = Field(default=2, ge=1, le=64)
    max_grad_norm: float = Field(default=1.0, gt=0, le=10.0)
    log_every_n_steps: int = Field(default=1, ge=1, le=100)
    max_steps: int = Field(default=0, ge=0, le=100_000)

    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    weight_decay: float = Field(default=0.01, ge=0, le=1.0)
    use_8bit_optimizer: bool = True

    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    min_lr_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    max_seq_length: int = Field(default=1024, ge=64, le=8192)
    train_on_response_only: bool = True

    lora_r: int = Field(default=16, ge=1, le=128)
    lora_alpha: int = Field(default=32, ge=1, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)

    use_4bit: bool = True
    compute_dtype: str = Field(default="bfloat16", description="'bfloat16' o 'float16'")

    save_every_n_steps: int = Field(default=0, ge=0, le=10_000)
    keep_last_n: int = Field(default=3, ge=1, le=20)

    finetuned_name: str | None = Field(default=None, max_length=255)


class TrainingStartResponseSchema(BaseModel):
    """Risposta a POST /api/training/start."""

    run_id: str = Field(description="ID semantico del run, es. train-20260507-a3f2")
    job_id: str = Field(description="ID del job nel JobManager (per status/cancel)")
    training_run_db_id: int


class TrainingRunSchema(BaseModel):
    """Dettaglio di un TrainingRun dal DB."""

    id: int
    run_id: str
    base_model_id: int
    base_model_name: str | None = None
    dataset_id: int | None
    dataset_name: str | None = None
    status: str
    config: dict | None = None
    metrics: dict | None = None
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str


class TrainingEstimateRequestSchema(BaseModel):
    """Body POST /api/training/estimate."""

    base_model_id: int
    dataset_id: int
    num_epochs: int = Field(default=3, ge=1)
    per_device_batch_size: int = Field(default=2, ge=1)
    grad_accum_steps: int = Field(default=2, ge=1)
    max_seq_length: int = Field(default=1024, ge=64)
    lora_r: int = Field(default=16, ge=1)
    use_4bit: bool = True


class TrainingEstimateResponseSchema(BaseModel):
    """Stima euristica pre-training (per UI)."""

    estimated_vram_mb: int
    estimated_time_seconds: int
    total_steps: int
    steps_per_epoch: int
    trainable_params_estimated: int
    notes: list[str] = Field(default_factory=list)


class TrainingJobSchema(BaseModel):
    """Job attivo del JobManager filtrato per kind=training."""

    job_id: str
    run_id: str | None
    training_run_db_id: int | None
    status: str
    progress: float
    progress_message: str
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    
    
    # ===========================================================================
# Inference (M6)
# ===========================================================================


class GenerationParamsSchema(BaseModel):
    """Parametri di sampling per generate."""

    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, gt=0.0, le=5.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = True


class InferenceGenerateRequestSchema(BaseModel):
    """Body POST /api/inference/generate."""
    
    model_config = ConfigDict(protected_namespaces=())

    prompt: str = Field(min_length=1, max_length=10_000)
    # Specifica un solo modello per generazione
    model_kind: str = Field(description="'base' | 'ft'")
    model_id: int
    params: GenerationParamsSchema = Field(default_factory=GenerationParamsSchema)


class InferenceGenerateResponseSchema(BaseModel):
    """Risposta a POST /api/inference/generate."""
    
    model_config = ConfigDict(protected_namespaces=())

    text: str
    tokens_generated: int
    elapsed_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str
    model_key: str
    model_display_name: str


class LoadedModelSchema(BaseModel):
    """Modello attualmente in cache (per GET /models/loaded)."""
    
    model_config = ConfigDict(protected_namespaces=())

    key: str
    kind: str
    model_id: int
    display_name: str
    base_model_id: int
    has_adapter: bool


class AvailableModelSchema(BaseModel):
    """Modello selezionabile per inference (base o ft)."""
    
    model_config = ConfigDict(protected_namespaces=())

    key: str                    # "base:N" o "ft:N"
    kind: str                   # "base" | "ft"
    model_id: int
    display_name: str
    base_model_id: int          # per i ft, il base sottostante
    base_model_name: str | None = None
    is_loaded: bool             # già in cache?
    metadata: dict | None = None  # info extra (loss finale per ft, etc.)
    
    
    # ===========================================================================
# Export (M7)
# ===========================================================================


class ExportStartRequestSchema(BaseModel):
    """Body POST /api/export/start."""

    model_config = ConfigDict(protected_namespaces=())

    ft_model_id: int = Field(description="ID del FineTunedModel da esportare")
    quantization: str = Field(default="Q4_K_M", description="Formato quantizzazione")
    output_name: str | None = Field(
        default=None,
        max_length=128,
        description="Nome file output (senza .gguf). Auto-generato se vuoto.",
    )


class ExportStartResponseSchema(BaseModel):
    """Response a POST /api/export/start."""

    job_id: str
    ft_model_id: int
    quantization: str
    expected_filename: str


class ExportJobSchema(BaseModel):
    """Stato di un export job."""

    job_id: str
    kind: str
    status: str
    progress: float
    message: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict | None = None


class ExportFileSchema(BaseModel):
    """File .gguf su disco."""

    filename: str
    path: str
    size_bytes: int
    quantization: str
    ft_name: str | None = None
    created_at: str


class QuantizationOptionSchema(BaseModel):
    """Opzione di quantizzazione disponibile."""

    value: str           # "Q4_K_M"
    label: str           # "Q4_K_M — bilanciato (consigliato)"
    description: str
    is_default: bool
    
    # ===========================================================================
# Export (M7)
# ===========================================================================


class ExportStartRequestSchema(BaseModel):
    """Body POST /api/export/start."""

    model_config = ConfigDict(protected_namespaces=())

    ft_model_id: int = Field(description="ID del FineTunedModel da esportare")
    quantization: str = Field(default="Q4_K_M", description="Formato quantizzazione")
    output_name: str | None = Field(
        default=None,
        max_length=128,
        description="Nome file output (senza .gguf). Auto-generato se vuoto.",
    )


class ExportStartResponseSchema(BaseModel):
    """Response a POST /api/export/start."""

    job_id: str
    ft_model_id: int
    quantization: str
    expected_filename: str


class ExportJobSchema(BaseModel):
    """Stato di un export job."""

    job_id: str
    kind: str
    status: str
    progress: float
    message: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict | None = None


class ExportFileSchema(BaseModel):
    """File .gguf su disco."""

    filename: str
    path: str
    size_bytes: int
    quantization: str
    ft_name: str | None = None
    created_at: str


class QuantizationOptionSchema(BaseModel):
    """Opzione di quantizzazione disponibile."""

    value: str
    label: str
    description: str
    is_default: bool