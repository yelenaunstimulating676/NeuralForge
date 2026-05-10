"""
Model loader del Training Engine.

Carica un base model HuggingFace con quantizzazione 4-bit (bitsandbytes)
e applica l'adapter LoRA via PEFT, restituendo un PeftModel pronto per
il training.

Backend di quantizzazione: bitsandbytes (4-bit NF4 + bf16 compute).
Backend di adapter: PEFT (LoraConfig + get_peft_model).

Per la determinazione dei `target_modules` LoRA usiamo una mappa
hardcoded per le famiglie di modelli supportati nella whitelist M2.
Per modelli custom, fallback ad auto-detection scansionando tutti i
moduli `nn.Linear`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------


class ModelLoadError(Exception):
    """Errore durante caricamento o adapter del modello."""


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraConfigParams:
    """Parametri LoRA passati a peft.LoraConfig."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    target_modules: list[str] | None = None  # None → auto-detect

    def __post_init__(self) -> None:
        if self.r < 1 or self.r > 256:
            raise ValueError("rank deve essere tra 1 e 256")
        if self.alpha < 1:
            raise ValueError("alpha deve essere ≥ 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve essere in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias deve essere 'none' | 'all' | 'lora_only'")


@dataclass(frozen=True)
class QuantizationConfig:
    """Parametri della quantizzazione 4-bit (bitsandbytes)."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" o "fp4"
    bnb_4bit_use_double_quant: bool = True
    # Compute dtype: bf16 su Ampere+, fp16 su Turing/Pascal
    bnb_4bit_compute_dtype: str = "bfloat16"  # "bfloat16" | "float16"

    def __post_init__(self) -> None:
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("bnb_4bit_quant_type deve essere 'nf4' o 'fp4'")
        if self.bnb_4bit_compute_dtype not in {"bfloat16", "float16"}:
            raise ValueError(
                "bnb_4bit_compute_dtype deve essere 'bfloat16' o 'float16'"
            )


# ---------------------------------------------------------------------------
# Mappatura target_modules per famiglia
# ---------------------------------------------------------------------------


# Per ogni famiglia, i moduli LINEAR su cui applicare LoRA. Lista basata
# sul paper QLoRA (Dettmers et al.) + best practice della community.
# Solo i proj dell'attention (q,k,v,o), no MLP/lm_head per minimizzare
# parametri trainable mantenendo qualità.
TARGET_MODULES_BY_FAMILY: dict[str, list[str]] = {
    "qwen2.5": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi3.5": ["qkv_proj", "o_proj"],
    "phi2": ["q_proj", "k_proj", "v_proj", "dense"],
    "smollm2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "smollm3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "tinyllama": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_target_modules_for_family(tag: str | None) -> list[str] | None:
    """
    Ritorna i target_modules LoRA per la famiglia data, o None se ignota.
    """
    if not tag:
        return None
    return TARGET_MODULES_BY_FAMILY.get(tag.lower())


def find_all_linear_names(model) -> list[str]:
    """
    Auto-detection dei moduli `nn.Linear` adatti per LoRA.

    Strategia conservativa:
      - Esclude `lm_head` (output projection — adapttarlo causa overfitting)
      - Esclude moduli di nome contenente "embed" (embeddings)
      - Esclude moduli `Linear8bitLt` o `Linear4bit` se non sono base
        (sono già quantizzati ma li includiamo: PEFT li gestisce)

    Returns:
        Lista di nomi unici (es. ["q_proj", "k_proj", "v_proj", "o_proj"]).
    """
    import torch.nn as nn

    linear_names: set[str] = set()
    excluded_keywords = {"embed", "lm_head", "embed_tokens"}

    for name, module in model.named_modules():
        # Considera nn.Linear classiche e quelle quantizzate di bnb
        is_linear = isinstance(module, nn.Linear)
        # bnb classes (se installato)
        try:
            import bitsandbytes as bnb  # type: ignore

            is_linear = is_linear or isinstance(
                module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
            )
        except ImportError:
            pass

        if not is_linear:
            continue

        # name è tipo "model.layers.0.self_attn.q_proj"
        # vogliamo solo l'ultima parte ("q_proj")
        leaf_name = name.split(".")[-1]
        if any(k in leaf_name.lower() for k in excluded_keywords):
            continue
        linear_names.add(leaf_name)

    return sorted(linear_names)


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------


def load_tokenizer(model_path: Path | str):
    """
    Carica il tokenizer HF dal path locale.

    Fix automatico: se manca `pad_token_id`, lo settiamo a `eos_token_id`.
    Necessario perché molti modelli (Llama-like, Qwen) non hanno pad
    di default ma il DataCollator ne ha bisogno.

    Returns:
        AutoTokenizer instance.

    Raises:
        ModelLoadError: se il path non esiste o tokenizer non caricabile.
    """
    from transformers import AutoTokenizer

    path = str(model_path)
    if not Path(path).exists():
        raise ModelLoadError(f"Model path non esiste: {path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=False, use_fast=True
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Impossibile caricare il tokenizer da {path}: {exc}"
        ) from exc

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(
                "Tokenizer non aveva pad_token_id, impostato a eos_token_id (%d).",
                tokenizer.eos_token_id,
            )
        else:
            raise ModelLoadError(
                "Tokenizer senza pad_token_id né eos_token_id."
            )

    return tokenizer


# ---------------------------------------------------------------------------
# Model loading + quantizzazione
# ---------------------------------------------------------------------------


def _build_bnb_config(quant: QuantizationConfig):
    """Costruisce BitsAndBytesConfig dai nostri parametri."""
    from transformers import BitsAndBytesConfig

    compute_dtype = (
        torch.bfloat16
        if quant.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_quantized_model(
    model_path: Path | str,
    quant_config: QuantizationConfig | None = None,
):
    """
    Carica il base model con quantizzazione 4-bit applicata.

    Args:
        model_path: path locale al modello scaricato.
        quant_config: parametri quantizzazione, default = QuantizationConfig().

    Returns:
        AutoModelForCausalLM con pesi in 4-bit, già su device CUDA se disponibile.

    Raises:
        ModelLoadError: errori di caricamento o config.
    """
    from transformers import AutoModelForCausalLM

    quant_config = quant_config or QuantizationConfig()
    path = str(model_path)
    if not Path(path).exists():
        raise ModelLoadError(f"Model path non esiste: {path}")

    bnb_config = _build_bnb_config(quant_config)

    logger.info(
        "Caricamento modello 4-bit da %s (quant=%s, compute=%s)…",
        path, quant_config.bnb_4bit_quant_type, quant_config.bnb_4bit_compute_dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto",  # auto-place su GPU se disponibile
            trust_remote_code=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Impossibile caricare il modello da {path}: {exc}"
        ) from exc

    # Disable cache: incompatibile con gradient checkpointing
    model.config.use_cache = False

    logger.info("Modello caricato. Tipo: %s", type(model).__name__)
    return model


# ---------------------------------------------------------------------------
# Applicazione LoRA
# ---------------------------------------------------------------------------


def apply_lora(
    model,
    lora_params: LoraConfigParams,
    family_tag: str | None = None,
):
    """
    Applica LoRA al modello base. Se `target_modules` non è specificato in
    `lora_params`, lo deriva da `family_tag` (whitelist) o da auto-detect.

    Args:
        model: base model (idealmente già quantizzato).
        lora_params: parametri LoRA.
        family_tag: tag della famiglia ("qwen2.5", "phi3.5", ...) per il
            lookup dei target_modules. Se None e `lora_params.target_modules`
            è None, fa auto-detect.

    Returns:
        PeftModel con adapter LoRA applicato e parametri trainable settati.

    Raises:
        ModelLoadError: errori di applicazione LoRA.
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # 1. Determina target_modules
    target = lora_params.target_modules
    if target is None:
        target = get_target_modules_for_family(family_tag)
    if target is None:
        logger.info(
            "target_modules non specificato e famiglia ignota: auto-detect…"
        )
        target = find_all_linear_names(model)
    if not target:
        raise ModelLoadError(
            "Impossibile determinare target_modules per LoRA."
        )

    logger.info("LoRA target_modules: %s", target)

    # 2. Prepara modello per training k-bit
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # 3. Costruisci LoraConfig
    try:
        config = LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.alpha,
            lora_dropout=lora_params.dropout,
            bias=lora_params.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"LoraConfig invalida: {exc}") from exc

    # 4. Applica
    try:
        peft_model = get_peft_model(model, config)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"get_peft_model fallito: {exc}") from exc

    return peft_model


# ---------------------------------------------------------------------------
# Helper: parametri trainable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainableParamsInfo:
    """Statistiche sui parametri trainable di un modello PEFT."""

    trainable_params: int
    total_params: int
    trainable_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_percent": round(self.trainable_percent, 4),
        }


def count_trainable_parameters(model) -> TrainableParamsInfo:
    """
    Conta i parametri trainable vs totali del modello.

    Per un modello QLoRA tipico ci aspettiamo:
      - trainable: solo i pesi LoRA (~0.1% – 1% del totale)
      - total: tutto incluso il base model quantizzato
    """
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    pct = (trainable / total * 100) if total > 0 else 0.0
    return TrainableParamsInfo(
        trainable_params=trainable,
        total_params=total,
        trainable_percent=pct,
    )


# ---------------------------------------------------------------------------
# Top-level helper: carica tutto in un colpo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """Bundle restituito da `prepare_model_for_training`."""

    model: Any            # PeftModel
    tokenizer: Any        # AutoTokenizer
    trainable_info: TrainableParamsInfo


def prepare_model_for_training(
    model_path: Path | str,
    family_tag: str | None = None,
    quant_config: QuantizationConfig | None = None,
    lora_params: LoraConfigParams | None = None,
) -> LoadedModel:
    """
    Pipeline completa: tokenizer + base model 4-bit + LoRA adapter.

    Args:
        model_path: path locale al modello.
        family_tag: tag famiglia (per target_modules). Se None, auto-detect.
        quant_config: parametri quantizzazione (default ragionevoli).
        lora_params: parametri LoRA (default r=16, alpha=32).

    Returns:
        LoadedModel con model + tokenizer + info parametri.
    """
    lora_params = lora_params or LoraConfigParams()

    tokenizer = load_tokenizer(model_path)
    base_model = load_quantized_model(model_path, quant_config)
    peft_model = apply_lora(base_model, lora_params, family_tag)

    info = count_trainable_parameters(peft_model)
    logger.info(
        "Modello pronto: %d / %d parametri trainable (%.4f%%)",
        info.trainable_params, info.total_params, info.trainable_percent,
    )

    return LoadedModel(
        model=peft_model,
        tokenizer=tokenizer,
        trainable_info=info,
    )
    
    
    # ---------------------------------------------------------------------------
# Inference variant
# ---------------------------------------------------------------------------


def apply_lora_for_inference(model, adapter_path: Path | str):
    """
    Carica un adapter LoRA su un base model in modalità inference.

    Args:
        model: base model già caricato (idealmente quantizzato).
        adapter_path: path locale alla cartella che contiene
            adapter_model.safetensors + adapter_config.json.

    Returns:
        PeftModel con l'adapter caricato e impostato in eval mode.

    Raises:
        ModelLoadError: errori di caricamento adapter.
    """
    from peft import PeftModel

    path = str(adapter_path)
    if not Path(path).exists():
        raise ModelLoadError(f"Adapter path non esiste: {path}")

    try:
        peft_model = PeftModel.from_pretrained(model, path)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Impossibile caricare adapter LoRA da {path}: {exc}"
        ) from exc

    peft_model.eval()
    logger.info("Adapter LoRA caricato per inference da %s", path)
    return peft_model


def prepare_base_for_inference(
    model_path: Path | str,
    quant_config: QuantizationConfig | None = None,
) -> LoadedModel:
    """
    Carica solo il base model + tokenizer per inference.

    A differenza di prepare_model_for_training:
      - NON applica `prepare_model_for_kbit_training` (no gradient setup)
      - NON applica LoRA
      - Mette il modello in `eval()` mode

    Returns:
        LoadedModel con `trainable_info` riempito a 0 (no LoRA).
    """
    tokenizer = load_tokenizer(model_path)
    base_model = load_quantized_model(model_path, quant_config)
    base_model.eval()

    info = TrainableParamsInfo(
        trainable_params=0,
        total_params=sum(p.numel() for p in base_model.parameters()),
        trainable_percent=0.0,
    )
    return LoadedModel(model=base_model, tokenizer=tokenizer, trainable_info=info)


def prepare_ft_for_inference(
    base_model_path: Path | str,
    adapter_path: Path | str,
    quant_config: QuantizationConfig | None = None,
) -> LoadedModel:
    """
    Carica base + adapter LoRA per inference.
    """
    tokenizer = load_tokenizer(base_model_path)
    base_model = load_quantized_model(base_model_path, quant_config)
    peft_model = apply_lora_for_inference(base_model, adapter_path)

    info = TrainableParamsInfo(
        trainable_params=0,
        total_params=sum(p.numel() for p in peft_model.parameters()),
        trainable_percent=0.0,
    )
    return LoadedModel(model=peft_model, tokenizer=tokenizer, trainable_info=info)