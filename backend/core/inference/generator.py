"""
Funzioni di generazione testo. Wrappa model.generate() con:
  - Chat template auto-applicato se disponibile
  - Sampling parameters tipizzati
  - Misurazione tempo + throughput
  - Esecuzione thread-friendly (sync, da chiamare in to_thread)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationParams:
    """Parametri di sampling per model.generate()."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens deve essere tra 1 e 4096")
        if not 0.0 < self.temperature <= 5.0:
            raise ValueError("temperature deve essere in (0, 5]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p deve essere in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k deve essere >= 0")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty deve essere >= 1.0")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    """Risultato di una generazione."""

    text: str                       # solo i token generati (no prompt)
    formatted_prompt: str           # prompt dopo chat_template
    tokens_generated: int
    elapsed_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str              # 'eos' | 'length' | 'unknown'

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
            "finish_reason": self.finish_reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _format_prompt(tokenizer, user_prompt: str) -> str:
    """
    Applica il chat template se disponibile, altrimenti ritorna il prompt
    grezzo (l'utente può scriverlo già in formato Alpaca se vuole).
    """
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def _determine_finish_reason(
    output_ids: torch.Tensor,
    input_length: int,
    eos_token_id: int | None,
    max_new_tokens: int,
) -> str:
    """Capisce perché la generazione si è fermata."""
    generated_length = output_ids.shape[0] - input_length
    if generated_length >= max_new_tokens:
        return "length"
    if eos_token_id is not None and output_ids[-1].item() == eos_token_id:
        return "eos"
    return "unknown"


# ---------------------------------------------------------------------------
# Generation function (sync, thread-friendly)
# ---------------------------------------------------------------------------


def generate_text(
    *,
    model,
    tokenizer,
    prompt: str,
    params: GenerationParams | None = None,
) -> GenerationResult:
    """
    Genera testo da un prompt usando il modello passato.

    Funzione SINCRONA, GPU-heavy. Va invocata in `asyncio.to_thread()`
    per non bloccare l'event loop FastAPI.

    Args:
        model: PeftModel o AutoModelForCausalLM in eval mode.
        tokenizer: tokenizer associato.
        prompt: testo dell'utente.
        params: parametri sampling.

    Returns:
        GenerationResult con testo + metadata.
    """
    params = params or GenerationParams()

    # 1. Format prompt con chat_template
    formatted = _format_prompt(tokenizer, prompt)

    # 2. Tokenize
    inputs = tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )

    # Sposta su device del modello
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    input_length = input_ids.shape[1]

    # 3. Generate
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            do_sample=params.do_sample,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    elapsed = time.time() - start

    # 4. Slice off the prompt
    generated_ids = output[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    tokens_generated = generated_ids.shape[0]
    throughput = tokens_generated / max(0.001, elapsed)

    finish_reason = _determine_finish_reason(
        output[0], input_length, eos_id, params.max_new_tokens
    )

    logger.info(
        "Generato testo: %d tokens in %.2fs (%.1f tok/s) | finish=%s",
        tokens_generated, elapsed, throughput, finish_reason,
    )

    return GenerationResult(
        text=generated_text,
        formatted_prompt=formatted,
        tokens_generated=tokens_generated,
        elapsed_seconds=elapsed,
        throughput_tokens_per_sec=throughput,
        finish_reason=finish_reason,
    )