"""
Data layer del Training Engine.

Trasforma un dataset Alpaca-format (JSONL) in tensori PyTorch pronti
per il forward del modello, applicando:

  1. Chat template specifico del modello (apply_chat_template di HF)
     o fallback Alpaca per base models
  2. Tokenizzazione con max_seq_length
  3. Loss masking: i token dell'instruction sono mascherati a -100
     così il modello impara solo a predire la response
  4. EOS al termine della response

Restituisce: torch.utils.data.Dataset + DataCollator per il DataLoader.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Costante PyTorch standard: token con label = IGNORE_INDEX vengono saltati
# da CrossEntropyLoss. È il meccanismo di loss masking.
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Parametri del data layer."""

    # Lunghezza massima sequenza (in token). Esempi più lunghi → troncati.
    max_seq_length: int = 1024
    # Se True, tokenizziamo "solo response" per il loss masking.
    # Se False, calcoliamo loss su TUTTO (instruction inclusa). Default True.
    train_on_response_only: bool = True
    # Se aggiungere EOS in fondo alla response. Aiuta il modello a fermarsi
    # in inferenza. Default True.
    add_eos: bool = True

    def __post_init__(self) -> None:
        if self.max_seq_length < 64:
            raise ValueError("max_seq_length deve essere ≥ 64")


# ---------------------------------------------------------------------------
# Loading da JSONL
# ---------------------------------------------------------------------------


def load_alpaca_examples(jsonl_path: Path) -> list[dict[str, str]]:
    """
    Carica esempi Alpaca dal file JSONL prodotto dal Dataset Engine (M3).

    Returns:
        Lista di dict con chiavi `instruction`, `input`, `output`.
        I record malformati vengono ignorati con warning.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {jsonl_path}")

    examples: list[dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Riga %d malformata in %s: %s", i, jsonl_path, exc)
                continue

            instr = obj.get("instruction", "")
            inp = obj.get("input", "")
            out = obj.get("output", "")
            if not instr or not out:
                logger.warning(
                    "Riga %d: instruction o output mancanti, salto.", i
                )
                continue
            examples.append({"instruction": instr, "input": inp, "output": out})

    logger.info("Caricati %d esempi da %s", len(examples), jsonl_path)
    return examples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


_ALPACA_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_ALPACA_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt_alpaca(instruction: str, input_: str) -> str:
    """
    Fallback per base models senza chat template: format Alpaca classico.
    Restituisce la parte "instruction" del prompt, FINO a "### Response:\\n"
    incluso. La response del dataset va concatenata dopo.
    """
    if input_:
        return _ALPACA_TEMPLATE_WITH_INPUT.format(
            instruction=instruction, input=input_
        )
    return _ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction)


def format_prompt_chat_template(
    tokenizer, instruction: str, input_: str
) -> str:
    """
    Per modelli istruiti con chat template (Qwen, Phi, SmolLM-Instruct, ecc.):
    usa `tokenizer.apply_chat_template` con `add_generation_prompt=True`,
    che produce il prefisso "fino al turno dell'assistente" (response esclusa).

    Args:
        tokenizer: HF tokenizer del modello base.
        instruction: testo dell'istruzione utente.
        input_: contesto opzionale. Se non vuoto, viene concatenato alla
            instruction nel ruolo "user" (ChatML standard non ha un campo
            "input" separato).

    Returns:
        Stringa testuale del prompt fino al turn assistente.
    """
    user_content = instruction
    if input_:
        user_content = f"{instruction}\n\n{input_}"

    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def has_chat_template(tokenizer) -> bool:
    """True se il tokenizer ha un chat_template configurato."""
    return getattr(tokenizer, "chat_template", None) is not None


# ---------------------------------------------------------------------------
# Tokenizzazione singolo esempio
# ---------------------------------------------------------------------------


def tokenize_example(
    tokenizer,
    example: dict[str, str],
    config: DataConfig,
) -> dict[str, list[int]]:
    """
    Tokenizza un singolo esempio Alpaca producendo input_ids + labels
    con loss masking applicato.

    Args:
        tokenizer: HF tokenizer.
        example: dict con instruction/input/output.
        config: DataConfig.

    Returns:
        Dict con 'input_ids' (list[int]) e 'labels' (list[int]).
        Labels = input_ids ma con i token del prompt mascherati a -100
        (se train_on_response_only=True).
    """
    instruction = example["instruction"]
    input_ = example.get("input", "")
    output = example["output"]

    # Step 1: prompt (parte di cui NON vogliamo calcolare la loss)
    if has_chat_template(tokenizer):
        prompt_text = format_prompt_chat_template(tokenizer, instruction, input_)
    else:
        prompt_text = format_prompt_alpaca(instruction, input_)

    # Step 2: full text (prompt + response + opz. EOS)
    full_text = prompt_text + output
    if config.add_eos and tokenizer.eos_token:
        full_text = full_text + tokenizer.eos_token

    # Step 3: tokenizzazione full text
    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=config.max_seq_length,
        add_special_tokens=False,  # il template/Alpaca include già BOS se serve
    )["input_ids"]

    # Step 4: tokenizzazione prompt-only per sapere dove inizia la response
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=config.max_seq_length,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # Edge case: troncamento ha tagliato anche parte del prompt
    # In tal caso, mascheriamo TUTTO (non c'è nulla di "response" da imparare)
    if prompt_len >= len(full_ids):
        prompt_len = len(full_ids)

    # Step 5: labels = full_ids con i primi prompt_len token mascherati
    labels = list(full_ids)
    if config.train_on_response_only:
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX

    return {"input_ids": full_ids, "labels": labels}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class InstructionTuningDataset(Dataset):
    """
    PyTorch Dataset che carica esempi Alpaca da JSONL e li tokenizza
    on-the-fly al primo accesso (lazy + cache).

    Lazy tokenization: per dataset grossi non vogliamo pre-tokenizzare
    tutto in memoria. Cachiamo solo gli esempi acceduti.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        config: DataConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self.examples = load_alpaca_examples(jsonl_path)
        self._cache: dict[int, dict[str, list[int]]] = {}

        if not self.examples:
            raise ValueError(f"Dataset vuoto o senza esempi validi: {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if idx in self._cache:
            return self._cache[idx]
        tokenized = tokenize_example(
            self.tokenizer, self.examples[idx], self.config
        )
        self._cache[idx] = tokenized
        return tokenized

    def stats(self) -> dict[str, Any]:
        """Statistiche utili per logging pre-training."""
        # Tokenizziamo TUTTO per le statistiche (può essere costoso ma è una tantum)
        lengths: list[int] = []
        for i in range(len(self)):
            lengths.append(len(self[i]["input_ids"]))
        return {
            "num_examples": len(self),
            "tokens_min": min(lengths),
            "tokens_max": max(lengths),
            "tokens_mean": sum(lengths) / len(lengths),
            "tokens_total": sum(lengths),
        }


# ---------------------------------------------------------------------------
# DataCollator (dynamic padding)
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorWithPadding:
    """
    Collator che fa padding dinamico al massimo del batch (non al
    max_seq_length globale). Più efficiente perché non sprechiamo VRAM
    su padding inutile.

    Padding token:
      - input_ids → tokenizer.pad_token_id
      - labels    → IGNORE_INDEX (i token di pad NON contribuiscono alla loss)
      - attention_mask → 0 sui pad, 1 sui token reali
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = None  # opzionale, alcuni hardware preferiscono 8

    def __call__(
        self, batch: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        # Determina max length nel batch
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Fallback: usa eos come pad (tipico per modelli LLaMA-like)
            pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError(
                    "Il tokenizer non ha pad_token_id né eos_token_id, "
                    "impossibile fare padding."
                )

        input_ids_batch: list[list[int]] = []
        labels_batch: list[list[int]] = []
        attention_batch: list[list[int]] = []

        for item in batch:
            ids = item["input_ids"]
            labels = item["labels"]
            n_pad = max_len - len(ids)

            input_ids_batch.append(list(ids) + [pad_id] * n_pad)
            labels_batch.append(list(labels) + [IGNORE_INDEX] * n_pad)
            attention_batch.append([1] * len(ids) + [0] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
        }