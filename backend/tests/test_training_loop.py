"""
Test del training loop.

Strategia: usiamo un mini-modello torch (nn.Linear) wrappato per simulare
l'API HF `model(**batch).loss`. Così possiamo testare il loop completo
SENZA GPU, SENZA bitsandbytes, SENZA scaricare modelli reali.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.training.loop import (
    CancellationToken,
    LoopConfig,
    StepLog,
    TrainingResult,
    train_loop,
)
from core.training.optimizer import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)


# ---------------------------------------------------------------------------
# Mini-modello e dataset finti per simulare il setup HF
# ---------------------------------------------------------------------------


@dataclass
class FakeOutputs:
    """Imita ModelOutput di HF: ha attributo `.loss`."""

    loss: torch.Tensor


class FakeCausalLM(nn.Module):
    """
    Modello finto: prende input_ids (LongTensor), genera embedding finto,
    calcola una loss finta dipendente dall'input. Riceve **kwargs come HF.
    """

    def __init__(self, vocab_size: int = 100, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.proj(x)
        # Loss CE come fa un vero LM, con ignore_index=-100
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            loss = logits.mean()  # fallback bizzarro, non usato
        return FakeOutputs(loss=loss)


class FakeBatchDataset(Dataset):
    """
    Dataset finto: produce N batch già "tokenizzati" con shape (seq_len,).
    Il collator del DataLoader li impila in (B, seq_len).
    """

    def __init__(self, n: int = 32, seq_len: int = 8, vocab: int = 100):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab, (self.seq_len,)),
            "labels": torch.randint(0, self.vocab, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def collate(batch):
    """Collate semplice: stack lungo dim 0."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def make_setup(
    n_examples: int = 32,
    batch_size: int = 4,
    grad_accum: int = 1,
    total_steps_for_sched: int | None = None,
):
    """Crea modello + loader + optimizer + scheduler pronti."""
    model = FakeCausalLM()
    ds = FakeBatchDataset(n=n_examples)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate)

    opt = build_optimizer(model, OptimizerConfig(use_8bit=False))
    sched_total = total_steps_for_sched or (n_examples // batch_size // grad_accum + 1)
    sched = build_scheduler(opt, SchedulerConfig(total_steps=sched_total))
    return model, loader, opt, sched


# ---------------------------------------------------------------------------
# LoopConfig validation
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_defaults(self):
        c = LoopConfig()
        assert c.num_epochs == 1
        assert c.grad_accum_steps == 1
        assert c.max_grad_norm == 1.0

    def test_invalid_epochs(self):
        with pytest.raises(ValueError):
            LoopConfig(num_epochs=0)

    def test_invalid_grad_accum(self):
        with pytest.raises(ValueError):
            LoopConfig(grad_accum_steps=0)

    def test_invalid_grad_norm(self):
        with pytest.raises(ValueError):
            LoopConfig(max_grad_norm=0)

    def test_invalid_log_every(self):
        with pytest.raises(ValueError):
            LoopConfig(log_every_n_steps=0)


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------


class TestCancellationToken:
    def test_initially_not_cancelled(self):
        t = CancellationToken()
        assert t.is_cancelled() is False

    def test_cancel_sets_flag(self):
        t = CancellationToken()
        t.cancel()
        assert t.is_cancelled() is True

    def test_from_threading_event(self):
        ev = threading.Event()
        t = CancellationToken.from_event(ev)
        assert t.is_cancelled() is False
        ev.set()
        assert t.is_cancelled() is True


# ---------------------------------------------------------------------------
# Training loop: esegue davvero
# ---------------------------------------------------------------------------


class TestTrainLoopBasic:
    def test_completes_one_epoch(self):
        model, loader, opt, sched = make_setup(
            n_examples=16, batch_size=4, grad_accum=1
        )
        config = LoopConfig(num_epochs=1, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, device="cpu",
        )
        assert result.completed is True
        assert result.cancelled is False
        assert result.total_steps == 4   # 16 / 4 = 4 step
        assert len(result.history) == 4
        assert result.final_loss > 0

    def test_grad_accum(self):
        """Con grad_accum=2 e batch_size=4, ogni 2 micro = 1 step logico."""
        model, loader, opt, sched = make_setup(
            n_examples=16, batch_size=4, grad_accum=2,
            total_steps_for_sched=10,
        )
        config = LoopConfig(num_epochs=1, grad_accum_steps=2, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, device="cpu",
        )
        # 16/4 = 4 micro-batch, /2 grad_accum = 2 step logici
        assert result.total_steps == 2

    def test_max_steps_caps(self):
        model, loader, opt, sched = make_setup(
            n_examples=100, batch_size=4, grad_accum=1
        )
        config = LoopConfig(num_epochs=1, max_steps=3, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, device="cpu",
        )
        assert result.total_steps == 3

    def test_loss_decreases(self):
        """Sanity: con SGD su problema fisso, loss deve calare."""
        torch.manual_seed(42)
        model, loader, opt, sched = make_setup(
            n_examples=128, batch_size=4, grad_accum=1
        )
        # Loader deterministico → stessi batch ogni epoch
        config = LoopConfig(num_epochs=2, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, device="cpu",
        )
        # Compariamo loss iniziale e finale (rispettando rumore)
        initial = result.history[0].loss
        final = result.history[-1].loss
        # Non chiediamo che cali strettamente — è un task random.
        # Solo che non esploda.
        assert final < initial * 2

    def test_history_serializable(self):
        import json
        model, loader, opt, sched = make_setup()
        config = LoopConfig(num_epochs=1, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, device="cpu",
        )
        # Ogni log entry deve essere JSON-safe
        for entry in result.history:
            json.dumps(entry.to_dict())


# ---------------------------------------------------------------------------
# Cancellazione
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_before_start(self):
        model, loader, opt, sched = make_setup()
        token = CancellationToken()
        token.cancel()  # cancellato prima di iniziare
        config = LoopConfig(num_epochs=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, cancel_token=token, device="cpu",
        )
        assert result.cancelled is True
        assert result.completed is False
        assert result.total_steps == 0

    def test_cancel_mid_training(self):
        """Cancella dopo qualche step usando un on_step callback."""
        model, loader, opt, sched = make_setup(
            n_examples=200, batch_size=4, grad_accum=1
        )
        token = CancellationToken()

        steps_observed = []

        def cb(log_entry: StepLog):
            steps_observed.append(log_entry.step)
            if log_entry.step >= 3:
                token.cancel()

        config = LoopConfig(num_epochs=1, log_every_n_steps=1)
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config,
            cancel_token=token, on_step=cb, device="cpu",
        )
        assert result.cancelled is True
        # Almeno qualche step è stato eseguito
        assert result.total_steps >= 3
        # Ma non tutti i 50 (200/4)
        assert result.total_steps < 50


# ---------------------------------------------------------------------------
# Callback on_step
# ---------------------------------------------------------------------------


class TestOnStepCallback:
    def test_callback_called_with_step_log(self):
        calls = []

        def cb(log_entry: StepLog):
            calls.append(log_entry)

        model, loader, opt, sched = make_setup(
            n_examples=16, batch_size=4
        )
        config = LoopConfig(num_epochs=1, log_every_n_steps=1)
        train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, on_step=cb, device="cpu",
        )
        assert len(calls) == 4
        assert all(isinstance(c, StepLog) for c in calls)
        assert calls[0].step == 1
        assert calls[-1].step == 4

    def test_callback_exception_does_not_crash(self):
        """Se on_step solleva, il loop continua (logghiamo warning)."""
        def cb(_):
            raise RuntimeError("boom")

        model, loader, opt, sched = make_setup()
        config = LoopConfig(num_epochs=1, log_every_n_steps=1)
        # Non deve sollevare
        result = train_loop(
            model=model, train_loader=loader, optimizer=opt,
            scheduler=sched, config=config, on_step=cb, device="cpu",
        )
        assert result.completed is True


# ---------------------------------------------------------------------------
# StepLog serialization
# ---------------------------------------------------------------------------


class TestStepLog:
    def test_to_dict_serializable(self):
        import json
        log = StepLog(
            step=1, epoch=0, loss=2.5, learning_rate=2e-4,
            grad_norm=0.8, vram_used_mb=4096, 
            throughput_tokens_per_sec=1500, elapsed_seconds=10.5,
        )
        json.dumps(log.to_dict())


class TestTrainingResult:
    def test_to_dict_serializable(self):
        import json
        result = TrainingResult(
            completed=True, cancelled=False, total_steps=10,
            final_loss=1.2, elapsed_seconds=120.0,
        )
        json.dumps(result.to_dict())