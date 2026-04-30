"""
Test unitari per `core/memory.py`.

Copertura:
  - suggest_training_config su VRAM diverse (edge cases delle soglie)
  - VRAM insufficiente → RuntimeError
  - Calcolo grad_accum coerente con target_effective_batch
  - Selezione dtype in base a bf16/fp16 supportati
  - Helper _bytes_to_mb
"""

from __future__ import annotations

import pytest

from core.memory import (
    GPUInfo,
    TrainingConfigSuggestion,
    _bytes_to_mb,
    suggest_training_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gpu(
    *,
    vram_mb: int,
    bf16: bool = True,
    fp16: bool = True,
    name: str = "Mock GPU",
) -> GPUInfo:
    """Costruisce una GPUInfo finta per i test."""
    return GPUInfo(
        index=0,
        name=name,
        compute_capability="8.9",
        vram_total_mb=vram_mb,
        vram_used_mb=0,
        vram_free_mb=vram_mb,
        driver_version="999.99",
        cuda_runtime_version="12.8",
        bf16_supported=bf16,
        fp16_supported=fp16,
    )


# ---------------------------------------------------------------------------
# _bytes_to_mb
# ---------------------------------------------------------------------------


class TestBytesToMb:
    def test_zero(self):
        assert _bytes_to_mb(0) == 0

    def test_one_mb(self):
        assert _bytes_to_mb(1024 * 1024) == 1

    def test_below_one_mb_truncates(self):
        # 1 MB - 1 byte → 0 MB (troncamento intero)
        assert _bytes_to_mb(1024 * 1024 - 1) == 0

    def test_twelve_gb_rtx_4070(self):
        # ~12 GB tipica RTX 4070
        assert _bytes_to_mb(12 * 1024 * 1024 * 1024) == 12 * 1024


# ---------------------------------------------------------------------------
# suggest_training_config — edge cases delle soglie
# ---------------------------------------------------------------------------


class TestSuggestTrainingConfig:
    def test_vram_too_low_raises(self):
        gpu = make_gpu(vram_mb=4 * 1024)  # 4 GB → < 6 GB
        with pytest.raises(RuntimeError, match="VRAM insufficiente"):
            suggest_training_config(gpu)

    def test_low_vram_6_to_7_gb(self):
        """6-7 GB → QLoRA aggressiva, batch=1, gc on."""
        gpu = make_gpu(vram_mb=6 * 1024 + 500)  # ~6.5 GB
        cfg = suggest_training_config(gpu)
        assert cfg.strategy == "qlora"
        assert cfg.batch_size == 1
        assert cfg.max_seq_length == 1024
        assert cfg.gradient_checkpointing is True
        assert cfg.lora_rank == 8
        assert cfg.use_4bit is True

    def test_mid_vram_8_gb(self):
        """8 GB → QLoRA, batch=2, gc on."""
        gpu = make_gpu(vram_mb=8 * 1024)
        cfg = suggest_training_config(gpu)
        assert cfg.strategy == "qlora"
        assert cfg.batch_size == 2
        assert cfg.max_seq_length == 2048
        assert cfg.gradient_checkpointing is True
        assert cfg.lora_rank == 16

    def test_rtx_4070_12gb(self):
        """RTX 4070 reale ha vram_total_mb=12282."""
        gpu = make_gpu(vram_mb=12282)
        cfg = suggest_training_config(gpu)
        assert cfg.strategy == "qlora"
        assert cfg.batch_size == 4
        assert cfg.max_seq_length == 2048
        assert cfg.gradient_checkpointing is False
        assert cfg.lora_rank == 16

    def test_high_end_24gb(self):
        """24 GB (RTX 4090 / A6000) → LoRA fp16/bf16, no quant."""
        gpu = make_gpu(vram_mb=24 * 1024)
        cfg = suggest_training_config(gpu)
        assert cfg.strategy == "lora"
        assert cfg.use_4bit is False
        assert cfg.batch_size == 4
        assert cfg.max_seq_length == 4096
        assert cfg.lora_rank == 32

    # ---- Boundary tests ----

    def test_boundary_at_11500_mb(self):
        """Soglia 11500 MB: <11500 → batch=2, ≥11500 → batch=4."""
        cfg_below = suggest_training_config(make_gpu(vram_mb=11499))
        cfg_at = suggest_training_config(make_gpu(vram_mb=11500))
        assert cfg_below.batch_size == 2
        assert cfg_at.batch_size == 4

    def test_boundary_at_15500_mb(self):
        """Soglia 15500 MB: <15500 → qlora, ≥15500 → lora."""
        cfg_below = suggest_training_config(make_gpu(vram_mb=15499))
        cfg_at = suggest_training_config(make_gpu(vram_mb=15500))
        assert cfg_below.strategy == "qlora"
        assert cfg_at.strategy == "lora"


# ---------------------------------------------------------------------------
# Mixed precision dtype selection
# ---------------------------------------------------------------------------


class TestDtypeSelection:
    def test_bf16_when_supported(self):
        gpu = make_gpu(vram_mb=12282, bf16=True, fp16=True)
        cfg = suggest_training_config(gpu)
        assert cfg.mixed_precision_dtype == "bf16"

    def test_fp16_when_no_bf16(self):
        gpu = make_gpu(vram_mb=12282, bf16=False, fp16=True)
        cfg = suggest_training_config(gpu)
        assert cfg.mixed_precision_dtype == "fp16"
        assert any("bf16 non supportato" in n for n in cfg.notes)

    def test_fp32_when_no_half_precision(self):
        gpu = make_gpu(vram_mb=12282, bf16=False, fp16=False)
        cfg = suggest_training_config(gpu)
        assert cfg.mixed_precision_dtype == "fp32"
        assert any("fp32" in n for n in cfg.notes)


# ---------------------------------------------------------------------------
# Gradient accumulation math
# ---------------------------------------------------------------------------


class TestGradAccumulation:
    def test_default_target_16_with_batch_4(self):
        """target_effective_batch=16, batch=4 → grad_accum=4."""
        gpu = make_gpu(vram_mb=12282)
        cfg = suggest_training_config(gpu, target_effective_batch=16)
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.effective_batch_size == 16

    def test_target_32_with_batch_4(self):
        gpu = make_gpu(vram_mb=12282)
        cfg = suggest_training_config(gpu, target_effective_batch=32)
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.effective_batch_size == 32

    def test_target_smaller_than_batch_floors_to_1(self):
        """target=2, batch=4 → grad_accum non può essere 0, deve essere min 1."""
        gpu = make_gpu(vram_mb=12282)
        cfg = suggest_training_config(gpu, target_effective_batch=2)
        assert cfg.gradient_accumulation_steps == 1
        # effective sarà batch_size (4), non 2
        assert cfg.effective_batch_size == cfg.batch_size


# ---------------------------------------------------------------------------
# LoRA alpha convention
# ---------------------------------------------------------------------------


class TestLoraAlpha:
    @pytest.mark.parametrize("vram_mb", [6500, 8 * 1024, 12282, 24 * 1024])
    def test_alpha_is_double_rank(self, vram_mb):
        gpu = make_gpu(vram_mb=vram_mb)
        cfg = suggest_training_config(gpu)
        assert cfg.lora_alpha == cfg.lora_rank * 2


# ---------------------------------------------------------------------------
# Smoke: il risultato è un TrainingConfigSuggestion
# ---------------------------------------------------------------------------


def test_returns_dataclass_instance():
    gpu = make_gpu(vram_mb=12282)
    cfg = suggest_training_config(gpu)
    assert isinstance(cfg, TrainingConfigSuggestion)