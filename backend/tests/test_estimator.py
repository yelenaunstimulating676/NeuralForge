"""Test dell'estimator pre-training."""

from __future__ import annotations

from core.training.estimator import estimate_training


class TestEstimateTraining:
    def test_basic_smollm_estimate(self):
        e = estimate_training(
            params_billions=0.135,
            num_examples=100,
            num_epochs=3,
            per_device_batch_size=4,
            grad_accum_steps=2,
            max_seq_length=1024,
            lora_r=16,
        )
        # Per SmolLM2 ~135M ci aspettiamo VRAM modesta (< 1 GB)
        assert 100 < e.estimated_vram_mb < 1000
        assert e.total_steps > 0
        assert e.steps_per_epoch > 0
        assert e.estimated_time_seconds > 0

    def test_large_model_warning(self):
        e = estimate_training(
            params_billions=70,
            num_examples=10000,
            num_epochs=3,
            per_device_batch_size=4,
            grad_accum_steps=4,
            max_seq_length=2048,
            lora_r=16,
        )
        # Modello 70B → VRAM molto alta → warning
        assert e.estimated_vram_mb > 12000
        assert any("VRAM" in n for n in e.notes)

    def test_too_few_steps_warning(self):
        e = estimate_training(
            params_billions=0.5,
            num_examples=2,
            num_epochs=1,
            per_device_batch_size=4,
            grad_accum_steps=4,
            max_seq_length=512,
            lora_r=8,
        )
        # 2 esempi / 16 batch_eff = 0 step/epoch → 1 (min 1)
        assert any("step" in n.lower() for n in e.notes)

    def test_total_steps_formula(self):
        e = estimate_training(
            params_billions=0.5,
            num_examples=160,
            num_epochs=2,
            per_device_batch_size=4,
            grad_accum_steps=2,
            max_seq_length=512,
            lora_r=8,
        )
        # 160 / (4*2) = 20 steps per epoch × 2 = 40
        assert e.steps_per_epoch == 20
        assert e.total_steps == 40

    def test_4bit_uses_less_vram(self):
        e_4bit = estimate_training(
            params_billions=1.0, num_examples=100, num_epochs=1,
            per_device_batch_size=2, grad_accum_steps=2,
            max_seq_length=1024, lora_r=16, use_4bit=True,
        )
        e_fp16 = estimate_training(
            params_billions=1.0, num_examples=100, num_epochs=1,
            per_device_batch_size=2, grad_accum_steps=2,
            max_seq_length=1024, lora_r=16, use_4bit=False,
        )
        assert e_4bit.estimated_vram_mb < e_fp16.estimated_vram_mb