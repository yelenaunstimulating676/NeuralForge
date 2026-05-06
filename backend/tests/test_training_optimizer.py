"""
Test di optimizer e scheduler del Training Engine.

Tutti i test usano mini-modelli torch.nn.Linear, niente bitsandbytes,
niente GPU richiesta.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from core.training.optimizer import (
    OptimizerConfig,
    SchedulerConfig,
    _cosine_warmup_lambda,
    build_optimizer,
    build_scheduler,
    get_trainable_parameters,
)


# ---------------------------------------------------------------------------
# Helper: mini-modello con parametri parzialmente congelati
# ---------------------------------------------------------------------------


class MiniModel(nn.Module):
    def __init__(self, freeze_first: bool = False):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)
        if freeze_first:
            for p in self.layer1.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# OptimizerConfig validation
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults(self):
        c = OptimizerConfig()
        assert c.learning_rate == 2e-4
        assert c.weight_decay == 0.01
        assert c.use_8bit is True

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            OptimizerConfig(learning_rate=0)
        with pytest.raises(ValueError):
            OptimizerConfig(learning_rate=2.0)

    def test_invalid_weight_decay(self):
        with pytest.raises(ValueError):
            OptimizerConfig(weight_decay=-0.01)

    def test_invalid_betas(self):
        with pytest.raises(ValueError):
            OptimizerConfig(betas=(1.5, 0.999))
        with pytest.raises(ValueError):
            OptimizerConfig(betas=(0.9, 1.0))

    def test_invalid_eps(self):
        with pytest.raises(ValueError):
            OptimizerConfig(eps=-1e-8)


# ---------------------------------------------------------------------------
# SchedulerConfig validation
# ---------------------------------------------------------------------------


class TestSchedulerConfig:
    def test_defaults(self):
        c = SchedulerConfig(total_steps=1000)
        assert c.total_steps == 1000
        assert c.warmup_ratio == 0.03
        assert c.min_lr_ratio == 0.0

    def test_warmup_steps_computed(self):
        c = SchedulerConfig(total_steps=1000, warmup_ratio=0.1)
        assert c.warmup_steps == 100

    def test_warmup_steps_at_least_one(self):
        # Anche con warmup_ratio=0, vogliamo almeno 1 step di warmup
        c = SchedulerConfig(total_steps=10, warmup_ratio=0.0)
        assert c.warmup_steps == 1

    def test_invalid_total_steps(self):
        with pytest.raises(ValueError):
            SchedulerConfig(total_steps=0)

    def test_invalid_warmup_ratio(self):
        with pytest.raises(ValueError):
            SchedulerConfig(total_steps=100, warmup_ratio=1.0)
        with pytest.raises(ValueError):
            SchedulerConfig(total_steps=100, warmup_ratio=-0.1)

    def test_invalid_min_lr_ratio(self):
        with pytest.raises(ValueError):
            SchedulerConfig(total_steps=100, min_lr_ratio=1.5)
        with pytest.raises(ValueError):
            SchedulerConfig(total_steps=100, min_lr_ratio=-0.1)


# ---------------------------------------------------------------------------
# get_trainable_parameters
# ---------------------------------------------------------------------------


class TestGetTrainableParameters:
    def test_all_trainable(self):
        model = MiniModel()
        params = get_trainable_parameters(model)
        # 2 Linear, ognuno ha weight + bias → 4 tensori
        assert len(params) == 4

    def test_partial_trainable(self):
        model = MiniModel(freeze_first=True)
        params = get_trainable_parameters(model)
        # Solo layer2 è trainable → 2 tensori (weight + bias)
        assert len(params) == 2

    def test_none_trainable(self):
        model = MiniModel()
        for p in model.parameters():
            p.requires_grad = False
        params = get_trainable_parameters(model)
        assert params == []


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_torch_fallback(self):
        """Se forziamo use_8bit=False, sicuramente torch AdamW."""
        model = MiniModel()
        opt = build_optimizer(model, OptimizerConfig(use_8bit=False))
        # AdamW di torch (non subclass tipo bnb)
        assert type(opt).__name__ == "AdamW"

    def test_optimizer_lr_set(self):
        model = MiniModel()
        opt = build_optimizer(
            model, OptimizerConfig(learning_rate=1e-3, use_8bit=False)
        )
        assert opt.param_groups[0]["lr"] == 1e-3

    def test_no_trainable_params_raises(self):
        model = MiniModel()
        for p in model.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="trainable"):
            build_optimizer(model, OptimizerConfig(use_8bit=False))

    def test_only_trainable_passed_to_optimizer(self):
        """Se freezo metà modello, l'optimizer deve avere solo metà params."""
        model = MiniModel(freeze_first=True)
        opt = build_optimizer(model, OptimizerConfig(use_8bit=False))
        # Solo i 2 tensori di layer2
        assert len(opt.param_groups[0]["params"]) == 2


# ---------------------------------------------------------------------------
# Cosine warmup math
# ---------------------------------------------------------------------------


class TestCosineWarmupLambda:
    def test_step_zero_is_zero(self):
        # All'inizio del warmup il LR è 0
        v = _cosine_warmup_lambda(0, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        assert v == 0.0

    def test_warmup_linear_midpoint(self):
        # A metà warmup il moltiplicatore è 0.5
        v = _cosine_warmup_lambda(5, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        assert v == 0.5

    def test_at_warmup_end_is_one(self):
        # Alla fine del warmup il moltiplicatore è 1.0
        v = _cosine_warmup_lambda(10, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        assert v == pytest.approx(1.0)

    def test_at_total_steps_is_min(self):
        # Alla fine il moltiplicatore è min_lr_ratio
        v = _cosine_warmup_lambda(100, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_at_total_steps_with_min_lr_ratio(self):
        v = _cosine_warmup_lambda(100, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
        assert v == pytest.approx(0.1, abs=1e-6)

    def test_decay_is_monotonic_after_warmup(self):
        """Dopo il warmup il LR deve solo decrescere."""
        warmup = 10
        total = 100
        previous = float("inf")
        for step in range(warmup, total + 1):
            v = _cosine_warmup_lambda(
                step, warmup_steps=warmup, total_steps=total, min_lr_ratio=0.0
            )
            assert v <= previous + 1e-9, (
                f"LR non monotono decrescente: step {step}, prev {previous}, cur {v}"
            )
            previous = v

    def test_overshooting_total_steps_clamped(self):
        # Step oltre total_steps: clampa a min_lr_ratio
        v = _cosine_warmup_lambda(
            200, warmup_steps=10, total_steps=100, min_lr_ratio=0.05
        )
        assert v == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# build_scheduler — integrazione con torch
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    @pytest.fixture
    def setup(self):
        model = MiniModel()
        opt_config = OptimizerConfig(learning_rate=1e-3, use_8bit=False)
        opt = build_optimizer(model, opt_config)
        return opt, opt_config.learning_rate

    def test_initial_lr_at_step_zero(self, setup):
        opt, base_lr = setup
        sched = build_scheduler(
            opt, SchedulerConfig(total_steps=100, warmup_ratio=0.1)
        )
        # LambdaLR alla creazione applica step 0
        # warmup_lambda(0) = 0 → LR = 0
        assert opt.param_groups[0]["lr"] == 0.0

    def test_lr_at_warmup_end(self, setup):
        opt, base_lr = setup
        sched = build_scheduler(
            opt, SchedulerConfig(total_steps=100, warmup_ratio=0.1)
        )
        # 10 step di warmup (10% di 100)
        for _ in range(10):
            sched.step()
        # Ora siamo a fine warmup: LR ≈ base_lr
        assert opt.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-4)

    def test_lr_decays_after_warmup(self, setup):
        opt, base_lr = setup
        sched = build_scheduler(
            opt, SchedulerConfig(total_steps=100, warmup_ratio=0.1)
        )
        # Avanza fino a step 50 (metà cosine decay)
        for _ in range(50):
            sched.step()
        # LR deve essere tra base_lr e 0
        cur_lr = opt.param_groups[0]["lr"]
        assert 0 < cur_lr < base_lr

    def test_lr_near_zero_at_end(self, setup):
        opt, base_lr = setup
        sched = build_scheduler(
            opt, SchedulerConfig(total_steps=20, warmup_ratio=0.1, min_lr_ratio=0.0)
        )
        for _ in range(20):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)

    def test_lr_at_min_ratio(self, setup):
        opt, base_lr = setup
        sched = build_scheduler(
            opt,
            SchedulerConfig(total_steps=20, warmup_ratio=0.1, min_lr_ratio=0.1),
        )
        for _ in range(20):
            sched.step()
        # LR finale = base_lr * 0.1
        assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * 0.1, rel=1e-4)


# ---------------------------------------------------------------------------
# E2E: optimizer + scheduler + dummy training step
# ---------------------------------------------------------------------------


class TestE2E:
    def test_one_training_step(self):
        """Sanity check: optimizer + scheduler + backward + step funzionano."""
        model = MiniModel()
        opt = build_optimizer(model, OptimizerConfig(use_8bit=False))
        sched = build_scheduler(opt, SchedulerConfig(total_steps=10))

        # Dummy forward + backward
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        out = model.layer2(model.layer1(x))
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # Step (deve non sollevare)
        opt.step()
        sched.step()

        # Verifica che il LR sia avanzato dal valore iniziale
        # Step 0 di sched al __init__, step manuale → step 1
        # warmup di 1 step su 10 → step 1 = LR pieno
        assert opt.param_groups[0]["lr"] > 0