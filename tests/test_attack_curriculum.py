import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.attacks import RetrievalAttackWrapper, attack_strength_scale


class StubProjection:
    def __init__(self, bound):
        self.bound = bound


class StubBackend(nn.Module):
    def __init__(self, model, bound=0.1, step=0.04):
        super().__init__()
        self.bound = bound
        self.step = step
        self.projection = StubProjection(bound)

    def forward(self, inputs, labels):
        return inputs


class StubAttack(RetrievalAttackWrapper):
    backend_cls = StubBackend


class BoundlessBackend(nn.Module):
    def __init__(self, model):
        super().__init__()

    def forward(self, inputs, labels):
        return inputs


class BoundlessAttack(RetrievalAttackWrapper):
    backend_cls = BoundlessBackend


class AttackStrengthScheduleTests(unittest.TestCase):
    def test_no_ramp_means_full_strength_immediately(self):
        for epoch in range(5):
            with self.subTest(epoch=epoch):
                self.assertEqual(attack_strength_scale(epoch, warmup_epochs=2, ramp_epochs=0), 1.0)

    def test_first_adversarial_epoch_starts_at_the_minimum_scale(self):
        scale = attack_strength_scale(2, warmup_epochs=2, ramp_epochs=4, min_scale=0.1)

        self.assertAlmostEqual(scale, 0.1)

    def test_scale_rises_linearly_across_the_ramp(self):
        scales = [attack_strength_scale(epoch, warmup_epochs=2, ramp_epochs=4, min_scale=0.2) for epoch in range(2, 7)]

        self.assertAlmostEqual(scales[0], 0.2)
        self.assertAlmostEqual(scales[1], 0.4)
        self.assertAlmostEqual(scales[2], 0.6)
        self.assertAlmostEqual(scales[3], 0.8)
        self.assertAlmostEqual(scales[4], 1.0)

    def test_scale_saturates_at_full_strength_after_the_ramp(self):
        self.assertAlmostEqual(attack_strength_scale(50, warmup_epochs=2, ramp_epochs=4), 1.0)

    def test_warmup_epochs_stay_at_the_minimum(self):
        # The adversarial branch is off during warm-up, so this only pins the boundary.
        self.assertAlmostEqual(attack_strength_scale(0, warmup_epochs=2, ramp_epochs=4, min_scale=0.1), 0.1)
        self.assertAlmostEqual(attack_strength_scale(1, warmup_epochs=2, ramp_epochs=4, min_scale=0.1), 0.1)

    def test_ramp_without_warmup_starts_at_the_first_epoch(self):
        self.assertAlmostEqual(attack_strength_scale(0, warmup_epochs=0, ramp_epochs=2, min_scale=0.5), 0.5)
        self.assertAlmostEqual(attack_strength_scale(1, warmup_epochs=0, ramp_epochs=2, min_scale=0.5), 0.75)
        self.assertAlmostEqual(attack_strength_scale(2, warmup_epochs=0, ramp_epochs=2, min_scale=0.5), 1.0)

    def test_min_scale_must_be_within_the_unit_interval(self):
        with self.assertRaises(ValueError):
            attack_strength_scale(0, warmup_epochs=0, ramp_epochs=2, min_scale=1.5)


class AttackStrengthScalingTests(unittest.TestCase):
    def make_attack(self):
        return StubAttack(nn.Identity(), margin=0.1, device="cpu", bound=0.1, step=0.04)

    def test_scaling_halves_the_backend_bound_and_step(self):
        attack = self.make_attack()

        attack.set_strength_scale(0.5)

        self.assertAlmostEqual(attack.backend.bound, 0.05)
        self.assertAlmostEqual(attack.backend.step, 0.02)

    def test_the_projection_bound_is_scaled_too(self):
        attack = self.make_attack()

        attack.set_strength_scale(0.25)

        # Leaving the projection at full bound would let the attack escape its own budget.
        self.assertAlmostEqual(attack.backend.projection.bound, 0.025)

    def test_scaling_is_measured_from_the_original_bound_not_the_current_one(self):
        attack = self.make_attack()

        attack.set_strength_scale(0.5)
        attack.set_strength_scale(0.25)

        self.assertAlmostEqual(attack.backend.bound, 0.025)

    def test_full_strength_restores_the_configured_bound(self):
        attack = self.make_attack()

        attack.set_strength_scale(0.1)
        attack.set_strength_scale(1.0)

        self.assertAlmostEqual(attack.backend.bound, 0.1)
        self.assertAlmostEqual(attack.backend.step, 0.04)

    def test_a_backend_without_a_bound_is_left_alone(self):
        attack = BoundlessAttack(nn.Identity(), margin=0.1, device="cpu")

        attack.set_strength_scale(0.5)

        self.assertEqual(attack.strength_scale, 0.5)


if __name__ == "__main__":
    unittest.main()
