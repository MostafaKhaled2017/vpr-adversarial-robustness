import functools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submodules" / "perceptual-advex"))

from src.attacks import sync_uar_backend_resolution


class FakeUarAttack(nn.Module):
    def __init__(self, nb_its, eps_max, step_size, resol, norm="linf", scale_each=False):
        super().__init__()
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.norm = norm
        self.scale_each = scale_each
        self.transform = nn.Identity()
        self.inverse_transform = nn.Identity()


def make_backend():
    return SimpleNamespace(
        attack_fn=functools.partial(FakeUarAttack, 5, 8.0, 2.0, 224, norm="linf", scale_each=1),
        attack=None,
    )


class SyncUarBackendResolutionTests(unittest.TestCase):
    def test_rebinds_hardcoded_resolution(self):
        backend = make_backend()
        sync_uar_backend_resolution(backend, 322, 322)
        self.assertIsNotNone(backend.attack)
        self.assertEqual(backend.attack.resol, 322)
        self.assertEqual(backend.attack.nb_its, 5)
        self.assertEqual(backend.attack.norm, "linf")

    def test_installs_pixel_scale_transforms(self):
        backend = make_backend()
        sync_uar_backend_resolution(backend, 322, 322)
        values = torch.tensor([255.0])
        self.assertTrue(torch.allclose(backend.attack.transform(values), torch.tensor([1.0])))
        self.assertTrue(torch.allclose(backend.attack.inverse_transform(torch.tensor([1.0])), values))

    def test_noop_when_resolution_matches(self):
        backend = make_backend()
        sync_uar_backend_resolution(backend, 322, 322)
        existing_attack = backend.attack
        sync_uar_backend_resolution(backend, 322, 322)
        self.assertIs(backend.attack, existing_attack)

    def test_rebuilds_on_resolution_change(self):
        backend = make_backend()
        sync_uar_backend_resolution(backend, 322, 322)
        sync_uar_backend_resolution(backend, 224, 224)
        self.assertEqual(backend.attack.resol, 224)

    def test_rejects_non_square_inputs(self):
        backend = make_backend()
        with self.assertRaises(ValueError):
            sync_uar_backend_resolution(backend, 322, 224)

    def test_ignores_non_uar_backends(self):
        backend = SimpleNamespace(attack="mister_ed_attack_state")
        sync_uar_backend_resolution(backend, 322, 322)
        self.assertEqual(backend.attack, "mister_ed_attack_state")


if __name__ == "__main__":
    unittest.main()
