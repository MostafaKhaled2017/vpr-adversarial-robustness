"""Token-descriptor DINOv2 model: descriptor, key layout, weight loading and block freezing."""

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from model import network  # noqa: E402

from src.cli import parse_arguments  # noqa: E402
from src.models import get_model_adapter, model_names  # noqa: E402
from src.train_loop import combine_losses  # noqa: E402

DIM = 4


class StubViT(nn.Module):
    """Two blocks and a norm; tokens are a linear map of the mean pixel of each image."""

    def __init__(self, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)])
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x):
        tokens = x.mean(dim=(2, 3)).repeat(1, 2)[:, :DIM].unsqueeze(1).repeat(1, 5, 1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens[:, 1:]}


def build(**overrides):
    args = Namespace(**{"foundation_model_path": None, "freeze_te": None, "device": "cpu", **overrides})
    with mock.patch.object(network, "vit_base", StubViT):
        return get_model_adapter("dinov2_tokens").build(args), args


class DinoV2TokensAdapterTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("dinov2_tokens", model_names())
        bundle, _ = build()
        self.assertIsInstance(bundle.model.backbone, StubViT)

    def test_descriptor_is_cls_and_mean_patch_unit_norm(self):
        bundle, _ = build()
        descriptors = bundle.model(torch.randn(3, 3, 8, 8), queryflag=1)
        self.assertEqual(descriptors.shape, (3, 2 * DIM))
        self.assertTrue(torch.allclose(descriptors.norm(dim=-1), torch.ones(3), atol=1e-6))

    def test_state_dict_keys_are_under_backbone(self):
        bundle, _ = build()
        self.assertTrue(all(key.startswith("backbone.") for key in bundle.model.state_dict()))

    def test_load_weights_accepts_prefixed_and_plain_vit_keys(self):
        source, args = build()
        with torch.no_grad():
            for parameter in source.model.parameters():
                parameter.add_(1.0)
        prefixed = source.model.state_dict()
        plain = {key[len("backbone."):]: value for key, value in prefixed.items()}
        adapter = get_model_adapter("dinov2_tokens")
        with tempfile.TemporaryDirectory() as tmp:
            for name, state in (("prefixed", {"model_state_dict": prefixed}), ("plain", plain)):
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.pth"
                    torch.save(state, path)
                    target, _ = build()
                    adapter.load_weights(target.model, str(path), args)
                    for key, value in target.model.state_dict().items():
                        self.assertTrue(torch.equal(value, prefixed[key]), key)

    def test_freeze_te_freezes_the_first_blocks(self):
        bundle, _ = build(freeze_te=1)
        backbone = bundle.model.backbone
        self.assertFalse(any(p.requires_grad for p in backbone.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.blocks[1].parameters()))
        self.assertFalse(any(p.requires_grad for p in backbone.norm.parameters()))


class CleanLossWeightTest(unittest.TestCase):
    def test_weight_scales_only_the_clean_term(self):
        base = ["--eval_datasets_folder", "/tmp", "--device", "cpu", "--model=dinov2_tokens"]
        default, off = parse_arguments(base), parse_arguments(base + ["--clean_loss_weight=0"])
        self.assertEqual(default.clean_loss_weight, 1.0)
        clean, adv, drift, ood = (torch.tensor(value) for value in (2.0, 3.0, 5.0, 7.0))
        default.clean_drift_weight = off.clean_drift_weight = 0.5
        self.assertAlmostEqual(combine_losses(default, clean, adv, drift, ood).item(), 2.0 + 3.0 + 2.5 + 7.0)
        self.assertAlmostEqual(combine_losses(off, clean, adv, drift, ood).item(), 3.0 + 2.5 + 7.0)
        with self.assertRaisesRegex(ValueError, "--clean_loss_weight"):
            parse_arguments(base + ["--clean_loss_weight=-1"])


if __name__ == "__main__":
    unittest.main()
