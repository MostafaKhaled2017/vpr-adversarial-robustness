import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "third_party" / "SuperVLAD"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.models import dinomix  # noqa: E402


def args(**overrides):
    values = {"freeze_te": None, "resize": None, "train_resize": None, "device": "cpu"}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def trainable(dino):
    return [i for i, block in enumerate(dino.blocks) if all(p.requires_grad for p in block.parameters())]


class DinoMixAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = dinomix.build(args()).model.eval()

    def test_descriptor_is_4096_and_l2_normalised(self):
        with torch.no_grad():
            descriptors = self.model(torch.randn(2, 3, 224, 224), queryflag=1)
        self.assertEqual(descriptors.shape, (2, dinomix.DESCRIPTOR_DIM))
        torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2))

    def test_state_dict_keys_match_release_layout(self):
        keys = list(self.model.state_dict())
        self.assertTrue(all(key.startswith(("backbone.dino_model.", "aggregator.")) for key in keys))
        self.assertFalse(any(key.startswith("backbone.dino_model.norm.") for key in keys))
        self.assertIn("backbone.dino_model.blocks.11.attn.qkv.weight", keys)
        self.assertIn("aggregator.row_proj.weight", keys)

    def test_freeze_te_none_trains_blocks_8_to_11(self):
        dino = dinomix.build(args()).model.backbone.dino_model
        self.assertEqual(trainable(dino), [8, 9, 10, 11])
        self.assertFalse(any(p.requires_grad for p in dino.patch_embed.parameters()))

    def test_freeze_te_zero_trains_everything(self):
        model = dinomix.build(args(freeze_te=0)).model
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_frozen_blocks_preserve_input_gradients(self):
        inputs = torch.randn(1, 3, 224, 224, requires_grad=True)
        self.model(inputs)[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.model.zero_grad(set_to_none=True)

    def test_non_224_size_is_rejected(self):
        for overrides in ({"resize": [322, 322]}, {"train_resize": [320, 320]}):
            with self.subTest(**overrides), self.assertRaisesRegex(ValueError, "224"):
                dinomix.build(args(**overrides))
        with self.assertRaisesRegex(ValueError, "224"):
            dinomix.configure_evaluation(types.SimpleNamespace(resize=[322, 322], test_method="hard_resize"))
        with self.assertRaisesRegex(ValueError, "224"):
            self.model(torch.randn(1, 3, 322, 322))

    def test_configure_evaluation_sets_224_and_requires_hard_resize(self):
        eval_args = types.SimpleNamespace(resize=None, test_method="hard_resize")
        dinomix.configure_evaluation(eval_args)
        self.assertEqual(eval_args.resize, [224, 224])
        with self.assertRaisesRegex(ValueError, "hard_resize"):
            dinomix.configure_evaluation(types.SimpleNamespace(resize=None, test_method="central_crop"))

    def test_download_is_mocked_and_uses_release_url(self):
        model = dinomix.build(args()).model
        release = {"state_dict": {key: value.clone() for key, value in model.state_dict().items()}}
        with mock.patch("torch.hub.load_state_dict_from_url", return_value=release) as download:
            dinomix.download_weights(model, args())
        self.assertEqual(download.call_args.args[0], dinomix.CHECKPOINT_URL)

    def test_download_rejects_a_checkpoint_with_missing_keys(self):
        model = dinomix.build(args()).model
        state = dict(model.state_dict())
        state.pop("aggregator.row_proj.weight")
        with mock.patch("torch.hub.load_state_dict_from_url", return_value={"state_dict": state}):
            with self.assertRaises(RuntimeError):
                dinomix.download_weights(model, args())


if __name__ == "__main__":
    unittest.main()
