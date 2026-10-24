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

from src.models import qaa  # noqa: E402


def args(**overrides):
    values = {"freeze_te": None, "device": "cpu"}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def trainable(dino):
    return [i for i, block in enumerate(dino.blocks) if all(p.requires_grad for p in block.parameters())]


class QAAAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = qaa.build(args()).model.eval()

    def test_descriptor_is_8192_and_l2_normalised_at_224_and_322(self):
        for size in (224, 322):
            with self.subTest(size=size), torch.no_grad():
                descriptors = self.model(torch.randn(2, 3, size, size), queryflag=1)
                self.assertEqual(descriptors.shape, (2, qaa.DESCRIPTOR_DIM))
                torch.testing.assert_close(descriptors.norm(dim=1), torch.ones(2))

    def test_state_dict_keys_match_release_layout(self):
        keys = list(self.model.state_dict())
        self.assertTrue(all(key.startswith(("backbone.model.", "aggregator.")) for key in keys))
        self.assertIn("backbone.model.norm.weight", keys)
        self.assertIn("backbone.model.blocks.11.attn.qkv.weight", keys)
        self.assertIn("aggregator.dust_bin", keys)

    def test_backbone_applies_final_norm(self):
        model = qaa.build(args()).model.eval()
        norm = model.backbone.model.norm
        with mock.patch.object(norm, "forward", wraps=norm.forward) as spy, torch.no_grad():
            model(torch.randn(1, 3, 224, 224))
        self.assertEqual(spy.call_count, 1)

    def test_freeze_te_none_trains_last_two_blocks(self):
        dino = qaa.build(args()).model.backbone.model
        self.assertEqual(trainable(dino), [10, 11])
        self.assertFalse(any(p.requires_grad for p in dino.patch_embed.parameters()))

    def test_freeze_te_zero_trains_everything(self):
        model = qaa.build(args(freeze_te=0)).model
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_frozen_blocks_preserve_input_gradients(self):
        inputs = torch.randn(1, 3, 224, 224, requires_grad=True)
        self.model(inputs)[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.model.zero_grad(set_to_none=True)

    def test_config_matches_pinned_release(self):
        config = qaa.MODEL_CONFIG["agg_config"]
        self.assertEqual(
            (config["num_clusters"], config["cluster_dim"], config["num_queries"], config["token_dim"]),
            (64, 128, 256, 0),
        )
        self.assertEqual(config["num_clusters"] * config["cluster_dim"], qaa.DESCRIPTOR_DIM)

    def test_configure_evaluation_sets_322_and_requires_hard_resize(self):
        eval_args = types.SimpleNamespace(resize=None, test_method="hard_resize")
        qaa.configure_evaluation(eval_args)
        self.assertEqual(eval_args.resize, [322, 322])
        with self.assertRaisesRegex(ValueError, "322"):
            qaa.configure_evaluation(types.SimpleNamespace(resize=[224, 224], test_method="hard_resize"))
        with self.assertRaisesRegex(ValueError, "hard_resize"):
            qaa.configure_evaluation(types.SimpleNamespace(resize=None, test_method="central_crop"))

    def test_download_is_mocked_and_uses_pinned_revision(self):
        model = qaa.build(args()).model
        release = {key: value.clone() for key, value in model.state_dict().items()}
        with mock.patch("huggingface_hub.hf_hub_download", return_value="weights.safetensors") as download, \
                mock.patch("safetensors.torch.load_file", return_value=release) as load_file:
            qaa.download_weights(model, args())
        download.assert_called_once_with(qaa.HF_REPO, qaa.HF_WEIGHTS_FILE, revision=qaa.HF_REVISION)
        load_file.assert_called_once_with("weights.safetensors")


if __name__ == "__main__":
    unittest.main()
