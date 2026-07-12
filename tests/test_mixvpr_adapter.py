import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from src.models import mixvpr


MODELS_INFO = {
    128: ("url-128", "checkpoint-128", 64, 2),
    512: ("url-512", "checkpoint-512", 256, 2),
    4096: ("url-4096", "checkpoint-4096", 1024, 4),
}


class FakeMixVPRModel(nn.Module):
    def __init__(self, agg_config):
        super().__init__()
        self.agg_config = agg_config
        self.backbone = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        descriptor_dim = agg_config["out_channels"] * agg_config["out_rows"]
        self.aggregator = nn.Linear(3, descriptor_dim, bias=False)

    def forward(self, inputs):
        features = self.backbone(inputs).mean(dim=(2, 3))
        return nn.functional.normalize(self.aggregator(features), dim=1)


class MixVPRAdapterTests(unittest.TestCase):
    def setUp(self):
        self.upstream = SimpleNamespace(MODELS_INFO=MODELS_INFO, MixVPRModel=FakeMixVPRModel)

    def args(self, dimension=None, freeze=False):
        return SimpleNamespace(
            mixvpr_descriptors_dimension=dimension,
            mixvpr_freeze_backbone=freeze,
            device="cpu",
        )

    def test_default_dimension_is_4096(self):
        self.assertEqual(mixvpr.resolve_descriptor_dimension(self.args()), 4096)

    def test_configs_match_all_published_variants(self):
        expected = {
            128: (64, 2),
            512: (256, 2),
            4096: (1024, 4),
        }
        for dimension, (out_channels, out_rows) in expected.items():
            with self.subTest(dimension=dimension):
                config = mixvpr.build_model_config(self.upstream, dimension)
                self.assertEqual(config["in_channels"], 1024)
                self.assertEqual(config["in_h"], 20)
                self.assertEqual(config["in_w"], 20)
                self.assertEqual(config["out_channels"], out_channels)
                self.assertEqual(config["out_rows"], out_rows)
                self.assertEqual(config["mix_depth"], 4)
                self.assertEqual(config["mlp_ratio"], 1)

    def test_descriptor_forward_accepts_queryflag_and_normalizes_output(self):
        descriptor_model = mixvpr.build_descriptor_model_class(self.upstream)
        model = descriptor_model(agg_config=mixvpr.build_model_config(self.upstream, 128))

        descriptors = model(torch.randn(2, 3, 4, 4), queryflag=0)

        self.assertEqual(descriptors.shape, (2, 128))
        self.assertTrue(torch.allclose(descriptors.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_build_trains_all_parameters_by_default(self):
        with mock.patch.object(mixvpr, "load_upstream_module", return_value=self.upstream):
            bundle = mixvpr.build(self.args(dimension=512))

        self.assertEqual(bundle.descriptor_dim, 512)
        self.assertTrue(all(parameter.requires_grad for parameter in bundle.model.parameters()))

    def test_freezing_backbone_preserves_input_gradients(self):
        with mock.patch.object(mixvpr, "load_upstream_module", return_value=self.upstream):
            model = mixvpr.build(self.args(dimension=128, freeze=True)).model
        inputs = torch.randn(2, 3, 4, 4, requires_grad=True)

        model(inputs)[:, 0].sum().backward()

        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.aggregator.parameters()))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))

    def test_plain_official_state_dict_loads_strictly(self):
        descriptor_model = mixvpr.build_descriptor_model_class(self.upstream)
        source = descriptor_model(agg_config=mixvpr.build_model_config(self.upstream, 128))
        destination = descriptor_model(agg_config=mixvpr.build_model_config(self.upstream, 128))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mixvpr.pth"
            torch.save(source.state_dict(), checkpoint_path)

            mixvpr.load_weights(destination, str(checkpoint_path), self.args(dimension=128))

        for source_parameter, destination_parameter in zip(source.parameters(), destination.parameters()):
            self.assertTrue(torch.equal(source_parameter, destination_parameter))

    def test_download_is_mocked_and_uses_official_url(self):
        descriptor_model = mixvpr.build_descriptor_model_class(self.upstream)
        model = descriptor_model(agg_config=mixvpr.build_model_config(self.upstream, 128))
        fake_gdown = types.ModuleType("gdown")
        download_calls = []

        def fake_download(*, url, output, fuzzy):
            download_calls.append((url, output, fuzzy))
            Path(output).write_bytes(b"mock checkpoint")
            return output

        fake_gdown.download = fake_download
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint-128"
            with (
                mock.patch.object(mixvpr, "load_upstream_module", return_value=self.upstream),
                mock.patch.object(mixvpr, "official_checkpoint_path", return_value=checkpoint_path),
                mock.patch.object(mixvpr, "load_model_weights") as load_model_weights,
                mock.patch.dict(sys.modules, {"gdown": fake_gdown}),
            ):
                mixvpr.download_weights(model, self.args(dimension=128))

        self.assertEqual(download_calls, [("url-128", str(checkpoint_path), True)])
        load_model_weights.assert_called_once_with(model, str(checkpoint_path), map_location="cpu", strict=True)


if __name__ == "__main__":
    unittest.main()
