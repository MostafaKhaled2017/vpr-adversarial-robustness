import unittest

import torch
from torch import nn

from src.models.supervlad import encode_cross_image


def make_encoder():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=False)
    return nn.TransformerEncoder(layer, num_layers=1).eval()


class CrossImageEncodingTests(unittest.TestCase):
    def test_batched_mode_matches_upstream_expression(self):
        encoder = make_encoder()
        tokens = torch.randn(3, 4, 8)
        expected = encoder(tokens.view(3, -1, 8)).view(3, -1)
        self.assertTrue(torch.allclose(encode_cross_image(encoder, tokens, independent=False), expected))

    def test_batched_mode_couples_images(self):
        encoder = make_encoder()
        tokens = torch.randn(3, 4, 8)
        alone = encode_cross_image(encoder, tokens[:1], independent=False)
        together = encode_cross_image(encoder, tokens, independent=False)[:1]
        self.assertFalse(torch.allclose(alone, together, atol=1e-5))

    def test_independent_mode_equals_batch_of_one(self):
        encoder = make_encoder()
        tokens = torch.randn(5, 4, 8)
        together = encode_cross_image(encoder, tokens, independent=True)
        singles = torch.cat([encode_cross_image(encoder, tokens[i : i + 1], independent=False) for i in range(5)])
        self.assertTrue(torch.allclose(together, singles, atol=1e-5))

    def test_independent_mode_output_shape(self):
        encoder = make_encoder()
        self.assertEqual(encode_cross_image(encoder, torch.randn(2, 4, 8), independent=True).shape, (2, 32))


from src.models.supervlad import _query_mode_model_class


class FakeBackbone(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        patches = x.flatten(1)[:, :16 * 8].reshape(batch, 16, 8)
        return {"x_prenorm": torch.zeros(batch, 17, 8), "x_norm_patchtokens": patches, "x_norm_clstoken": patches[:, 0]}


class FakeAggregation(nn.Module):
    def forward(self, x):  # (B, D, 4, 4) -> (B, 4 * D) as 4 cluster tokens
        return x.flatten(2)[:, :, :4].permute(0, 2, 1).reshape(x.shape[0], -1)


class QueryModeForwardTests(unittest.TestCase):
    def make_model(self):
        cls = _query_mode_model_class()
        model = nn.Module.__new__(cls)
        nn.Module.__init__(model)
        model.backbone = FakeBackbone()
        model.aggregation = FakeAggregation()
        model.encoder = make_encoder()
        model.crossimage_encoder = True
        model.arch_name = "dino"
        return model.eval()

    def test_queryflag_one_is_batch_independent(self):
        model = self.make_model()
        images = torch.randn(4, 3, 8, 8)
        batched = model(images, queryflag=1)
        singles = torch.cat([model(images[i : i + 1], queryflag=1) for i in range(4)])
        self.assertTrue(torch.allclose(batched, singles, atol=1e-5))

    def test_queryflag_zero_keeps_upstream_coupling(self):
        model = self.make_model()
        images = torch.randn(4, 3, 8, 8)
        self.assertFalse(torch.allclose(model(images, queryflag=0)[:1], model(images[:1], queryflag=0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
