import types
import unittest

import torch
from torch import nn

from src.models.boq import _build_model_classes, apply_freeze_te


class TokenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, tokens):
        return self.projection(tokens)


class FakeDinoCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TokenBlock(), TokenBlock()])

    def prepare_tokens_with_masks(self, inputs):
        patches = inputs.flatten(2).transpose(1, 2)
        class_token = torch.zeros(inputs.shape[0], 1, inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
        return torch.cat([class_token, patches], dim=1)


class FakeDinoV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = FakeDinoCore()
        self.reshape_output = True
        self.out_channels = 3
        self.dino.requires_grad_(False)

    @property
    def patch_size(self):
        return 1


class FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_layers = nn.Conv2d(3, 3, 1, bias=False)
        self.unfrozen_layers = nn.Identity()
        self.frozen_layers.requires_grad_(False)


class FakeVPRModel(nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, inputs):
        descriptors, _ = self.aggregator(self.backbone(inputs))
        return descriptors


class FakeAggregator(nn.Module):
    def forward(self, features):
        descriptors = features.mean(dim=(2, 3))
        return torch.nn.functional.normalize(descriptors, dim=1), []


class BoQAdapterTests(unittest.TestCase):
    def setUp(self):
        upstream = types.SimpleNamespace(DinoV2=FakeDinoV2, ResNet=FakeResNet, VPRModel=FakeVPRModel)
        self.gradient_dino, self.gradient_resnet, self.descriptor_model = _build_model_classes(upstream)

    def test_descriptor_forward_accepts_queryflag(self):
        model = self.descriptor_model(self.gradient_dino(), FakeAggregator())
        descriptors = model(torch.randn(2, 3, 2, 2), queryflag=0)

        self.assertEqual(descriptors.shape, (2, 3))
        self.assertTrue(torch.allclose(descriptors.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_frozen_dino_blocks_preserve_input_gradients(self):
        model = self.descriptor_model(self.gradient_dino(), FakeAggregator())
        inputs = torch.randn(2, 3, 2, 2, requires_grad=True)

        model(inputs)[:, 0].sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))

    def test_frozen_resnet_layers_preserve_input_gradients(self):
        backbone = self.gradient_resnet()
        inputs = torch.randn(1, 3, 2, 2, requires_grad=True)

        backbone(inputs).sum().backward()

        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in backbone.frozen_layers.parameters()))



class FakeTwelveBlockDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(12))


def trainable(dino):
    return [i for i, block in enumerate(dino.blocks) if all(p.requires_grad for p in block.parameters())]


class BoQFreezeTeTests(unittest.TestCase):
    def setUp(self):
        self.dino = FakeTwelveBlockDino()

    def test_freeze_te_none_keeps_last_two_blocks(self):
        apply_freeze_te(self.dino, None)
        self.assertEqual(trainable(self.dino), [10, 11])
        self.assertFalse(any(p.requires_grad for p in self.dino.patch_embed.parameters()))

    def test_freeze_te_zero_trains_everything(self):
        apply_freeze_te(self.dino, 0)
        self.assertTrue(all(p.requires_grad for p in self.dino.parameters()))

    def test_freeze_te_eight_trains_blocks_from_eight(self):
        apply_freeze_te(self.dino, 8)
        self.assertEqual(trainable(self.dino), [8, 9, 10, 11])
        self.assertFalse(any(p.requires_grad for p in self.dino.patch_embed.parameters()))

if __name__ == "__main__":
    unittest.main()
