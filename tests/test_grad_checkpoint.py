import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grad_checkpoint import CheckpointedBlock, enable_backbone_grad_checkpointing


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return x + torch.relu(self.linear(x))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(3))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()

    def forward(self, x):
        return self.backbone(x)


class GradCheckpointTests(unittest.TestCase):
    def test_outputs_and_input_gradients_are_identical(self):
        torch.manual_seed(0)
        model = _Model().eval()
        x = torch.randn(2, 8)

        x_ref = x.clone().requires_grad_(True)
        ref_out = model(x_ref)
        ref_grad = torch.autograd.grad(ref_out.sum(), x_ref)[0]

        enable_backbone_grad_checkpointing(model)

        x_ckpt = x.clone().requires_grad_(True)
        ckpt_out = model(x_ckpt)
        ckpt_grad = torch.autograd.grad(ckpt_out.sum(), x_ckpt)[0]

        torch.testing.assert_close(ckpt_out, ref_out, rtol=0.0, atol=0.0)
        torch.testing.assert_close(ckpt_grad, ref_grad, rtol=0.0, atol=0.0)

    def test_inference_mode_bypasses_checkpoint_and_matches_reference(self):
        torch.manual_seed(0)
        model = _Model().eval()
        reference = copy.deepcopy(model)
        enable_backbone_grad_checkpointing(model)
        x = torch.randn(2, 8)

        with torch.inference_mode():
            out = model(x)
            ref_out = reference(x)

        torch.testing.assert_close(out, ref_out, rtol=0.0, atol=0.0)

    def test_wrapping_is_idempotent(self):
        model = _Model()
        enable_backbone_grad_checkpointing(model)
        enable_backbone_grad_checkpointing(model)

        for block in model.backbone.blocks:
            self.assertIsInstance(block, CheckpointedBlock)
            self.assertNotIsInstance(block.block, CheckpointedBlock)

    def test_dataparallel_wrapped_model_is_supported(self):
        model = nn.DataParallel(_Model())
        enable_backbone_grad_checkpointing(model)

        for block in model.module.backbone.blocks:
            self.assertIsInstance(block, CheckpointedBlock)

    def test_model_without_backbone_blocks_is_rejected(self):
        with self.assertRaises(ValueError):
            enable_backbone_grad_checkpointing(nn.Linear(4, 4))


if __name__ == "__main__":
    unittest.main()
