import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grad_checkpoint import (
    CheckpointedBlock,
    checkpoint_backbone_blocks_in_place,
    enable_backbone_grad_checkpointing,
)


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


class InPlaceCheckpointTests(unittest.TestCase):
    def test_state_dict_keys_are_unchanged(self):
        model = _Model()
        keys = list(model.state_dict())
        checkpoint_backbone_blocks_in_place(model)
        self.assertEqual(list(model.state_dict()), keys)

    def test_outputs_and_parameter_gradients_match_reference(self):
        torch.manual_seed(0)
        model = _Model()
        reference = copy.deepcopy(model)
        for frozen in (model.backbone.blocks[0], reference.backbone.blocks[0]):
            frozen.requires_grad_(False)  # like --freeze_te: next block's input has no grad
        checkpoint_backbone_blocks_in_place(model)
        x = torch.randn(2, 8)
        out, ref_out = model(x), reference(x)
        torch.testing.assert_close(out, ref_out, rtol=0.0, atol=0.0)
        out.sum().backward()
        ref_out.sum().backward()
        for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
            if ref_param.grad is None:
                self.assertIsNone(param.grad, name)
            else:
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=0.0, atol=0.0)

    def test_trainable_blocks_are_checkpointed_once(self):
        model = _Model()
        checkpoint_backbone_blocks_in_place(model)
        checkpoint_backbone_blocks_in_place(model)
        with mock.patch("src.grad_checkpoint.checkpoint", wraps=torch.utils.checkpoint.checkpoint) as spy:
            model(torch.randn(2, 8)).sum().backward()
        self.assertEqual(spy.call_count, 3)

    def test_no_grad_bypasses_checkpoint(self):
        model = _Model()
        checkpoint_backbone_blocks_in_place(model)
        with mock.patch("src.grad_checkpoint.checkpoint") as spy, torch.no_grad():
            model(torch.randn(2, 8))
        spy.assert_not_called()

    def test_deepcopy_uses_the_copys_own_weights(self):
        torch.manual_seed(0)
        model = _Model()
        checkpoint_backbone_blocks_in_place(model)
        clone = copy.deepcopy(model)
        for parameter in clone.parameters():
            nn.init.zeros_(parameter)
        x = torch.randn(2, 8, requires_grad=True)
        torch.testing.assert_close(clone(x), x)  # zero Linear -> each block is identity (x + relu(0))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for nn.parallel.replicate")
    def test_dataparallel_replica_still_checkpoints_every_block(self):
        # A DataParallel replica has empty self.parameters(); checkpointing must not depend on it.
        model = _Model().cuda()
        checkpoint_backbone_blocks_in_place(model)
        replica = torch.nn.parallel.replicate(model, [0, 0])[0]
        with mock.patch("src.grad_checkpoint.checkpoint", wraps=torch.utils.checkpoint.checkpoint) as spy:
            replica(torch.randn(2, 8, device="cuda")).sum().backward()
        self.assertEqual(spy.call_count, 3)


if __name__ == "__main__":
    unittest.main()
