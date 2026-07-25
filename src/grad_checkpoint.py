import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    """Recomputes the wrapped block's activations during backward instead of storing them.

    The checkpointed path replays the exact same operations, so outputs and
    gradients are unchanged; only peak memory and backward-time compute differ.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


def enable_backbone_grad_checkpointing(model: nn.Module) -> nn.Module:
    """Wrap every ``model.backbone.blocks`` entry in :class:`CheckpointedBlock` in place.

    Accepts either a bare model or one wrapped in ``nn.DataParallel``. Safe to
    call more than once. Raises ``ValueError`` for models without a
    DINOv2-style ``backbone.blocks`` module list.
    """
    target = model.module if isinstance(model, nn.DataParallel) else model
    backbone = getattr(target, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise ValueError(
            "Gradient checkpointing requires a model exposing backbone.blocks "
            "(DINOv2-style transformer backbone)."
        )
    backbone.blocks = nn.ModuleList(
        block if isinstance(block, CheckpointedBlock) else CheckpointedBlock(block)
        for block in blocks
    )
    return model
