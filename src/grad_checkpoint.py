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


_CHECKPOINTED_CLASSES = {}


def _checkpointed_class(cls):
    """Subclass of ``cls`` whose forward checkpoints ``cls.forward`` (cached per class)."""
    if cls not in _CHECKPOINTED_CLASSES:

        def forward(self, x, *args, **kwargs):
            trainable = x.requires_grad if isinstance(x, torch.Tensor) else False
            trainable = trainable or any(parameter.requires_grad for parameter in self.parameters())
            if torch.is_grad_enabled() and trainable:
                return checkpoint(cls.forward, self, x, *args, use_reentrant=False, **kwargs)
            return cls.forward(self, x, *args, **kwargs)

        _CHECKPOINTED_CLASSES[cls] = type(f"Checkpointed{cls.__name__}", (cls,), {"forward": forward})
    return _CHECKPOINTED_CLASSES[cls]


def checkpoint_backbone_blocks_in_place(model: nn.Module) -> nn.Module:
    """Checkpoint every ``model.backbone.blocks`` entry without changing the module tree.

    Unlike :func:`enable_backbone_grad_checkpointing`, ``state_dict`` keys stay unchanged, so
    checkpoints saved during training load into plain models with ``strict=True``. Blocks
    with trainable parameters are checkpointed even when their input needs no gradient
    (the first trainable block after ``--freeze_te``). Safe to call more than once. Each block's
    class is swapped for a checkpointing subclass rather than patching its bound ``forward``, so
    ``forward`` is resolved on the class and always sees the actual module (including copies made
    by ``copy.deepcopy`` or ``nn.DataParallel`` replication); whole-model pickling of a
    checkpointed model is therefore not supported, though ``state_dict`` saving is unaffected.
    """
    target = model.module if isinstance(model, nn.DataParallel) else model
    blocks = getattr(getattr(target, "backbone", None), "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise ValueError(
            "Gradient checkpointing requires a model exposing backbone.blocks "
            "(DINOv2-style transformer backbone)."
        )
    for block in blocks:
        if type(block) not in _CHECKPOINTED_CLASSES.values():
            block.__class__ = _checkpointed_class(type(block))
    return model
