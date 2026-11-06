"""DINOv2 ViT-B/14 whose descriptor is its own tokens: the CLS token and the mean patch token.

No aggregation head, so the backbone can be trained for robustness on its own and then moved
under another model's head (scripts/transplant.py): the ViT sits at ``self.backbone`` with
the same module layout as SuperVLAD's, so the ``backbone.*`` keys match.
"""
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from ..checkpoints import extract_model_state_dict, load_trusted_checkpoint, normalize_state_dict_keys
from .registry import ModelAdapter, ModelBundle, register_model

TOKEN_DIM = 768
DESCRIPTOR_DIM = 2 * TOKEN_DIM


def add_arguments(parser) -> None:
    # --foundation_model_path and --freeze_te are shared with SuperVLAD.
    del parser


class DinoV2TokenModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: Tensor, queryflag: int = 0) -> Tensor:
        # Every image is encoded on its own, so queryflag changes nothing here.
        del queryflag
        features = self.backbone(x)
        descriptor = torch.cat([features["x_norm_clstoken"], features["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        return torch.nn.functional.normalize(descriptor, p=2, dim=-1)


def build(args) -> ModelBundle:
    from model import network

    # SuperVLAD's backbone builder: the same ViT, the same weight loading and the same
    # --freeze_te block freezing. It records side settings on its argument, so it gets a copy.
    backbone_args = SimpleNamespace(backbone="dino", freeze_te=getattr(args, "freeze_te", None))
    foundation_model_path = getattr(args, "foundation_model_path", None)
    backbone = network.get_backbone(backbone_args, bool(foundation_model_path), foundation_model_path)
    return ModelBundle(model=DinoV2TokenModel(backbone), descriptor_dim=DESCRIPTOR_DIM)


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    """Load a checkpoint of this model, or a plain DINOv2 release (keys without ``backbone.``)."""
    state_dict = normalize_state_dict_keys(
        extract_model_state_dict(load_trusted_checkpoint(checkpoint_path, map_location=args.device))
    )
    if not any(key.startswith("backbone.") for key in state_dict):
        state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


ADAPTER = ModelAdapter(
    name="dinov2_tokens",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
)


def register() -> None:
    register_model(ADAPTER)
