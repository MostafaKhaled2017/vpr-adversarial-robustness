"""DINOMix: MixVPR's aggregator on a DINOv2 ViT-B/14 backbone (arXiv 2311.00230).

The vendored third_party/VPR-methods-evaluation/vpr_models/dinomix.py loads DINOv2 through
torch.hub and imports MixVPR relatively, so it cannot be loaded on its own. This adapter rebuilds
its backbone wrapper on SuperVLAD's local vit_base (as BoQ does) and reuses the vendored MixVPR
aggregator. Module names match the release checkpoint, which loads with strict=True.
"""

from typing import Optional

import torch
from torch import nn

from ..checkpoints import load_model_state_dict, load_model_weights
from ..evaluation_data import build_vpr_evaluation_dataset
from . import mixvpr
from .boq import apply_freeze_te
from .registry import ModelAdapter, ModelBundle, register_model


CHECKPOINT_URL = "https://github.com/GaoShuang98/DINO-Mix/releases/download/v1.0.0/dinov2_vitb14_mix.ckpt"
DESCRIPTOR_DIM = 4096
PATCH_SIZE = 14
# The MixVPR aggregator is built for a 16 x 16 token grid, so DINOMix runs at 224 x 224 only.
IMAGE_SIZE = [224, 224]
AGG_CONFIG = {
    "in_channels": 768,
    "in_h": IMAGE_SIZE[0] // PATCH_SIZE,
    "in_w": IMAGE_SIZE[1] // PATCH_SIZE,
    "out_channels": 1024,
    "mix_depth": 2,
    "mlp_ratio": 1,
    "out_rows": 4,
}
# Upstream freezes patch_embed and blocks 0-7 (layer1=7), so blocks 8-11 train.
UPSTREAM_FREEZE_TE = 8


def add_arguments(parser) -> None:
    del parser  # DINOMix has a single published configuration.


def _check_size(name: str, size) -> None:
    if size is not None and list(size) != IMAGE_SIZE:
        raise ValueError(
            f"DINOMix runs at 224×224 only (its MixVPR aggregator expects a 16×16 token grid); "
            f"got {name} {' '.join(map(str, size))}."
        )


class DinoMixBackbone(nn.Module):
    """DINOv2 patch tokens without the final norm, as a (B, C, H/14, W/14) feature map."""

    def __init__(self):
        super().__init__()
        from model.vision_transformer import vit_base

        self.dino_model = vit_base(patch_size=PATCH_SIZE, img_size=518, init_values=1, block_chunks=0)
        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()

    def forward(self, inputs):
        _check_size("input", inputs.shape[-2:])
        batch_size, _, height, width = inputs.shape
        tokens = self.dino_model.forward_features(inputs)["x_norm_patchtokens"]
        tokens = tokens.reshape(batch_size, height // PATCH_SIZE, width // PATCH_SIZE, tokens.shape[-1])
        return tokens.permute(0, 3, 1, 2)


class DescriptorModel(nn.Module):
    def __init__(self, freeze_te: Optional[int] = None):
        super().__init__()
        self.backbone = DinoMixBackbone()
        self.aggregator = mixvpr.load_upstream_module().MixVPR(**AGG_CONFIG)
        apply_freeze_te(self.backbone.dino_model, UPSTREAM_FREEZE_TE if freeze_te is None else freeze_te)

    def forward(self, inputs, queryflag=0):
        del queryflag
        return self.aggregator(self.backbone(inputs))


def build(args) -> ModelBundle:
    _check_size("--resize", getattr(args, "resize", None))
    _check_size("--train_resize", getattr(args, "train_resize", None))
    model = DescriptorModel(freeze_te=getattr(args, "freeze_te", None))
    return ModelBundle(model=model, descriptor_dim=DESCRIPTOR_DIM)


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


def configure_evaluation(args) -> None:
    _check_size("--resize", args.resize)
    if args.test_method != "hard_resize":
        raise ValueError("DINOMix rank evaluation requires --test_method=hard_resize.")
    args.resize = list(IMAGE_SIZE)


def download_weights(model: nn.Module, args) -> None:
    del args
    checkpoint = torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL,
        map_location=torch.device("cpu"),
        progress=True,
    )
    load_model_state_dict(model, checkpoint, strict=True)


ADAPTER = ModelAdapter(
    name="dinomix",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
    download_weights=download_weights,
    configure_evaluation=configure_evaluation,
    build_evaluation_dataset=build_vpr_evaluation_dataset,
)


def register() -> None:
    register_model(ADAPTER)
