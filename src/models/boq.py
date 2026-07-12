import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from ..checkpoints import load_model_state_dict, load_model_weights
from .registry import ModelAdapter, ModelBundle, register_model


SUPPORTED_DIMENSIONS = {"Dinov2": 12288, "ResNet50": 16384}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--boq_backbone",
        choices=tuple(SUPPORTED_DIMENSIONS),
        default="Dinov2",
        help="Backbone for the Bag-of-Queries model.",
    )
    parser.add_argument(
        "--boq_descriptors_dimension",
        type=int,
        default=None,
        help="BoQ descriptor dimension. Defaults to the published dimension for the selected backbone.",
    )


def resolve_descriptor_dimension(args) -> int:
    expected = SUPPORTED_DIMENSIONS[args.boq_backbone]
    requested = args.boq_descriptors_dimension
    if requested is None:
        return expected
    if requested != expected:
        raise ValueError(
            f"BoQ backbone {args.boq_backbone!r} supports descriptor dimension {expected}, "
            f"not {requested}."
        )
    return requested


def load_upstream_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "VPR-methods-evaluation"
        / "vpr_models"
        / "boq.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"BoQ implementation was not found at {module_path}")

    spec = importlib.util.spec_from_file_location("vpr_methods_evaluation_boq", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model_classes(upstream):
    class GradientDinoV2(upstream.DinoV2):
        def forward(self, inputs):
            batch_size, _, height, width = inputs.shape
            outputs = self.dino.prepare_tokens_with_masks(inputs)
            for block in self.dino.blocks:
                outputs = block(outputs)
            outputs = outputs[:, 1:]
            if self.reshape_output:
                channels = outputs.shape[-1]
                patch_size = self.patch_size
                outputs = outputs.permute(0, 2, 1).reshape(
                    batch_size,
                    channels,
                    height // patch_size,
                    width // patch_size,
                )
            return outputs

    class GradientResNet(upstream.ResNet):
        def forward(self, inputs):
            return self.unfrozen_layers(self.frozen_layers(inputs))

    class DescriptorModel(upstream.VPRModel):
        def forward(self, inputs, queryflag=0):
            del queryflag
            return super().forward(inputs)

    return GradientDinoV2, GradientResNet, DescriptorModel


def _build_dinov2_backbone(backbone_class):
    from model.vision_transformer import vit_base

    backbone = backbone_class.__new__(backbone_class)
    nn.Module.__init__(backbone)
    backbone.backbone_name = "dinov2_vitb14"
    backbone.unfreeze_n_blocks = 2
    backbone.reshape_output = True
    backbone.dino = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    backbone.dino.requires_grad_(False)
    for block in backbone.dino.blocks[-backbone.unfreeze_n_blocks :]:
        block.requires_grad_(True)
    backbone.out_channels = backbone.dino.embed_dim
    return backbone


def _build_resnet50_backbone(backbone_class):
    import torchvision

    backbone = backbone_class.__new__(backbone_class)
    nn.Module.__init__(backbone)
    backbone.backbone_name = "resnet50"
    backbone.pretrained = True
    backbone.unfreeze_n_blocks = 1
    backbone.crop_last_block = True

    resnet = torchvision.models.resnet50(weights=None)
    layers = [
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    ]
    backbone.frozen_layers = nn.Sequential(*layers[:-1])
    backbone.unfrozen_layers = nn.Sequential(layers[-1])
    backbone.frozen_layers.requires_grad_(False)
    backbone.out_channels = resnet.layer3[-1].conv3.out_channels
    return backbone


def build(args) -> ModelBundle:
    upstream = load_upstream_module()
    gradient_dino, gradient_resnet, descriptor_model = _build_model_classes(upstream)
    descriptor_dim = resolve_descriptor_dimension(args)

    if args.boq_backbone == "Dinov2":
        backbone = _build_dinov2_backbone(gradient_dino)
        projection_channels = 384
    else:
        backbone = _build_resnet50_backbone(gradient_resnet)
        projection_channels = 512

    aggregator = upstream.BoQ(
        in_channels=backbone.out_channels,
        proj_channels=projection_channels,
        num_queries=64,
        num_layers=2,
        row_dim=descriptor_dim // projection_channels,
    )
    return ModelBundle(
        model=descriptor_model(backbone=backbone, aggregator=aggregator),
        descriptor_dim=descriptor_dim,
    )


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


def download_weights(model: nn.Module, args) -> None:
    upstream = load_upstream_module()
    descriptor_dim = resolve_descriptor_dimension(args)
    checkpoint_url = upstream.MODEL_URLS[f"{args.boq_backbone}_{descriptor_dim}"]
    state_dict = torch.hub.load_state_dict_from_url(
        checkpoint_url,
        map_location=torch.device("cpu"),
        progress=True,
    )
    load_model_state_dict(model, state_dict, strict=True)


ADAPTER = ModelAdapter(
    name="boq",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
    download_weights=download_weights,
)


def register() -> None:
    register_model(ADAPTER)
