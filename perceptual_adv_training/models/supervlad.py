from torch import nn

from ..checkpoints import load_model_weights
from .registry import ModelAdapter, ModelBundle, register_model


def add_arguments(parser) -> None:
    del parser


def build(args) -> ModelBundle:
    from model import network

    model = network.SuperVLADModel(
        args,
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path,
    )
    descriptor_dim = int(args.features_dim * args.supervlad_clusters)
    return ModelBundle(model=model, descriptor_dim=descriptor_dim)


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


ADAPTER = ModelAdapter(
    name="supervlad",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
)


def register() -> None:
    register_model(ADAPTER)
