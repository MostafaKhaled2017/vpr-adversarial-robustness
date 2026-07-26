from torch import nn

from ..checkpoints import load_model_weights
from .registry import ModelAdapter, ModelBundle, register_model


def add_arguments(parser) -> None:
    del parser


def _build(args, pretrained_foundation: bool) -> ModelBundle:
    from model import network

    model = network.SuperVLADModel(
        args,
        pretrained_foundation=pretrained_foundation,
        foundation_model_path=args.foundation_model_path,
    )
    descriptor_dim = int(args.features_dim * args.supervlad_clusters)
    return ModelBundle(model=model, descriptor_dim=descriptor_dim)


def build(args) -> ModelBundle:
    return _build(args, pretrained_foundation=True)


def build_evaluation(args) -> ModelBundle:
    return _build(args, pretrained_foundation=bool(args.foundation_model_path))


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


def configure_evaluation(args) -> None:
    if args.resize is None:
        args.resize = [322, 322]


def build_evaluation_dataset(args, dataset_name: str):
    import datasets_ws

    return datasets_ws.BaseDataset(args, args.eval_datasets_folder, dataset_name, "test")


def load_evaluation_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    import util

    args.resume = checkpoint_path
    util.resume_model(args, model)


ADAPTER = ModelAdapter(
    name="supervlad",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
    build_evaluation=build_evaluation,
    configure_evaluation=configure_evaluation,
    build_evaluation_dataset=build_evaluation_dataset,
    load_evaluation_weights=load_evaluation_weights,
)


def register() -> None:
    register_model(ADAPTER)
