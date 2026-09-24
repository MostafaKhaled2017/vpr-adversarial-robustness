import math

import torch
from torch import Tensor, nn

from ..checkpoints import load_model_weights
from .registry import ModelAdapter, ModelBundle, register_model


def add_arguments(parser) -> None:
    del parser


def encode_cross_image(encoder: nn.Module, tokens: Tensor, independent: bool) -> Tensor:
    """Run SuperVLAD's cross-image encoder on ``(B, clusters, D)`` cluster tokens.

    Upstream calls ``encoder(x.view(B, -1, D))`` with ``batch_first=False``, so the batch is
    the attention sequence and every image attends to the others in its batch. That makes a
    query's descriptor depend on its batch neighbours. ``independent=True`` uses a length-1
    sequence, which is exactly upstream's computation at batch size 1, at batched speed.
    """
    batch_size = tokens.shape[0]
    if independent:
        return encoder(tokens.reshape(1, -1, tokens.shape[-1])).reshape(batch_size, -1)
    return encoder(tokens.reshape(batch_size, -1, tokens.shape[-1])).reshape(batch_size, -1)


def _query_mode_model_class():
    from model import network

    class QueryModeSuperVLAD(network.SuperVLADModel):
        """SuperVLAD whose ``queryflag=1`` encodes each image independently (spec D1)."""

        def forward(self, x, queryflag=0):
            if not (self.crossimage_encoder and self.arch_name.startswith("dino")):
                return super().forward(x, queryflag)
            features = self.backbone(x)
            _, patches, dim = features["x_prenorm"].shape
            side = int(math.sqrt(patches - 1))
            patch_tokens = features["x_norm_patchtokens"].view(-1, side, side, dim).permute(0, 3, 1, 2)
            aggregated = self.aggregation(patch_tokens)
            batch_size = aggregated.shape[0]
            encoded = encode_cross_image(self.encoder, aggregated.view(batch_size, -1, dim), independent=bool(queryflag))
            return torch.nn.functional.normalize(encoded, p=2, dim=-1)

    return QueryModeSuperVLAD


def _build(args, pretrained_foundation: bool) -> ModelBundle:
    model = _query_mode_model_class()(
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

    return datasets_ws.BaseDataset(args, args.eval_datasets_folder, dataset_name, args.dataset_split)


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
