"""QAA: Query-based Adaptive Aggregation on a DINOv2 ViT-B/14 backbone (arXiv 2507.03831).

The model classes come from third_party/VPR-methods-evaluation/vpr_models/qaa.py. Its backbone
loads DINOv2 through torch.hub, so this adapter builds it on SuperVLAD's local vit_base instead (as
BoQ does), and takes the configuration from MODEL_CONFIG rather than from HuggingFace. Module names
match the release checkpoint, which loads with strict=True.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

from torch import nn

from ..checkpoints import load_model_state_dict, load_model_weights
from ..evaluation_data import build_vpr_evaluation_dataset
from .boq import apply_freeze_te
from .registry import ModelAdapter, ModelBundle, register_model


DESCRIPTOR_DIM = 8192
HF_REPO = "xjh19972/QAA-8192"
HF_REVISION = "37d406838b24e11c97d9d38249626cbf58bfe3c8"
HF_WEIGHTS_FILE = "model.safetensors"
# `model_config_dict` of config.json in HF_REPO at HF_REVISION, verbatim.
MODEL_CONFIG = {
    "agg_arch": "QAA",
    "agg_config": {
        "attn_arch": "conv",
        "cluster_dim": 128,
        "divide": 1,
        "divide_ratio": [1, 1, 1, 0],
        "dropout": 0.3,
        "feature_nheads": 16,
        "freeze": "none",
        "num_channels": 768,
        "num_clusters": 64,
        "num_queries": 256,
        "out_norm": False,
        "score_nheads": 16,
        "score_norm": "none",
        "self_attn_out_norm": True,
        "skip_connection": "none",
        "token_dim": 0,
    },
    "backbone_arch": "dinov2_vitb14",
    "backbone_config": {
        "cluster_dim": 16,
        "divide": 1,
        "divide_ratio": [1, 1, 1, 0],
        "domain_prompt": "none",
        "dropout": 0.3,
        "multi_scale": "1",
        "norm_layer": True,
        "num_clusters": 16,
        "num_trainable_blocks": 2,
        "return_token": True,
        "token_dim": 256,
    },
}
# Upstream trains at 224 and evaluates at 322 (paper, implementation details), as BoQ.
EVAL_SIZE = [322, 322]
UPSTREAM_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "VPR-methods-evaluation"
    / "vpr_models"
    / "qaa.py"
)


def add_arguments(parser) -> None:
    del parser  # Only the 8192-d release is supported.


def load_upstream_module() -> ModuleType:
    if not UPSTREAM_MODULE_PATH.is_file():
        raise FileNotFoundError(f"QAA implementation was not found at {UPSTREAM_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("vpr_methods_evaluation_qaa", UPSTREAM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {UPSTREAM_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            raise ModuleNotFoundError(
                "Loading the vendored QAA module requires 'huggingface_hub'. Install "
                "third_party/VPR-methods-evaluation/requirements.txt and retry."
            ) from exc
        raise
    return module


def _build_backbone(upstream: ModuleType) -> nn.Module:
    from model.vision_transformer import vit_base

    config = MODEL_CONFIG["backbone_config"]
    backbone = upstream.DINOv2Backbone.__new__(upstream.DINOv2Backbone)
    nn.Module.__init__(backbone)
    backbone.model = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    backbone.num_channels = upstream.DINOV2_ARCHS[MODEL_CONFIG["backbone_arch"]]
    backbone.norm_layer = config["norm_layer"]
    backbone.domain_prompt = config["domain_prompt"]
    return backbone


class DescriptorModel(nn.Module):
    def __init__(self, backbone: nn.Module, aggregator: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, inputs, queryflag=0):
        del queryflag
        return self.aggregator(self.backbone(inputs))


def build(args) -> ModelBundle:
    upstream = load_upstream_module()
    backbone = _build_backbone(upstream)
    # None keeps upstream's depth: the last 2 blocks train (num_trainable_blocks).
    apply_freeze_te(backbone.model, getattr(args, "freeze_te", None))
    aggregator = upstream.QAAAggregator(**MODEL_CONFIG["agg_config"])
    return ModelBundle(model=DescriptorModel(backbone, aggregator), descriptor_dim=DESCRIPTOR_DIM)


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


def configure_evaluation(args) -> None:
    if args.resize is not None and list(args.resize) != EVAL_SIZE:
        raise ValueError("QAA requires --resize 322 322 to match the reference evaluation pipeline.")
    if args.test_method != "hard_resize":
        raise ValueError("QAA rank evaluation requires --test_method=hard_resize.")
    args.resize = list(EVAL_SIZE)


def download_weights(model: nn.Module, args) -> None:
    del args
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Downloading official QAA weights requires 'huggingface_hub' and 'safetensors'. Install "
            "third_party/VPR-methods-evaluation/requirements.txt and retry."
        ) from exc
    weights_path = hf_hub_download(HF_REPO, HF_WEIGHTS_FILE, revision=HF_REVISION)
    load_model_state_dict(model, load_file(weights_path), strict=True)


ADAPTER = ModelAdapter(
    name="qaa",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
    download_weights=download_weights,
    configure_evaluation=configure_evaluation,
    build_evaluation_dataset=build_vpr_evaluation_dataset,
)


def register() -> None:
    register_model(ADAPTER)
