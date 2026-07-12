import importlib.util
from pathlib import Path
from types import ModuleType

from torch import nn

from ..checkpoints import load_model_weights
from .registry import ModelAdapter, ModelBundle, register_model


SUPPORTED_DIMENSIONS = (128, 512, 4096)
DEFAULT_DIMENSION = 4096
UPSTREAM_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "VPR-methods-evaluation"
    / "vpr_models"
    / "mixvpr.py"
)


def add_arguments(parser) -> None:
    parser.add_argument(
        "--mixvpr_descriptors_dimension",
        type=int,
        choices=SUPPORTED_DIMENSIONS,
        default=None,
        help="MixVPR descriptor dimension. Defaults to 4096.",
    )
    parser.add_argument(
        "--mixvpr_freeze_backbone",
        action="store_true",
        help="Freeze the MixVPR ResNet50 backbone and train only the aggregator.",
    )


def resolve_descriptor_dimension(args) -> int:
    dimension = args.mixvpr_descriptors_dimension
    if dimension is None:
        return DEFAULT_DIMENSION
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"Unsupported MixVPR descriptor dimension {dimension}. "
            f"Supported dimensions are {SUPPORTED_DIMENSIONS}."
        )
    return dimension


def load_upstream_module() -> ModuleType:
    if not UPSTREAM_MODULE_PATH.is_file():
        raise FileNotFoundError(f"MixVPR implementation was not found at {UPSTREAM_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("vpr_methods_evaluation_mixvpr", UPSTREAM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {UPSTREAM_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "gdown":
            raise ModuleNotFoundError(
                "Loading the vendored MixVPR module requires 'gdown'. Install "
                "third_party/VPR-methods-evaluation/requirements.txt and retry."
            ) from exc
        raise
    return module


def build_model_config(upstream: ModuleType, descriptor_dim: int):
    try:
        _, _, out_channels, out_rows = upstream.MODELS_INFO[descriptor_dim]
    except KeyError as exc:
        raise ValueError(f"The vendored MixVPR implementation does not support dimension {descriptor_dim}") from exc
    return {
        "in_channels": 1024,
        "in_h": 20,
        "in_w": 20,
        "out_channels": out_channels,
        "mix_depth": 4,
        "mlp_ratio": 1,
        "out_rows": out_rows,
    }


def build_descriptor_model_class(upstream: ModuleType):
    class DescriptorModel(upstream.MixVPRModel):
        def forward(self, inputs, queryflag=0):
            del queryflag
            return super().forward(inputs)

    return DescriptorModel


def build(args) -> ModelBundle:
    upstream = load_upstream_module()
    descriptor_dim = resolve_descriptor_dimension(args)
    model_config = build_model_config(upstream, descriptor_dim)
    descriptor_model = build_descriptor_model_class(upstream)
    model = descriptor_model(agg_config=model_config)
    if args.mixvpr_freeze_backbone:
        model.backbone.requires_grad_(False)
    return ModelBundle(model=model, descriptor_dim=descriptor_dim)


def load_weights(model: nn.Module, checkpoint_path: str, args) -> None:
    load_model_weights(model, checkpoint_path, map_location=args.device, strict=True)


def official_checkpoint_path(upstream: ModuleType, descriptor_dim: int) -> Path:
    _, filename, _, _ = upstream.MODELS_INFO[descriptor_dim]
    return UPSTREAM_MODULE_PATH.parents[1] / "trained_models" / "mixvpr" / filename


def download_weights(model: nn.Module, args) -> None:
    upstream = load_upstream_module()
    descriptor_dim = resolve_descriptor_dimension(args)
    url, _, _, _ = upstream.MODELS_INFO[descriptor_dim]
    checkpoint_path = official_checkpoint_path(upstream, descriptor_dim)

    if not checkpoint_path.is_file():
        try:
            import gdown
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Downloading official MixVPR weights requires 'gdown'. Install "
                "third_party/VPR-methods-evaluation/requirements.txt and retry."
            ) from exc
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = gdown.download(url=url, output=str(checkpoint_path), fuzzy=True)
        if downloaded_path is None or not checkpoint_path.is_file():
            raise RuntimeError(f"Failed to download official MixVPR weights from {url}")

    load_model_weights(model, str(checkpoint_path), map_location=args.device, strict=True)


ADAPTER = ModelAdapter(
    name="mixvpr",
    add_arguments=add_arguments,
    build=build,
    load_weights=load_weights,
    download_weights=download_weights,
)


def register() -> None:
    register_model(ADAPTER)
