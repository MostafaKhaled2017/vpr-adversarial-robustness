from dataclasses import dataclass
from typing import Callable, Dict, Optional

from torch import nn


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    descriptor_dim: int


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    add_arguments: Callable[[object], None]
    build: Callable[[object], ModelBundle]
    load_weights: Callable[[nn.Module, str, object], None]
    download_weights: Optional[Callable[[nn.Module, object], None]] = None


_REGISTRY: Dict[str, ModelAdapter] = {}


def register_model(adapter: ModelAdapter) -> None:
    if adapter.name in _REGISTRY:
        if _REGISTRY[adapter.name] == adapter:
            return
        raise ValueError(f"Model adapter {adapter.name!r} is already registered")
    _REGISTRY[adapter.name] = adapter


def get_model_adapter(name: str) -> ModelAdapter:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Available models: {model_names()}") from exc


def model_names():
    return tuple(sorted(_REGISTRY))


def add_model_arguments(parser) -> None:
    for adapter in _REGISTRY.values():
        adapter.add_arguments(parser)
