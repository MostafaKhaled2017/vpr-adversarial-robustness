from .registry import ModelAdapter, ModelBundle, add_model_arguments, get_model_adapter, model_names, register_model


def register_builtin_models() -> None:
    from . import boq, mixvpr, supervlad

    boq.register()
    mixvpr.register()
    supervlad.register()


register_builtin_models()

__all__ = [
    "ModelAdapter",
    "ModelBundle",
    "add_model_arguments",
    "get_model_adapter",
    "model_names",
    "register_model",
]
