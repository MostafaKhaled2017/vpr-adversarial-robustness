import functools
import inspect
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from .config import IMAGENET_MEAN, IMAGENET_STD, amp_autocast, unwrap_model
from .losses import compute_attack_score, compute_listwise_scores
from .targets import RetrievalAttackBatch


def normalized_to_pixels(inputs: Tensor) -> Tensor:
    mean = IMAGENET_MEAN.to(device=inputs.device, dtype=inputs.dtype)
    std = IMAGENET_STD.to(device=inputs.device, dtype=inputs.dtype)
    return inputs * std + mean


def pixels_to_normalized(inputs: Tensor) -> Tensor:
    mean = IMAGENET_MEAN.to(device=inputs.device, dtype=inputs.dtype)
    std = IMAGENET_STD.to(device=inputs.device, dtype=inputs.dtype)
    return (inputs - mean) / std


# Attributes that carry an attack's perturbation budget. The perceptual backends keep the
# budget on the attack itself (bound, step) and again on the projection that enforces it;
# scaling one without the other would let the attack exceed its own curriculum budget.
SCALABLE_STRENGTH_ATTRIBUTES = ("bound", "step")


def attack_strength_scale(
    epoch_num: int,
    warmup_epochs: int,
    ramp_epochs: int,
    min_scale: float = 0.1,
) -> float:
    """Attack-strength multiplier for an epoch (Task 2.5).

    Jumping straight to full-strength attacks after the clean warm-up destabilises the
    listwise objective, so the budget ramps linearly from ``min_scale`` at the first
    adversarial epoch to 1.0 at ``warmup_epochs + ramp_epochs``. ``ramp_epochs=0`` disables
    the curriculum and returns full strength.
    """
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError("min_scale must be between 0 and 1 (inclusive)")
    if ramp_epochs <= 0:
        return 1.0

    position = (epoch_num - warmup_epochs) / float(ramp_epochs)
    fraction = min(1.0, max(0.0, position))
    return min_scale + (1.0 - min_scale) * fraction


class RetrievalAttackProxy(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        margin: float,
        mixed_precision: bool,
        device: str,
        objective: str = "hinge",
        listwise_tau: float = 0.05,
        listwise_k: int = 1,
    ):
        super().__init__()
        self.model = model
        self.margin = margin
        self.mixed_precision = mixed_precision
        self.device = device
        self.objective = objective
        self.listwise_tau = listwise_tau
        self.listwise_k = listwise_k
        self.current_targets: RetrievalAttackBatch | None = None

    def attack_scores(self, query_descriptors: Tensor) -> Tensor:
        """Per-query attack objective. Mirrors the defense loss so the two fight over the
        same quantity (Task 2.3): a hinge defense gets a hinge attack, a listwise defense
        gets the same smooth top-K surrogate."""
        targets = self.current_targets
        if self.objective == "listwise":
            return compute_listwise_scores(
                query_descriptors,
                targets.positive_descriptors,
                targets.negative_descriptors,
                self.listwise_tau,
                self.listwise_k,
                targets.positive_mask,
            )
        return compute_attack_score(
            query_descriptors,
            targets.positive_descriptors,
            targets.negative_descriptors,
            self.margin,
            targets.positive_mask,
        )

    def set_targets(self, targets: RetrievalAttackBatch) -> None:
        self.current_targets = targets

    def clear_targets(self) -> None:
        self.current_targets = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self.current_targets is None:
            raise RuntimeError("RetrievalAttackProxy.forward() called without targets.")

        normalized_inputs = pixels_to_normalized(inputs)
        with amp_autocast(False, self.device):
            query_descriptors = self.model(normalized_inputs, queryflag=0)
        query_descriptors = query_descriptors.float()
        attack_scores = self.attack_scores(query_descriptors)
        zeros = torch.zeros_like(attack_scores)
        return torch.stack([zeros, attack_scores], dim=1)


class UnsupportedAttack(nn.Module):
    def __init__(self, *args, attack_name: str, **kwargs):
        super().__init__()
        self.attack_name = attack_name

    def forward(self, inputs: Tensor, targets) -> Tensor:
        raise NotImplementedError(
            f"{self.attack_name} is not supported in train.py because it depends on "
            "classification-specific AutoAttack behavior."
        )


@contextmanager
def attack_generation_context(model: nn.Module):
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    original_training = [(module, module.training) for module in model.modules()]

    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        model.eval()
        yield
    finally:
        for module, training in original_training:
            module.train(training)
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)


def sync_uar_backend_resolution(backend: nn.Module, height: int, width: int) -> None:
    """Rebuild an advex_uar-backed attack at the actual input resolution.

    advex_uar hardcodes resol=224 for imagenet inside get_attack, baking it into
    the functools.partial stored as UARAttack.attack_fn, so any other input size
    crashes on the [1, 3, 224, 224] normalization tensors. mister_ed-backed
    attacks (StAdv, ReColorAdv) have no attack_fn partial and need no fix.
    """
    attack_fn = getattr(backend, "attack_fn", None)
    if not isinstance(attack_fn, functools.partial):
        return
    if height != width:
        raise ValueError(f"advex_uar attacks require square inputs, got {height}x{width}.")
    current_attack = getattr(backend, "attack", None)
    if current_attack is not None and current_attack.resol == height:
        return

    parameter_names = list(inspect.signature(attack_fn.func).parameters)
    resol_index = parameter_names.index("resol")
    positional_args = list(attack_fn.args)
    if resol_index < len(positional_args):
        positional_args[resol_index] = height
        keyword_args = attack_fn.keywords
    else:
        keyword_args = {**attack_fn.keywords, "resol": height}
    backend.attack = attack_fn.func(*positional_args, **keyword_args)

    # UARAttack.forward installs these lambdas only on lazy first creation, so
    # replicate them here; the wrapper otherwise re-applies 224-sized
    # ImagenetTransforms around the [0, 255] pixel interface.
    from perceptual_advex.utilities import LambdaLayer

    backend.attack.transform = LambdaLayer(lambda x: x / 255)
    backend.attack.inverse_transform = LambdaLayer(lambda x: x * 255)


class RetrievalAttackWrapper(nn.Module):
    backend_cls = None
    default_kwargs: Dict[str, object] = {}

    def __init__(
        self,
        model: nn.Module,
        margin: float = 0.1,
        mixed_precision: bool = False,
        device: str = "cuda",
        objective: str = "hinge",
        listwise_tau: float = 0.05,
        listwise_k: int = 1,
        **kwargs,
    ):
        super().__init__()
        if self.backend_cls is None:
            raise RuntimeError("backend_cls must be defined by subclasses.")
        self.model = model
        self.margin = margin
        self.device = device
        self.proxy_model = RetrievalAttackProxy(
            unwrap_model(model),
            margin,
            mixed_precision=mixed_precision,
            device=device,
            objective=objective,
            listwise_tau=listwise_tau,
            listwise_k=listwise_k,
        )
        merged_kwargs = dict(self.default_kwargs)
        merged_kwargs.update(kwargs)
        self.backend = self.backend_cls(self.proxy_model, **merged_kwargs)
        self.strength_scale = 1.0
        self._base_strength = self._capture_strength()

    def _strength_holders(self):
        holders = [self.backend]
        projection = getattr(self.backend, "projection", None)
        if projection is not None:
            holders.append(projection)
        return holders

    def _capture_strength(self):
        captured = []
        for holder in self._strength_holders():
            for name in SCALABLE_STRENGTH_ATTRIBUTES:
                value = getattr(holder, name, None)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                captured.append((holder, name, float(value)))
        return captured

    def set_strength_scale(self, scale: float) -> None:
        """Scale the perturbation budget, always relative to the configured value.

        Reading from the captured baseline rather than the current value keeps the
        curriculum from compounding across epochs.
        """
        self.strength_scale = float(scale)
        for holder, name, base_value in self._base_strength:
            setattr(holder, name, base_value * self.strength_scale)

    def forward(self, inputs: Tensor, targets: RetrievalAttackBatch) -> Tensor:
        self.proxy_model.set_targets(targets)
        fake_labels = torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device)
        try:
            # The mister_ed/advex backends are not autocast-safe (e.g. StAdv's
            # fp16 flow field crashes stAdv_norm), so force fp32 even when the
            # caller is inside a mixed-precision training region.
            with amp_autocast(False, self.device):
                pixel_inputs = normalized_to_pixels(inputs.float()).clamp(0.0, 1.0)
                sync_uar_backend_resolution(self.backend, inputs.shape[-2], inputs.shape[-1])
                with attack_generation_context(self.proxy_model.model):
                    adv_pixels = self.backend(pixel_inputs, fake_labels)
            adv_pixels = torch.nan_to_num(adv_pixels, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            return pixels_to_normalized(adv_pixels).detach()
        finally:
            self.proxy_model.clear_targets()


class NoAttack(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor, targets) -> Tensor:
        return inputs


def build_attack_namespace(model: nn.Module, args) -> Dict[str, object]:
    submodule_root = Path(__file__).resolve().parent.parent / "submodules" / "perceptual-advex"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))

    from perceptual_advex.attacks import (
        FogAttack as BackendFogAttack,
        JPEGLinfAttack as BackendJPEGLinfAttack,
        L1Attack as BackendL1Attack,
        L2Attack as BackendL2Attack,
        LinfAttack as BackendLinfAttack,
        ReColorAdvAttack as BackendReColorAdvAttack,
        StAdvAttack as BackendStAdvAttack,
    )
    from perceptual_advex.perceptual_attacks import (
        FastLagrangePerceptualAttack as BackendFastLagrangePerceptualAttack,
        LagrangePerceptualAttack as BackendLagrangePerceptualAttack,
        PerceptualPGDAttack as BackendPerceptualPGDAttack,
    )

    class _ConfiguredWrapper(RetrievalAttackWrapper):
        """Binds the run's defense objective to every attack defined below.

        The wrappers are created per call with ``args`` in scope, so the objective is
        injected once here rather than threaded through ten near-identical __init__s."""

        def __init__(self, *positional, **keyword):
            keyword.setdefault("objective", getattr(args, "defense_loss", "hinge"))
            keyword.setdefault("listwise_tau", getattr(args, "listwise_tau", 0.05))
            keyword.setdefault("listwise_k", getattr(args, "listwise_k", 1))
            super().__init__(*positional, **keyword)

    class LinfAttack(_ConfiguredWrapper):
        backend_cls = BackendLinfAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class L2Attack(_ConfiguredWrapper):
        backend_cls = BackendL2Attack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class L1Attack(_ConfiguredWrapper):
        backend_cls = BackendL1Attack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class JPEGLinfAttack(_ConfiguredWrapper):
        backend_cls = BackendJPEGLinfAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class FogAttack(_ConfiguredWrapper):
        backend_cls = BackendFogAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class StAdvAttack(_ConfiguredWrapper):
        backend_cls = BackendStAdvAttack

        def __init__(self, model, margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                **kwargs,
            )

    class ReColorAdvAttack(_ConfiguredWrapper):
        backend_cls = BackendReColorAdvAttack

        def __init__(self, model, margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                **kwargs,
            )

    class FastLagrangePerceptualAttack(_ConfiguredWrapper):
        backend_cls = BackendFastLagrangePerceptualAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("train.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    class PerceptualPGDAttack(_ConfiguredWrapper):
        backend_cls = BackendPerceptualPGDAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("train.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    class LagrangePerceptualAttack(_ConfiguredWrapper):
        backend_cls = BackendLagrangePerceptualAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("train.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    return {
        "model": unwrap_model(model),
        "NoAttack": NoAttack,
        "LinfAttack": LinfAttack,
        "L2Attack": L2Attack,
        "L1Attack": L1Attack,
        "JPEGLinfAttack": JPEGLinfAttack,
        "FogAttack": FogAttack,
        "StAdvAttack": StAdvAttack,
        "ReColorAdvAttack": ReColorAdvAttack,
        "FastLagrangePerceptualAttack": FastLagrangePerceptualAttack,
        "PerceptualPGDAttack": PerceptualPGDAttack,
        "LagrangePerceptualAttack": LagrangePerceptualAttack,
        "AutoAttack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoAttack", **k),
        "AutoLinfAttack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoLinfAttack", **k),
        "AutoL2Attack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoL2Attack", **k),
        "math": math,
        "torch": torch,
    }


def instantiate_attacks(model: nn.Module, attack_strings: Sequence[str], args):
    namespace = build_attack_namespace(model, args)
    attacks = []
    for attack_string in attack_strings:
        attack = eval(attack_string, {"__builtins__": {}}, namespace)
        attacks.append(attack)
    return attacks
