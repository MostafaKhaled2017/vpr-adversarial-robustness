from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn

from .config import get_normalized_bounds
from .losses import compute_attack_score
from .targets import RetrievalAttackBatch


RankNorm = Literal["linf", "l2"]


@dataclass(frozen=True)
class RankAttackConfig:
    epsilon: float
    steps: int = 20
    restarts: int = 1
    step_size: Optional[float] = None
    norm: RankNorm = "linf"
    margin: float = 0.1
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.restarts < 1:
            raise ValueError("restarts must be at least 1")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size must be positive when provided")
        if self.norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")

    @property
    def resolved_step_size(self) -> float:
        if self.step_size is not None:
            return float(self.step_size)
        if self.steps == 0:
            return 0.0
        return 2.0 * float(self.epsilon) / float(self.steps)


@dataclass
class RankAttackResult:
    adversarial: Tensor
    metadata: Dict[str, Tensor]


def _as_batch_shape(values: Tensor, reference: Tensor) -> Tensor:
    return values.view(values.shape[0], *([1] * (reference.ndim - 1)))


def _l2_norm(values: Tensor) -> Tensor:
    return values.flatten(1).norm(p=2, dim=1)


def project_linf(clean_inputs: Tensor, candidate_inputs: Tensor, epsilon: float) -> Tensor:
    delta = (candidate_inputs - clean_inputs).clamp(min=-epsilon, max=epsilon)
    return clean_inputs + delta


def project_l2(clean_inputs: Tensor, candidate_inputs: Tensor, epsilon: float) -> Tensor:
    delta = candidate_inputs - clean_inputs
    flat_delta = delta.flatten(1)
    norms = flat_delta.norm(p=2, dim=1).clamp_min(1e-12)
    scale = torch.clamp(torch.full_like(norms, float(epsilon)) / norms, max=1.0)
    projected = flat_delta * scale.unsqueeze(1)
    return clean_inputs + projected.view_as(delta)


class RankPGDAttack(nn.Module):
    def __init__(self, model: nn.Module, config: RankAttackConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, inputs: Tensor, targets: RetrievalAttackBatch) -> RankAttackResult:
        clean_inputs = inputs.detach()
        batch_size = clean_inputs.shape[0]
        best_adv = clean_inputs.clone()
        best_loss = torch.full((batch_size,), -torch.inf, device=clean_inputs.device)
        best_restart = torch.full((batch_size,), -1, dtype=torch.long, device=clean_inputs.device)
        final_loss = torch.zeros(batch_size, device=clean_inputs.device)

        was_training = self.model.training
        self.model.eval()
        try:
            for restart_index in range(self.config.restarts):
                adv_inputs = self._initial_inputs(clean_inputs, restart_index)
                initial_loss = self._rank_loss_per_sample(adv_inputs, targets).detach()
                best_adv, best_loss, best_restart = self._update_best(
                    adv_inputs,
                    initial_loss,
                    restart_index,
                    best_adv,
                    best_loss,
                    best_restart,
                )

                for _ in range(self.config.steps):
                    adv_inputs = adv_inputs.detach().requires_grad_(True)
                    losses = self._rank_loss_per_sample(adv_inputs, targets)
                    objective = losses.mean()
                    gradient = torch.autograd.grad(objective, adv_inputs)[0]

                    with torch.no_grad():
                        adv_inputs = self._step(clean_inputs, adv_inputs.detach(), gradient, self._step_sizes(clean_inputs))
                        final_loss = self._rank_loss_per_sample(adv_inputs, targets).detach()
                        best_adv, best_loss, best_restart = self._update_best(
                            adv_inputs,
                            final_loss,
                            restart_index,
                            best_adv,
                            best_loss,
                            best_restart,
                        )
        finally:
            self.model.train(was_training)

        metadata = {
            "final_loss": final_loss.detach().cpu(),
            "best_loss": best_loss.detach().cpu(),
            "perturbation_norm": self._perturbation_norm(clean_inputs, best_adv).detach().cpu(),
            "restart_index": best_restart.detach().cpu(),
        }
        return RankAttackResult(adversarial=best_adv.detach(), metadata=metadata)

    def _rank_loss_per_sample(self, inputs: Tensor, targets: RetrievalAttackBatch) -> Tensor:
        descriptors = self.model(inputs, queryflag=0).float()
        positive_descriptors = targets.positive_descriptors.to(device=inputs.device, dtype=descriptors.dtype)
        negative_descriptors = targets.negative_descriptors.to(device=inputs.device, dtype=descriptors.dtype)
        return torch.relu(
            compute_attack_score(
                descriptors,
                positive_descriptors,
                negative_descriptors,
                self.config.margin,
            )
        )

    def _initial_inputs(self, clean_inputs: Tensor, restart_index: int) -> Tensor:
        if restart_index == 0 or self.config.epsilon == 0:
            return self._clamp_to_valid_range(clean_inputs)
        if self.config.norm == "linf":
            noise = torch.empty_like(clean_inputs).uniform_(-self.config.epsilon, self.config.epsilon)
            return self._clamp_to_valid_range(clean_inputs + noise)

        flat_noise = torch.randn_like(clean_inputs).flatten(1)
        noise_norm = flat_noise.norm(p=2, dim=1).clamp_min(1e-12)
        radius = torch.rand(clean_inputs.shape[0], device=clean_inputs.device, dtype=clean_inputs.dtype) * self.config.epsilon
        scaled_noise = flat_noise / noise_norm.unsqueeze(1) * radius.unsqueeze(1)
        return self._clamp_to_valid_range(clean_inputs + scaled_noise.view_as(clean_inputs))

    def _step(self, clean_inputs: Tensor, adv_inputs: Tensor, gradient: Tensor, step_sizes: Tensor) -> Tensor:
        if self.config.norm == "linf":
            step = _as_batch_shape(step_sizes, adv_inputs) * gradient.sign()
            stepped = adv_inputs + step
            projected = project_linf(clean_inputs, stepped, self.config.epsilon)
        else:
            flat_gradient = gradient.flatten(1)
            gradient_norm = flat_gradient.norm(p=2, dim=1).clamp_min(1e-12)
            normalized_gradient = flat_gradient / gradient_norm.unsqueeze(1)
            step = normalized_gradient * step_sizes.unsqueeze(1)
            stepped = adv_inputs + step.view_as(adv_inputs)
            projected = project_l2(clean_inputs, stepped, self.config.epsilon)
        return self._clamp_to_valid_range(projected)

    def _step_sizes(self, inputs: Tensor) -> Tensor:
        return torch.full(
            (inputs.shape[0],),
            self.config.resolved_step_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )

    def _update_best(
        self,
        adv_inputs: Tensor,
        losses: Tensor,
        restart_index: int,
        best_adv: Tensor,
        best_loss: Tensor,
        best_restart: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        should_update = losses > best_loss
        if should_update.any():
            best_adv = best_adv.clone()
            best_loss = best_loss.clone()
            best_restart = best_restart.clone()
            best_adv[should_update] = adv_inputs.detach()[should_update]
            best_loss[should_update] = losses.detach()[should_update]
            best_restart[should_update] = restart_index
        return best_adv, best_loss, best_restart

    def _perturbation_norm(self, clean_inputs: Tensor, adv_inputs: Tensor) -> Tensor:
        delta = adv_inputs - clean_inputs
        if self.config.norm == "linf":
            return delta.flatten(1).abs().max(dim=1).values
        return _l2_norm(delta)

    def _clamp_to_valid_range(self, inputs: Tensor) -> Tensor:
        min_value, max_value = get_normalized_bounds(self.config.device)
        min_value = min_value.to(device=inputs.device, dtype=inputs.dtype)
        max_value = max_value.to(device=inputs.device, dtype=inputs.dtype)
        return torch.max(torch.min(inputs, max_value), min_value)


class RankAPGDLinfAttack(RankPGDAttack):
    def __init__(self, model: nn.Module, config: RankAttackConfig):
        if config.norm != "linf":
            raise ValueError("RankAPGDLinfAttack supports only Linf constraints")
        super().__init__(model, config)

    def forward(self, inputs: Tensor, targets: RetrievalAttackBatch) -> RankAttackResult:
        clean_inputs = inputs.detach()
        batch_size = clean_inputs.shape[0]
        best_adv = clean_inputs.clone()
        best_loss = torch.full((batch_size,), -torch.inf, device=clean_inputs.device)
        best_restart = torch.full((batch_size,), -1, dtype=torch.long, device=clean_inputs.device)
        final_loss = torch.zeros(batch_size, device=clean_inputs.device)
        adaptation_window = max(1, self.config.steps // 4)

        was_training = self.model.training
        self.model.eval()
        try:
            for restart_index in range(self.config.restarts):
                step_sizes = self._step_sizes(clean_inputs)
                previous_window_best = torch.full((batch_size,), -torch.inf, device=clean_inputs.device)
                adv_inputs = self._initial_inputs(clean_inputs, restart_index)

                initial_loss = self._rank_loss_per_sample(adv_inputs, targets).detach()
                restart_best_loss = initial_loss.clone()
                best_adv, best_loss, best_restart = self._update_best(
                    adv_inputs,
                    initial_loss,
                    restart_index,
                    best_adv,
                    best_loss,
                    best_restart,
                )

                for step_index in range(self.config.steps):
                    adv_inputs = adv_inputs.detach().requires_grad_(True)
                    losses = self._rank_loss_per_sample(adv_inputs, targets)
                    gradient = torch.autograd.grad(losses.mean(), adv_inputs)[0]

                    with torch.no_grad():
                        adv_inputs = self._step(clean_inputs, adv_inputs.detach(), gradient, step_sizes)
                        final_loss = self._rank_loss_per_sample(adv_inputs, targets).detach()
                        best_adv, best_loss, best_restart = self._update_best(
                            adv_inputs,
                            final_loss,
                            restart_index,
                            best_adv,
                            best_loss,
                            best_restart,
                        )

                        if (step_index + 1) % adaptation_window == 0:
                            restart_best_loss = torch.maximum(restart_best_loss, final_loss)
                            no_improvement = restart_best_loss <= previous_window_best + 1e-12
                            step_sizes = torch.where(no_improvement, step_sizes * 0.5, step_sizes)
                            previous_window_best = restart_best_loss.clone()
        finally:
            self.model.train(was_training)

        metadata = {
            "final_loss": final_loss.detach().cpu(),
            "best_loss": best_loss.detach().cpu(),
            "perturbation_norm": self._perturbation_norm(clean_inputs, best_adv).detach().cpu(),
            "restart_index": best_restart.detach().cpu(),
        }
        return RankAttackResult(adversarial=best_adv.detach(), metadata=metadata)
