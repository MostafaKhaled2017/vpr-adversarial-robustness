from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

import torch
from torch import Tensor, nn

from .config import denormalize_imagenet, get_normalized_bounds
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
    audit: bool = False
    trace_query_indices: Optional[Sequence[int]] = None

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
    traces: Optional[list[Dict[str, object]]] = None


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
        self._trace_query_index_set = (
            {int(index) for index in config.trace_query_indices}
            if config.trace_query_indices is not None
            else None
        )

    def forward(self, inputs: Tensor, targets: RetrievalAttackBatch) -> RankAttackResult:
        clean_inputs = inputs.detach()
        batch_size = clean_inputs.shape[0]
        best_adv = clean_inputs.clone()
        best_loss = torch.full((batch_size,), -torch.inf, device=clean_inputs.device)
        best_restart = torch.full((batch_size,), -1, dtype=torch.long, device=clean_inputs.device)
        final_loss = torch.zeros(batch_size, device=clean_inputs.device)
        gradient_norm_sum = torch.zeros(batch_size, device=clean_inputs.device)
        gradient_norm_max = torch.zeros(batch_size, device=clean_inputs.device)
        gradient_norm_count = 0
        traces: list[Dict[str, object]] = []

        was_training = self.model.training
        self.model.eval()
        try:
            clean_audit = self._audit_components(clean_inputs, targets) if self.config.audit else {}
            for restart_index in range(self.config.restarts):
                adv_inputs = self._initial_inputs(clean_inputs, restart_index)
                initial_loss, initial_positive, initial_hard_negative, initial_descriptors = self._rank_components(
                    adv_inputs,
                    targets,
                )
                initial_loss = initial_loss.detach()
                initial_best_flags = initial_loss > best_loss
                best_adv, best_loss, best_restart = self._update_best(
                    adv_inputs,
                    initial_loss,
                    restart_index,
                    best_adv,
                    best_loss,
                    best_restart,
                )
                self._append_traces(
                    traces,
                    targets,
                    clean_inputs,
                    adv_inputs,
                    initial_descriptors,
                    initial_loss,
                    initial_positive,
                    initial_hard_negative,
                    restart_index,
                    0,
                    initial_best_flags,
                )

                for step_index in range(self.config.steps):
                    adv_inputs = adv_inputs.detach().requires_grad_(True)
                    losses = self._rank_loss_per_sample(adv_inputs, targets)
                    objective = losses.mean()
                    gradient = torch.autograd.grad(objective, adv_inputs)[0]
                    if self.config.audit:
                        gradient_norm = _l2_norm(gradient.detach())
                        gradient_norm_sum = gradient_norm_sum + gradient_norm
                        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm)
                        gradient_norm_count += 1

                    with torch.no_grad():
                        adv_inputs = self._step(clean_inputs, adv_inputs.detach(), gradient, self._step_sizes(clean_inputs))
                        final_loss, final_positive, final_hard_negative, final_descriptors = self._rank_components(
                            adv_inputs,
                            targets,
                        )
                        final_loss = final_loss.detach()
                        final_best_flags = final_loss > best_loss
                        best_adv, best_loss, best_restart = self._update_best(
                            adv_inputs,
                            final_loss,
                            restart_index,
                            best_adv,
                            best_loss,
                            best_restart,
                        )
                        self._append_traces(
                            traces,
                            targets,
                            clean_inputs,
                            adv_inputs,
                            final_descriptors,
                            final_loss,
                            final_positive,
                            final_hard_negative,
                            restart_index,
                            step_index + 1,
                            final_best_flags,
                        )
        finally:
            self.model.train(was_training)

        metadata = {
            "final_loss": final_loss.detach().cpu(),
            "best_loss": best_loss.detach().cpu(),
            "perturbation_norm": self._perturbation_norm(clean_inputs, best_adv).detach().cpu(),
            "restart_index": best_restart.detach().cpu(),
        }
        if self.config.audit:
            metadata.update(
                self._audit_metadata(
                    clean_inputs,
                    best_adv,
                    targets,
                    clean_audit,
                    gradient_norm_sum,
                    gradient_norm_max,
                    gradient_norm_count,
                )
            )
        return RankAttackResult(adversarial=best_adv.detach(), metadata=metadata, traces=traces or None)

    def _rank_loss_per_sample(self, inputs: Tensor, targets: RetrievalAttackBatch) -> Tensor:
        return self._rank_components(inputs, targets)[0]

    def _rank_components(self, inputs: Tensor, targets: RetrievalAttackBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        descriptors = self.model(inputs, queryflag=0).float()
        positive_descriptors = targets.positive_descriptors.detach().to(device=inputs.device, dtype=descriptors.dtype)
        negative_descriptors = targets.negative_descriptors.detach().to(device=inputs.device, dtype=descriptors.dtype)
        positive_distance = torch.norm(descriptors - positive_descriptors, p=2, dim=1)
        negative_distance = torch.norm(descriptors.unsqueeze(1) - negative_descriptors, p=2, dim=2)
        hard_negative_distance = negative_distance.min(dim=1).values
        score = self.config.margin + positive_distance - hard_negative_distance
        return torch.relu(score), positive_distance, hard_negative_distance, descriptors

    def _audit_components(self, inputs: Tensor, targets: RetrievalAttackBatch) -> Dict[str, Tensor]:
        with torch.no_grad():
            loss, positive_distance, hard_negative_distance, _ = self._rank_components(inputs, targets)
        return {
            "loss": loss.detach(),
            "positive_distance": positive_distance.detach(),
            "hard_negative_distance": hard_negative_distance.detach(),
        }

    def _audit_metadata(
        self,
        clean_inputs: Tensor,
        best_adv: Tensor,
        targets: RetrievalAttackBatch,
        clean_audit: Dict[str, Tensor],
        gradient_norm_sum: Tensor,
        gradient_norm_max: Tensor,
        gradient_norm_count: int,
    ) -> Dict[str, Tensor]:
        attacked_audit = self._audit_components(best_adv, targets)
        denormalized = denormalize_imagenet(best_adv.detach())
        gradient_norm = (
            gradient_norm_sum / float(gradient_norm_count)
            if gradient_norm_count > 0
            else torch.zeros_like(gradient_norm_sum)
        )
        return {
            "initial_loss": clean_audit["loss"].detach().cpu(),
            "gradient_norm": gradient_norm.detach().cpu(),
            "gradient_norm_max": gradient_norm_max.detach().cpu(),
            "positive_distance_before": clean_audit["positive_distance"].detach().cpu(),
            "positive_distance_after": attacked_audit["positive_distance"].detach().cpu(),
            "hard_negative_distance_before": clean_audit["hard_negative_distance"].detach().cpu(),
            "hard_negative_distance_after": attacked_audit["hard_negative_distance"].detach().cpu(),
            "denormalized_min": denormalized.flatten(1).min(dim=1).values.detach().cpu(),
            "denormalized_max": denormalized.flatten(1).max(dim=1).values.detach().cpu(),
        }

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

    def _perturbation_linf_norm(self, clean_inputs: Tensor, adv_inputs: Tensor) -> Tensor:
        return (adv_inputs - clean_inputs).flatten(1).abs().max(dim=1).values

    def _perturbation_raw_linf_norm(self, clean_inputs: Tensor, adv_inputs: Tensor) -> Tensor:
        raw_delta = denormalize_imagenet(adv_inputs) - denormalize_imagenet(clean_inputs)
        return raw_delta.flatten(1).abs().max(dim=1).values

    def _append_traces(
        self,
        traces: list[Dict[str, object]],
        targets: RetrievalAttackBatch,
        clean_inputs: Tensor,
        adv_inputs: Tensor,
        descriptors: Tensor,
        losses: Tensor,
        positive_distances: Tensor,
        hard_negative_distances: Tensor,
        restart_index: int,
        step_index: int,
        best_flags: Tensor,
    ) -> None:
        if self._trace_query_index_set is None:
            return

        query_indices = targets.query_indices.detach().cpu().tolist()
        linf_norms = self._perturbation_linf_norm(clean_inputs, adv_inputs).detach().cpu()
        raw_linf_norms = self._perturbation_raw_linf_norm(clean_inputs, adv_inputs).detach().cpu()
        cpu_losses = losses.detach().cpu()
        cpu_positive = positive_distances.detach().cpu()
        cpu_hard_negative = hard_negative_distances.detach().cpu()
        cpu_descriptors = descriptors.detach().cpu()
        cpu_best_flags = best_flags.detach().cpu()

        for batch_index, query_index in enumerate(query_indices):
            original_query_index = int(query_index)
            if original_query_index not in self._trace_query_index_set:
                continue
            traces.append(
                {
                    "query_index": original_query_index,
                    "restart": int(restart_index),
                    "step": int(step_index),
                    "loss": float(cpu_losses[batch_index].item()),
                    "positive_distance": float(cpu_positive[batch_index].item()),
                    "hard_negative_distance": float(cpu_hard_negative[batch_index].item()),
                    "perturbation_linf_normalized": float(linf_norms[batch_index].item()),
                    "perturbation_linf_raw": float(raw_linf_norms[batch_index].item()),
                    "best_so_far": bool(cpu_best_flags[batch_index].item()),
                    "descriptor": cpu_descriptors[batch_index].clone(),
                }
            )

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
        gradient_norm_sum = torch.zeros(batch_size, device=clean_inputs.device)
        gradient_norm_max = torch.zeros(batch_size, device=clean_inputs.device)
        gradient_norm_count = 0
        traces: list[Dict[str, object]] = []
        adaptation_window = max(1, self.config.steps // 4)

        was_training = self.model.training
        self.model.eval()
        try:
            clean_audit = self._audit_components(clean_inputs, targets) if self.config.audit else {}
            for restart_index in range(self.config.restarts):
                step_sizes = self._step_sizes(clean_inputs)
                previous_window_best = torch.full((batch_size,), -torch.inf, device=clean_inputs.device)
                adv_inputs = self._initial_inputs(clean_inputs, restart_index)

                initial_loss, initial_positive, initial_hard_negative, initial_descriptors = self._rank_components(
                    adv_inputs,
                    targets,
                )
                initial_loss = initial_loss.detach()
                restart_best_loss = initial_loss.clone()
                initial_best_flags = initial_loss > best_loss
                best_adv, best_loss, best_restart = self._update_best(
                    adv_inputs,
                    initial_loss,
                    restart_index,
                    best_adv,
                    best_loss,
                    best_restart,
                )
                self._append_traces(
                    traces,
                    targets,
                    clean_inputs,
                    adv_inputs,
                    initial_descriptors,
                    initial_loss,
                    initial_positive,
                    initial_hard_negative,
                    restart_index,
                    0,
                    initial_best_flags,
                )

                for step_index in range(self.config.steps):
                    adv_inputs = adv_inputs.detach().requires_grad_(True)
                    losses = self._rank_loss_per_sample(adv_inputs, targets)
                    gradient = torch.autograd.grad(losses.mean(), adv_inputs)[0]
                    if self.config.audit:
                        gradient_norm = _l2_norm(gradient.detach())
                        gradient_norm_sum = gradient_norm_sum + gradient_norm
                        gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm)
                        gradient_norm_count += 1

                    with torch.no_grad():
                        adv_inputs = self._step(clean_inputs, adv_inputs.detach(), gradient, step_sizes)
                        final_loss, final_positive, final_hard_negative, final_descriptors = self._rank_components(
                            adv_inputs,
                            targets,
                        )
                        final_loss = final_loss.detach()
                        final_best_flags = final_loss > best_loss
                        best_adv, best_loss, best_restart = self._update_best(
                            adv_inputs,
                            final_loss,
                            restart_index,
                            best_adv,
                            best_loss,
                            best_restart,
                        )
                        self._append_traces(
                            traces,
                            targets,
                            clean_inputs,
                            adv_inputs,
                            final_descriptors,
                            final_loss,
                            final_positive,
                            final_hard_negative,
                            restart_index,
                            step_index + 1,
                            final_best_flags,
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
        if self.config.audit:
            metadata.update(
                self._audit_metadata(
                    clean_inputs,
                    best_adv,
                    targets,
                    clean_audit,
                    gradient_norm_sum,
                    gradient_norm_max,
                    gradient_norm_count,
                )
            )
        return RankAttackResult(adversarial=best_adv.detach(), metadata=metadata, traces=traces or None)
