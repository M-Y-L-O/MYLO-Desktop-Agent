from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from Core.ModelTransition import ModelTransition, TransitionEvidence


@dataclass(frozen=True)
class RecoveryExperimentConfig:
    max_steps: int = 100
    evaluation_interval: int = 10
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    recovery_tolerance: float = 0.0
    early_stopping_patience_evaluations: int = 3
    minimum_relative_improvement: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps cannot be negative.")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.recovery_tolerance < 0:
            raise ValueError("recovery_tolerance cannot be negative.")
        if self.early_stopping_patience_evaluations < 0:
            raise ValueError("early_stopping_patience_evaluations cannot be negative.")
        if not 0.0 <= self.minimum_relative_improvement < 1.0:
            raise ValueError("minimum_relative_improvement must be in [0, 1).")
        if self.optimizer.lower() not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "evaluation_interval": self.evaluation_interval,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.lower(),
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "recovery_tolerance": self.recovery_tolerance,
            "early_stopping_patience_evaluations": self.early_stopping_patience_evaluations,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass
class RecoveryCheckpoint:
    step: int
    examples_seen: int
    validation_loss: float
    validation_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "examples_seen": self.examples_seen,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }


@dataclass
class RecoveryExperimentResult:
    config: RecoveryExperimentConfig
    parent_validation_loss: float
    parent_validation_accuracy: Optional[float]
    child_initial_loss: float
    child_initial_accuracy: Optional[float]
    checkpoints: List[RecoveryCheckpoint] = field(default_factory=list)
    recovery_step: Optional[int] = None
    optimizer_steps: int = 0
    examples_seen: int = 0
    elapsed_seconds: float = 0.0
    selected_step: int = 0
    best_validation_loss: Optional[float] = None
    stopped_early: bool = False
    stop_reason: Optional[str] = None

    @property
    def recovered(self) -> bool:
        return self.recovery_step is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "parent_validation_loss": self.parent_validation_loss,
            "parent_validation_accuracy": self.parent_validation_accuracy,
            "child_initial_loss": self.child_initial_loss,
            "child_initial_accuracy": self.child_initial_accuracy,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "recovery_step": self.recovery_step,
            "recovered": self.recovered,
            "optimizer_steps": self.optimizer_steps,
            "examples_seen": self.examples_seen,
            "elapsed_seconds": self.elapsed_seconds,
            "selected_step": self.selected_step,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


class TransitionRecoveryHarness:
    @classmethod
    def run(
        cls,
        parent_model: nn.Module,
        child_model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        criterion: nn.Module,
        config: RecoveryExperimentConfig,
        *,
        transition: Optional[ModelTransition] = None,
    ) -> RecoveryExperimentResult:
        config.validate()
        cls.seed_all(config.seed)
        device = torch.device(config.device)
        parent_model.to(device)
        child_model.to(device)

        started = time.perf_counter()
        parent_loss, parent_accuracy = cls.evaluate(
            parent_model, validation_loader, criterion, device
        )
        child_loss, child_accuracy = cls.evaluate(
            child_model, validation_loader, criterion, device
        )
        result = RecoveryExperimentResult(
            config=config,
            parent_validation_loss=parent_loss,
            parent_validation_accuracy=parent_accuracy,
            child_initial_loss=child_loss,
            child_initial_accuracy=child_accuracy,
            checkpoints=[
                RecoveryCheckpoint(
                    step=0,
                    examples_seen=0,
                    validation_loss=child_loss,
                    validation_accuracy=child_accuracy,
                )
            ],
        )
        if cls._has_recovered(child_loss, parent_loss, config.recovery_tolerance):
            result.recovery_step = 0
        best_validation_loss = child_loss
        best_step = 0
        best_state = copy.deepcopy(child_model.state_dict())

        optimizer = cls._optimizer(child_model, config)
        child_was_training = child_model.training
        steps = 0
        examples_seen = 0
        evaluations_without_improvement = 0
        stop_requested = False

        try:
            while steps < config.max_steps and not stop_requested:
                progressed = False
                for batch in train_loader:
                    progressed = True
                    inputs, targets = cls._split_batch(batch)
                    inputs = cls._to_device(inputs, device)
                    targets = cls._to_device(targets, device)

                    child_model.train()
                    optimizer.zero_grad(set_to_none=True)
                    output = cls._primary_tensor(cls._call_model(child_model, inputs))
                    if output is None:
                        raise ValueError("Child model did not produce a tensor output.")
                    loss = criterion(output, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite recovery loss encountered at step {steps + 1}."
                        )
                    loss.backward()
                    optimizer.step()

                    steps += 1
                    examples_seen += cls._batch_size(targets)
                    should_evaluate = (
                        steps % config.evaluation_interval == 0
                        or steps == config.max_steps
                    )
                    if should_evaluate:
                        validation_loss, validation_accuracy = cls.evaluate(
                            child_model, validation_loader, criterion, device
                        )
                        result.checkpoints.append(
                            RecoveryCheckpoint(
                                step=steps,
                                examples_seen=examples_seen,
                                validation_loss=validation_loss,
                                validation_accuracy=validation_accuracy,
                            )
                        )
                        meaningful_improvement = validation_loss < (
                            best_validation_loss
                            * (1.0 - config.minimum_relative_improvement)
                        )
                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            best_step = steps
                            best_state = copy.deepcopy(child_model.state_dict())
                        if meaningful_improvement:
                            evaluations_without_improvement = 0
                        else:
                            evaluations_without_improvement += 1
                        if result.recovery_step is None and cls._has_recovered(
                            validation_loss,
                            parent_loss,
                            config.recovery_tolerance,
                        ):
                            result.recovery_step = steps
                        if (
                            config.early_stopping_patience_evaluations > 0
                            and evaluations_without_improvement
                            >= config.early_stopping_patience_evaluations
                            and steps < config.max_steps
                        ):
                            result.stopped_early = True
                            result.stop_reason = "validation_plateau"
                            stop_requested = True
                            break
                    if steps >= config.max_steps:
                        break
                if not progressed:
                    raise ValueError("Training loader produced no batches.")
        finally:
            child_model.train(child_was_training)

        result.optimizer_steps = steps
        result.examples_seen = examples_seen
        result.elapsed_seconds = round(time.perf_counter() - started, 8)
        child_model.load_state_dict(best_state)
        result.selected_step = best_step
        result.best_validation_loss = best_validation_loss
        if result.stop_reason is None:
            result.stop_reason = (
                "no_training_budget" if config.max_steps == 0 else "max_steps_reached"
            )

        if transition is not None:
            if transition.evidence is None:
                transition.evidence = TransitionEvidence()
            transition.evidence.recovery = result.to_dict()
            transition.outcome.update(
                {
                    "recovered": result.recovered,
                    "recovery_step": result.recovery_step,
                    "final_validation_loss": best_validation_loss,
                    "selected_step": best_step,
                    "optimizer_steps": result.optimizer_steps,
                    "examples_seen": result.examples_seen,
                    "stopped_early": result.stopped_early,
                    "stop_reason": result.stop_reason,
                }
            )
            transition.status = "trained"
        return result

    @classmethod
    def evaluate(
        cls,
        model: nn.Module,
        loader: Any,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, Optional[float]]:
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_examples = 0
        correct = 0
        classified_examples = 0

        try:
            with torch.inference_mode():
                for batch in loader:
                    inputs, targets = cls._split_batch(batch)
                    inputs = cls._to_device(inputs, device)
                    targets = cls._to_device(targets, device)
                    output = cls._primary_tensor(cls._call_model(model, inputs))
                    if output is None:
                        raise ValueError("Model did not produce a tensor output.")
                    batch_size = cls._batch_size(targets)
                    loss_value = float(criterion(output, targets).item())
                    if not math.isfinite(loss_value):
                        raise RuntimeError("Evaluation produced a non-finite loss.")
                    total_loss += loss_value * batch_size
                    total_examples += batch_size

                    is_classification = (
                        isinstance(targets, torch.Tensor)
                        and not targets.dtype.is_floating_point
                    )
                    if is_classification and output.ndim >= 2 and output.shape[-1] > 1:
                        predictions = output.argmax(dim=-1)
                        comparable_targets = targets
                        if comparable_targets.ndim == predictions.ndim + 1:
                            comparable_targets = comparable_targets.squeeze(-1)
                        if comparable_targets.shape == predictions.shape:
                            correct += int(
                                (predictions == comparable_targets).sum().item()
                            )
                            classified_examples += predictions.numel()
        finally:
            model.train(was_training)

        if total_examples == 0:
            raise ValueError("Validation loader produced no examples.")
        accuracy = (
            correct / classified_examples if classified_examples > 0 else None
        )
        return total_loss / total_examples, accuracy

    @staticmethod
    def _optimizer(
        model: nn.Module, config: RecoveryExperimentConfig
    ) -> torch.optim.Optimizer:
        name = config.optimizer.lower()
        if name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @staticmethod
    def _split_batch(batch: Any) -> Tuple[Any, Any]:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
            return batch["inputs"], batch["targets"]
        raise ValueError(
            "A batch must be (inputs, targets) or contain inputs/targets keys."
        )

    @staticmethod
    def _call_model(model: nn.Module, inputs: Any) -> Any:
        if isinstance(inputs, dict):
            return model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return model(*inputs)
        return model(inputs)

    @classmethod
    def _primary_tensor(cls, value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = cls._primary_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = cls._primary_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @classmethod
    def _to_device(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: cls._to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._to_device(item, device) for item in value)
        if isinstance(value, list):
            return [cls._to_device(item, device) for item in value]
        return value

    @staticmethod
    def _batch_size(targets: Any) -> int:
        if isinstance(targets, torch.Tensor) and targets.ndim > 0:
            return int(targets.shape[0])
        return 1

    @staticmethod
    def _has_recovered(child_loss: float, parent_loss: float, tolerance: float) -> bool:
        return child_loss <= parent_loss * (1.0 + tolerance)

    @staticmethod
    def seed_all(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
