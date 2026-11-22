from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from Core.ArchitectureDescriptor import ArchitectureDescriptor
from Core.ModelTransition import ModelTransition
from Core.SemanticTransforms import LinearWidthPlan, LinearWidthTransformation
from Processing.Optimization.RecoveryHarness import (
    RecoveryExperimentConfig,
    RecoveryExperimentResult,
    TransitionRecoveryHarness,
)
from Processing.Optimization.TransitionEvidence import TransitionEvidenceCollector
from Processing.Optimization.TransitionDecision import (
    EvidenceDecisionPolicy,
    TransitionDecision,
)


@dataclass
class LinearWidthExperimentResult:
    child_descriptor: ArchitectureDescriptor
    child_model: nn.Module
    plan: LinearWidthPlan
    transition: ModelTransition
    recovery: RecoveryExperimentResult
    decision: TransitionDecision
    held_out_test: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "child_descriptor": self.child_descriptor.to_dict(),
            "transition": self.transition.to_dict(),
            "recovery": self.recovery.to_dict(),
            "decision": self.decision.to_dict(),
            "held_out_test": self.held_out_test,
        }


class LinearWidthRecoveryExperiment:
    @classmethod
    def run(
        cls,
        descriptor: ArchitectureDescriptor,
        parent_model: nn.Module,
        node_id: str,
        new_width: int,
        train_loader: Any,
        validation_loader: Any,
        criterion: nn.Module,
        config: RecoveryExperimentConfig,
        *,
        parent_revision_id: str,
        calibration_batch: Optional[Any] = None,
        calibration_batches: int = 3,
        test_loader: Optional[Any] = None,
        decision_policy: Optional[EvidenceDecisionPolicy] = None,
    ) -> LinearWidthExperimentResult:
        TransitionRecoveryHarness.seed_all(config.seed)
        applied = LinearWidthTransformation.apply(
            descriptor,
            parent_model.state_dict(),
            node_id,
            new_width,
            parent_revision_id=parent_revision_id,
        )

        policy = decision_policy or EvidenceDecisionPolicy(mode="always_train")
        mode = policy.mode.strip().lower()
        uses_execution_evidence = mode in {"evidence", "execution_evidence"}
        evidence_arguments: Dict[str, Any] = {}
        if uses_execution_evidence:
            calibration_batch_count = 1
            if calibration_batch is None:
                batches = []
                for index, batch in enumerate(validation_loader):
                    batches.append(batch)
                    if index + 1 >= max(1, calibration_batches):
                        break
                if not batches:
                    raise ValueError("Validation loader produced no calibration batch.")
                calibration_batch = cls._combine_batches(batches)
                calibration_batch_count = len(batches)

            calibration_inputs, calibration_targets = cls._split_batch(calibration_batch)
            device = torch.device(config.device)
            parent_model.to(device)
            applied.child_model.to(device)
            criterion.to(device)
            calibration_inputs = cls._to_device(calibration_inputs, device)
            calibration_targets = cls._to_device(calibration_targets, device)
            evidence_arguments = {
                "parent_model": parent_model,
                "child_model": applied.child_model,
                "inputs": calibration_inputs,
                "targets": calibration_targets,
                "criterion": criterion,
            }

        applied.transition.evidence = TransitionEvidenceCollector.collect(
            descriptor,
            applied.child_descriptor,
            state_mapping=applied.transition.state_mapping,
            transformation=applied.plan.to_dict(),
            **evidence_arguments,
        )
        if uses_execution_evidence:
            applied.transition.evidence.execution["calibration_batches"] = calibration_batch_count
            applied.transition.evidence.execution["calibration_examples"] = cls._batch_size(
                calibration_targets
            )
        applied.transition.status = "measured"

        decision = policy.decide(
            applied.transition.evidence.execution,
            semantic_class=applied.plan.semantic_class,
            static=applied.transition.evidence.static,
        )
        applied.transition.decision = decision.to_dict()
        applied.transition.outcome["decision"] = decision.to_dict()
        recovery_config = config if decision.accepted else replace(config, max_steps=0)

        recovery = TransitionRecoveryHarness.run(
            parent_model,
            applied.child_model,
            train_loader,
            validation_loader,
            criterion,
            recovery_config,
            transition=applied.transition,
        )
        if not decision.accepted:
            applied.transition.status = "rejected"
        held_out_test = None
        if test_loader is not None and decision.accepted:
            device = torch.device(config.device)
            parent_loss, parent_accuracy = TransitionRecoveryHarness.evaluate(
                parent_model, test_loader, criterion, device
            )
            child_loss, child_accuracy = TransitionRecoveryHarness.evaluate(
                applied.child_model, test_loader, criterion, device
            )
            held_out_test = {
                "parent_loss": parent_loss,
                "parent_accuracy": parent_accuracy,
                "child_loss": child_loss,
                "child_accuracy": child_accuracy,
                "loss_delta": child_loss - parent_loss,
                "used_for_selection": False,
            }
            applied.transition.outcome["held_out_test"] = held_out_test
        return LinearWidthExperimentResult(
            child_descriptor=applied.child_descriptor,
            child_model=applied.child_model,
            plan=applied.plan,
            transition=applied.transition,
            recovery=recovery,
            decision=decision,
            held_out_test=held_out_test,
        )

    @staticmethod
    def _split_batch(batch: Any) -> Tuple[Any, Any]:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
            return batch["inputs"], batch["targets"]
        raise ValueError(
            "A calibration batch must be (inputs, targets) or contain inputs/targets keys."
        )

    @classmethod
    def _combine_batches(cls, batches: Any) -> Tuple[Any, Any]:
        split = [cls._split_batch(batch) for batch in batches]
        return (
            cls._concatenate([item[0] for item in split]),
            cls._concatenate([item[1] for item in split]),
        )

    @classmethod
    def _concatenate(cls, values: Any) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
        if isinstance(first, dict):
            return {
                key: cls._concatenate([value[key] for value in values])
                for key in first
            }
        if isinstance(first, tuple):
            return tuple(
                cls._concatenate([value[index] for value in values])
                for index in range(len(first))
            )
        if isinstance(first, list):
            return [
                cls._concatenate([value[index] for value in values])
                for index in range(len(first))
            ]
        raise ValueError("Calibration batches contain unsupported values.")

    @staticmethod
    def _batch_size(value: Any) -> int:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        return 1

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
