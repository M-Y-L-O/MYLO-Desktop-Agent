from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from Core.ArchitectureDescriptor import ArchitectureDescriptor
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.ModelTransition import canonical_fingerprint, state_dict_fingerprint
from Core.SemanticTransforms import TransformationError
from Processing.Data.DataPipeline import DataPipelineResult
from Processing.Optimization.RecoveryHarness import (
    RecoveryExperimentConfig,
    TransitionRecoveryHarness,
)
from Processing.Optimization.TransitionDecision import EvidenceDecisionPolicy
from Processing.Optimization.TransitionExperiment import LinearWidthRecoveryExperiment


@dataclass(frozen=True)
class LinearWidthCandidate:
    node_id: str
    new_width: int

    @property
    def candidate_id(self) -> str:
        return f"{self.node_id}:{self.new_width}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "new_width": self.new_width,
        }


@dataclass
class _CandidateProbe:
    index: int
    candidate: LinearWidthCandidate
    result: Any
    parameter_count: int
    parameter_ratio: float
    priority_score: Optional[float]
    priority_components: Dict[str, float]
    probe_elapsed_seconds: float
    training_elapsed_seconds: float = 0.0
    optimizer_steps_allocated: int = 0
    training_rank: Optional[int] = None


class LinearWidthBenchmark:
    @classmethod
    def run(
        cls,
        descriptor: ArchitectureDescriptor,
        parent_state: Mapping[str, torch.Tensor],
        candidates: Sequence[LinearWidthCandidate],
        pipeline: DataPipelineResult,
        criterion: nn.Module,
        recovery_config: RecoveryExperimentConfig,
        *,
        parent_revision_id: str,
        decision_policies: Mapping[str, EvidenceDecisionPolicy],
        optimizer_step_budget: int,
        steps_per_candidate: int,
        calibration_batches: int = 3,
        max_validation_loss_ratio: float = 2.0,
        test_evaluation_scope: str = "none",
    ) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("At least one benchmark candidate is required.")
        if optimizer_step_budget < 0:
            raise ValueError("optimizer_step_budget cannot be negative.")
        if steps_per_candidate < 0:
            raise ValueError("steps_per_candidate cannot be negative.")
        if max_validation_loss_ratio < 1.0:
            raise ValueError("max_validation_loss_ratio must be at least 1.")
        normalized_test_scope = test_evaluation_scope.strip().lower()
        if normalized_test_scope not in {"none", "champion", "feasible_frontier"}:
            raise ValueError(
                "test_evaluation_scope must be 'none', 'champion', or "
                "'feasible_frontier'."
            )
        if normalized_test_scope != "none" and pipeline.test_loader is None:
            raise ValueError(
                "Held-out evaluation was requested, but the data pipeline has no "
                "test partition."
            )

        parent_reference = DescriptorModelBuilder.build(descriptor)
        parent_reference.load_state_dict(dict(parent_state), strict=True)
        parent_parameter_count = sum(
            parameter.numel() for parameter in parent_reference.parameters()
        )
        modes = {}
        for mode, policy in decision_policies.items():
            modes[mode] = cls._run_mode(
                descriptor,
                parent_state,
                candidates,
                pipeline,
                criterion,
                recovery_config,
                parent_revision_id=parent_revision_id,
                policy=policy,
                optimizer_step_budget=optimizer_step_budget,
                steps_per_candidate=steps_per_candidate,
                calibration_batches=calibration_batches,
                parent_parameter_count=parent_parameter_count,
                max_validation_loss_ratio=max_validation_loss_ratio,
                test_evaluation_scope=normalized_test_scope,
            )

        early_stop_opportunities = max(steps_per_candidate - 1, 0) // max(
            recovery_config.evaluation_interval, 1
        )
        early_stop_can_trigger = (
            recovery_config.early_stopping_patience_evaluations > 0
            and early_stop_opportunities
            >= recovery_config.early_stopping_patience_evaluations
        )
        protocol_warnings = []
        if not early_stop_can_trigger:
            protocol_warnings.append(
                "Early stopping cannot trigger before the per-candidate budget ends "
                "with the current steps, evaluation interval, and patience."
            )
        if normalized_test_scope != "none":
            protocol_warnings.append(
                "Held-out results are for final reporting only and must not be used "
                "to revise candidates, thresholds, or selection policy."
            )
        protocol = {
            "schema_version": "linear_width_benchmark_protocol_v1",
            "candidate_order": [candidate.to_dict() for candidate in candidates],
            "decision_modes": list(decision_policies),
            "decision_policies": {
                name: policy.to_dict() for name, policy in decision_policies.items()
            },
            "optimizer_step_budget_per_mode": optimizer_step_budget,
            "steps_per_candidate": steps_per_candidate,
            "calibration_batches": calibration_batches,
            "selection_data": "validation",
            "final_evaluation_data": (
                "not_evaluated"
                if normalized_test_scope == "none"
                else "held_out_test"
            ),
            "test_evaluation_scope": normalized_test_scope,
            "test_used_for_selection": False,
            "held_out_evaluation_locked_after_validation_selection": True,
            "parent_revision_id": parent_revision_id,
            "parent_state_fingerprint": state_dict_fingerprint(parent_state),
            "parent_architecture_fingerprint": canonical_fingerprint(descriptor),
            "parent_parameter_count": parent_parameter_count,
            "dataset_split": pipeline.split_metadata,
            "preprocessing": pipeline.preprocessing_metadata(),
            "criterion": (
                f"{criterion.__class__.__module__}."
                f"{criterion.__class__.__qualname__}"
            ),
            "runtime": {
                "torch_version": str(torch.__version__),
                "device": recovery_config.device,
            },
            "feasibility_constraint": {
                "metric": "validation_loss / parent_validation_loss",
                "maximum_ratio": max_validation_loss_ratio,
                "raw_frontier_is_also_reported": True,
            },
            "adaptive_recovery": recovery_config.to_dict(),
            "unused_optimizer_steps_remain_available_to_later_candidates": True,
            "early_stopping_opportunities_before_budget_end": early_stop_opportunities,
            "early_stopping_can_trigger": early_stop_can_trigger,
            "warnings": protocol_warnings,
        }
        protocol["protocol_fingerprint"] = canonical_fingerprint(protocol)
        return {
            "protocol": protocol,
            "modes": modes,
            "comparison": cls._comparison(modes),
        }

    @classmethod
    def _run_mode(
        cls,
        descriptor: ArchitectureDescriptor,
        parent_state: Mapping[str, torch.Tensor],
        candidates: Sequence[LinearWidthCandidate],
        pipeline: DataPipelineResult,
        criterion: nn.Module,
        recovery_config: RecoveryExperimentConfig,
        *,
        parent_revision_id: str,
        policy: EvidenceDecisionPolicy,
        optimizer_step_budget: int,
        steps_per_candidate: int,
        calibration_batches: int,
        parent_parameter_count: int,
        max_validation_loss_ratio: float,
        test_evaluation_scope: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        steps_remaining = optimizer_step_budget
        records: List[Dict[str, Any]] = []
        probes: List[_CandidateProbe] = []
        _, parent_validation_loader, _ = pipeline.fresh_loaders(recovery_config.seed)
        device = torch.device(recovery_config.device)
        parent_validation_model = DescriptorModelBuilder.build(descriptor)
        parent_validation_model.load_state_dict(dict(parent_state), strict=True)
        parent_validation_model.to(device)
        criterion.to(device)
        parent_validation_loss, parent_validation_accuracy = TransitionRecoveryHarness.evaluate(
            parent_validation_model, parent_validation_loader, criterion, device
        )
        parent_record = {
            "candidate_id": "parent",
            "node_id": None,
            "new_width": None,
            "status": "baseline",
            "accepted": True,
            "parameter_count": parent_parameter_count,
            "parameter_ratio": 1.0,
            "validation_loss": parent_validation_loss,
            "validation_accuracy": parent_validation_accuracy,
            "validation_loss_ratio_to_parent": 1.0,
            "feasible": True,
            "optimizer_steps_allocated": 0,
            "optimizer_steps_used": 0,
            "budget_exhausted_before_training": False,
        }
        probe_started = time.perf_counter()
        for index, candidate in enumerate(candidates):
            probe_config = replace(
                recovery_config,
                max_steps=0,
                seed=recovery_config.seed + index,
            )
            train_loader, validation_loader, _ = pipeline.fresh_loaders(
                probe_config.seed
            )
            parent_model = DescriptorModelBuilder.build(descriptor)
            parent_model.load_state_dict(dict(parent_state), strict=True)
            candidate_started = time.perf_counter()

            try:
                result = LinearWidthRecoveryExperiment.run(
                    descriptor,
                    parent_model,
                    candidate.node_id,
                    candidate.new_width,
                    train_loader,
                    validation_loader,
                    criterion,
                    probe_config,
                    parent_revision_id=parent_revision_id,
                    calibration_batches=calibration_batches,
                    test_loader=None,
                    decision_policy=policy,
                )
            except (TransformationError, RuntimeError, ValueError) as exc:
                records.append(
                    {
                        **candidate.to_dict(),
                        "status": "failed",
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "parameter_count": None,
                        "parameter_ratio": None,
                        "validation_loss": None,
                        "validation_accuracy": None,
                        "optimizer_steps_allocated": 0,
                        "optimizer_steps_used": 0,
                        "budget_exhausted_before_training": False,
                        "candidate_order": index,
                        "training_rank": None,
                    }
                )
                continue
            parameter_count = sum(
                parameter.numel() for parameter in result.child_model.parameters()
            )
            parameter_ratio = parameter_count / max(parent_parameter_count, 1)
            priority_score, priority_components = cls._priority_score(
                result,
                parameter_ratio,
                policy,
            )
            probes.append(
                _CandidateProbe(
                    index=index,
                    candidate=candidate,
                    result=result,
                    parameter_count=parameter_count,
                    parameter_ratio=parameter_ratio,
                    priority_score=priority_score,
                    priority_components=priority_components,
                    probe_elapsed_seconds=round(
                        time.perf_counter() - candidate_started, 8
                    ),
                )
            )

        probe_elapsed_seconds = round(time.perf_counter() - probe_started, 8)

        scheduled = sorted(
            (probe for probe in probes if probe.result.decision.accepted),
            key=lambda probe: cls._scheduling_key(probe, policy),
        )
        for training_rank, probe in enumerate(scheduled, start=1):
            probe.training_rank = training_rank
            allocated_steps = min(steps_per_candidate, steps_remaining)
            probe.optimizer_steps_allocated = allocated_steps
            if allocated_steps <= 0:
                continue

            config = replace(
                recovery_config,
                max_steps=allocated_steps,
                seed=recovery_config.seed + probe.index,
            )
            train_loader, validation_loader, _ = pipeline.fresh_loaders(config.seed)
            parent_model = DescriptorModelBuilder.build(descriptor)
            parent_model.load_state_dict(dict(parent_state), strict=True)
            probe.result.recovery = TransitionRecoveryHarness.run(
                parent_model,
                probe.result.child_model,
                train_loader,
                validation_loader,
                criterion,
                config,
                transition=probe.result.transition,
            )
            probe.training_elapsed_seconds = probe.result.recovery.elapsed_seconds
            used_steps = probe.result.recovery.optimizer_steps
            steps_remaining = max(0, steps_remaining - used_steps)

        for probe in sorted(probes, key=lambda item: item.index):
            result = probe.result
            validation_loss = result.recovery.best_validation_loss
            validation_loss_ratio = (
                float(validation_loss) / max(abs(parent_validation_loss), 1e-12)
                if validation_loss is not None
                else None
            )
            used_steps = result.recovery.optimizer_steps
            budget_exhausted = (
                result.decision.accepted
                and steps_per_candidate > 0
                and probe.optimizer_steps_allocated == 0
            )
            status = result.transition.status
            if budget_exhausted:
                status = "not_trained_budget_exhausted"
            elif result.decision.accepted and steps_per_candidate == 0:
                status = "measured_no_training_requested"
            record = {
                **probe.candidate.to_dict(),
                "transition_id": result.transition.transition_id,
                "status": status,
                "accepted": result.decision.accepted,
                "decision": result.decision.to_dict(),
                "semantic_class": result.plan.semantic_class,
                "mode": result.plan.mode,
                "parameter_count": probe.parameter_count,
                "parameter_ratio": probe.parameter_ratio,
                "validation_loss": validation_loss,
                "validation_accuracy": cls._selected_accuracy(result),
                "validation_loss_ratio_to_parent": validation_loss_ratio,
                "feasible": bool(
                    result.decision.accepted
                    and validation_loss_ratio is not None
                    and validation_loss_ratio <= max_validation_loss_ratio
                ),
                "recovery_step": result.recovery.recovery_step,
                "selected_step": result.recovery.selected_step,
                "optimizer_steps_allocated": probe.optimizer_steps_allocated,
                "optimizer_steps_used": used_steps,
                "stopped_early": result.recovery.stopped_early,
                "stop_reason": result.recovery.stop_reason,
                "budget_exhausted_before_training": budget_exhausted,
                "candidate_order": probe.index,
                "training_rank": probe.training_rank,
                "priority_score": probe.priority_score,
                "priority_components": probe.priority_components,
                "probe_elapsed_seconds": probe.probe_elapsed_seconds,
                "execution_evidence": result.transition.evidence.execution,
                "evidence_collection_cost": result.transition.evidence.collection_cost,
                "recovery_elapsed_seconds": probe.training_elapsed_seconds,
                "state_mapping": result.transition.state_mapping.summary(),
            }
            records.append(record)

        records.sort(key=lambda record: record.get("candidate_order", 0))
        frontier = cls._pareto_frontier([parent_record, *records])
        feasible_frontier = cls._pareto_frontier(
            [
                record
                for record in [parent_record, *records]
                if record.get("feasible", False)
            ]
        )
        champion = cls._select_champion(feasible_frontier)
        held_out_test = None
        held_out_frontier = None
        test_evaluation_count = 0
        if (
            champion is not None
            and pipeline.test_loader is not None
            and test_evaluation_scope != "none"
        ):
            models_by_candidate_id = {
                "parent": parent_validation_model,
                **{
                    probe.candidate.candidate_id: probe.result.child_model
                    for probe in probes
                },
            }
            selected_records = (
                feasible_frontier
                if test_evaluation_scope == "feasible_frontier"
                else [champion]
            )
            held_out_results = cls._evaluate_held_out_records(
                selected_records,
                models_by_candidate_id,
                pipeline.test_loader,
                criterion,
                device,
            )
            test_evaluation_count = len(held_out_results)
            champion_result = next(
                result
                for result in held_out_results
                if result["candidate_id"] == champion["candidate_id"]
            )
            held_out_test = {
                "candidate_id": champion["candidate_id"],
                "parent_loss": champion_result["parent_test_loss"],
                "parent_accuracy": champion_result["parent_test_accuracy"],
                "champion_loss": champion_result["test_loss"],
                "champion_accuracy": champion_result["test_accuracy"],
                "loss_delta": champion_result["test_loss_delta_from_parent"],
                "used_for_selection": False,
            }
            if test_evaluation_scope == "feasible_frontier":
                held_out_frontier = held_out_results

        used = optimizer_step_budget - steps_remaining
        return {
            "decision_mode": policy.mode,
            "scheduling": {
                "strategy": (
                    "provided_order"
                    if policy.mode.strip().lower() == "always_train"
                    else "evidence_priority_v1"
                ),
                "scheduled_order": [
                    probe.candidate.candidate_id for probe in scheduled
                ],
                "trained_order": [
                    probe.candidate.candidate_id
                    for probe in scheduled
                    if probe.result.recovery.optimizer_steps > 0
                ],
                "priority_formula": {
                    "risk_weight": 1.0,
                    "expansion_penalty_weight": 1.5,
                    "compression_reward_weight": 0.5,
                    "deferred_candidates_after_preferred": True,
                },
            },
            "parent_baseline": parent_record,
            "candidates": records,
            "pareto_frontier": frontier,
            "feasible_pareto_frontier": feasible_frontier,
            "champion": champion,
            "held_out_test": held_out_test,
            "held_out_frontier": held_out_frontier,
            "test_evaluation_count": test_evaluation_count,
            "budget": {
                "optimizer_steps_available": optimizer_step_budget,
                "optimizer_steps_used": used,
                "optimizer_steps_remaining": steps_remaining,
                "candidates_evaluated": len(records),
                "candidates_trained": sum(
                    record["optimizer_steps_used"] > 0 for record in records
                ),
                "candidates_rejected": sum(
                    record["status"] == "rejected" for record in records
                ),
                "candidates_failed": sum(
                    record["status"] == "failed" for record in records
                ),
                "candidates_stopped_early": sum(
                    bool(record.get("stopped_early")) for record in records
                ),
                "candidates_deferred": sum(
                    record.get("decision", {}).get("action") == "defer"
                    for record in records
                ),
                "probe_seconds": probe_elapsed_seconds,
                "evidence_collection_seconds": round(
                    sum(
                        float(record.get("evidence_collection_cost", {}).get("elapsed_seconds", 0.0))
                        for record in records
                    ),
                    8,
                ),
                "recovery_seconds": round(
                    sum(float(record.get("recovery_elapsed_seconds", 0.0)) for record in records),
                    8,
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 8),
            },
        }

    @classmethod
    def _evaluate_held_out_records(
        cls,
        records: Sequence[Dict[str, Any]],
        models_by_candidate_id: Mapping[str, nn.Module],
        test_loader: Any,
        criterion: nn.Module,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        parent_model = models_by_candidate_id.get("parent")
        if parent_model is None:
            raise RuntimeError("Parent model is unavailable for held-out evaluation.")
        parent_model.to(device)
        criterion.to(device)
        parent_loss, parent_accuracy = TransitionRecoveryHarness.evaluate(
            parent_model, test_loader, criterion, device
        )

        results = []
        seen = set()
        for record in records:
            candidate_id = str(record["candidate_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            model = models_by_candidate_id.get(candidate_id)
            if model is None:
                raise RuntimeError(
                    f"Model for selected candidate '{candidate_id}' is unavailable."
                )
            if candidate_id == "parent":
                test_loss, test_accuracy = parent_loss, parent_accuracy
            else:
                model.to(device)
                test_loss, test_accuracy = TransitionRecoveryHarness.evaluate(
                    model, test_loader, criterion, device
                )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "parameter_count": record["parameter_count"],
                    "parameter_ratio": record["parameter_ratio"],
                    "validation_loss": record["validation_loss"],
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "parent_test_loss": parent_loss,
                    "parent_test_accuracy": parent_accuracy,
                    "test_loss_delta_from_parent": test_loss - parent_loss,
                    "selected_on": "validation",
                    "used_for_selection": False,
                }
            )
        return results

    @staticmethod
    def _priority_score(
        result: Any,
        parameter_ratio: float,
        policy: EvidenceDecisionPolicy,
    ) -> tuple[Optional[float], Dict[str, float]]:
        if policy.mode.strip().lower() == "always_train":
            return None, {}
        risk = result.decision.observed.get("risk_score")
        risk_value = float(risk) if risk is not None else 0.0
        expansion_penalty = 1.5 * max(0.0, parameter_ratio - 1.0)
        compression_reward = 0.5 * max(0.0, 1.0 - parameter_ratio)
        components = {
            "risk": risk_value,
            "expansion_penalty": expansion_penalty,
            "compression_reward": compression_reward,
        }
        return risk_value + expansion_penalty - compression_reward, components

    @staticmethod
    def _scheduling_key(
        probe: _CandidateProbe,
        policy: EvidenceDecisionPolicy,
    ) -> tuple[Any, ...]:
        if policy.mode.strip().lower() == "always_train":
            return (probe.index,)
        tier = 1 if probe.result.decision.action == "defer" else 0
        score = probe.priority_score if probe.priority_score is not None else 0.0
        return (tier, score, probe.index)

    @staticmethod
    def _selected_accuracy(result: Any) -> Optional[float]:
        selected_step = result.recovery.selected_step
        for checkpoint in result.recovery.checkpoints:
            if checkpoint.step == selected_step:
                return checkpoint.validation_accuracy
        return None

    @classmethod
    def _pareto_frontier(cls, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        eligible = [
            record
            for record in records
            if record["accepted"] and record["validation_loss"] is not None
        ]
        frontier = []
        for candidate in eligible:
            dominated = any(
                other is not candidate
                and cls._dominates(other, candidate)
                for other in eligible
            )
            if not dominated:
                frontier.append(candidate)
        return sorted(
            frontier,
            key=lambda record: (record["parameter_count"], record["validation_loss"]),
        )

    @staticmethod
    def _dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        no_worse = (
            left["validation_loss"] <= right["validation_loss"]
            and left["parameter_count"] <= right["parameter_count"]
        )
        strictly_better = (
            left["validation_loss"] < right["validation_loss"]
            or left["parameter_count"] < right["parameter_count"]
        )
        return no_worse and strictly_better

    @staticmethod
    def _select_champion(frontier: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not frontier:
            return None
        return min(
            frontier,
            key=lambda record: (record["validation_loss"], record["parameter_count"]),
        )

    @classmethod
    def _comparison(
        cls, modes: Mapping[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        evidence = modes.get("execution_evidence") or modes.get("evidence")
        static = modes.get("static_only")
        baseline = modes.get("always_train")
        if evidence is None or baseline is None:
            return None
        evidence_test = evidence.get("held_out_test") or {}
        static_test = (static or {}).get("held_out_test") or {}
        baseline_test = baseline.get("held_out_test") or {}
        comparison = {
            "evidence_champion": (
                evidence.get("champion", {}).get("candidate_id")
                if evidence.get("champion")
                else None
            ),
            "always_train_champion": (
                baseline.get("champion", {}).get("candidate_id")
                if baseline.get("champion")
                else None
            ),
            "evidence_test_loss": evidence_test.get("champion_loss"),
            "always_train_test_loss": baseline_test.get("champion_loss"),
            "test_loss_delta_evidence_minus_baseline": cls._difference(
                evidence_test.get("champion_loss"),
                baseline_test.get("champion_loss"),
            ),
            "evidence_optimizer_steps": evidence["budget"]["optimizer_steps_used"],
            "always_train_optimizer_steps": baseline["budget"]["optimizer_steps_used"],
            "optimizer_steps_saved_vs_always_train": (
                baseline["budget"]["optimizer_steps_used"]
                - evidence["budget"]["optimizer_steps_used"]
            ),
            "evidence_rejections": evidence["budget"]["candidates_rejected"],
            "evidence_deferred": evidence["budget"]["candidates_deferred"],
            "always_train_rejections": baseline["budget"]["candidates_rejected"],
        }
        if static is not None:
            comparison.update(
                {
                    "static_only_champion": (
                        static.get("champion", {}).get("candidate_id")
                        if static.get("champion")
                        else None
                    ),
                    "static_only_test_loss": static_test.get("champion_loss"),
                    "test_loss_delta_evidence_minus_static_only": cls._difference(
                        evidence_test.get("champion_loss"),
                        static_test.get("champion_loss"),
                    ),
                    "static_only_optimizer_steps": static["budget"][
                        "optimizer_steps_used"
                    ],
                    "static_only_rejections": static["budget"][
                        "candidates_rejected"
                    ],
                    "static_only_deferred": static["budget"][
                        "candidates_deferred"
                    ],
                }
            )
        comparison["validation_pareto"] = cls._frontier_comparison(modes)
        held_out_comparison = cls._held_out_frontier_comparison(modes)
        if held_out_comparison is not None:
            comparison["held_out_selected_frontier"] = held_out_comparison
        return comparison

    @classmethod
    def _held_out_frontier_comparison(
        cls, modes: Mapping[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        evaluated = {
            name: mode.get("held_out_frontier")
            for name, mode in modes.items()
            if mode.get("held_out_frontier") is not None
        }
        if not evaluated:
            return None
        all_records = [record for records in evaluated.values() for record in records]
        if not all_records:
            return None

        reference_complexity = (
            max(float(record["parameter_ratio"]) for record in all_records) * 1.05
            + 1e-12
        )
        reference_loss = (
            max(float(record["test_loss"]) for record in all_records) * 1.05
            + 1e-12
        )
        summaries = {}
        for name, records in evaluated.items():
            test_frontier = cls._pareto_frontier(
                [
                    {
                        **record,
                        "accepted": True,
                        "validation_loss": record["test_loss"],
                    }
                    for record in records
                ]
            )
            summary = cls._frontier_summary(
                test_frontier,
                reference_complexity=reference_complexity,
                reference_loss=reference_loss,
            )
            summary["best_test_loss"] = summary.pop("best_validation_loss")
            summaries[name] = summary
        return {
            "reference_point": {
                "parameter_ratio": reference_complexity,
                "test_loss": reference_loss,
                "derived_from": "validation-selected feasible frontier members",
            },
            "modes": summaries,
            "selection_data": "validation",
            "test_used_for_selection": False,
            "higher_hypervolume_is_better": True,
        }

    @classmethod
    def _frontier_comparison(
        cls, modes: Mapping[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        eligible = [
            record
            for mode in modes.values()
            for record in [mode.get("parent_baseline"), *mode.get("candidates", [])]
            if record
            and record.get("accepted")
            and record.get("validation_loss") is not None
            and record.get("parameter_ratio") is not None
        ]
        if not eligible:
            return {"reference_point": None, "modes": {}}

        max_complexity = max(float(record["parameter_ratio"]) for record in eligible)
        max_loss = max(float(record["validation_loss"]) for record in eligible)
        reference_complexity = max_complexity * 1.05 + 1e-12
        reference_loss = max_loss * 1.05 + 1e-12
        summaries = {}
        for name, mode in modes.items():
            raw_frontier = mode.get("pareto_frontier", [])
            feasible_frontier = mode.get("feasible_pareto_frontier", [])
            summaries[name] = {
                "raw": cls._frontier_summary(
                    raw_frontier,
                    reference_complexity=reference_complexity,
                    reference_loss=reference_loss,
                ),
                "feasible": cls._frontier_summary(
                    feasible_frontier,
                    reference_complexity=reference_complexity,
                    reference_loss=reference_loss,
                ),
            }
        return {
            "reference_point": {
                "parameter_ratio": reference_complexity,
                "validation_loss": reference_loss,
                "derived_from": "all accepted candidates across compared modes",
            },
            "modes": summaries,
            "higher_hypervolume_is_better": True,
        }

    @classmethod
    def _frontier_summary(
        cls,
        frontier: Sequence[Dict[str, Any]],
        *,
        reference_complexity: float,
        reference_loss: float,
    ) -> Dict[str, Any]:
        return {
            "hypervolume": cls._hypervolume(
                frontier,
                reference_complexity=reference_complexity,
                reference_loss=reference_loss,
            ),
            "frontier_size": len(frontier),
            "compressed_frontier_size": sum(
                float(record["parameter_ratio"]) < 1.0 for record in frontier
            ),
            "minimum_parameter_ratio": (
                min(float(record["parameter_ratio"]) for record in frontier)
                if frontier
                else None
            ),
            "best_validation_loss": (
                min(float(record["validation_loss"]) for record in frontier)
                if frontier
                else None
            ),
        }

    @staticmethod
    def _hypervolume(
        frontier: Sequence[Dict[str, Any]],
        *,
        reference_complexity: float,
        reference_loss: float,
    ) -> float:
        ordered = sorted(
            (
                (float(record["parameter_ratio"]), float(record["validation_loss"]))
                for record in frontier
            ),
            key=lambda point: point[0],
        )
        area = 0.0
        previous_loss = reference_loss
        for complexity, loss in ordered:
            bounded_loss = min(previous_loss, loss)
            area += max(0.0, reference_complexity - complexity) * max(
                0.0, previous_loss - bounded_loss
            )
            previous_loss = bounded_loss
        return round(area, 12)

    @staticmethod
    def _difference(left: Any, right: Any) -> Optional[float]:
        if left is None or right is None:
            return None
        return float(left) - float(right)
