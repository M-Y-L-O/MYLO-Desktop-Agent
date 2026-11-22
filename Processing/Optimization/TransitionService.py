from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.ArchitectureDescriptor import ArchitectureDescriptor
from Core.ModelTransition import canonical_fingerprint, state_dict_fingerprint
from Core.SemanticTransforms import LinearWidthTransformation, TransformationError
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Processing.Data.DataPipeline import DataPipeline
from Processing.Models.DescriptorHandling import extractStateDict, loadDescriptorFromBytes
from Processing.Models.ModelEditing import loadProjectDescriptor
from Processing.Optimization.Neuroevolution import TieredEvaluator
from Processing.Optimization.RecoveryHarness import RecoveryExperimentConfig
from Processing.Optimization.TransitionExperiment import (
    LinearWidthRecoveryExperiment,
)
from Processing.Optimization.TransitionDecision import EvidenceDecisionPolicy
from Processing.Optimization.TransitionBenchmark import (
    LinearWidthBenchmark,
    LinearWidthCandidate,
)
from Processing.Optimization.ResearchRunStore import ResearchRunStore
from Utils.FileHandler import readBinary


class ResearchExperimentError(ValueError):
    def __init__(self, message: str, *, details: Any = None):
        super().__init__(message)
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        result = {"error": str(self)}
        if self.details is not None:
            result["details"] = self.details
        return result


@dataclass
class PreparedResearchContext:
    descriptor: Any
    parent_model: Any
    parent_state: Dict[str, Any]
    pipeline: Any
    criterion: Any
    recovery_config: RecoveryExperimentConfig
    parent_revision_id: str
    problem_type: str
    parent_load: Dict[str, Any]
    dataset: Dict[str, Any]


def research_api_enabled() -> bool:
    return os.getenv("MYLO_RESEARCH_API_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def research_status() -> Dict[str, Any]:
    return {
        "enabled": research_api_enabled(),
        "experimental": True,
        "production_optimizer_uses_transitions": False,
        "supported_experiments": ["linear_width", "linear_width_benchmark"],
        "supported_path": "Linear -> elementwise activation(s)/Identity -> Linear",
        "decision_modes": [
            "execution_evidence",
            "static_only",
            "always_train",
            "evidence",
        ],
        "evaluation_protocol": "train/validation/held-out-test",
        "evidence_policy": "evidence_policy_v3",
        "benchmark_protocol": "linear_width_benchmark_protocol_v1",
        "test_evaluation_scopes": ["none", "champion", "feasible_frontier"],
        "adaptive_recovery": True,
        "switch": "MYLO_RESEARCH_API_ENABLED",
    }


def discover_linear_width_candidates(
    project: Any,
    *,
    project_directory: str = "temp_project",
) -> Dict[str, Any]:
    if not research_api_enabled():
        raise ResearchExperimentError(
            "Research API is disabled. Set MYLO_RESEARCH_API_ENABLED=true and restart the agent."
        )
    model_filename = os.path.basename(getattr(project, "modelFilepath", "") or "")
    if not model_filename:
        raise ResearchExperimentError("The current project has no model.")
    descriptor = _load_descriptor(project_directory, model_filename)
    eligible = []
    ineligible = []
    for node in descriptor.nodes:
        if node.type != "Linear":
            continue
        old_width = int(node.params.get("out_features", 0) or 0)
        suggestions = []
        last_error = None
        for new_width in _suggest_widths(old_width):
            try:
                _, plan = LinearWidthTransformation.plan(
                    descriptor, node.id, new_width
                )
            except TransformationError as exc:
                last_error = str(exc)
                continue
            suggestions.append(
                {
                    "nodeId": node.id,
                    "newWidth": new_width,
                    "mode": plan.mode,
                    "semanticClass": plan.semantic_class,
                    "downstreamNodeId": plan.downstream_node_id,
                    "pathNodeIds": list(plan.path_node_ids),
                }
            )
        if suggestions:
            eligible.append(
                {
                    "nodeId": node.id,
                    "currentWidth": old_width,
                    "suggestions": suggestions,
                }
            )
        else:
            ineligible.append(
                {
                    "nodeId": node.id,
                    "currentWidth": old_width,
                    "reason": last_error or "No valid width suggestions were produced.",
                }
            )
    return {
        "supportedPath": "Linear -> elementwise activation(s)/Identity -> Linear",
        "eligibleNodes": eligible,
        "ineligibleLinearNodes": ineligible,
        "candidateCount": sum(len(node["suggestions"]) for node in eligible),
    }


def _suggest_widths(old_width: int) -> List[int]:
    if old_width <= 0:
        return []
    values = {
        max(1, old_width // 2),
        max(1, round(old_width * 0.75)),
        min(65536, max(1, round(old_width * 1.25))),
        min(65536, max(1, round(old_width * 1.5))),
        min(65536, old_width * 2),
    }
    values.discard(old_width)
    return sorted(values)


def _load_descriptor(project_directory: str, model_filename: str) -> ArchitectureDescriptor:
    descriptor_path = os.path.join(project_directory, "descriptor.json")
    if os.path.isfile(descriptor_path):
        with open(descriptor_path, "r", encoding="utf-8") as handle:
            descriptor = ArchitectureDescriptor.from_dict(json.load(handle))
        descriptor.validate()
        return descriptor
    if os.path.abspath(project_directory) == os.path.abspath("temp_project"):
        return loadProjectDescriptor(model_filename)
    model_path = os.path.join(project_directory, model_filename)
    model_bytes = readBinary(model_path)
    is_pytorch = model_filename.lower().endswith(
        (".pt", ".pth", ".pt2", ".ptpkg", ".torchpackage")
    )
    return loadDescriptorFromBytes(model_bytes, is_pytorch, input_dim=1, output_dim=1)


def _prepare_research_context(
    project: Any,
    request: Dict[str, Any],
    *,
    project_directory: str,
) -> PreparedResearchContext:
    model_filename = os.path.basename(getattr(project, "modelFilepath", "") or "")
    csv_filename = os.path.basename(getattr(project, "csvFilepath", "") or "")
    weights_filename = os.path.basename(
        getattr(project, "weightsFilepath", "") or ""
    )
    if not model_filename:
        raise ResearchExperimentError("The current project has no model.")
    if not csv_filename:
        raise ResearchExperimentError("The current project has no CSV dataset.")

    model_path = os.path.join(project_directory, model_filename)
    csv_path = os.path.join(project_directory, csv_filename)
    state_path = os.path.join(project_directory, weights_filename or model_filename)
    for label, path in (("model", model_path), ("CSV", csv_path), ("weights", state_path)):
        if not os.path.isfile(path):
            raise ResearchExperimentError(f"Current project {label} file is missing: {path}")

    feature_columns = _string_list(request.get("inputFeatures"), "inputFeatures")
    target_columns = _string_list(
        request.get("targetFeatures", request.get("targetFeature")),
        "targetFeatures",
    )
    if not feature_columns:
        raise ResearchExperimentError("At least one input feature is required.")
    if not target_columns:
        raise ResearchExperimentError("At least one target feature is required.")

    problem_type = str(request.get("problemType", "regression")).strip().lower()
    if problem_type not in {"regression", "classification"}:
        raise ResearchExperimentError(
            "problemType must be 'regression' or 'classification'."
        )
    batch_size = _bounded_int(request, "batchSize", minimum=1, maximum=4096, default=32)
    validation_split = _bounded_float(
        request, "validationSplit", minimum=0.05, maximum=0.5, default=0.2
    )
    test_split = _bounded_float(
        request, "testSplit", minimum=0.05, maximum=0.4, default=0.1
    )
    if validation_split + test_split >= 0.8:
        raise ResearchExperimentError(
            "validationSplit + testSplit must leave at least 20% for training."
        )
    random_state = _bounded_int(
        request, "randomState", minimum=0, maximum=2**31 - 1, default=42
    )

    descriptor = _load_descriptor(project_directory, model_filename)
    sequence_length = (
        int(descriptor.input_shape[1])
        if len(descriptor.input_shape) == 3 and descriptor.input_shape[1] > 1
        else None
    )
    encoding = str(request.get("encoding", "none"))
    pipeline = DataPipeline.prepare_data(
        csv_path,
        feature_columns,
        target_columns,
        problem_type=problem_type,
        batch_size=batch_size,
        val_split=validation_split,
        test_split=test_split,
        sequence_length=sequence_length,
        random_state=random_state,
        encoding=encoding,
    )
    if descriptor.input_shape[-1] != pipeline.input_shape[-1]:
        raise ResearchExperimentError(
            "Selected input features do not match the loaded model input width.",
            details={
                "model_input_width": descriptor.input_shape[-1],
                "dataset_input_width": pipeline.input_shape[-1],
            },
        )
    if descriptor.output_shape[-1] != pipeline.output_shape[-1]:
        raise ResearchExperimentError(
            "Selected target features/classes do not match the loaded model output width.",
            details={
                "model_output_width": descriptor.output_shape[-1],
                "dataset_output_width": pipeline.output_shape[-1],
            },
        )

    state_bytes = readBinary(state_path)
    parent_state = extractStateDict(state_bytes, is_pytorch=True)
    if not parent_state:
        raise ResearchExperimentError("Could not extract a PyTorch state_dict.")
    parent_model = DescriptorModelBuilder.build(descriptor)
    parent_load = WeightCompatibilityEngine.transfer_weights(
        parent_state,
        parent_model,
        allow_ambiguous_key_mapping=False,
    )
    parent_summary = parent_load["mapping_summary"]
    approximate_parent = any(
        parent_summary.get(key, 0) > 0
        for key in (
            "overlap_tensor_count",
            "dtype_cast_tensor_count",
            "initialized_target_tensor_count",
            "unmapped_source_tensor_count",
        )
    )
    if approximate_parent and not bool(request.get("allowApproximateParent", False)):
        raise ResearchExperimentError(
            "Parent reconstruction was not exact. Inspect the mapping or explicitly set allowApproximateParent=true.",
            details={
                "mapping_summary": parent_summary,
                "mapping_warnings": parent_load.get("mapping_warnings", []),
            },
        )

    device = _resolve_device(str(request.get("device", "cpu")))
    recovery_config = RecoveryExperimentConfig(
        max_steps=_bounded_int(request, "maxSteps", minimum=0, maximum=10000, default=25),
        evaluation_interval=_bounded_int(
            request, "evaluationInterval", minimum=1, maximum=10000, default=5
        ),
        learning_rate=_bounded_float(
            request, "learningRate", minimum=1e-8, maximum=10.0, default=1e-3
        ),
        optimizer=str(request.get("optimizer", "adam")),
        weight_decay=_bounded_float(
            request, "weightDecay", minimum=0.0, maximum=10.0, default=0.0
        ),
        momentum=_bounded_float(
            request, "momentum", minimum=0.0, maximum=1.0, default=0.9
        ),
        recovery_tolerance=_bounded_float(
            request, "recoveryTolerance", minimum=0.0, maximum=1.0, default=1e-5
        ),
        early_stopping_patience_evaluations=_bounded_int(
            request,
            "earlyStoppingPatienceEvaluations",
            minimum=0,
            maximum=1000,
            default=3,
        ),
        minimum_relative_improvement=_bounded_float(
            request,
            "minimumRelativeImprovement",
            minimum=0.0,
            maximum=0.999999,
            default=1e-4,
        ),
        seed=_bounded_int(request, "seed", minimum=0, maximum=2**31 - 1, default=0),
        device=device,
    )
    criterion = TieredEvaluator.select_criterion(problem_type)
    canonical_parent_state = {
        key: value.detach().clone() for key, value in parent_model.state_dict().items()
    }
    parent_revision_id = str(request.get("parentRevisionId", "")).strip()
    if not parent_revision_id:
        parent_revision_id = canonical_fingerprint(
            {
                "project_id": getattr(project, "id", ""),
                "model_file": model_filename,
                "weights_file": weights_filename,
                "architecture": descriptor,
                "state_dict_fingerprint": state_dict_fingerprint(
                    canonical_parent_state
                ),
            }
        )
    return PreparedResearchContext(
        descriptor=descriptor,
        parent_model=parent_model,
        parent_state=canonical_parent_state,
        pipeline=pipeline,
        criterion=criterion,
        recovery_config=recovery_config,
        parent_revision_id=parent_revision_id,
        problem_type=problem_type,
        parent_load={
            "exact": not approximate_parent,
            "mapping_summary": parent_summary,
            "mapping_warnings": parent_load.get("mapping_warnings", []),
        },
        dataset={
            "input_features": feature_columns,
            "target_features": target_columns,
            "problem_type": problem_type,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "test_split": test_split,
            "random_state": random_state,
            "preprocessing": "train-fitted imputation, encoding, and standardization",
            "transformed_input_features": pipeline.feature_names,
            "split": pipeline.split_metadata,
            "preprocessing_metadata": pipeline.preprocessing_metadata(),
            "parent_preprocessing_compatibility": "unverified",
        },
    )


def _decision_policy(
    request: Dict[str, Any], problem_type: str, mode: str
) -> EvidenceDecisionPolicy:
    supported_modes = {
        "evidence",
        "execution_evidence",
        "static_only",
        "always_train",
    }
    if mode not in supported_modes:
        raise ResearchExperimentError(
            "Decision mode must be 'execution_evidence', 'static_only', "
            "'always_train', or the backwards-compatible 'evidence' alias."
        )
    return EvidenceDecisionPolicy(
        mode=mode,
        max_relative_l2_delta=_bounded_float(
            request, "maxRelativeL2Delta", minimum=0.0, maximum=1e6, default=0.5
        ),
        function_preserving_tolerance=_bounded_float(
            request,
            "functionPreservingTolerance",
            minimum=0.0,
            maximum=1.0,
            default=1e-5,
        ),
        max_calibration_loss_ratio=_bounded_float(
            request,
            "maxCalibrationLossRatio",
            minimum=1.0,
            maximum=1e6,
            default=2.0,
        ),
        min_prediction_agreement=_bounded_float(
            request,
            "minPredictionAgreement",
            minimum=0.0,
            maximum=1.0,
            default=0.75 if problem_type == "classification" else 0.0,
        ),
        risk_threshold=_bounded_float(
            request, "riskThreshold", minimum=1e-9, maximum=100.0, default=0.75
        ),
        hard_risk_threshold=_bounded_float(
            request,
            "hardRiskThreshold",
            minimum=1e-9,
            maximum=100.0,
            default=1.0,
        ),
        min_static_width_retention=_bounded_float(
            request,
            "minStaticWidthRetention",
            minimum=0.0,
            maximum=1.0,
            default=0.5,
        ),
    )


def run_project_linear_width_experiment(
    project: Any,
    request: Dict[str, Any],
    *,
    project_directory: str = "temp_project",
) -> Dict[str, Any]:
    if not research_api_enabled():
        raise ResearchExperimentError(
            "Research API is disabled. Set MYLO_RESEARCH_API_ENABLED=true and restart the agent."
        )

    node_id = str(request.get("nodeId", "")).strip()
    if not node_id:
        raise ResearchExperimentError("nodeId is required.")
    new_width = _bounded_int(request, "newWidth", minimum=1, maximum=65536)
    context = _prepare_research_context(
        project, request, project_directory=project_directory
    )
    decision_mode = str(
        request.get("decisionMode", "execution_evidence")
    ).strip().lower()

    try:
        result = LinearWidthRecoveryExperiment.run(
            context.descriptor,
            context.parent_model,
            node_id,
            new_width,
            context.pipeline.train_loader,
            context.pipeline.val_loader,
            context.criterion,
            context.recovery_config,
            parent_revision_id=context.parent_revision_id,
            calibration_batches=_bounded_int(
                request, "calibrationBatches", minimum=1, maximum=100, default=3
            ),
            test_loader=context.pipeline.test_loader,
            decision_policy=_decision_policy(
                request, context.problem_type, decision_mode
            ),
        )
    except TransformationError as exc:
        raise ResearchExperimentError(str(exc)) from exc
    response = {
        "status": "success",
        "experimental": True,
        "parent_load": context.parent_load,
        "dataset": context.dataset,
        "result": result.to_dict(),
    }
    response["research_run"] = _persist_result(
        project,
        request,
        response,
        run_type="linear_width_experiment",
        project_directory=project_directory,
    )
    return response


def run_project_linear_width_benchmark(
    project: Any,
    request: Dict[str, Any],
    *,
    project_directory: str = "temp_project",
) -> Dict[str, Any]:
    if not research_api_enabled():
        raise ResearchExperimentError(
            "Research API is disabled. Set MYLO_RESEARCH_API_ENABLED=true and restart the agent."
        )
    candidates = _linear_width_candidates(request.get("candidates"))
    decision_modes = _decision_modes(request.get("decisionModes"))
    context = _prepare_research_context(
        project, request, project_directory=project_directory
    )
    steps_per_candidate = _bounded_int(
        request,
        "stepsPerCandidate",
        minimum=0,
        maximum=10000,
        default=context.recovery_config.max_steps,
    )
    optimizer_step_budget = _bounded_int(
        request,
        "optimizerStepBudget",
        minimum=0,
        maximum=1_000_000,
        default=len(candidates) * steps_per_candidate,
    )
    calibration_batches = _bounded_int(
        request, "calibrationBatches", minimum=1, maximum=100, default=3
    )
    policies = {
        mode: _decision_policy(request, context.problem_type, mode)
        for mode in decision_modes
    }
    benchmark = LinearWidthBenchmark.run(
        context.descriptor,
        context.parent_state,
        candidates,
        context.pipeline,
        context.criterion,
        context.recovery_config,
        parent_revision_id=context.parent_revision_id,
        decision_policies=policies,
        optimizer_step_budget=optimizer_step_budget,
        steps_per_candidate=steps_per_candidate,
        calibration_batches=calibration_batches,
        max_validation_loss_ratio=_bounded_float(
            request,
            "maxValidationLossRatio",
            minimum=1.0,
            maximum=1e6,
            default=2.0,
        ),
        test_evaluation_scope=_test_evaluation_scope(
            request.get("testEvaluationScope")
        ),
    )
    response = {
        "status": "success",
        "experimental": True,
        "parent_load": context.parent_load,
        "dataset": context.dataset,
        "benchmark": benchmark,
    }
    response["research_run"] = _persist_result(
        project,
        request,
        response,
        run_type="linear_width_benchmark",
        project_directory=project_directory,
    )
    return response


def list_research_runs(
    project: Any,
    *,
    project_directory: str = "temp_project",
) -> List[Dict[str, Any]]:
    store = ResearchRunStore(os.path.join(project_directory, "research_runs"))
    return store.list(project_id=str(getattr(project, "id", "") or ""))


def load_research_run(
    run_id: str,
    *,
    project: Any = None,
    project_directory: str = "temp_project",
) -> Dict[str, Any]:
    store = ResearchRunStore(os.path.join(project_directory, "research_runs"))
    record = store.load(run_id)
    if project is not None:
        project_id = str(getattr(project, "id", "") or "")
        if record.get("project_id") != project_id:
            raise FileNotFoundError(f"Research run '{run_id}' does not exist.")
    return record


def _persist_result(
    project: Any,
    request: Dict[str, Any],
    result: Dict[str, Any],
    *,
    run_type: str,
    project_directory: str,
) -> Dict[str, Any]:
    if request.get("persistResult", True) is False:
        return {"saved": False, "reason": "disabled_by_request"}
    store = ResearchRunStore(os.path.join(project_directory, "research_runs"))
    try:
        metadata = store.save(
            run_type=run_type,
            project_id=str(getattr(project, "id", "") or ""),
            request=dict(request),
            result=result,
        )
        return {"saved": True, **metadata}
    except Exception as exc:
        return {
            "saved": False,
            "reason": "persistence_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _linear_width_candidates(value: Any) -> List[LinearWidthCandidate]:
    if not isinstance(value, list) or not value:
        raise ResearchExperimentError(
            "candidates must be a non-empty list of {nodeId, newWidth} objects."
        )
    if len(value) > 100:
        raise ResearchExperimentError("A benchmark supports at most 100 candidates.")
    candidates = []
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ResearchExperimentError(f"candidates[{index}] must be an object.")
        node_id = str(item.get("nodeId", "")).strip()
        if not node_id:
            raise ResearchExperimentError(f"candidates[{index}].nodeId is required.")
        try:
            new_width = int(item.get("newWidth"))
        except (TypeError, ValueError) as exc:
            raise ResearchExperimentError(
                f"candidates[{index}].newWidth must be an integer."
            ) from exc
        if not 1 <= new_width <= 65536:
            raise ResearchExperimentError(
                f"candidates[{index}].newWidth must be between 1 and 65536."
            )
        key = (node_id, new_width)
        if key in seen:
            raise ResearchExperimentError(
                f"Duplicate benchmark candidate '{node_id}:{new_width}'."
            )
        seen.add(key)
        candidates.append(LinearWidthCandidate(node_id=node_id, new_width=new_width))
    return candidates


def _decision_modes(value: Any) -> List[str]:
    if value is None:
        return ["execution_evidence", "always_train"]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ResearchExperimentError("decisionModes must be a non-empty list.")
    modes = []
    for mode in value:
        normalized = str(mode).strip().lower()
        if normalized not in {
            "evidence",
            "execution_evidence",
            "static_only",
            "always_train",
        }:
            raise ResearchExperimentError(
                "decisionModes may only contain 'execution_evidence', "
                "'static_only', 'always_train', or 'evidence'."
            )
        if normalized not in modes:
            modes.append(normalized)
    return modes


def _test_evaluation_scope(value: Any) -> str:
    normalized = str(value or "none").strip().lower()
    if normalized not in {"none", "champion", "feasible_frontier"}:
        raise ResearchExperimentError(
            "testEvaluationScope must be 'none', 'champion', or "
            "'feasible_frontier'."
        )
    return normalized


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ResearchExperimentError(f"{name} must be a list of column names.")
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_int(
    request: Dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
    default: Any = None,
) -> int:
    value = request.get(key, default)
    if value is None:
        raise ResearchExperimentError(f"{key} is required.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchExperimentError(f"{key} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ResearchExperimentError(
            f"{key} must be between {minimum} and {maximum}."
        )
    return parsed


def _bounded_float(
    request: Dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    default: Any,
) -> float:
    value = request.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchExperimentError(f"{key} must be numeric.") from exc
    if not minimum <= parsed <= maximum:
        raise ResearchExperimentError(
            f"{key} must be between {minimum} and {maximum}."
        )
    return parsed


def _resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ResearchExperimentError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ResearchExperimentError("device must be 'cpu', 'cuda', or 'auto'.")
    return normalized
