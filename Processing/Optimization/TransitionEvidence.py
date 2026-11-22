from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from Core.ModelTransition import StateMappingReport, TransitionEvidence


class TransitionEvidenceCollector:
    @classmethod
    def collect(
        cls,
        parent_architecture: Any,
        child_architecture: Any,
        *,
        state_mapping: Optional[Any] = None,
        transformation: Optional[Mapping[str, Any]] = None,
        parent_model: Optional[nn.Module] = None,
        child_model: Optional[nn.Module] = None,
        inputs: Optional[Any] = None,
        targets: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
    ) -> TransitionEvidence:
        started = time.perf_counter()
        evidence = TransitionEvidence(
            static=cls.collect_static(
                parent_architecture,
                child_architecture,
                state_mapping=state_mapping,
                transformation=transformation,
            )
        )

        execution_requested = any(
            item is not None for item in (parent_model, child_model, inputs)
        )
        if execution_requested:
            if parent_model is None or child_model is None or inputs is None:
                evidence.warnings.append(
                    "Execution evidence requires parent_model, child_model, and inputs."
                )
            else:
                try:
                    evidence.execution = cls.collect_execution(
                        parent_model,
                        child_model,
                        inputs,
                        targets=targets,
                        criterion=criterion,
                    )
                except Exception as exc:
                    evidence.warnings.append(
                        f"Execution evidence collection failed: {type(exc).__name__}: {exc}"
                    )

        evidence.collection_cost["elapsed_seconds"] = round(
            time.perf_counter() - started, 8
        )
        return evidence

    @classmethod
    def collect_static(
        cls,
        parent_architecture: Any,
        child_architecture: Any,
        *,
        state_mapping: Optional[Any] = None,
        transformation: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        parent = cls._architecture_dict(parent_architecture)
        child = cls._architecture_dict(child_architecture)
        parent_nodes = cls._nodes_by_id(parent)
        child_nodes = cls._nodes_by_id(child)

        parent_types = Counter(
            str(node.get("type", "unknown")) for node in parent_nodes.values()
        )
        child_types = Counter(
            str(node.get("type", "unknown")) for node in child_nodes.values()
        )
        changed_nodes = sorted(
            node_id
            for node_id in parent_nodes.keys() & child_nodes.keys()
            if parent_nodes[node_id] != child_nodes[node_id]
        )

        evidence = {
            "transformation": dict(transformation or {}),
            "parent_node_count": len(parent_nodes),
            "child_node_count": len(child_nodes),
            "node_count_delta": len(child_nodes) - len(parent_nodes),
            "parent_edge_count": len(parent.get("edges", [])),
            "child_edge_count": len(child.get("edges", [])),
            "edge_count_delta": len(child.get("edges", []))
            - len(parent.get("edges", [])),
            "added_node_ids": sorted(child_nodes.keys() - parent_nodes.keys()),
            "removed_node_ids": sorted(parent_nodes.keys() - child_nodes.keys()),
            "changed_node_ids": changed_nodes,
            "node_type_delta": {
                node_type: child_types[node_type] - parent_types[node_type]
                for node_type in sorted(parent_types.keys() | child_types.keys())
                if child_types[node_type] != parent_types[node_type]
            },
            "parent_input_shape": parent.get("input_shape"),
            "child_input_shape": child.get("input_shape"),
            "parent_output_shape": parent.get("output_shape"),
            "child_output_shape": child.get("output_shape"),
        }

        mapping_dict = cls._state_mapping_dict(state_mapping)
        if mapping_dict:
            evidence["state_mapping"] = mapping_dict.get(
                "summary", mapping_dict.get("mapping_summary", {})
            )
        return evidence

    @classmethod
    def collect_execution(
        cls,
        parent_model: nn.Module,
        child_model: nn.Module,
        inputs: Any,
        *,
        targets: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        parent_was_training = parent_model.training
        child_was_training = child_model.training
        parent_model.eval()
        child_model.eval()

        try:
            with torch.inference_mode():
                parent_raw = cls._call_model(parent_model, inputs)
                child_raw = cls._call_model(child_model, inputs)
                parent_output = cls._primary_tensor(parent_raw)
                child_output = cls._primary_tensor(child_raw)

                if parent_output is None or child_output is None:
                    raise ValueError("Both models must produce at least one tensor output.")

                metrics: Dict[str, Any] = {
                    "parent_output_shape": list(parent_output.shape),
                    "child_output_shape": list(child_output.shape),
                    "parent_finite": bool(torch.isfinite(parent_output).all().item()),
                    "child_finite": bool(torch.isfinite(child_output).all().item()),
                }

                if parent_output.shape != child_output.shape:
                    metrics["comparable_outputs"] = False
                    return metrics

                metrics["comparable_outputs"] = True
                parent_flat = parent_output.detach().float().reshape(-1)
                child_flat = child_output.detach().float().reshape(-1)
                delta = child_flat - parent_flat

                metrics.update(
                    {
                        "mean_absolute_delta": float(delta.abs().mean().item())
                        if delta.numel()
                        else 0.0,
                        "mean_squared_delta": float((delta * delta).mean().item())
                        if delta.numel()
                        else 0.0,
                        "relative_l2_delta": cls._relative_l2(parent_flat, delta),
                        "cosine_similarity": cls._cosine_similarity(
                            parent_flat, child_flat
                        ),
                    }
                )

                is_classification = (
                    isinstance(targets, torch.Tensor)
                    and not targets.dtype.is_floating_point
                )
                if (
                    is_classification
                    and parent_output.ndim >= 2
                    and parent_output.shape[-1] > 1
                ):
                    parent_prediction = parent_output.argmax(dim=-1)
                    child_prediction = child_output.argmax(dim=-1)
                    metrics["prediction_agreement"] = float(
                        (parent_prediction == child_prediction).float().mean().item()
                    )

                if criterion is not None and targets is not None:
                    parent_loss = criterion(parent_output, targets)
                    child_loss = criterion(child_output, targets)
                    parent_loss_value = float(parent_loss.detach().item())
                    child_loss_value = float(child_loss.detach().item())
                    metrics.update(
                        {
                            "parent_loss": parent_loss_value,
                            "child_loss": child_loss_value,
                            "loss_delta": child_loss_value - parent_loss_value,
                        }
                    )
                return metrics
        finally:
            parent_model.train(parent_was_training)
            child_model.train(child_was_training)

    @staticmethod
    def _architecture_dict(architecture: Any) -> Dict[str, Any]:
        if architecture is None:
            return {}
        if isinstance(architecture, dict):
            return architecture
        to_dict = getattr(architecture, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        raise TypeError("Architecture must be a mapping or expose to_dict().")

    @staticmethod
    def _nodes_by_id(architecture: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            str(node.get("id")): node
            for node in architecture.get("nodes", [])
            if isinstance(node, dict) and node.get("id") is not None
        }

    @staticmethod
    def _state_mapping_dict(state_mapping: Any) -> Dict[str, Any]:
        if state_mapping is None:
            return {}
        if isinstance(state_mapping, StateMappingReport):
            return state_mapping.to_dict()
        if isinstance(state_mapping, dict):
            nested = state_mapping.get("state_mapping")
            return nested if isinstance(nested, dict) else state_mapping
        raise TypeError("State mapping must be a StateMappingReport or mapping.")

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

    @staticmethod
    def _relative_l2(reference: torch.Tensor, delta: torch.Tensor) -> float:
        denominator = float(torch.linalg.vector_norm(reference).item())
        numerator = float(torch.linalg.vector_norm(delta).item())
        return numerator / max(denominator, 1e-12)

    @staticmethod
    def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        if not left.numel():
            return 1.0
        if torch.equal(left, right):
            return 1.0
        denominator = float(
            (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).item()
        )
        if denominator <= 1e-12:
            return 1.0 if torch.equal(left, right) else 0.0
        similarity = float(torch.dot(left, right).item()) / denominator
        return max(-1.0, min(1.0, similarity))
