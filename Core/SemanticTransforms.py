from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .ArchitectureDescriptor import ArchitectureDescriptor, Node
from .DescriptorModelBuilder import DescriptorModelBuilder
from .ModelTransition import (
    ModelTransition,
    ParameterMapping,
    StateMappingReport,
    canonical_fingerprint,
)
from .WeightCompatibilityEngine import WeightCompatibilityEngine


class TransformationError(ValueError):
    pass


@dataclass(frozen=True)
class LinearWidthPlan:
    node_id: str
    downstream_node_id: str
    path_node_ids: Tuple[str, ...]
    old_width: int
    new_width: int
    index_map: Tuple[int, ...]
    replication_counts: Tuple[int, ...]
    mode: str
    semantic_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "linear_width",
            "node_id": self.node_id,
            "downstream_node_id": self.downstream_node_id,
            "path_node_ids": list(self.path_node_ids),
            "old_width": self.old_width,
            "new_width": self.new_width,
            "index_map": list(self.index_map),
            "replication_counts": list(self.replication_counts),
            "mode": self.mode,
            "semantic_class": self.semantic_class,
        }


@dataclass
class AppliedModelTransition:
    child_descriptor: ArchitectureDescriptor
    child_model: nn.Module
    plan: LinearWidthPlan
    transition: ModelTransition


class LinearWidthTransformation:
    _TRANSPARENT_NODE_TYPES = {
        "ELU",
        "GELU",
        "Identity",
        "LeakyReLU",
        "ReLU",
        "Sigmoid",
        "SiLU",
        "Softplus",
        "Softsign",
        "Tanh",
        "Tanhshrink",
    }

    @classmethod
    def plan(
        cls,
        descriptor: ArchitectureDescriptor,
        node_id: str,
        new_width: int,
    ) -> Tuple[ArchitectureDescriptor, LinearWidthPlan]:
        try:
            descriptor.validate()
        except ValueError as exc:
            raise TransformationError(f"Parent descriptor is invalid: {exc}") from exc
        if not isinstance(new_width, int) or isinstance(new_width, bool):
            raise TransformationError("new_width must be an integer.")
        if new_width <= 0:
            raise TransformationError("new_width must be positive.")

        nodes = {node.id: node for node in descriptor.nodes}
        target = nodes.get(node_id)
        if target is None:
            raise TransformationError(f"Node '{node_id}' does not exist.")
        if target.type != "Linear":
            raise TransformationError("Linear width transformation requires a Linear node.")

        old_width = int(target.params.get("out_features", 0))
        if old_width <= 0:
            raise TransformationError(
                f"Linear node '{node_id}' has no valid out_features value."
            )
        if old_width == new_width:
            raise TransformationError("The requested width is unchanged.")

        downstream, path = cls._find_downstream_linear(descriptor, nodes, node_id)
        downstream_in = int(downstream.params.get("in_features", 0))
        if downstream_in != old_width:
            raise TransformationError(
                f"Downstream Linear '{downstream.id}' expects {downstream_in} features, "
                f"not the target width {old_width}."
            )

        if new_width > old_width:
            index_map = tuple(range(old_width)) + tuple(
                index % old_width for index in range(new_width - old_width)
            )
            mode = "widen"
            semantic_class = "function_preserving"
        else:
            index_map = tuple(range(new_width))
            mode = "narrow"
            semantic_class = "approximate"

        counts = Counter(index_map)
        replication_counts = tuple(counts[index] for index in range(old_width))

        child = copy.deepcopy(descriptor)
        child_nodes = {node.id: node for node in child.nodes}
        child_nodes[node_id].params["out_features"] = new_width
        child_nodes[downstream.id].params["in_features"] = new_width
        child.normalize_inplace()
        child.validate()

        plan = LinearWidthPlan(
            node_id=node_id,
            downstream_node_id=downstream.id,
            path_node_ids=tuple(path),
            old_width=old_width,
            new_width=new_width,
            index_map=index_map,
            replication_counts=replication_counts,
            mode=mode,
            semantic_class=semantic_class,
        )
        return child, plan

    @classmethod
    def apply(
        cls,
        descriptor: ArchitectureDescriptor,
        parent_state_dict: Dict[str, torch.Tensor],
        node_id: str,
        new_width: int,
        *,
        parent_revision_id: str,
        child_revision_id: Optional[str] = None,
    ) -> AppliedModelTransition:
        child_descriptor, plan = cls.plan(descriptor, node_id, new_width)
        child_model = DescriptorModelBuilder.build(child_descriptor)
        state_mapping = cls.transport_state(parent_state_dict, child_model, plan)

        if child_revision_id is None:
            child_revision_id = canonical_fingerprint(
                {
                    "parent_revision_id": parent_revision_id,
                    "transformation": plan.to_dict(),
                    "child_architecture": child_descriptor,
                }
            )

        transition = ModelTransition.create(
            parent_revision_id=parent_revision_id,
            child_revision_id=child_revision_id,
            transformation=plan.to_dict(),
            parent_architecture=descriptor,
            child_architecture=child_descriptor,
        )
        transition.state_mapping = state_mapping
        transition.status = "initialized"
        return AppliedModelTransition(
            child_descriptor=child_descriptor,
            child_model=child_model,
            plan=plan,
            transition=transition,
        )

    @classmethod
    def transport_state(
        cls,
        parent_state_dict: Dict[str, torch.Tensor],
        child_model: nn.Module,
        plan: LinearWidthPlan,
    ) -> StateMappingReport:
        base_transfer = WeightCompatibilityEngine.transfer_weights(
            parent_state_dict,
            child_model,
            allow_partial=False,
            allow_ambiguous_key_mapping=False,
        )
        child_state = child_model.state_dict()

        source_target_weight_key = cls._state_key(
            parent_state_dict, plan.node_id, "weight"
        )
        child_target_weight_key = cls._state_key(
            child_state, plan.node_id, "weight"
        )
        source_downstream_weight_key = cls._state_key(
            parent_state_dict, plan.downstream_node_id, "weight"
        )
        child_downstream_weight_key = cls._state_key(
            child_state, plan.downstream_node_id, "weight"
        )

        source_target_weight = parent_state_dict[source_target_weight_key]
        source_downstream_weight = parent_state_dict[source_downstream_weight_key]
        cls._validate_weight_shapes(
            source_target_weight, source_downstream_weight, child_state, plan
        )

        index = torch.tensor(
            plan.index_map,
            dtype=torch.long,
            device=source_target_weight.device,
        )
        transformed_target_weight = source_target_weight.index_select(0, index)
        transformed_downstream_weight = source_downstream_weight.index_select(1, index)

        if plan.mode == "widen":
            counts = torch.tensor(
                plan.replication_counts,
                dtype=transformed_downstream_weight.dtype,
                device=transformed_downstream_weight.device,
            )
            transformed_downstream_weight = transformed_downstream_weight / counts[
                index.to(counts.device)
            ].unsqueeze(0)

        updates = {
            child_target_weight_key: transformed_target_weight,
            child_downstream_weight_key: transformed_downstream_weight,
        }

        source_target_bias_key = cls._optional_state_key(
            parent_state_dict, plan.node_id, "bias"
        )
        child_target_bias_key = cls._optional_state_key(
            child_state, plan.node_id, "bias"
        )
        if bool(source_target_bias_key) != bool(child_target_bias_key):
            raise TransformationError(
                "Parent and child disagree about the transformed Linear bias."
            )
        if source_target_bias_key and child_target_bias_key:
            updates[child_target_bias_key] = parent_state_dict[
                source_target_bias_key
            ].index_select(0, index)

        for target_key, tensor in updates.items():
            target = child_state[target_key]
            child_state[target_key] = tensor.to(device=target.device, dtype=target.dtype)
        child_model.load_state_dict(child_state, strict=True)

        report = StateMappingReport.from_dict(base_transfer["state_mapping"])
        affected_source_keys = {
            key
            for key in (
                source_target_weight_key,
                source_downstream_weight_key,
                source_target_bias_key,
            )
            if key is not None
        }
        affected_target_keys = {
            key
            for key in (
                child_target_weight_key,
                child_downstream_weight_key,
                child_target_bias_key,
            )
            if key is not None
        }
        report.parameter_mappings = [
            mapping
            for mapping in report.parameter_mappings
            if mapping.source_key not in affected_source_keys
            and mapping.target_key not in affected_target_keys
        ]

        mapping_kind = (
            "net2wider_transform"
            if plan.mode == "widen"
            else "structured_narrow_transform"
        )
        report.parameter_mappings.extend(
            [
                cls._semantic_mapping(
                    source_target_weight_key,
                    child_target_weight_key,
                    parent_state_dict[source_target_weight_key],
                    child_state[child_target_weight_key],
                    mapping_kind,
                    plan,
                    source_elements_used=(
                        parent_state_dict[source_target_weight_key].numel()
                        if plan.mode == "widen"
                        else child_state[child_target_weight_key].numel()
                    ),
                ),
                cls._semantic_mapping(
                    source_downstream_weight_key,
                    child_downstream_weight_key,
                    parent_state_dict[source_downstream_weight_key],
                    child_state[child_downstream_weight_key],
                    mapping_kind,
                    plan,
                    source_elements_used=(
                        parent_state_dict[source_downstream_weight_key].numel()
                        if plan.mode == "widen"
                        else child_state[child_downstream_weight_key].numel()
                    ),
                ),
            ]
        )
        if source_target_bias_key and child_target_bias_key:
            report.parameter_mappings.append(
                cls._semantic_mapping(
                    source_target_bias_key,
                    child_target_bias_key,
                    parent_state_dict[source_target_bias_key],
                    child_state[child_target_bias_key],
                    mapping_kind,
                    plan,
                    source_elements_used=(
                        parent_state_dict[source_target_bias_key].numel()
                        if plan.mode == "widen"
                        else child_state[child_target_bias_key].numel()
                    ),
                )
            )
        report.policy["semantic_transformation"] = plan.to_dict()
        return report

    @classmethod
    def _find_downstream_linear(
        cls,
        descriptor: ArchitectureDescriptor,
        nodes: Dict[str, Node],
        node_id: str,
    ) -> Tuple[Node, List[str]]:
        current = node_id
        path = []
        while True:
            outgoing = [edge for edge in descriptor.edges if edge.source == current]
            if len(outgoing) != 1:
                raise TransformationError(
                    f"Node '{current}' must have exactly one outgoing edge."
                )
            next_id = outgoing[0].target
            if next_id == "output":
                raise TransformationError(
                    "A hidden Linear transformation requires a downstream Linear."
                )
            incoming = [edge for edge in descriptor.edges if edge.target == next_id]
            if len(incoming) != 1:
                raise TransformationError(
                    f"Node '{next_id}' participates in a branch or merge."
                )
            next_node = nodes.get(next_id)
            if next_node is None:
                raise TransformationError(f"Downstream node '{next_id}' is missing.")
            if next_node.type == "Linear":
                return next_node, path
            if next_node.type not in cls._TRANSPARENT_NODE_TYPES:
                raise TransformationError(
                    f"Unsupported node '{next_id}' ({next_node.type}) between Linear layers."
                )
            path.append(next_id)
            current = next_id

    @staticmethod
    def _state_key(
        state_dict: Dict[str, torch.Tensor], node_id: str, parameter: str
    ) -> str:
        key = LinearWidthTransformation._optional_state_key(
            state_dict, node_id, parameter
        )
        if key is None:
            raise TransformationError(
                f"Could not resolve state key for '{node_id}.{parameter}'."
            )
        return key

    @staticmethod
    def _optional_state_key(
        state_dict: Dict[str, torch.Tensor], node_id: str, parameter: str
    ) -> Optional[str]:
        preferred = [
            f"node_modules.{node_id}.{parameter}",
            f"{node_id}.{parameter}",
        ]
        for key in preferred:
            if key in state_dict:
                return key

        candidates = [
            key
            for key in state_dict
            if key.endswith(f".{parameter}") and node_id in key.split(".")
        ]
        if len(candidates) > 1:
            raise TransformationError(
                f"State key for '{node_id}.{parameter}' is ambiguous: {candidates}."
            )
        return candidates[0] if candidates else None

    @staticmethod
    def _validate_weight_shapes(
        source_target_weight: torch.Tensor,
        source_downstream_weight: torch.Tensor,
        child_state: Dict[str, torch.Tensor],
        plan: LinearWidthPlan,
    ) -> None:
        if source_target_weight.ndim != 2 or source_downstream_weight.ndim != 2:
            raise TransformationError("Linear width transport requires rank-2 weights.")
        if source_target_weight.shape[0] != plan.old_width:
            raise TransformationError("Parent target weight does not match the plan width.")
        if source_downstream_weight.shape[1] != plan.old_width:
            raise TransformationError(
                "Parent downstream weight does not match the plan width."
            )

        child_target_key = LinearWidthTransformation._state_key(
            child_state, plan.node_id, "weight"
        )
        child_downstream_key = LinearWidthTransformation._state_key(
            child_state, plan.downstream_node_id, "weight"
        )
        if child_state[child_target_key].shape[0] != plan.new_width:
            raise TransformationError("Child target weight has an unexpected width.")
        if child_state[child_downstream_key].shape[1] != plan.new_width:
            raise TransformationError(
                "Child downstream weight has an unexpected input width."
            )

    @staticmethod
    def _semantic_mapping(
        source_key: str,
        target_key: str,
        source_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        mapping_kind: str,
        plan: LinearWidthPlan,
        *,
        source_elements_used: int,
    ) -> ParameterMapping:
        return ParameterMapping(
            source_key=source_key,
            target_key=target_key,
            mapping_kind=mapping_kind,
            key_resolution="semantic_node_identity",
            source_shape=tuple(source_tensor.shape),
            target_shape=tuple(target_tensor.shape),
            source_dtype=str(source_tensor.dtype),
            target_dtype=str(target_tensor.dtype),
            source_elements=source_tensor.numel(),
            target_elements=target_tensor.numel(),
            source_elements_used=int(source_elements_used),
            target_elements_initialized=target_tensor.numel(),
            note=(
                "Deterministic Net2Wider replication with downstream compensation."
                if plan.mode == "widen"
                else "Deterministic prefix selection; narrowing is not function preserving."
            ),
        )
