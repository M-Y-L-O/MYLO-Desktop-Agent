import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Processing.Models.DescriptorHandling import (
    descriptorToGraph,
    descriptorToDetailedGraph,
    descriptorToHybridGraph,
    loadDescriptorFromBytes,
    extractStateDict,
)
from Processing.Optimization.Neuroevolution import MutationGrammar
from Utils.Other import make_json_serializable

logger = logging.getLogger(__name__)

DESCRIPTOR_FILENAME = "descriptor.json"
MODEL_INFO_FILENAME = "modelInfo.json"


def project_path(filename: str) -> str:
    return os.path.join("temp_project", filename)


def loadProjectDescriptor(model_filepath: str = "") -> ArchitectureDescriptor:
    """Load the canonical descriptor for the current project."""
    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    if os.path.exists(descriptor_path):
        with open(descriptor_path, "r", encoding="utf-8") as f:
            descriptor = ArchitectureDescriptor.from_dict(json.load(f))
            try:
                descriptor.validate()
            except Exception as e:
                logger.warning("Bypassing validation error during load: %s", e)
            return descriptor

    if not model_filepath:
        info_path = project_path("projectInfo.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                model_filepath = json.load(f).get("modelFilepath", "")

    if not model_filepath:
        raise ValueError("No model loaded in the current project")

    full_path = project_path(model_filepath)
    if not os.path.exists(full_path):
        raise ValueError(f"Model file not found: {model_filepath}")

    with open(full_path, "rb") as f:
        model_bytes = f.read()

    is_pytorch = model_filepath.lower().endswith((".pt", ".pth", ".pt2"))
    descriptor = loadDescriptorFromBytes(model_bytes, is_pytorch, input_dim=1, output_dim=1)
    saveProjectDescriptor(descriptor)
    return descriptor


def saveProjectDescriptor(descriptor: ArchitectureDescriptor) -> None:
    os.makedirs("temp_project", exist_ok=True)
    with open(project_path(DESCRIPTOR_FILENAME), "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)


def refreshModelVisualization(
    descriptor: ArchitectureDescriptor,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    expanded_nodes = expanded_nodes or []
    if view_mode == "detailed":
        graph = descriptorToDetailedGraph(descriptor)
    elif view_mode == "hybrid" and expanded_nodes:
        graph = descriptorToHybridGraph(descriptor, expanded_nodes)
    else:
        graph = descriptorToGraph(descriptor)

    if "error" in graph:
        return graph

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = view_mode
    graph["expandedNodes"] = expanded_nodes

    with open(project_path(MODEL_INFO_FILENAME), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(graph), f)

    return graph


def validateDescriptorPayload(descriptor_dict: Dict[str, Any]) -> Dict[str, Any]:
    try:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.normalize_inplace()
        descriptor.validate()
        shapes = descriptor._propagate_shapes(mutate=False)
        return {
            "valid": True,
            "descriptor": descriptor.to_dict(),
            "shapes": {k: v for k, v in shapes.items()},
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


class ModelEditEngine:
    """Deterministic descriptor edit operations for the visual model editor."""

    INSERTABLE_TYPES = [
        "Linear", "ReLU", "Tanh", "Sigmoid", "Dropout", "Identity",
        "LayerNorm", "BatchNorm1d", "BatchNorm2d", "Flatten",
    ]
    SWAPPABLE_ACTIVATIONS = MutationGrammar.ACTIVATION_TYPES
    PROTECTED_TYPES = {"LSTM", "GRU", "MultiheadAttention"}

    @classmethod
    def apply(cls, descriptor: ArchitectureDescriptor, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        working = copy.deepcopy(descriptor)
        handler = {
            "add_node": cls._add_node,
            "remove_node": cls._remove_node,
            "update_node": cls._update_node,
            "swap_node_type": cls._swap_node_type,
            "add_edge": cls._add_edge,
            "remove_edge": cls._remove_edge,
            "insert_after": cls._insert_after,
            "connect": cls._add_edge,
            "scale_width": cls._scale_width,
            "add_skip_connection": cls._add_skip_connection,
        }.get(operation)

        if not handler:
            return {"success": False, "error": f"Unknown operation: {operation}"}

        try:
            result = handler(working, payload)
            if not result.get("success"):
                return result

            try:
                working.normalize_inplace()
            except Exception as e:
                logger.warning("Bypassing shape propagation/normalization error inside edit apply: %s", e)

            try:
                working.validate()
            except Exception as e:
                logger.warning("Bypassing validation error inside edit apply: %s", e)

            return {
                "success": True,
                "descriptor": working.to_dict(),
                "message": result.get("message", f"Applied {operation}"),
                "affected": result.get("affected", {}),
            }
        except Exception as exc:
            logger.exception("Model edit failed: %s", operation)
            return {"success": False, "error": str(exc)}

    @classmethod
    def _add_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        node_type = payload.get("type")
        params = payload.get("params", {})
        connect_from = payload.get("connectFrom") or payload.get("connect_from")
        connect_to = payload.get("connectTo") or payload.get("connect_to")

        if not node_id or not node_type:
            return {"success": False, "error": "nodeId and type are required"}

        if any(n.id == node_id for n in descriptor.nodes):
            return {"success": False, "error": f"Node id already exists: {node_id}"}

        descriptor.nodes.append(Node(id=node_id, type=node_type, params=dict(params)))

        if connect_from:
            descriptor.edges.append(Edge(source=connect_from, target=node_id))
        if connect_to:
            descriptor.edges.append(Edge(source=node_id, target=connect_to))

        return {"success": True, "affected": {"nodeId": node_id}, "message": f"Added node {node_id}"}

    @classmethod
    def _remove_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        force = bool(payload.get("force", False))

        if not node_id:
            return {"success": False, "error": "nodeId is required"}

        protected = set() if force else cls.PROTECTED_TYPES
        if MutationGrammar.remove_layer(descriptor, node_id, protected_types=protected):
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Removed node {node_id}"}

        return {"success": False, "error": f"Could not remove node {node_id}"}

    @classmethod
    def _update_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        new_type = payload.get("type")
        params = payload.get("params")
        new_id = payload.get("newNodeId") or payload.get("new_node_id")

        node = next((n for n in descriptor.nodes if n.id == node_id), None)
        if not node:
            return {"success": False, "error": f"Node not found: {node_id}"}

        if new_type:
            node.type = new_type
        if params is not None:
            node.params = dict(params)

        if new_id and new_id != node_id:
            if any(n.id == new_id for n in descriptor.nodes):
                return {"success": False, "error": f"Node id already exists: {new_id}"}
            old_id = node.id
            node.id = new_id
            for edge in descriptor.edges:
                if edge.source == old_id:
                    edge.source = new_id
                if edge.target == old_id:
                    edge.target = new_id
            node_id = new_id

        return {"success": True, "affected": {"nodeId": node_id}, "message": f"Updated node {node_id}"}

    @classmethod
    def _swap_node_type(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        new_type = payload.get("type")

        if not node_id or not new_type:
            return {"success": False, "error": "nodeId and type are required"}

        node = next((n for n in descriptor.nodes if n.id == node_id), None)
        if not node:
            return {"success": False, "error": f"Node not found: {node_id}"}

        if new_type in cls.SWAPPABLE_ACTIVATIONS and node.type in cls.SWAPPABLE_ACTIVATIONS:
            node.type = new_type
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Swapped activation to {new_type}"}

        if new_type in MutationGrammar.RECURRENT_TYPES and node.type in MutationGrammar.RECURRENT_TYPES:
            if MutationGrammar.swap_recurrent_type(descriptor, node_id):
                return {"success": True, "affected": {"nodeId": node_id}, "message": f"Swapped recurrent type to {new_type}"}
            return {"success": False, "error": "Recurrent swap failed"}

        node.type = new_type
        return {"success": True, "affected": {"nodeId": node_id}, "message": f"Set node type to {new_type}"}

    @classmethod
    def _add_edge(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = payload.get("source") or payload.get("from")
        target = payload.get("target") or payload.get("to")
        source_port = payload.get("sourcePort", payload.get("source_port", "output"))
        target_port = payload.get("targetPort", payload.get("target_port", "input"))

        if not source or not target:
            return {"success": False, "error": "source and target are required"}

        if any(e.source == source and e.target == target for e in descriptor.edges):
            return {"success": False, "error": "Edge already exists"}

        descriptor.edges.append(Edge(source=source, target=target, source_port=source_port, target_port=target_port))
        return {"success": True, "affected": {"source": source, "target": target}, "message": "Edge added"}

    @classmethod
    def _remove_edge(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = payload.get("source") or payload.get("from")
        target = payload.get("target") or payload.get("to")

        if not source or not target:
            return {"success": False, "error": "source and target are required"}

        before = len(descriptor.edges)
        descriptor.edges = [e for e in descriptor.edges if not (e.source == source and e.target == target)]
        if len(descriptor.edges) == before:
            return {"success": False, "error": "Edge not found"}

        return {"success": True, "affected": {"source": source, "target": target}, "message": "Edge removed"}

    @classmethod
    def _insert_after(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        after_node_id = payload.get("afterNodeId") or payload.get("after_node_id")
        node_type = payload.get("type")
        node_id = payload.get("nodeId") or payload.get("node_id")
        params = payload.get("params")

        if not after_node_id or not node_type:
            return {"success": False, "error": "afterNodeId and type are required"}

        if node_id:
            out_dim = MutationGrammar._get_node_output_dim(descriptor, after_node_id) or 64
            if params is None:
                params = cls._default_params_for_type(node_type, out_dim)
            descriptor.nodes.append(Node(id=node_id, type=node_type, params=params))
            outgoing = [e for e in descriptor.edges if e.source == after_node_id]
            if not outgoing:
                return {"success": False, "error": f"Node {after_node_id} has no outgoing edges"}
            original_targets = [e.target for e in outgoing]
            descriptor.edges = [e for e in descriptor.edges if e not in outgoing]
            descriptor.edges.append(Edge(source=after_node_id, target=node_id))
            for target in original_targets:
                descriptor.edges.append(Edge(source=node_id, target=target))
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Inserted {node_type} after {after_node_id}"}

        before_ids = {n.id for n in descriptor.nodes}
        if MutationGrammar.add_layer(descriptor, after_node_id, node_type):
            new_ids = [n.id for n in descriptor.nodes if n.id not in before_ids]
            inserted_id = new_ids[0] if new_ids else after_node_id
            return {"success": True, "affected": {"nodeId": inserted_id}, "message": f"Inserted {node_type} after {after_node_id}"}

        return {"success": False, "error": f"Could not insert {node_type} after {after_node_id}"}

    @classmethod
    def _scale_width(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        factor = float(payload.get("factor", 1.0))

        if not node_id:
            return {"success": False, "error": "nodeId is required"}
        if factor <= 0:
            return {"success": False, "error": "factor must be positive"}

        if MutationGrammar.scale_width(descriptor, node_id, factor):
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Scaled width of {node_id} by {factor}"}
        return {"success": False, "error": f"Could not scale node {node_id}"}

    @classmethod
    def _add_skip_connection(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        from_id = payload.get("from") or payload.get("fromId") or payload.get("from_id")
        to_id = payload.get("to") or payload.get("toId") or payload.get("to_id")

        if not from_id or not to_id:
            return {"success": False, "error": "from and to are required"}

        if MutationGrammar.add_skip_connection(descriptor, from_id, to_id):
            return {"success": True, "affected": {"from": from_id, "to": to_id}, "message": "Skip connection added"}
        return {"success": False, "error": "Skip connection could not be added"}

    @staticmethod
    def _default_params_for_type(node_type: str, in_dim: int) -> Dict[str, Any]:
        if node_type == "Linear":
            return {"in_features": in_dim, "out_features": in_dim}
        if node_type == "Dropout":
            return {"p": 0.2}
        if node_type == "LayerNorm":
            return {"normalized_shape": in_dim}
        if node_type in ("BatchNorm1d", "BatchNorm2d"):
            return {"num_features": in_dim}
        return {}


def applyModelEdit(
    descriptor_dict: Optional[Dict[str, Any]],
    operation: str,
    payload: Dict[str, Any],
    persist: bool = False,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if descriptor_dict:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
    else:
        descriptor = loadProjectDescriptor()

    edit_result = ModelEditEngine.apply(descriptor, operation, payload)
    if not edit_result.get("success"):
        return edit_result

    updated = ArchitectureDescriptor.from_dict(edit_result["descriptor"])

    if persist:
        saveProjectDescriptor(updated)
        _sync_weights_after_edit(updated)

    graph = refreshModelVisualization(updated, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    edit_result["graph"] = graph
    return edit_result


def saveModelDescriptor(
    descriptor_dict: Dict[str, Any],
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    validation = validateDescriptorPayload(descriptor_dict)
    if not validation.get("valid"):
        logger.warning("Saving descriptor with validation errors: %s", validation.get("error"))
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
    else:
        descriptor = ArchitectureDescriptor.from_dict(validation["descriptor"])
    saveProjectDescriptor(descriptor)
    _sync_weights_after_edit(descriptor)
    graph = refreshModelVisualization(descriptor, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    return {"success": True, "descriptor": descriptor.to_dict(), "graph": graph}


def visualizeModel(
    descriptor_dict: Optional[Dict[str, Any]] = None,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if descriptor_dict:
        validation = validateDescriptorPayload(descriptor_dict)
        if not validation.get("valid"):
            return validation
        descriptor = ArchitectureDescriptor.from_dict(validation["descriptor"])
    else:
        descriptor = loadProjectDescriptor()

    expanded_nodes = expanded_nodes or []
    if view_mode == "detailed":
        graph = descriptorToDetailedGraph(descriptor)
    elif view_mode == "hybrid":
        graph = descriptorToHybridGraph(descriptor, expanded_nodes)
    else:
        graph = descriptorToGraph(descriptor)

    if "error" in graph:
        return graph

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = view_mode
    graph["expandedNodes"] = expanded_nodes
    return graph


def expandNodes(
    descriptor_dict: Optional[Dict[str, Any]],
    node_ids: List[str],
    current_expanded: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Expand one or more descriptor nodes into ONNX sub-graphs while keeping others collapsed."""
    if descriptor_dict:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.validate()
    else:
        descriptor = loadProjectDescriptor()

    expanded = list(dict.fromkeys((current_expanded or []) + node_ids))
    graph = descriptorToHybridGraph(descriptor, expanded)
    if "error" in graph:
        return graph

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = "hybrid"
    graph["expandedNodes"] = expanded
    return graph


def collapseNodes(
    descriptor_dict: Optional[Dict[str, Any]],
    node_ids: List[str],
    current_expanded: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collapse previously expanded nodes back to high-level descriptor nodes."""
    if descriptor_dict:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
    else:
        descriptor = loadProjectDescriptor()

    expanded = [nid for nid in (current_expanded or []) if nid not in node_ids]
    graph = descriptorToHybridGraph(descriptor, expanded)
    if "error" in graph:
        return graph

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = "hybrid" if expanded else "summary"
    graph["expandedNodes"] = expanded
    return graph


def getEditorCatalog() -> Dict[str, Any]:
    """Return metadata the visual editor needs (supported ops, limits, etc.)."""
    return {
        "insertableTypes": ModelEditEngine.INSERTABLE_TYPES,
        "swappableActivations": ModelEditEngine.SWAPPABLE_ACTIVATIONS,
        "protectedTypes": sorted(ModelEditEngine.PROTECTED_TYPES),
        "limits": {
            "maxNodes": MutationGrammar.MAX_NODES,
            "maxDepth": MutationGrammar.MAX_DEPTH,
        },
        "operations": [
            "add_node", "remove_node", "update_node", "swap_node_type",
            "add_edge", "remove_edge", "insert_after", "scale_width", "add_skip_connection",
        ],
        "viewModes": ["summary", "detailed", "hybrid"],
    }


def _sync_weights_after_edit(descriptor: ArchitectureDescriptor) -> Optional[Dict[str, Any]]:
    """Best-effort weight transfer when a descriptor is saved after structural edits."""
    info_path = project_path("projectInfo.json")
    if not os.path.exists(info_path):
        return None

    with open(info_path, "r", encoding="utf-8") as f:
        project_info = json.load(f)

    model_filepath = project_info.get("modelFilepath", "")
    if not model_filepath or model_filepath.lower().endswith(".onnx"):
        return None

    weights_path = project_info.get("weightsFilepath", "")
    read_path = project_path(weights_path) if weights_path else project_path(model_filepath)
    if not os.path.exists(read_path):
        return None

    with open(read_path, "rb") as f:
        state_dict = extractStateDict(f.read(), is_pytorch=True)
    if not state_dict:
        return None

    try:
        model = DescriptorModelBuilder.build(descriptor)
        transfer_report = WeightCompatibilityEngine.transfer_weights(state_dict, model)
        checkpoint = {
            "model_config": descriptor.to_dict(),
            "state_dict": model.state_dict(),
        }
        out_path = project_path(model_filepath)
        torch.save(checkpoint, out_path)
        return transfer_report
    except Exception as exc:
        logger.warning("Weight sync after edit failed: %s", exc)
        return None
