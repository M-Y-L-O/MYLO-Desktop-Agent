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

# ---------------------------------------------------------------------------
# Project-scoped undo/redo history
# ---------------------------------------------------------------------------

class _ProjectHistory:
    """Per-project undo/redo stacks with a global registry keyed by project path."""

    _MAX_HISTORY = 50
    _stacks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    @classmethod
    def _key(cls) -> str:
        return project_path("")

    @classmethod
    def _ensure(cls, key: str) -> None:
        if key not in cls._stacks:
            cls._stacks[key] = {"undo": [], "redo": []}

    @classmethod
    def push_undo(cls, descriptor_dict: Dict[str, Any]) -> None:
        key = cls._key()
        cls._ensure(key)
        cls._stacks[key]["undo"].append(copy.deepcopy(descriptor_dict))
        if len(cls._stacks[key]["undo"]) > cls._MAX_HISTORY:
            cls._stacks[key]["undo"].pop(0)
        cls._stacks[key]["redo"].clear()

    @classmethod
    def pop_undo(cls) -> Optional[Dict[str, Any]]:
        key = cls._key()
        cls._ensure(key)
        if not cls._stacks[key]["undo"]:
            return None
        return cls._stacks[key]["undo"].pop()

    @classmethod
    def push_redo(cls, descriptor_dict: Dict[str, Any]) -> None:
        key = cls._key()
        cls._ensure(key)
        cls._stacks[key]["redo"].append(copy.deepcopy(descriptor_dict))

    @classmethod
    def pop_redo(cls) -> Optional[Dict[str, Any]]:
        key = cls._key()
        cls._ensure(key)
        if not cls._stacks[key]["redo"]:
            return None
        return cls._stacks[key]["redo"].pop()

    @classmethod
    def can_undo(cls) -> bool:
        key = cls._key()
        cls._ensure(key)
        return bool(cls._stacks[key]["undo"])

    @classmethod
    def can_redo(cls) -> bool:
        key = cls._key()
        cls._ensure(key)
        return bool(cls._stacks[key]["redo"])

    @classmethod
    def clear_project(cls, key: Optional[str] = None) -> None:
        target = key or cls._key()
        cls._stacks.pop(target, None)


# ---------------------------------------------------------------------------
# Project path helpers
# ---------------------------------------------------------------------------

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
    """Save descriptor to disk. Undo snapshot is taken *after* validation."""
    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    os.makedirs("temp_project", exist_ok=True)

    # Strict validation before persisting — no draft nodes allowed at save time
    descriptor.validate()

    # Save the new descriptor
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)

    # Push previous state to undo (if there was one)
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                current_dict = json.load(f)
                _ProjectHistory.push_undo(current_dict)
        except Exception:
            pass


def undoModelEdit(view_mode: str = "summary", expanded_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _ProjectHistory.can_undo():
        return {"success": False, "error": {"code": "NO_UNDO_HISTORY", "message": "No actions to undo"}}

    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    current_dict = None
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                current_dict = json.load(f)
        except Exception:
            pass

    previous_dict = _ProjectHistory.pop_undo()
    if previous_dict is None:
        return {"success": False, "error": {"code": "NO_UNDO_HISTORY", "message": "No actions to undo"}}

    if current_dict:
        _ProjectHistory.push_redo(current_dict)

    descriptor = ArchitectureDescriptor.from_dict(previous_dict)
    os.makedirs("temp_project", exist_ok=True)
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)

    _sync_weights_after_edit(descriptor)
    graph = refreshModelVisualization(descriptor, view_mode=view_mode, expanded_nodes=expanded_nodes or [])

    try:
        shapes = descriptor._propagate_shapes(mutate=False)
        shapes_dict = {k: v for k, v in shapes.items()}
    except Exception:
        shapes_dict = {}

    return {
        "success": True,
        "descriptor": descriptor.to_dict(),
        "shapes": shapes_dict,
        "graph": graph,
        "message": "Undo applied successfully"
    }


def redoModelEdit(view_mode: str = "summary", expanded_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _ProjectHistory.can_redo():
        return {"success": False, "error": {"code": "NO_REDO_HISTORY", "message": "No actions to redo"}}

    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    current_dict = None
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                current_dict = json.load(f)
        except Exception:
            pass

    next_dict = _ProjectHistory.pop_redo()
    if next_dict is None:
        return {"success": False, "error": {"code": "NO_REDO_HISTORY", "message": "No actions to redo"}}

    if current_dict:
        _ProjectHistory.push_undo(current_dict)

    descriptor = ArchitectureDescriptor.from_dict(next_dict)
    os.makedirs("temp_project", exist_ok=True)
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)

    _sync_weights_after_edit(descriptor)
    graph = refreshModelVisualization(descriptor, view_mode=view_mode, expanded_nodes=expanded_nodes or [])

    try:
        shapes = descriptor._propagate_shapes(mutate=False)
        shapes_dict = {k: v for k, v in shapes.items()}
    except Exception:
        shapes_dict = {}

    return {
        "success": True,
        "descriptor": descriptor.to_dict(),
        "shapes": shapes_dict,
        "graph": graph,
        "message": "Redo applied successfully"
    }


# ---------------------------------------------------------------------------
# Graph reachability / draft node utilities
# ---------------------------------------------------------------------------

def _get_input_node_ids(descriptor: ArchitectureDescriptor) -> Set[str]:
    explicit_inputs = {
        n.id for n in descriptor.nodes
        if n.type.lower() == "input"
    }
    # Always include the virtual 'input' — it's the canonical graph source
    explicit_inputs.add("input")
    return explicit_inputs


def _get_reachable_nodes(descriptor: ArchitectureDescriptor) -> Set[str]:
    """Return all node IDs reachable from any Input source via outgoing edges."""
    # Build adjacency list from all edges
    adj: Dict[str, List[str]] = {}
    all_node_ids = {n.id for n in descriptor.nodes}

    # Initialize adjacency for all known nodes + virtual input/output
    for node_id in all_node_ids:
        adj[node_id] = []
    adj["input"] = []
    adj["output"] = []

    for edge in descriptor.edges:
        if edge.source in adj:
            adj[edge.source].append(edge.target)

    # BFS/DFS from all input sources
    sources = _get_input_node_ids(descriptor)
    reachable: Set[str] = set()
    stack = list(sources)

    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        # Only follow edges from real nodes (not from virtual output)
        for neighbor in adj.get(node_id, []):
            if neighbor not in reachable:
                stack.append(neighbor)

    return reachable


def _get_active_subgraph(descriptor: ArchitectureDescriptor) -> ArchitectureDescriptor:
    reachable = _get_reachable_nodes(descriptor)
    # Filter to only real nodes (exclude virtual input/output)
    active_node_ids = reachable - {"input", "output"}

    active = copy.deepcopy(descriptor)
    active.nodes = [n for n in active.nodes if n.id in active_node_ids]
    active.edges = [
        e for e in active.edges 
        if (e.source in active_node_ids or e.source == "input")
        and (e.target in active_node_ids or e.target == "output")
    ]
    return active


def _get_draft_node_ids(descriptor: ArchitectureDescriptor) -> Set[str]:
    """Return the set of real node IDs that are NOT reachable from any Input source."""
    all_ids = {n.id for n in descriptor.nodes}
    reachable = _get_reachable_nodes(descriptor)
    # Remove virtual nodes from consideration
    reachable_real = reachable - {"input", "output"}
    return all_ids - reachable_real


# ---------------------------------------------------------------------------
# Validation helpers with draft-node awareness
# ---------------------------------------------------------------------------

def _validate_with_drafts(
    descriptor: ArchitectureDescriptor,
    strict: bool = False
) -> None:
    if strict:
        # Legacy behavior: every node must be valid and connected
        descriptor.validate()
        return

    # Lenient mode: only validate the active (reachable) subgraph
    draft_ids = _get_draft_node_ids(descriptor)

    if not draft_ids:
        # No draft nodes — validate everything normally
        descriptor.validate()
        return

    # Extract active subgraph and validate it
    active = _get_active_subgraph(descriptor)

    if active.nodes:
        active.validate()


def _propagate_shapes_with_drafts(
    descriptor: ArchitectureDescriptor
) -> Dict[str, Any]:
    draft_ids = _get_draft_node_ids(descriptor)

    if not draft_ids:
        # No drafts — normal shape propagation
        try:
            return {k: v for k, v in descriptor._propagate_shapes(mutate=False).items()}
        except Exception:
            return {}

    # Work on active subgraph for shape propagation
    active = _get_active_subgraph(descriptor)

    try:
        active_shapes = active._propagate_shapes(mutate=False)
    except Exception:
        active_shapes = {}

    # Merge: active nodes get real shapes, draft nodes get None
    all_shapes: Dict[str, Any] = {}
    for node in descriptor.nodes:
        if node.id in draft_ids:
            all_shapes[node.id] = None  # Draft — no shape yet
        else:
            all_shapes[node.id] = active_shapes.get(node.id)

    return all_shapes


# ---------------------------------------------------------------------------
# Visualization helpers (defined in this module so they are importable)
# ---------------------------------------------------------------------------

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

    draft_ids = _get_draft_node_ids(descriptor)
    graph["draftNodes"] = sorted(draft_ids)
    graph["activeNodes"] = sorted({n.id for n in descriptor.nodes} - draft_ids)

    # Per-node draft flag for convenience
    if "nodes" in graph:
        for node_entry in graph["nodes"]:
            node_entry["isDraft"] = node_entry.get("id") in draft_ids

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = view_mode
    graph["expandedNodes"] = expanded_nodes

    with open(project_path(MODEL_INFO_FILENAME), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(graph), f)

    return graph


def validateDescriptorPayload(descriptor_dict: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """Validate a descriptor payload. 

    By default uses lenient validation (allows draft nodes).
    Pass strict=True for save-time enforcement.
    """
    try:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.normalize_inplace()
        _validate_with_drafts(descriptor, strict=strict)
        shapes = _propagate_shapes_with_drafts(descriptor)
        return {
            "valid": True,
            "descriptor": descriptor.to_dict(),
            "shapes": shapes,
            "draftNodes": sorted(_get_draft_node_ids(descriptor)),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# EditError with structured codes
# ---------------------------------------------------------------------------

class EditError(Exception):
    def __init__(self, code: str, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.field = field
        self.details = details or {}
        super().__init__(message)


# ---------------------------------------------------------------------------
# ModelEditEngine
# ---------------------------------------------------------------------------

class ModelEditEngine:
    """Deterministic descriptor edit operations for the visual model editor.

    EDITING MODE (default):
        - Disconnected/draft nodes are ALLOWED
        - Only the connected subgraph from Input sources is validated
        - Users can freely place, move, and configure nodes before wiring

    STRICT MODE (save/persist):
        - All nodes must be valid and connected
        - Triggered by saveProjectDescriptor() or explicit strict validation
    """

    INSERTABLE_TYPES = [
        "Input", "Output", "Linear", "ReLU", "Tanh", "Sigmoid", "Dropout", "Identity",
        "LayerNorm", "BatchNorm1d", "BatchNorm2d", "Flatten",
    ]
    SWAPPABLE_ACTIVATIONS = MutationGrammar.ACTIVATION_TYPES
    PROTECTED_TYPES = {"LSTM", "GRU", "MultiheadAttention"}

    @classmethod
    def apply(cls, descriptor: ArchitectureDescriptor, operation: str, payload: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": {
                    "code": "UNKNOWN_OPERATION",
                    "message": f"Unknown operation: {operation}",
                    "field": None,
                    "details": {}
                }
            }

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

            working.normalize_inplace()

            # --- KEY CHANGE: use draft-aware validation ---
            _validate_with_drafts(working, strict=strict)

            shapes = _propagate_shapes_with_drafts(working)
            draft_ids = _get_draft_node_ids(working)

            return {
                "success": True,
                "descriptor": working.to_dict(),
                "shapes": shapes,
                "message": result.get("message", f"Applied {operation}"),
                "affected": result.get("affected", {}),
                "draftNodes": sorted(draft_ids),
                "activeNodes": sorted({n.id for n in working.nodes} - draft_ids),
            }
        except EditError as exc:
            return {
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "field": exc.field,
                    "details": exc.details
                }
            }
        except ValueError as exc:
            msg = str(exc)
            code = "VALIDATION_ERROR"
            field = None
            if "Duplicate node ids" in msg:
                code = "DUPLICATE_NODE_ID"
                field = "nodeId"
            elif "graph contains cycles" in msg:
                code = "CYCLE_DETECTED"
            elif "no path to output" in msg:
                code = "DISCONNECTED_GRAPH"
            elif "disconnected nodes" in msg:
                code = "DISCONNECTED_NODES"
            elif "Output shape mismatch" in msg:
                code = "SHAPE_MISMATCH"
            elif "requires compatible input shapes" in msg:
                code = "INCOMPATIBLE_SHAPES"
            elif "concat dimension" in msg:
                code = "INCOMPATIBLE_SHAPES"
            return {
                "success": False,
                "error": {
                    "code": code,
                    "message": msg,
                    "field": field,
                    "details": {}
                }
            }
        except Exception as exc:
            logger.exception("Model edit failed: %s", operation)
            return {
                "success": False,
                "error": {
                    "code": "UNKNOWN_ERROR",
                    "message": str(exc),
                    "field": None,
                    "details": {}
                }
            }

    @classmethod
    def _add_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        node_type = payload.get("type")
        params = payload.get("params", {})
        connect_from = payload.get("connectFrom") or payload.get("connect_from")
        connect_to = payload.get("connectTo") or payload.get("connect_to")
        source_port = payload.get("sourcePort") or payload.get("source_port", "output")
        target_port = payload.get("targetPort") or payload.get("target_port", "input")

        if not node_id:
            raise EditError("MISSING_ARGUMENT", "nodeId is required", field="nodeId")
        if not node_type:
            raise EditError("MISSING_ARGUMENT", "type is required", field="type")

        if any(n.id == node_id for n in descriptor.nodes):
            raise EditError(
                "DUPLICATE_NODE_ID",
                f"Node id already exists: {node_id}",
                field="nodeId",
                details={"existingId": node_id},
            )

        descriptor.nodes.append(Node(id=node_id, type=node_type, params=dict(params)))

        if connect_from:
            descriptor.edges.append(Edge(
                source=connect_from,
                target=node_id,
                source_port=source_port,
                target_port=target_port
            ))
        if connect_to:
            descriptor.edges.append(Edge(
                source=node_id,
                target=connect_to,
                source_port=source_port,
                target_port=target_port
            ))

        return {"success": True, "affected": {"nodeId": node_id}, "message": f"Added node {node_id}"}

    @classmethod
    def _remove_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        force = bool(payload.get("force", False))

        if not node_id:
            raise EditError("MISSING_ARGUMENT", "nodeId is required", field="nodeId")

        if not any(n.id == node_id for n in descriptor.nodes):
            raise EditError("NODE_NOT_FOUND", f"Node not found: {node_id}", field="nodeId", details={"nodeId": node_id})

        protected = set() if force else cls.PROTECTED_TYPES
        if MutationGrammar.remove_layer(descriptor, node_id, protected_types=protected):
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Removed node {node_id}"}

        raise EditError("OPERATION_FAILED", f"Could not remove node {node_id}", details={"nodeId": node_id})

    @classmethod
    def _update_node(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        new_type = payload.get("type")
        params = payload.get("params")
        new_id = payload.get("newNodeId") or payload.get("new_node_id")

        if not node_id:
            raise EditError("MISSING_ARGUMENT", "nodeId is required", field="nodeId")

        node = next((n for n in descriptor.nodes if n.id == node_id), None)
        if not node:
            raise EditError("NODE_NOT_FOUND", f"Node not found: {node_id}", field="nodeId", details={"nodeId": node_id})

        if new_type:
            node.type = new_type
        if params is not None:
            node.params = dict(params)

        if new_id and new_id != node_id:
            if any(n.id == new_id for n in descriptor.nodes):
                raise EditError(
                    "DUPLICATE_NODE_ID",
                    f"Node id already exists: {new_id}",
                    field="newNodeId",
                    details={"existingId": new_id},
                )
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

        if not node_id:
            raise EditError("MISSING_ARGUMENT", "nodeId is required", field="nodeId")
        if not new_type:
            raise EditError("MISSING_ARGUMENT", "type is required", field="type")

        node = next((n for n in descriptor.nodes if n.id == node_id), None)
        if not node:
            raise EditError("NODE_NOT_FOUND", f"Node not found: {node_id}", field="nodeId", details={"nodeId": node_id})

        if new_type in cls.SWAPPABLE_ACTIVATIONS and node.type in cls.SWAPPABLE_ACTIVATIONS:
            node.type = new_type
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Swapped activation to {new_type}"}

        if new_type in MutationGrammar.RECURRENT_TYPES and node.type in MutationGrammar.RECURRENT_TYPES:
            if MutationGrammar.swap_recurrent_type(descriptor, node_id):
                return {"success": True, "affected": {"nodeId": node_id}, "message": f"Swapped recurrent type to {new_type}"}
            raise EditError("OPERATION_FAILED", "Recurrent swap failed", details={"nodeId": node_id, "type": new_type})

        node.type = new_type
        return {"success": True, "affected": {"nodeId": node_id}, "message": f"Set node type to {new_type}"}

    @classmethod
    def _add_edge(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = payload.get("source") or payload.get("from")
        target = payload.get("target") or payload.get("to")
        source_port = payload.get("sourcePort", payload.get("source_port", "output"))
        target_port = payload.get("targetPort", payload.get("target_port", "input"))

        if not source:
            raise EditError("MISSING_ARGUMENT", "source is required", field="source")
        if not target:
            raise EditError("MISSING_ARGUMENT", "target is required", field="target")

        node_ids = {n.id for n in descriptor.nodes}
        if source not in node_ids and source != "input":
            raise EditError("NODE_NOT_FOUND", f"Source node not found: {source}", field="source", details={"nodeId": source})
        if target not in node_ids and target != "output":
            raise EditError("NODE_NOT_FOUND", f"Target node not found: {target}", field="target", details={"nodeId": target})

        if any(e.source == source and e.target == target for e in descriptor.edges):
            raise EditError("DUPLICATE_EDGE", "Edge already exists", details={"source": source, "target": target})

        descriptor.edges.append(Edge(source=source, target=target, source_port=source_port, target_port=target_port))
        return {"success": True, "affected": {"source": source, "target": target}, "message": "Edge added"}

    @classmethod
    def _remove_edge(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = payload.get("source") or payload.get("from")
        target = payload.get("target") or payload.get("to")

        if not source:
            raise EditError("MISSING_ARGUMENT", "source is required", field="source")
        if not target:
            raise EditError("MISSING_ARGUMENT", "target is required", field="target")

        before = len(descriptor.edges)
        descriptor.edges = [e for e in descriptor.edges if not (e.source == source and e.target == target)]
        if len(descriptor.edges) == before:
            raise EditError("EDGE_NOT_FOUND", "Edge not found", details={"source": source, "target": target})

        return {"success": True, "affected": {"source": source, "target": target}, "message": "Edge removed"}

    @classmethod
    def _insert_after(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        after_node_id = payload.get("afterNodeId") or payload.get("after_node_id")
        node_type = payload.get("type")
        node_id = payload.get("nodeId") or payload.get("node_id")
        params = payload.get("params")

        if not after_node_id:
            raise EditError("MISSING_ARGUMENT", "afterNodeId is required", field="afterNodeId")
        if not node_type:
            raise EditError("MISSING_ARGUMENT", "type is required", field="type")

        if not any(n.id == after_node_id for n in descriptor.nodes):
            raise EditError("NODE_NOT_FOUND", f"Node not found: {after_node_id}", field="afterNodeId", details={"nodeId": after_node_id})

        if node_id:
            if any(n.id == node_id for n in descriptor.nodes):
                raise EditError("DUPLICATE_NODE_ID", f"Node id already exists: {node_id}", field="nodeId", details={"existingId": node_id})

            out_dim = MutationGrammar._get_node_output_dim(descriptor, after_node_id) or 64
            if params is None:
                params = cls._default_params_for_type(node_type, out_dim)
            descriptor.nodes.append(Node(id=node_id, type=node_type, params=params))
            outgoing = [e for e in descriptor.edges if e.source == after_node_id]
            if not outgoing:
                raise EditError("OPERATION_FAILED", f"Node {after_node_id} has no outgoing edges", details={"nodeId": after_node_id})
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

        raise EditError("OPERATION_FAILED", f"Could not insert {node_type} after {after_node_id}", details={"afterNodeId": after_node_id, "type": node_type})

    @classmethod
    def _scale_width(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("nodeId") or payload.get("node_id")
        factor = payload.get("factor", 1.0)

        if not node_id:
            raise EditError("MISSING_ARGUMENT", "nodeId is required", field="nodeId")
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            raise EditError("INVALID_ARGUMENT", "factor must be a number", field="factor")
        if factor <= 0:
            raise EditError("INVALID_ARGUMENT", "factor must be positive", field="factor", details={"factor": factor})

        if not any(n.id == node_id for n in descriptor.nodes):
            raise EditError("NODE_NOT_FOUND", f"Node not found: {node_id}", field="nodeId", details={"nodeId": node_id})

        if MutationGrammar.scale_width(descriptor, node_id, factor):
            return {"success": True, "affected": {"nodeId": node_id}, "message": f"Scaled width of {node_id} by {factor}"}
        raise EditError("OPERATION_FAILED", f"Could not scale node {node_id}", details={"nodeId": node_id})

    @classmethod
    def _add_skip_connection(cls, descriptor: ArchitectureDescriptor, payload: Dict[str, Any]) -> Dict[str, Any]:
        from_id = payload.get("from") or payload.get("fromId") or payload.get("from_id")
        to_id = payload.get("to") or payload.get("toId") or payload.get("to_id")

        if not from_id:
            raise EditError("MISSING_ARGUMENT", "from is required", field="from")
        if not to_id:
            raise EditError("MISSING_ARGUMENT", "to is required", field="to")

        node_ids = {n.id for n in descriptor.nodes}
        if from_id not in node_ids:
            raise EditError("NODE_NOT_FOUND", f"Source node not found: {from_id}", field="from", details={"nodeId": from_id})
        if to_id not in node_ids:
            raise EditError("NODE_NOT_FOUND", f"Target node not found: {to_id}", field="to", details={"nodeId": to_id})

        if MutationGrammar.add_skip_connection(descriptor, from_id, to_id):
            return {"success": True, "affected": {"from": from_id, "to": to_id}, "message": "Skip connection added"}
        raise EditError("OPERATION_FAILED", "Skip connection could not be added", details={"from": from_id, "to": to_id})

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


# ---------------------------------------------------------------------------
# High-level API functions
# ---------------------------------------------------------------------------

def applyModelEdit(
    descriptor_dict: Optional[Dict[str, Any]],
    operation: str,
    payload: Dict[str, Any],
    persist: bool = False,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if descriptor_dict:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
    else:
        descriptor = loadProjectDescriptor()

    # Use strict validation only when actually persisting to disk
    strict = persist

    edit_result = ModelEditEngine.apply(descriptor, operation, payload, strict=strict)
    if not edit_result.get("success"):
        return edit_result

    updated = ArchitectureDescriptor.from_dict(edit_result["descriptor"])

    if not dry_run and persist:
        saveProjectDescriptor(updated)
        _sync_weights_after_edit(updated)

    if dry_run:
        graph = visualizeModel(updated.to_dict(), view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    else:
        graph = refreshModelVisualization(updated, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    edit_result["graph"] = graph
    return edit_result


def saveModelDescriptor(
    descriptor_dict: Dict[str, Any],
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Save a model descriptor. Uses STRICT validation — no draft nodes allowed."""
    validation = validateDescriptorPayload(descriptor_dict, strict=True)
    if not validation.get("valid"):
        logger.warning("Saving descriptor with validation errors: %s", validation.get("error"))
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
    else:
        descriptor = ArchitectureDescriptor.from_dict(validation["descriptor"])
    saveProjectDescriptor(descriptor)
    _sync_weights_after_edit(descriptor)
    graph = refreshModelVisualization(descriptor, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    return {"success": True, "descriptor": descriptor.to_dict(), "graph": graph, "shapes": validation.get("shapes", {})}


def checkEdgeCompatibility(
    descriptor_dict: Dict[str, Any],
    source: str,
    target: str,
    source_port: str = "output",
    target_port: str = "input"
) -> Dict[str, Any]:
    from Processing.Models.node_catalog import PORT_DEFINITIONS, DEFAULT_PORTS, COMPATIBILITY_HINTS

    def _ports_for(node_type: str) -> Dict[str, Any]:
        return PORT_DEFINITIONS.get(node_type, DEFAULT_PORTS)

    try:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        node_map = {node.id: node for node in descriptor.nodes}

        src_node = node_map.get(source)
        tgt_node = node_map.get(target)

        if src_node is None and source != "input":
            return {"compatible": False, "code": "NODE_NOT_FOUND", "field": "source",
                    "error": f"Source node not found: {source}"}
        if tgt_node is None and target != "output":
            return {"compatible": False, "code": "NODE_NOT_FOUND", "field": "target",
                    "error": f"Target node not found: {target}"}

        # For virtual input/output, we can't check ports — skip port validation
        if source != "input" and src_node is not None:
            src_ports = _ports_for(src_node.type)
            if not any(p["id"] == source_port for p in src_ports["outputs"]):
                return {
                    "compatible": False,
                    "code": "PORT_NOT_FOUND",
                    "field": "sourcePort",
                    "error": f"Source port '{source_port}' does not exist on {src_node.type}",
                    "availablePorts": [p["id"] for p in src_ports["outputs"]],
                }

        if target != "output" and tgt_node is not None:
            tgt_ports = _ports_for(tgt_node.type)
            if not any(p["id"] == target_port for p in tgt_ports["inputs"]):
                return {
                    "compatible": False,
                    "code": "PORT_NOT_FOUND",
                    "field": "targetPort",
                    "error": f"Target port '{target_port}' does not exist on {tgt_node.type}",
                    "availablePorts": [p["id"] for p in tgt_ports["inputs"]],
                }

            # Check compatibility hints only for real target nodes
            hint = COMPATIBILITY_HINTS.get(tgt_node.type)
            if hint and src_node is not None and src_node.type not in hint.get("accepts", []) and "*" not in hint.get("accepts", []):
                return {
                    "compatible": False,
                    "code": "COMPATIBILITY_VIOLATION",
                    "field": None,
                    "error": f"{tgt_node.type} does not accept connections from {src_node.type}. {hint.get('note', '')}",
                    "hint": hint,
                }

        if any(e.source == source and e.target == target for e in descriptor.edges):
            return {"compatible": False, "code": "DUPLICATE_EDGE", "field": None,
                    "error": "Edge already exists"}

        descriptor.edges.append(Edge(source=source, target=target, source_port=source_port, target_port=target_port))
        descriptor.normalize_inplace()
        _validate_with_drafts(descriptor, strict=False)

        shapes = _propagate_shapes_with_drafts(descriptor)
        return {
            "compatible": True,
            "sourceShape": shapes.get(source),
            "targetShape": shapes.get(target),
            "warnings": []
        }
    except Exception as exc:
        msg = str(exc)
        code = "INCOMPATIBLE_EDGE"
        if "contains cycles" in msg:
            code = "CYCLE_DETECTED"
        elif "multiple inputs" in msg:
            code = "MULTI_INPUT_VIOLATION"
        return {"compatible": False, "error": msg, "code": code, "field": None}


def visualizeModel(
    descriptor_dict: Optional[Dict[str, Any]] = None,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if descriptor_dict:
        validation = validateDescriptorPayload(descriptor_dict, strict=False)
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

    # --- DRAFT NODE METADATA ---
    draft_ids = _get_draft_node_ids(descriptor)
    graph["draftNodes"] = sorted(draft_ids)
    graph["activeNodes"] = sorted({n.id for n in descriptor.nodes} - draft_ids)
    if "nodes" in graph:
        for node_entry in graph["nodes"]:
            node_entry["isDraft"] = node_entry.get("id") in draft_ids

    graph["descriptor"] = descriptor.to_dict()
    graph["viewMode"] = view_mode
    graph["expandedNodes"] = expanded_nodes
    return graph


def expandNodes(
    descriptor_dict: Optional[Dict[str, Any]],
    node_ids: List[str],
    current_expanded: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Expand one or more descriptor nodes into ONNX op-level sub-graphs while keeping others collapsed."""
    if descriptor_dict:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        _validate_with_drafts(descriptor, strict=False)
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
