import copy
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Processing.Models.DescriptorHandling import (
    descriptorToGraph,
    descriptorToDetailedGraph,
    descriptorToHybridGraph,
    descriptorToOnnx,
    loadDescriptorFromBytes,
    extractStateDict,
)
from Processing.Optimization.Neuroevolution import MutationGrammar
from Utils.Other import make_json_serializable

logger = logging.getLogger(__name__)

DESCRIPTOR_FILENAME = "descriptor.json"
MODEL_INFO_FILENAME = "modelInfo.json"


# ---------------------------------------------------------------------------
# Blocked-action reasons (Python console + future user-facing details)
# ---------------------------------------------------------------------------

def log_blocked(
    action: str,
    reason: str,
    *,
    code: Optional[str] = None,
    **details: Any,
) -> None:
    """Print why an editor action cannot complete (agent console + logger).

    ``reason`` / ``code`` / ``details`` are structured so the same payload can
    later drive UI toasts without rewriting call sites.
    """
    code_s = f" [{code}]" if code else ""
    detail_s = f" | {details}" if details else ""
    msg = f'[ModelEditor] blocked "{action}"{code_s}: {reason}{detail_s}'
    print(msg, flush=True)
    logger.warning("%s", msg)


def blocked_edit_result(
    action: str,
    code: str,
    message: str,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Log a block and return the standard ``{success: False, error: {...}}`` shape."""
    details = details or {}
    log_blocked(action, message, code=code, field=field, **details)
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "field": field,
            "details": details,
        },
    }

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
    def push_undo(cls, descriptor_dict: Dict[str, Any], *, clear_redo: bool = True) -> None:
        key = cls._key()
        cls._ensure(key)
        cls._stacks[key]["undo"].append(copy.deepcopy(descriptor_dict))
        if len(cls._stacks[key]["undo"]) > cls._MAX_HISTORY:
            cls._stacks[key]["undo"].pop(0)
        if clear_redo:
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
    def status(cls) -> Dict[str, bool]:
        return {"canUndo": cls.can_undo(), "canRedo": cls.can_redo()}

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

    is_pytorch = model_filepath.lower().endswith((".pt", ".pth", ".pt2", ".ptpkg", ".torchpackage"))
    if model_filepath.lower().endswith(".pt2"):
        try:
            from Processing.Models.TorchExportProcessing import (
                NotTorchExportArtifact,
                TorchExportImportError,
                import_torch_export,
            )
            descriptor, _, _ = import_torch_export(full_path)
            saveProjectDescriptor(descriptor)
            return descriptor
        except NotTorchExportArtifact:
            pass
        except TorchExportImportError:
            raise
    descriptor = loadDescriptorFromBytes(model_bytes, is_pytorch, input_dim=1, output_dim=1)
    saveProjectDescriptor(descriptor)
    return descriptor


def saveProjectDescriptor(
    descriptor: ArchitectureDescriptor,
    *,
    record_undo: bool = False,
    strict: bool = True,
) -> None:
    """Save descriptor to disk.

    ``record_undo`` is opt-in. Edit operations push the pre-edit descriptor
    themselves; an explicit Save should not invent a fake history entry from
    the post-write file (that previously made undo jump back to older saves).

    ``strict`` controls validation: canvas edits may write draft/disconnected
    graphs, while Save / persist paths should fail closed.
    """
    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    os.makedirs("temp_project", exist_ok=True)

    try:
        descriptor.validate()
    except Exception as exc:
        if strict:
            raise
        logger.warning("Saving draft descriptor with validation errors: %s", exc)

    previous_dict: Optional[Dict[str, Any]] = None
    if record_undo and os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                previous_dict = json.load(f)
        except Exception:
            previous_dict = None

    new_dict = descriptor.to_dict()
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(new_dict, f, indent=2)

    if record_undo and previous_dict is not None and previous_dict != new_dict:
        _ProjectHistory.push_undo(previous_dict)


def undoModelEdit(view_mode: str = "summary", expanded_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _ProjectHistory.can_undo():
        return {
            **blocked_edit_result(
                "undo",
                "NO_UNDO_HISTORY",
                "No actions to undo — undo stack is empty",
            ),
            **_ProjectHistory.status(),
        }

    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    current_dict = None
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                current_dict = json.load(f)
        except Exception as exc:
            log_blocked("undo", "Could not read current descriptor before undo (continuing)", code="READ_FAILED", error=str(exc))

    previous_dict = _ProjectHistory.pop_undo()
    if previous_dict is None:
        return {
            **blocked_edit_result(
                "undo",
                "NO_UNDO_HISTORY",
                "No actions to undo — pop_undo returned empty after can_undo was true",
            ),
            **_ProjectHistory.status(),
        }

    if current_dict:
        _ProjectHistory.push_redo(current_dict)

    descriptor = ArchitectureDescriptor.from_dict(previous_dict)
    os.makedirs("temp_project", exist_ok=True)
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)

    _sync_model_file_after_edit(descriptor)
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
        "warnings": graph.get("warnings") or _collect_descriptor_warnings(descriptor),
        "graph": graph,
        "message": "Undo applied successfully",
        **_ProjectHistory.status(),
    }


def redoModelEdit(view_mode: str = "summary", expanded_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not _ProjectHistory.can_redo():
        return {
            **blocked_edit_result(
                "redo",
                "NO_REDO_HISTORY",
                "No actions to redo — redo stack is empty",
            ),
            **_ProjectHistory.status(),
        }

    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    current_dict = None
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                current_dict = json.load(f)
        except Exception as exc:
            log_blocked("redo", "Could not read current descriptor before redo (continuing)", code="READ_FAILED", error=str(exc))

    next_dict = _ProjectHistory.pop_redo()
    if next_dict is None:
        return {
            **blocked_edit_result(
                "redo",
                "NO_REDO_HISTORY",
                "No actions to redo — pop_redo returned empty after can_redo was true",
            ),
            **_ProjectHistory.status(),
        }

    # Preserve remaining redo entries — push_undo(clear_redo=True) would wipe them.
    if current_dict:
        _ProjectHistory.push_undo(current_dict, clear_redo=False)

    descriptor = ArchitectureDescriptor.from_dict(next_dict)
    os.makedirs("temp_project", exist_ok=True)
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)

    _sync_model_file_after_edit(descriptor)
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
        "warnings": graph.get("warnings") or _collect_descriptor_warnings(descriptor),
        "graph": graph,
        "message": "Redo applied successfully",
        **_ProjectHistory.status(),
    }


# ---------------------------------------------------------------------------
# Graph reachability / draft node utilities
# ---------------------------------------------------------------------------

def _explicit_input_ids(descriptor: ArchitectureDescriptor) -> Set[str]:
    return {n.id for n in descriptor.nodes if n.type.lower() == "input"}


def _explicit_output_ids(descriptor: ArchitectureDescriptor) -> Set[str]:
    return {n.id for n in descriptor.nodes if n.type.lower() == "output"}


def _uses_virtual_input(descriptor: ArchitectureDescriptor) -> bool:
    return any(e.source == "input" or e.target == "input" for e in descriptor.edges)


def _uses_virtual_output(descriptor: ArchitectureDescriptor) -> bool:
    return any(e.source == "output" or e.target == "output" for e in descriptor.edges)


def _shows_virtual_input(descriptor: ArchitectureDescriptor) -> bool:
    """Virtual input is shown when wired, or when no explicit Input nodes exist."""
    return _uses_virtual_input(descriptor) or not _explicit_input_ids(descriptor)


def _shows_virtual_output(descriptor: ArchitectureDescriptor) -> bool:
    """Virtual output is shown when wired, or when no explicit Output nodes exist."""
    return _uses_virtual_output(descriptor) or not _explicit_output_ids(descriptor)


def _input_role_count(descriptor: ArchitectureDescriptor) -> int:
    return len(_explicit_input_ids(descriptor)) + (1 if _shows_virtual_input(descriptor) else 0)


def _output_role_count(descriptor: ArchitectureDescriptor) -> int:
    return len(_explicit_output_ids(descriptor)) + (1 if _shows_virtual_output(descriptor) else 0)


def _get_input_node_ids(descriptor: ArchitectureDescriptor) -> Set[str]:
    sources = set(_explicit_input_ids(descriptor))
    if _shows_virtual_input(descriptor):
        sources.add("input")
    return sources


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
        descriptor.validate(strict=True)
        return

    # Lenient mode: only validate the active (reachable) subgraph
    draft_ids = _get_draft_node_ids(descriptor)

    if not draft_ids:
        # No draft nodes — validate everything normally
        descriptor.validate(strict=False)
        return

    # Extract active subgraph and validate it
    active = _get_active_subgraph(descriptor)

    if active.nodes:
        active.validate(strict=False)


def _propagate_shapes_with_drafts(
    descriptor: ArchitectureDescriptor,
    strict: bool = False
) -> Dict[str, Any]:
    draft_ids = _get_draft_node_ids(descriptor)

    if not draft_ids:
        # No drafts — normal shape propagation
        try:
            return {k: v for k, v in descriptor._propagate_shapes(mutate=False, strict=strict).items()}
        except Exception:
            return {}

    # Work on active subgraph for shape propagation
    active = _get_active_subgraph(descriptor)

    try:
        active_shapes = active._propagate_shapes(mutate=False, strict=strict)
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


def _collect_descriptor_warnings(descriptor: ArchitectureDescriptor) -> List[str]:
    """Soft shape / contract warnings for the editor (never fail the edit)."""
    draft_ids = _get_draft_node_ids(descriptor)
    target = _get_active_subgraph(descriptor) if draft_ids else descriptor
    if not target.nodes and not any(e.target == "output" for e in target.edges):
        return []
    try:
        return target.collect_shape_warnings()
    except Exception as exc:
        return [str(exc)]


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
    graph["warnings"] = _collect_descriptor_warnings(descriptor)
    graph["shapes"] = _propagate_shapes_with_drafts(descriptor, strict=False)

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
    Soft shape issues are always returned in ``warnings`` so the editor can
    show them even when the descriptor is structurally valid.
    """
    try:
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.normalize_inplace(strict=strict)
        _validate_with_drafts(descriptor, strict=strict)
        shapes = _propagate_shapes_with_drafts(descriptor, strict=strict)
        warnings = _collect_descriptor_warnings(descriptor)
        result = {
            "valid": True,
            "descriptor": descriptor.to_dict(),
            "shapes": shapes,
            "draftNodes": sorted(_get_draft_node_ids(descriptor)),
            "warnings": warnings,
        }
        # Strict callers (export/compile) treat shape warnings as hard failures
        if strict and warnings:
            result["valid"] = False
            result["error"] = {
                "code": "SHAPE_MISMATCH" if any("shape mismatch" in w.lower() for w in warnings) else "VALIDATION_WARNING",
                "message": warnings[0],
                "details": {"warnings": warnings},
            }
            log_blocked("validate", warnings[0], code=result["error"]["code"], strict=strict)
        return result
    except Exception as exc:
        log_blocked("validate", str(exc), code="VALIDATION_ERROR", strict=strict)
        return {
            "valid": False,
            "error": {"code": "VALIDATION_ERROR", "message": str(exc)},
            "warnings": [str(exc)],
        }


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
    PROTECTED_TYPES = {} # "LSTM", "GRU", "MultiheadAttention" - why not allow edits to these? maybe allow edits but warn user

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
            return blocked_edit_result(
                operation or "apply",
                "UNKNOWN_OPERATION",
                f"Unknown operation: {operation}",
                details={"payloadKeys": list(payload.keys()) if isinstance(payload, dict) else []},
            )

        try:
            result = handler(working, payload)
            if not result.get("success"):
                err = result.get("error") or {}
                if isinstance(err, dict):
                    log_blocked(
                        operation,
                        err.get("message") or "Handler returned success=False",
                        code=err.get("code"),
                        field=err.get("field"),
                        **(err.get("details") or {}),
                    )
                else:
                    log_blocked(operation, str(err) or "Handler returned success=False")
                return result

            try:
                working.normalize_inplace(strict=strict)
            except Exception as e:
                logger.warning("Bypassing shape propagation/normalization error inside edit apply: %s", e)

            try:
                working.validate(strict=strict)
            except Exception as e:
                logger.warning("Bypassing validation error inside edit apply: %s", e)

            try:
                working.normalize_inplace(strict=strict)
            except Exception as e:
                logger.warning("Bypassing second shape propagation/normalization error inside edit apply: %s", e)

            # --- KEY CHANGE: use draft-aware validation ---
            _validate_with_drafts(working, strict=strict)

            shapes = _propagate_shapes_with_drafts(working, strict=strict)
            draft_ids = _get_draft_node_ids(working)
            warnings = _collect_descriptor_warnings(working)

            return {
                "success": True,
                "descriptor": working.to_dict(),
                "shapes": shapes,
                "warnings": warnings,
                "message": result.get("message", f"Applied {operation}"),
                "affected": result.get("affected", {}),
                "draftNodes": sorted(draft_ids),
                "activeNodes": sorted({n.id for n in working.nodes} - draft_ids),
            }
        except EditError as exc:
            return blocked_edit_result(
                operation,
                exc.code,
                exc.message,
                field=exc.field,
                details=exc.details,
            )
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
            return blocked_edit_result(operation, code, msg, field=field)
        except Exception as exc:
            logger.exception("Model edit failed: %s", operation)
            return blocked_edit_result(
                operation,
                "UNKNOWN_ERROR",
                str(exc),
                details={"exceptionType": type(exc).__name__},
            )

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

        # Virtual terminals live outside descriptor.nodes. Removable only when
        # another Input/Output-typed node can take the role (keep ≥1 of each).
        if node_id == "input":
            if _input_role_count(descriptor) <= 1:
                raise EditError(
                    "PROTECTED_NODE",
                    "Cannot remove the only input node",
                    field="nodeId",
                    details={"nodeId": node_id},
                )
            descriptor.edges = [
                e for e in descriptor.edges
                if e.source != "input" and e.target != "input"
            ]
            return {
                "success": True,
                "affected": {"nodeId": node_id},
                "message": f"Removed virtual node {node_id}",
            }

        if node_id == "output":
            if _output_role_count(descriptor) <= 1:
                raise EditError(
                    "PROTECTED_NODE",
                    "Cannot remove the only output node",
                    field="nodeId",
                    details={"nodeId": node_id},
                )
            descriptor.edges = [
                e for e in descriptor.edges
                if e.source != "output" and e.target != "output"
            ]
            return {
                "success": True,
                "affected": {"nodeId": node_id},
                "message": f"Removed virtual node {node_id}",
            }

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
            return {
                "success": True,
                "affected": {"source": source, "target": target},
                "message": "Edge already absent",
            }

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

    # Snapshot pre-edit state for session undo (previews must not pollute history)
    before_dict = descriptor.to_dict()

    # Use strict validation only when actually persisting to disk
    strict = persist

    edit_result = ModelEditEngine.apply(descriptor, operation, payload, strict=strict)
    if not edit_result.get("success"):
        # ModelEditEngine.apply already logs; note dry_run/persist context for the console
        err = (edit_result.get("error") or {}) if isinstance(edit_result.get("error"), dict) else {}
        log_blocked(
            "applyModelEdit",
            err.get("message") or "Edit rejected",
            code=err.get("code"),
            operation=operation,
            persist=persist,
            dry_run=dry_run,
            strict=strict,
        )
        return {**edit_result, **_ProjectHistory.status()}

    updated = ArchitectureDescriptor.from_dict(edit_result["descriptor"])

    if not dry_run:
        # Record the pre-edit descriptor so Undo reverses this edit (not an older save).
        if before_dict != updated.to_dict():
            _ProjectHistory.push_undo(before_dict)
        # Always write descriptor.json so undo/redo's "current" state matches the
        # canvas. ``persist`` additionally syncs the .pt2/weights artifact.
        # History is already recorded above — never push again from the file.
        saveProjectDescriptor(updated, record_undo=False, strict=persist)
        if persist:
            _sync_model_file_after_edit(updated)

    if dry_run:
        graph = visualizeModel(updated.to_dict(), view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    else:
        graph = refreshModelVisualization(updated, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    edit_result["graph"] = graph
    if isinstance(graph, dict) and graph.get("warnings") is not None:
        edit_result["warnings"] = graph["warnings"]
    elif "warnings" not in edit_result:
        edit_result["warnings"] = _collect_descriptor_warnings(updated)
    edit_result.update(_ProjectHistory.status())
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

    # Edits already push undo snapshots. Save only commits descriptor.json +
    # the model artifact so reopen/export cannot fall back to a stale .pt2.
    previous_disk: Optional[Dict[str, Any]] = None
    descriptor_path = project_path(DESCRIPTOR_FILENAME)
    if os.path.exists(descriptor_path):
        try:
            with open(descriptor_path, "r", encoding="utf-8") as f:
                previous_disk = json.load(f)
        except Exception:
            previous_disk = None

    new_dict = descriptor.to_dict()
    if previous_disk is not None and previous_disk != new_dict:
        _ProjectHistory.push_undo(previous_disk)

    saveProjectDescriptor(descriptor, record_undo=False, strict=True)
    _sync_model_file_after_edit(descriptor)
    graph = refreshModelVisualization(descriptor, view_mode=view_mode, expanded_nodes=expanded_nodes or [])
    warnings = validation.get("warnings") or graph.get("warnings") or _collect_descriptor_warnings(descriptor)
    return {
        "success": True,
        "descriptor": descriptor.to_dict(),
        "graph": graph,
        "shapes": validation.get("shapes", {}) or graph.get("shapes", {}),
        "warnings": warnings,
        **_ProjectHistory.status(),
    }


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
            reason = f"Source node not found: {source}"
            log_blocked("checkEdge", reason, code="NODE_NOT_FOUND", source=source, target=target)
            return {"compatible": False, "code": "NODE_NOT_FOUND", "field": "source", "error": reason}
        if tgt_node is None and target != "output":
            reason = f"Target node not found: {target}"
            log_blocked("checkEdge", reason, code="NODE_NOT_FOUND", source=source, target=target)
            return {"compatible": False, "code": "NODE_NOT_FOUND", "field": "target", "error": reason}

        # For virtual input/output, we can't check ports — skip port validation
        if source != "input" and src_node is not None:
            src_ports = _ports_for(src_node.type)
            if not any(p["id"] == source_port for p in src_ports["outputs"]):
                reason = f"Source port '{source_port}' does not exist on {src_node.type}"
                available = [p["id"] for p in src_ports["outputs"]]
                log_blocked(
                    "checkEdge", reason, code="PORT_NOT_FOUND",
                    source=source, target=target, sourcePort=source_port, availablePorts=available,
                )
                return {
                    "compatible": False,
                    "code": "PORT_NOT_FOUND",
                    "field": "sourcePort",
                    "error": reason,
                    "availablePorts": available,
                }

        if target != "output" and tgt_node is not None:
            tgt_ports = _ports_for(tgt_node.type)
            if not any(p["id"] == target_port for p in tgt_ports["inputs"]):
                reason = f"Target port '{target_port}' does not exist on {tgt_node.type}"
                available = [p["id"] for p in tgt_ports["inputs"]]
                log_blocked(
                    "checkEdge", reason, code="PORT_NOT_FOUND",
                    source=source, target=target, targetPort=target_port, availablePorts=available,
                )
                return {
                    "compatible": False,
                    "code": "PORT_NOT_FOUND",
                    "field": "targetPort",
                    "error": reason,
                    "availablePorts": available,
                }

            # Check compatibility hints only for real target nodes
            hint = COMPATIBILITY_HINTS.get(tgt_node.type)
            if hint and src_node is not None and src_node.type not in hint.get("accepts", []) and "*" not in hint.get("accepts", []):
                reason = (
                    f"{tgt_node.type} does not accept connections from {src_node.type}. "
                    f"{hint.get('note', '')}"
                ).strip()
                log_blocked(
                    "checkEdge", reason, code="COMPATIBILITY_VIOLATION",
                    source=source, target=target, sourceType=src_node.type, targetType=tgt_node.type,
                )
                return {
                    "compatible": False,
                    "code": "COMPATIBILITY_VIOLATION",
                    "field": None,
                    "error": reason,
                    "hint": hint,
                }

        if any(e.source == source and e.target == target for e in descriptor.edges):
            reason = "Edge already exists"
            log_blocked("checkEdge", reason, code="DUPLICATE_EDGE", source=source, target=target)
            return {"compatible": False, "code": "DUPLICATE_EDGE", "field": None, "error": reason}

        descriptor.edges.append(Edge(source=source, target=target, source_port=source_port, target_port=target_port))
        descriptor.normalize_inplace(strict=False)
        _validate_with_drafts(descriptor, strict=False)

        shapes = _propagate_shapes_with_drafts(descriptor, strict=False)
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
        log_blocked("checkEdge", msg, code=code, source=source, target=target)
        return {"compatible": False, "error": msg, "code": code, "field": None}


def visualizeModel(
    descriptor_dict: Optional[Dict[str, Any]] = None,
    view_mode: str = "summary",
    expanded_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if descriptor_dict:
        validation = validateDescriptorPayload(descriptor_dict, strict=False)
        if not validation.get("valid"):
            # validateDescriptorPayload already logs
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
        log_blocked(
            "visualize",
            str(graph.get("error")),
            code=graph.get("code") or "VISUALIZE_FAILED",
            viewMode=view_mode,
        )
        return graph

    # --- DRAFT NODE METADATA ---
    draft_ids = _get_draft_node_ids(descriptor)
    graph["draftNodes"] = sorted(draft_ids)
    graph["activeNodes"] = sorted({n.id for n in descriptor.nodes} - draft_ids)
    graph["warnings"] = _collect_descriptor_warnings(descriptor)
    graph["shapes"] = _propagate_shapes_with_drafts(descriptor, strict=False)
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
        from Processing.Models.TorchPackageProcessing import (
            export_mylo_torch_package,
            is_torch_package_archive,
        )
        if is_torch_package_archive(read_path):
            fd, temporary_path = tempfile.mkstemp(
                prefix="mylo_package_",
                suffix=os.path.splitext(out_path)[1] or ".ptpkg",
                dir=os.path.dirname(out_path) or ".",
            )
            os.close(fd)
            try:
                export_mylo_torch_package(model, descriptor, temporary_path)
                os.replace(temporary_path, out_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        else:
            torch.save(checkpoint, out_path)
        return transfer_report
    except Exception as exc:
        logger.warning("Weight sync after edit failed: %s", exc)
        return None


def _sync_json_pt2_after_edit(descriptor: ArchitectureDescriptor) -> bool:
    """Rewrite JSON-architecture .pt2 files so reopen/analyse sees the latest save.

    Editor-created models are plain JSON written as ``.pt2``. Saving only
    ``descriptor.json`` left the ``.pt2`` stale; a later analyse/visualise of
    that file would clobber the descriptor back to the first version.
    """
    info_path = project_path("projectInfo.json")
    if not os.path.exists(info_path):
        return False

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            project_info = json.load(f)
    except Exception:
        return False

    model_filepath = (
        project_info.get("modelFilepath", "")
        or project_info.get("uploadedPt2Filepath", "")
    )
    if not model_filepath or not model_filepath.lower().endswith(".pt2"):
        return False

    out_path = project_path(model_filepath)
    if not os.path.exists(out_path):
        return False

    try:
        with open(out_path, "rb") as f:
            raw = f.read()
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return False

    if not isinstance(payload, dict) or "nodes" not in payload:
        return False

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(descriptor.to_dict(), f, indent=2)
        return True
    except OSError as exc:
        logger.warning("JSON .pt2 sync after edit failed: %s", exc)
        return False


def _sync_model_file_after_edit(descriptor: ArchitectureDescriptor) -> Optional[Dict[str, Any]]:
    """Keep the on-disk model artifact aligned with descriptor.json after saves."""
    report = _sync_weights_after_edit(descriptor)
    if report is not None:
        return report
    _sync_json_pt2_after_edit(descriptor)
    return None


def exportUploadedPt2ToOnnx(pt2_filename: str, previous_onnx_filename: str = "") -> Dict[str, Any]:
    """Build the uploaded PT2 into an ONNX file for the uploaded ONNX slot."""
    if not pt2_filename:
        return {"error": "No uploaded PT2 model found"}

    pt2_path = project_path(pt2_filename)
    if not os.path.exists(pt2_path):
        return {"error": f"PT2 file not found: {pt2_filename}"}

    try:
        descriptor = loadProjectDescriptor(pt2_filename)
        descriptor.validate()
    except Exception as exc:
        return {"error": f"Invalid PT2 descriptor: {exc}"}

    try:
        model = DescriptorModelBuilder.build(descriptor)
    except Exception as exc:
        return {"error": f"Failed to build model from PT2: {exc}"}

    with open(pt2_path, "rb") as f:
        state_dict = extractStateDict(f.read(), is_pytorch=True)
    if state_dict:
        WeightCompatibilityEngine.transfer_weights(state_dict, model)

    stem = os.path.splitext(os.path.basename(pt2_filename))[0] or "model"
    onnx_filename = f"{stem}.onnx"
    onnx_path = project_path(onnx_filename)

    if previous_onnx_filename and previous_onnx_filename != onnx_filename:
        previous_path = project_path(previous_onnx_filename)
        if os.path.exists(previous_path):
            os.remove(previous_path)

    try:
        descriptorToOnnx(model, descriptor, onnx_path)
    except Exception as exc:
        logger.exception("ONNX export from uploaded PT2 failed")
        return {"error": f"ONNX export failed: {exc}"}

    if not os.path.exists(onnx_path):
        return {"error": f"ONNX export did not create file: {onnx_filename}"}

    return {
        "success": True,
        "onnxFile": onnx_filename,
        "path": onnx_filename,
        "name": onnx_filename,
        "exists": True,
    }
