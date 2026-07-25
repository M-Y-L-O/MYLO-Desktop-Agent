import copy
import torch
import onnx
import json
from typing import Any, Dict, List, Set, Tuple
from Utils.Other import make_json_serializable
from Utils.FileHandler import createTempFile
import logging
import os

def descriptorToOnnx(model, descriptor, outputPath, device=None):
    if device is None:
        device = torch.device("cpu")

    modelCpu = model.to("cpu")
    modelCpu.eval()

    dummyShape = dummyShapeFromDescriptor(descriptor.input_shape)
    dummyInput = torch.randn(*dummyShape, dtype=torch.float32)

    dynamicAxes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    # dynamo=False: legacy exporter. The dynamo path can fail on Windows consoles
    # (Unicode logging) and is unnecessary for these descriptor models.
    torch.onnx.export(
        modelCpu,
        dummyInput,
        outputPath,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamicAxes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )

    if not os.path.exists(outputPath):
        raise FileNotFoundError(f"ONNX export reported success but file missing: {outputPath}")

    onnx.checker.check_model(outputPath)
    print(f"ONNX model exported to {outputPath}")
    return outputPath

def dummyShapeFromDescriptor(input_shape):
    dummy_shape = []
    for index, dim in enumerate(input_shape):
        if dim is None or dim <= 0:
            dummy_shape.append(1 if index == 0 else 1)
        else:
            dummy_shape.append(int(dim))
    if not dummy_shape:
        dummy_shape = [1, 1]
    return dummy_shape

def descriptorToGraph(descriptor):
    nodes = [
        {"id": "input", "label": "input", "title": json.dumps({"shape": descriptor.input_shape}, indent=2), "group": "input"},
        {"id": "output", "label": "output", "title": json.dumps({"shape": descriptor.output_shape}, indent=2), "group": "output"},
    ]

    for node in descriptor.nodes:
        nodes.append({
            "id": node.id,
            "label": node.id,
            "title": json.dumps({
                "type": node.type,
                "params": node.params,
                "execution": getattr(node, 'execution', {}),
            }, indent=2),
            "group": node.type,
        })

    edges = [
        {"from": edge.source, "to": edge.target}
        for edge in descriptor.edges
    ]

    node_summary = []
    node_summary.append({
        "name": "input",
        "op_type": "Input",
    })
    for node in descriptor.nodes:
        node_summary.append({
            "name": node.id if node.id else "Unnamed",
            "op_type": node.type,
        })
    node_summary.append({
        "name": "output",
        "op_type": "Output",
    })

    summary = {
        "ir_version": None,
        "producer": "descriptor",
        "inputs": [{"name": "input"}],
        "outputs": [{"name": "output"}],
        "nodes": node_summary,
        "node_count": len(node_summary),
        "filename": f"{descriptor.model_name}.json" if descriptor.model_name else "descriptor.json",
        "model_name": descriptor.model_name,
        "input_shape": descriptor.input_shape,
        "output_shape": descriptor.output_shape,
        "edge_count": len(descriptor.edges),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": make_json_serializable(summary),
    }

def descriptorToDetailedGraph(descriptor):
    """Generate a detailed graph visualization from descriptor by exporting to ONNX and analyzing."""
    return _exportDescriptorToOnnxGraph(descriptor)


def descriptorToHybridGraph(descriptor, expanded_node_ids=None):
    """Visualize descriptor at mixed granularity — expand only selected nodes to ONNX ops."""
    expanded_node_ids = list(dict.fromkeys(expanded_node_ids or []))
    valid_ids = {node.id for node in descriptor.nodes}
    expanded_node_ids = [node_id for node_id in expanded_node_ids if node_id in valid_ids]

    if not expanded_node_ids:
        return descriptorToGraph(descriptor)

    if len(expanded_node_ids) == len(descriptor.nodes):
        return descriptorToDetailedGraph(descriptor)

    base = descriptorToGraph(descriptor)
    hybrid_nodes = [n for n in base["nodes"] if n["id"] not in expanded_node_ids]
    hybrid_edges = [e for e in base["edges"] if e["from"] not in expanded_node_ids and e["to"] not in expanded_node_ids]

    expansion_meta: Dict[str, Any] = {}

    for node_id in expanded_node_ids:
        subgraph = expandNodeToOnnxSubgraph(descriptor, node_id)
        if "error" in subgraph:
            return subgraph

        hybrid_nodes.extend(subgraph["nodes"])
        hybrid_edges.extend(subgraph["edges"])

        for external_source in base["edges"]:
            if external_source["to"] == node_id and external_source["from"] not in expanded_node_ids:
                for entry_id in subgraph["entryIds"]:
                    hybrid_edges.append({"from": external_source["from"], "to": entry_id})

        for external_target in base["edges"]:
            if external_target["from"] == node_id and external_target["to"] not in expanded_node_ids:
                for exit_id in subgraph["exitIds"]:
                    hybrid_edges.append({"from": exit_id, "to": external_target["to"]})

        expansion_meta[node_id] = {
            "entryIds": subgraph["entryIds"],
            "exitIds": subgraph["exitIds"],
            "opCount": len(subgraph["nodes"]),
        }

    hybrid_edges = _deduplicate_visual_edges(hybrid_edges)

    summary = dict(base["summary"])
    summary["view_mode"] = "hybrid"
    summary["expanded_nodes"] = expanded_node_ids
    summary["expansion"] = expansion_meta

    return {
        "nodes": hybrid_nodes,
        "edges": hybrid_edges,
        "summary": make_json_serializable(summary),
    }


def expandNodeToOnnxSubgraph(descriptor, node_id: str) -> Dict[str, Any]:
    """Export a single descriptor node to ONNX and return a prefixed visualization sub-graph."""
    from Core.ArchitectureDescriptor import ArchitectureDescriptor, Edge

    node = next((n for n in descriptor.nodes if n.id == node_id), None)
    if not node:
        return {"error": f"Node not found: {node_id}"}

    try:
        shapes = descriptor._propagate_shapes(mutate=False)
    except Exception as exc:
        return {"error": f"Could not propagate shapes for {node_id}: {exc}"}

    in_shape = _input_shape_for_node(descriptor, node_id, shapes)
    out_shape = shapes.get(node_id)
    if not in_shape or not out_shape:
        return {"error": f"Could not infer tensor shapes for node {node_id}"}

    sub_descriptor = ArchitectureDescriptor(
        model_name=f"{node_id}_expanded",
        input_shape=list(in_shape),
        output_shape=list(out_shape),
        nodes=[copy.deepcopy(node)],
        edges=[
            Edge(source="input", target=node_id),
            Edge(source=node_id, target="output"),
        ],
    )

    try:
        sub_descriptor.normalize_inplace()
        sub_descriptor.validate()
    except Exception as exc:
        return {"error": f"Invalid sub-graph for {node_id}: {exc}"}

    detail_graph = _exportDescriptorToOnnxGraph(sub_descriptor)
    if "error" in detail_graph:
        return detail_graph

    return _prefix_subgraph(detail_graph, node_id)


def _exportDescriptorToOnnxGraph(descriptor):
    import tempfile
    from Core.DescriptorModelBuilder import DescriptorModelBuilder
    from Processing.Models.ONNXProcessing import analyseOnnx

    try:
        model = DescriptorModelBuilder.build(descriptor)
    except Exception as e:
        logging.error(f"Failed to build model from descriptor for detailed visualization: {e}")
        return {"error": f"Failed to build model: {str(e)}"}

    temp_fd, temp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(temp_fd)

    try:
        descriptorToOnnx(model, descriptor, temp_path)
        result = analyseOnnx(temp_path)

        if "summary" in result and "filename" in result["summary"]:
            model_name = getattr(descriptor, "model_name", None)
            if not model_name:
                model_name = "descriptor_model"
            result["summary"]["filename"] = f"{model_name}.onnx"

        return result
    except Exception as e:
        logging.exception(f"Failed to export descriptor to ONNX for detailed graph visualization: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _input_shape_for_node(descriptor, node_id: str, shapes: Dict[str, List[int]]) -> List[int]:
    incoming = [edge for edge in descriptor.edges if edge.target == node_id]
    if not incoming:
        return list(descriptor.input_shape)

    source_shapes = [shapes.get(edge.source) for edge in incoming if edge.source in shapes]
    source_shapes = [shape for shape in source_shapes if shape]
    if not source_shapes:
        return list(descriptor.input_shape)
    if len(source_shapes) == 1:
        return list(source_shapes[0])
    return list(source_shapes[0])


def _prefix_subgraph(subgraph: Dict[str, Any], parent_id: str) -> Dict[str, Any]:
    prefix = f"{parent_id}::"
    id_map = {node["id"]: f"{prefix}{node['id']}" for node in subgraph["nodes"]}

    nodes = []
    for node in subgraph["nodes"]:
        new_id = id_map[node["id"]]
        nodes.append({
            **node,
            "id": new_id,
            "label": f"{parent_id}/{node.get('label', node['id'])}",
            "group": node.get("group", "operation"),
            "expandedFrom": parent_id,
            "parentNode": parent_id,
        })

    edges = [{"from": id_map[edge["from"]], "to": id_map[edge["to"]]} for edge in subgraph["edges"]]
    entry_ids, exit_ids = _find_subgraph_boundaries(subgraph, id_map)

    return {
        "nodes": nodes,
        "edges": edges,
        "entryIds": entry_ids,
        "exitIds": exit_ids,
    }


def _find_subgraph_boundaries(subgraph: Dict[str, Any], id_map: Dict[Any, str]) -> Tuple[List[str], List[str]]:
    internal = {id_map[node["id"]] for node in subgraph["nodes"]}
    incoming: Set[str] = set()
    outgoing: Set[str] = set()

    for edge in subgraph["edges"]:
        incoming.add(id_map[edge["to"]])
        outgoing.add(id_map[edge["from"]])

    entry = sorted(internal - incoming)
    exit_nodes = sorted(internal - outgoing)
    if not entry:
        entry = [sorted(internal)[0]]
    if not exit_nodes:
        exit_nodes = [sorted(internal)[-1]]
    return entry, exit_nodes


def _deduplicate_visual_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for edge in edges:
        key = (edge["from"], edge["to"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique

def saveDescriptorToProject(descriptor) -> None:
    os.makedirs("temp_project", exist_ok=True)
    with open(os.path.join("temp_project", "descriptor.json"), "w", encoding="utf-8") as f:
        json.dump(descriptor.to_dict(), f, indent=2)


def loadDescriptorFromBytes(model_bytes, is_pytorch: bool, input_dim: int, output_dim: int):
    from Core.ArchitectureDescriptor import ArchitectureDescriptor

    try:
        payload = json.loads(model_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict) and "nodes" in payload:
        # Explicit JSON architecture (.pt2 descriptor) — do not silently fall back
        descriptor = ArchitectureDescriptor.from_dict(payload)
        descriptor.validate()
        return descriptor

    if is_pytorch:
        temp_in = createTempFile(model_bytes, ".pt")
        try:
            checkpoint = torch.load(temp_in, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(checkpoint[key])
                        descriptor.validate()
                        return descriptor
        except Exception as exc:
            logging.warning(f"Could not load descriptor from checkpoint: {exc}")
        finally:
            if os.path.exists(temp_in):
                os.remove(temp_in)

    descriptor = ArchitectureDescriptor.default_feedforward(input_dim, output_dim)
    descriptor.validate()
    return descriptor

def extractStateDict(model_bytes, is_pytorch: bool):
    if not is_pytorch:
        return None

    temp_in = createTempFile(model_bytes, ".pt")
    try:
        checkpoint = torch.load(temp_in, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
    except Exception as exc:
        logging.warning(f"Could not extract state dict from uploaded model: {exc}")
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)
    return None