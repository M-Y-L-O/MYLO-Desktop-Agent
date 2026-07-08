import torch
import onnx
import json
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
    )

    onnx.checker.check_model(outputPath)
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
    """Generate graph visualization from descriptor."""
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
        {"from": edge.source, "to": edge.target, "label": f"{getattr(edge, 'source_port', '')}→{getattr(edge, 'target_port', '')}"}
        for edge in descriptor.edges
    ]

    summary = {
        "format": "pt2_descriptor_bundle",
        "model_name": descriptor.model_name,
        "input_shape": descriptor.input_shape,
        "output_shape": descriptor.output_shape,
        "node_count": len(descriptor.nodes),
        "edge_count": len(descriptor.edges),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": make_json_serializable(summary),
    }

def loadDescriptorFromBytes(model_bytes, is_pytorch: bool, input_dim: int, output_dim: int):
    from Core.ArchitectureDescriptor import ArchitectureDescriptor

    try:
        payload = json.loads(model_bytes.decode("utf-8"))
        if isinstance(payload, dict) and "nodes" in payload:
            descriptor = ArchitectureDescriptor.from_dict(payload)
            descriptor.validate()
            return descriptor
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError):
        pass

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