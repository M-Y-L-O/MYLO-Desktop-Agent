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
        
        # Override the filename in the summary to match the descriptor's name if available
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