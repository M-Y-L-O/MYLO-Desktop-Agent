import os
import torch

from Processing.Models.DescriptorHandling import descriptorToGraph, loadDescriptorFromBytes
from Processing.Models.ONNXProcessing import analyseOnnx
from Core.ArchitectureDescriptor import ArchitectureDescriptor


def _try_extract_descriptor(model_bytes: bytes, is_pytorch: bool):
    try:
        return loadDescriptorFromBytes(
            model_bytes=model_bytes,
            is_pytorch=is_pytorch,
            input_dim=1,
            output_dim=1,
        )
    except Exception:
        return None


def _pytorch_visualization_response(filePath: str, message: str, has_state_dict: bool = False):
    return {
        "nodes": [],
        "edges": [],
        "summary": {
            "ir_version": None,
            "producer": "pytorch",
            "inputs": [],
            "outputs": [],
            "nodes": [],
            "node_count": 0,
            "filename": os.path.basename(filePath),
            "format": "pytorch_weights" if has_state_dict else "raw_pytorch",
            "message": message,
            "has_state_dict": has_state_dict,
        },
    }


def analyseModel(filePath: str, originalFilename: str = "", weightsPath: str = ""):
    filename = (originalFilename or os.path.basename(filePath)).lower()
    weights_present = bool(weightsPath)

    if filename.endswith(".onnx"):
        return analyseOnnx(filePath)

    with open(filePath, "rb") as model_file:
        model_bytes = model_file.read()

    if filename.endswith(".pt2"):
        descriptor = _try_extract_descriptor(model_bytes, True)
        if not descriptor:
            return {"error": "Invalid .pt2 descriptor"}
        return descriptorToGraph(descriptor)

    if weights_present:
        descriptor = _try_extract_descriptor(model_bytes, False)
        if not descriptor:
            return {"error": "Descriptor extraction failed for model + weights upload"}
        return descriptorToGraph(descriptor)

    if filename.endswith((".pt", ".pth")):
        try:
            loaded = torch.load(filePath, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in loaded and isinstance(loaded[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(loaded[key])
                        descriptor.validate()
                        return descriptorToGraph(descriptor)

                if "state_dict" in loaded or all(isinstance(v, torch.Tensor) for v in loaded.values()):
                    return _pytorch_visualization_response(
                        filePath,
                        "Weights loaded successfully. Please upload model_config.json separately for full analysis.",
                        has_state_dict=True,
                    )

        except Exception:
            pass

        descriptor = _try_extract_descriptor(model_bytes, True)
        if descriptor:
            return descriptorToGraph(descriptor)

        return _pytorch_visualization_response(
            filePath,
            "Limited visualization",
        )

    return {"error": f"Unsupported model format: {filename}"}


def processUploadedModel(filePath: str):
    return analyseModel(filePath)