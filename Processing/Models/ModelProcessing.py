import os
import torch

from Processing.Models.DescriptorHandling import descriptorToGraph, loadDescriptorFromBytes, saveDescriptorToProject
from Processing.Models.ONNXProcessing import analyseOnnx
from Core.ArchitectureDescriptor import ArchitectureDescriptor
from Processing.Models.TorchPackageProcessing import (
    TORCH_PACKAGE_EXTENSIONS,
    TorchPackageImportError,
    import_torch_package,
)
from Processing.Models.TorchExportProcessing import (
    NotTorchExportArtifact,
    TorchExportImportError,
    import_torch_export,
)


def _try_extract_descriptor(model_bytes: bytes, is_pytorch: bool):
    try:
        return loadDescriptorFromBytes(
            model_bytes=model_bytes,
            is_pytorch=is_pytorch,
            input_dim=1,
            output_dim=1,
        )
    except Exception as exc:
        return {"error": str(exc)}


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
            "editable": False,
            "message": message,
            "has_state_dict": has_state_dict,
        },
    }


def _analyse_torch_package(file_path: str):
    imported = import_torch_package(file_path)
    saveDescriptorToProject(imported.descriptor)
    graph = descriptorToGraph(imported.descriptor)
    graph["import"] = imported.report
    graph["summary"].update({
        "format": "torch.package",
        "producer": "torch.package + torch.fx",
        "editable": True,
        "source_filename": os.path.basename(file_path),
        "conversion": imported.report.get("conversion", {}),
        "verification": imported.report.get("verification", {}),
    })
    return graph


def _analyse_torch_export(file_path: str):
    descriptor, _, report = import_torch_export(file_path)
    saveDescriptorToProject(descriptor)
    graph = descriptorToGraph(descriptor)
    graph["import"] = report
    graph["summary"].update({
        "format": "torch.export",
        "producer": "torch.export + ATen-to-MYLO lowering",
        "editable": True,
        "source_filename": os.path.basename(file_path),
        "verification": report.get("verification", {}),
    })
    return graph


def analyseModel(filePath: str, originalFilename: str = "", weightsPath: str = ""):
    filename = (originalFilename or os.path.basename(filePath)).lower()
    extension = os.path.splitext(filename)[1]
    weights_present = bool(weightsPath)

    if filename.endswith(".onnx"):
        return analyseOnnx(filePath)

    with open(filePath, "rb") as model_file:
        model_bytes = model_file.read()

    if filename.endswith(".pt2"):
        try:
            return _analyse_torch_export(filePath)
        except NotTorchExportArtifact:
            pass
        except TorchExportImportError as exc:
            return {"error": str(exc), "format": "torch.export", "editable": False}

        descriptor = _try_extract_descriptor(model_bytes, True)
        if isinstance(descriptor, dict) and "error" in descriptor:
            return descriptor
        if not descriptor:
            return {"error": "Invalid .pt2 descriptor"}
        saveDescriptorToProject(descriptor)
        return descriptorToGraph(descriptor)

    if extension in TORCH_PACKAGE_EXTENSIONS:
        try:
            return _analyse_torch_package(filePath)
        except TorchPackageImportError as exc:
            return {"error": str(exc), "format": "torch.package", "editable": False}

    if weights_present:
        descriptor = _try_extract_descriptor(model_bytes, False)
        if isinstance(descriptor, dict) and "error" in descriptor:
            return descriptor
        if not descriptor:
            return {"error": "Descriptor extraction failed for model + weights upload"}
        saveDescriptorToProject(descriptor)
        return descriptorToGraph(descriptor)

    if filename.endswith((".pt", ".pth")):
        try:
            return _analyse_torch_package(filePath)
        except TorchPackageImportError:
            pass

        try:
            loaded = torch.load(filePath, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in loaded and isinstance(loaded[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(loaded[key])
                        descriptor.validate()
                        saveDescriptorToProject(descriptor)
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
        if isinstance(descriptor, dict) and "error" in descriptor:
            # Binary torch files may fail JSON parse; fall through to limited viz
            pass
        elif descriptor:
            saveDescriptorToProject(descriptor)
            return descriptorToGraph(descriptor)

        return _pytorch_visualization_response(
            filePath,
            "Limited visualization",
        )

    return {"error": f"Unsupported model format: {filename}"}


def processUploadedModel(filePath: str):
    return analyseModel(filePath)
