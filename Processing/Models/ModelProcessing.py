import torch
from Models.DescriptorHandling import descriptorToGraph
from Models.ONNXProcessing import analyseOnnx
from Core.ArchitectureDescriptor import ArchitectureDescriptor

def processUploadedModel(filePath):

    if filePath.endswith((".pt", ".pth", ".pt2")):
        try:
           
            loaded = torch.load(filePath, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in loaded and isinstance(loaded[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(loaded[key])
                        descriptor.validate()
                        return descriptorToGraph(descriptor)

                if "state_dict" in loaded or all(isinstance(v, torch.Tensor) for v in loaded.values()):
                    return {
                        "summary": {
                            "format": "pytorch_weights",
                            "message": "Weights loaded successfully. Please upload model_config.json separately for full analysis.",
                            "has_state_dict": True,
                        },
                        "nodes": [],
                        "edges": [],
                    }

            return {"error": "Unsupported PyTorch file. Upload a bundle with model_config or use ONNX for visualization."}

        except Exception as e:
            return {"error": str(e)}

    else:
        # ONNX visualization only
        return analyseOnnx(filePath)