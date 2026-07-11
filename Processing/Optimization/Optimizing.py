from Processing.Optimization.Neuroevolution import NeuroevolutionEngine
from Types.Types import ProjectData, OptimizationRequest
from Utils.FileHandler import loadData, readBinary
from Utils.Other import getDevice
import asyncio
import concurrent.futures
from Processing.Data.DataProcessingForTraining import encodeData, mapOriginalToEncodedColumns
import torch
from Processing.Data.DataPipeline import DataPipeline
from Processing.Models.DescriptorHandling import loadDescriptorFromBytes, descriptorToOnnx, extractStateDict
import os
from Core.AdaptedModel import AdaptedModel


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _temp_project_path(filename: str) -> str:
    if not filename:
        return os.path.join(_repo_root(), "temp_project")

    if os.path.isabs(filename):
        return filename

    return os.path.join(_repo_root(), "temp_project", filename)


async def startOptimization(project: ProjectData, requestInfo: OptimizationRequest, statusCallback):
    statusCallback = statusCallback or (lambda *_args, **_kwargs: None)
    print(requestInfo)

    if not project.csvFilepath:
        return {"error": "CSV file is missing from the current project"}

    if not project.modelFilepath:
        return {"error": "Model file is missing from the current project"}

    csv_path = _temp_project_path(project.csvFilepath)
    model_path = _temp_project_path(project.modelFilepath)

    if not os.path.exists(csv_path):
        return {"error": f"Data file not found: {csv_path}"}

    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    statusCallback({"status": "Processing request...", "progress": 0})
    data = loadData(csv_path)

    statusCallback({"status": "Encoding data...", "progress": 3})
    result = encodeData(data, requestInfo.encoding)
    encodedDf = result[0]
    encodedMetadata = result[1]

    targetFeature = requestInfo.targetFeature if isinstance(requestInfo.targetFeature, list) else [requestInfo.targetFeature]
    mappedInputFeatures = mapOriginalToEncodedColumns(requestInfo.inputFeatures, encodedMetadata, encodedDf)
    mappedTargetFeature = mapOriginalToEncodedColumns(targetFeature, encodedMetadata, encodedDf)

    statusCallback({"status": "Loading model...", "progress": 5})
    requestInfo.inputFeatures = mappedInputFeatures
    requestInfo.targetFeature = mappedTargetFeature

    isPytorchModel = model_path.endswith(".pt") or model_path.endswith(".pth") or model_path.endswith(".pt2")

    weightBytes = None
    weightsPath = getattr(project, "weightsFilepath", None)
    if weightsPath:
        full_weights_path = _temp_project_path(weightsPath)
        if os.path.exists(full_weights_path):
            weightBytes = readBinary(full_weights_path)

    statusCallback({"status": "Optimizing model...", "progress": 10})
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: findOptimalArchitecture(
                project,
                encodedDf,
                requestInfo,
                isPytorchModel,
                statusCallback=statusCallback,
                weightBytes=weightBytes,
            ),
        )

    if "model_path" not in result:
        return {"error": "Optimization failed", "details": result}

    return result


def findOptimalArchitecture(project: ProjectData, df, requestInfo: OptimizationRequest, isPytorchModel, statusCallback=None, weightBytes=None):
    statusCallback = statusCallback or (lambda *_args, **_kwargs: None)

    try:
        model_path = _temp_project_path(project.modelFilepath)
        modelBytes = readBinary(model_path)
        device = getDevice()

        torch.set_num_threads(4)

        if df is None or df.empty:
            return {"error": "Encoded DataFrame is empty"}

        if modelBytes is None or len(modelBytes) == 0:
            return {"error": "Model bytes are empty"}

        statusCallback({"status": "Analysing model...", "progress": 15})

        featureCols = requestInfo.inputFeatures
        targetCol = requestInfo.targetFeature if isinstance(requestInfo.targetFeature, list) else [requestInfo.targetFeature]
        inputDim = len(featureCols)
        outputDim = len(targetCol)
        problem_type = getattr(requestInfo, "problem_type", "regression")

        statusCallback({"status": "Loading architecture descriptor...", "progress": 20})
        initialDescriptor = loadDescriptorFromBytes(modelBytes, isPytorchModel, inputDim, outputDim)
        parent_state_dict = None
        if weightBytes is not None:
            statusCallback({"status": "Loading parent weights...", "progress": 7})
            parent_state_dict = extractStateDict(weightBytes, True)
        elif isPytorchModel:
            parent_state_dict = extractStateDict(modelBytes, isPytorchModel)

        sequence_length = None
        if len(initialDescriptor.input_shape) == 3:
            sequence_length = initialDescriptor.input_shape[1]

        statusCallback({"status": "Preparing leakage-free data pipeline...", "progress": 10})
        pipeline = DataPipeline.prepare_data(
            _temp_project_path(project.csvFilepath),
            featureCols,
            targetCol,
            problem_type=problem_type,
            batch_size=32 if device.type == "cpu" else min(128, max(32, len(df) // 8)),
            sequence_length=sequence_length,
        )

        if initialDescriptor.input_shape[-1] != pipeline.input_shape[-1]:
            initialDescriptor.input_shape = pipeline.input_shape
            for node in initialDescriptor.nodes:
                if node.type == "Linear" and any(
                    edge.source == "input" and edge.target == node.id for edge in initialDescriptor.edges
                ):
                    node.params["in_features"] = pipeline.input_shape[-1]

        if initialDescriptor.output_shape[-1] != pipeline.output_shape[-1]:
            initialDescriptor.output_shape = pipeline.output_shape
            output_nodes = [edge.source for edge in initialDescriptor.edges if edge.target == "output"]
            if output_nodes:
                out_node = next(n for n in initialDescriptor.nodes if n.id == output_nodes[0])
                if out_node.type == "Linear":
                    out_node.params["out_features"] = pipeline.output_shape[-1]

        initialDescriptor.validate()

        statusCallback({"status": "Starting neuroevolution...", "progress": 20})
        population_size = min(20, max(4, requestInfo.epochs * 4))
        engine = NeuroevolutionEngine(initialDescriptor, population_size=population_size)
        best_descriptor, best_model, diagnostics = engine.evolve(
    train_loader=pipeline.train_loader,
    val_loader=pipeline.val_loader,
    generations=max(2, requestInfo.epochs), 
    max_epochs=requestInfo.epochs,
    device=str(device),
    parent_state_dict=parent_state_dict,
    problem_type=requestInfo.problem_type,
    complexity_penalty=1e-7,
)

        expected_input = best_descriptor.input_shape[-1]
        actual_input = pipeline.input_shape[-1]
        expected_output = best_descriptor.output_shape[-1]
        actual_output = pipeline.output_shape[-1]
        if expected_input != actual_input or expected_output != actual_output:
            best_model = AdaptedModel.from_shape_mismatch(
                best_model,
                actual_input_dim=actual_input,
                expected_input_dim=expected_input,
                actual_output_dim=actual_output,
                expected_output_dim=expected_output,
            )

        statusCallback({"status": "Saving optimized model bundle...", "progress": 90})
        output_dir = _temp_project_path("")
        config_path = os.path.join(output_dir, "model_config.json")
        weights_path = os.path.join(output_dir, "model_weights.pth")
        bundle_path = os.path.join(output_dir, "optimized_model.pt2")
        onnx_path = os.path.join(output_dir, "optimized_model.onnx")

        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(best_descriptor.to_json())

        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "model_config": best_descriptor.to_dict(),
            },
            weights_path,
        )

        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "model_config": best_descriptor.to_dict(),
            },
            bundle_path,
        )
        onnx_path = os.path.join(output_dir, project.modelFilepath.split(".")[0] + "_optimized.onnx")
        try:
            descriptorToOnnx(best_model, best_descriptor, onnx_path, device=device)
        except Exception as e:
            print("ONNX export failed. Continuing without ONNX export. " + str(e))
            onnx_path = None

        statusCallback({"status": "Optimization complete", "progress": 100})
        return {
    "status": "success",
    "model_path": bundle_path,
    "model_config_path": config_path,
    "model_weights_path": weights_path,
    "model_onnx_path": onnx_path,
    "onnx_exported": onnx_path is not None,
    "summary": {
        "strategy_used": "neuroevolution",
        "requested_strategy": requestInfo.strategy,
        "generations": max(2, requestInfo.epochs),
        "population_size": population_size,
        "input_features": featureCols,
        "output_features": targetCol,
    },
    "best_config": best_descriptor.to_dict(),
    "original_descriptor": diagnostics["original_descriptor"],  # <-- NEW
    "optimization_diagnostics": diagnostics,  # <-- NEW
}
        
    except Exception as e:
        statusCallback({"status": f"Error: {str(e)}", "error": True, "progress": 0})
        return {"error": str(e)}


