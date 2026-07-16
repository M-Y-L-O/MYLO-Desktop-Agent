from Processing.Optimization.Neuroevolution import NeuroevolutionEngine
from Processing.Optimization.OptunaSearch import OptunaSearchEngine
from Types.Types import ProjectData, OptimizationRequest
from Utils.FileHandler import loadData, readBinary
from Utils.Other import getDevice
import asyncio
import concurrent.futures
from Processing.Data.DataProcessingForTraining import encodeData, mapOriginalToEncodedColumns
import torch
from Processing.Models.ModelEditing import loadProjectDescriptor
from Processing.Data.DataPipeline import DataPipeline
from Processing.Models.DescriptorHandling import loadDescriptorFromBytes, descriptorToOnnx, extractStateDict
import os
import json
import logging

from Core.AdaptedModel import AdaptedModel

logger = logging.getLogger(__name__)

NEUROEVOLUTION_POPULATION_SIZE = 30

SUPPORTED_STRATEGIES = ("optuna", "neuroevolution") # Orice altceva se duce la default
DEFAULT_STRATEGY = "neuroevolution"

REPORT_FILENAME = "optimization_report.json"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _temp_project_path(filename: str) -> str:
    if not filename:
        return os.path.join(_repo_root(), "temp_project")

    if os.path.isabs(filename):
        return filename

    return os.path.join(_repo_root(), "temp_project", filename)


def _suggest_batch_size(device, dataset_size, descriptor):
    """Auto-tune batch size based on device and model complexity."""
    if device.type == "cpu":
        return 32

    # Estimate model memory footprint
    param_count = sum(
        n.params.get("out_features", 64) * n.params.get("in_features", 64)
        for n in descriptor.nodes if n.type == "Linear"
    )

    # Start conservative, could be made smarter with actual GPU memory query
    base_batch = min(128, max(32, dataset_size // 8))

    # Reduce batch for large models
    if param_count > 1000000:
        base_batch = max(16, base_batch // 2)
    if param_count > 5000000:
        base_batch = max(8, base_batch // 2)

    return base_batch


def _normalize_strategy(raw_strategy):
    """Map the requested strategy to a supported engine, or fall back to default."""
    if not raw_strategy:
        return DEFAULT_STRATEGY
    normalized = str(raw_strategy).strip().lower()
    if normalized in SUPPORTED_STRATEGIES:
        return normalized
    logger.warning(
        f"Unknown strategy '{raw_strategy}' requested; falling back to '{DEFAULT_STRATEGY}'."
    )
    return DEFAULT_STRATEGY


def _request_generations(requestInfo) -> int:
    raw = getattr(requestInfo, "generations", None)
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        raw = None
    if not raw or raw <= 0:
        raw = getattr(requestInfo, "epochs", 5) or 5
    return max(3, int(raw))


def _request_problem_type(requestInfo) -> str:
    return (
        getattr(requestInfo, "problem_type", None)
        or getattr(requestInfo, "problemType", None)
        or "regression"
    )


def _diff_descriptors(original: dict, best: dict) -> dict:
    original = original or {}
    best = best or {}

    orig_nodes = {n.get("id"): n for n in original.get("nodes", [])}
    best_nodes = {n.get("id"): n for n in best.get("nodes", [])}

    nodes_added = [n for nid, n in best_nodes.items() if nid not in orig_nodes]
    nodes_removed = [n for nid, n in orig_nodes.items() if nid not in best_nodes]
    nodes_changed = []
    for nid in orig_nodes.keys() & best_nodes.keys():
        o, b = orig_nodes[nid], best_nodes[nid]
        if o.get("type") != b.get("type") or o.get("params") != b.get("params"):
            nodes_changed.append({
                "id": nid,
                "before": {"type": o.get("type"), "params": o.get("params")},
                "after": {"type": b.get("type"), "params": b.get("params")},
            })

    orig_edges = {(e.get("source"), e.get("target")) for e in original.get("edges", [])}
    best_edges = {(e.get("source"), e.get("target")) for e in best.get("edges", [])}
    edges_added = [{"source": s, "target": t} for (s, t) in sorted(best_edges - orig_edges, key=str)]
    edges_removed = [{"source": s, "target": t} for (s, t) in sorted(orig_edges - best_edges, key=str)]

    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_changed": nodes_changed,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "counts": {
            "nodes_added": len(nodes_added),
            "nodes_removed": len(nodes_removed),
            "nodes_changed": len(nodes_changed),
            "edges_added": len(edges_added),
            "edges_removed": len(edges_removed),
        },
    }


def _build_engine(strategy, initial_descriptor, requestInfo, statusCallback):

    if strategy == "optuna":
        statusCallback({"type": "phase", "phase": "optimizing", "status": "Using Optuna architecture search...", "progress": 42})
        engine = OptunaSearchEngine(
            initial_descriptor=initial_descriptor,
            n_trials=max(20, requestInfo.epochs * 2),
            epochs_per_trial=min(10, requestInfo.epochs),
            min_resource=2,
            reduction_factor=3,
            sampler_type="tpe",
            pruner_type="hyperband",
            n_startup_trials=8,
            seed=42,
            statusCallback=statusCallback,
        )

        def run(train_loader, val_loader, device, parent_state_dict, problem_type):
            return engine.search(
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=requestInfo.epochs,
                device=str(device),
                parent_state_dict=parent_state_dict,
                problem_type=problem_type,
                complexity_penalty=1e-5,
            )

        def summary_builder(diagnostics):
            return {
                "sampler": diagnostics.get("sampler", "TPESampler"),
                "pruner": diagnostics.get("pruner", "HyperbandPruner"),
                "n_trials_requested": diagnostics.get("n_trials_requested"),
                "n_trials_completed": diagnostics.get("n_trials_completed"),
                "n_trials_pruned": diagnostics.get("n_trials_pruned"),
                "prune_rate": diagnostics.get("prune_rate"),
                "elapsed_seconds": diagnostics.get("elapsed_seconds"),
            }

        return "optuna", run, summary_builder

    # Default: neuroevolution
    statusCallback({"type": "phase", "phase": "optimizing", "status": "Using neuroevolution optimization...", "progress": 42})

    
    engine = NeuroevolutionEngine(
        initial_descriptor,
        population_size=NEUROEVOLUTION_POPULATION_SIZE,
        statusCallback=statusCallback,
    )
    n_generations = _request_generations(requestInfo)

    def run(train_loader, val_loader, device, parent_state_dict, problem_type):
        return engine.evolve(
            train_loader=train_loader,
            val_loader=val_loader,
            generations=n_generations,
            max_epochs=requestInfo.epochs,
            device=str(device),
            parent_state_dict=parent_state_dict,
            problem_type=problem_type,
            complexity_penalty=1e-5,
        )

    def summary_builder(diagnostics):
        d = diagnostics if isinstance(diagnostics, dict) else {}
        improvement = d.get("improvement") or {}
        return {
            "generations": n_generations,
            "population_size": NEUROEVOLUTION_POPULATION_SIZE,
            "best_train_loss": d.get("best_train_loss"),
            "best_val_loss": d.get("best_val_loss"),
            "best_score": d.get("best_score"),
            "val_loss_change_pct": improvement.get("val_loss_change_pct"),
            "verdict": improvement.get("verdict"),
            "total_mutation_successes": d.get("total_mutation_successes"),
            "overall_mutation_success_rate": d.get("overall_mutation_success_rate"),
            "stagnation_events": d.get("stagnation_events"),
            "elapsed_seconds": d.get("elapsed_seconds"),
        }

    return "neuroevolution", run, summary_builder


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

    statusCallback({"type": "phase", "phase": "processing", "status": "Processing request...", "progress": 0})
    data = loadData(csv_path)

    statusCallback({"type": "phase", "phase": "encoding", "status": "Encoding data...", "progress": 5})
    result = encodeData(data, requestInfo.encoding)
    encodedDf = result[0]
    encodedMetadata = result[1]

    targetFeature = requestInfo.targetFeature if isinstance(requestInfo.targetFeature, list) else [requestInfo.targetFeature]
    mappedInputFeatures = mapOriginalToEncodedColumns(requestInfo.inputFeatures, encodedMetadata, encodedDf)
    mappedTargetFeature = mapOriginalToEncodedColumns(targetFeature, encodedMetadata, encodedDf)

    statusCallback({"type": "phase", "phase": "loading_model", "status": "Loading model...", "progress": 10})
    requestInfo.inputFeatures = mappedInputFeatures
    requestInfo.targetFeature = mappedTargetFeature

    isPytorchModel = model_path.endswith(".pt") or model_path.endswith(".pth") or model_path.endswith(".pt2")

    weightBytes = None
    weightsPath = getattr(project, "weightsFilepath", None)
    if weightsPath:
        full_weights_path = _temp_project_path(weightsPath)
        if os.path.exists(full_weights_path):
            weightBytes = readBinary(full_weights_path)

    statusCallback({"type": "phase", "phase": "optimizing", "status": "Optimizing model...", "progress": 15})
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

        # Cap CPU threads so we don't fight the executor pool
        torch.set_num_threads(min(4, os.cpu_count() or 4))

        if df is None or df.empty:
            return {"error": "Encoded DataFrame is empty"}

        if modelBytes is None or len(modelBytes) == 0:
            return {"error": "Model bytes are empty"}

        statusCallback({"type": "phase", "phase": "analysing", "status": "Analysing model...", "progress": 20})

        featureCols = requestInfo.inputFeatures
        targetCol = requestInfo.targetFeature if isinstance(requestInfo.targetFeature, list) else [requestInfo.targetFeature]
        inputDim = len(featureCols)
        outputDim = len(targetCol)
        problem_type = _request_problem_type(requestInfo)

        statusCallback({"type": "phase", "phase": "loading_descriptor", "status": "Loading architecture descriptor...", "progress": 25})
        initialDescriptor = loadProjectDescriptor()
        parent_state_dict = None
        if weightBytes is not None:
            statusCallback({"type": "phase", "phase": "loading_weights", "status": "Loading parent weights...", "progress": 30})
            parent_state_dict = extractStateDict(weightBytes, True)
        elif isPytorchModel:
            parent_state_dict = extractStateDict(modelBytes, isPytorchModel)

        sequence_length = None
        if len(initialDescriptor.input_shape) == 3:
            sequence_length = initialDescriptor.input_shape[1]

        statusCallback({"type": "phase", "phase": "data_pipeline", "status": "Preparing leakage-free data pipeline...", "progress": 35})

        # Auto-tune batch size based on model size and available memory
        batch_size = _suggest_batch_size(device, len(df), initialDescriptor)

        pipeline = DataPipeline.prepare_data(
            _temp_project_path(project.csvFilepath),
            featureCols,
            targetCol,
            problem_type=problem_type,
            batch_size=batch_size,
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

        # Snapshot the true "before" architecture for the diff/report
        original_descriptor_dict = initialDescriptor.to_dict()

        statusCallback({"type": "phase", "phase": "optimizing", "status": "Starting optimization...", "progress": 40})

        # Strategy dispatch -- each branch is fully self-contained in _build_engine.
        strategy, run_search, build_summary = _build_engine(
            strategy=_normalize_strategy(getattr(requestInfo, "strategy", None)),
            initial_descriptor=initialDescriptor,
            requestInfo=requestInfo,
            statusCallback=statusCallback,
        )

        best_descriptor, best_model, diagnostics = run_search(
            train_loader=pipeline.train_loader,
            val_loader=pipeline.val_loader,
            device=device,
            parent_state_dict=parent_state_dict,
            problem_type=problem_type,
        )

        # Validate descriptor/model shape alignment; fall back to adapter on mismatch.
        expected_input = best_descriptor.input_shape[-1]
        actual_input = pipeline.input_shape[-1]
        expected_output = best_descriptor.output_shape[-1]
        actual_output = pipeline.output_shape[-1]

        if expected_input != actual_input or expected_output != actual_output:
            logger.warning(
                f"Shape mismatch detected after {strategy}: "
                f"input({expected_input}!={actual_input}), "
                f"output({expected_output}!={actual_output}). "
                f"Using AdaptedModel as fallback."
            )
            best_model = AdaptedModel.from_shape_mismatch(
                best_model,
                actual_input_dim=actual_input,
                expected_input_dim=expected_input,
                actual_output_dim=actual_output,
                expected_output_dim=expected_output,
            )

        statusCallback({"type": "phase", "phase": "saving", "status": "Saving optimized model bundle...", "progress": 90})
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
            onnx_path = descriptorToOnnx(best_model, best_descriptor, onnx_path, device=device)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
            onnx_path = None

        if isinstance(diagnostics, dict) and diagnostics.get("original_descriptor"):
            original_descriptor_dict = diagnostics["original_descriptor"]

        architecture_diff = _diff_descriptors(original_descriptor_dict, best_descriptor.to_dict())

        summary = {
            "strategy_used": strategy,
            "requested_strategy": getattr(requestInfo, "strategy", None),
            "input_features": featureCols,
            "output_features": targetCol,
        }
        summary.update(build_summary(diagnostics))

        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

        result = {
            "status": "success",
            "model_path": bundle_path,
            "model_config_path": config_path,
            "model_weights_path": weights_path,
            "model_onnx_path": onnx_path,
            "onnx_exported": onnx_path is not None,
            "summary": summary,
            "best_config": best_descriptor.to_dict(),
            "original_descriptor": original_descriptor_dict,
            "architecture_diff": architecture_diff,
            "optimization_diagnostics": diagnostics,
            "baseline": diagnostics.get("baseline"),
            "champion": diagnostics.get("champion"),
            "improvement": diagnostics.get("improvement"),
            "champion_lineage": diagnostics.get("champion_lineage"),
            "training_history": diagnostics.get("champion_training"),
            "mutation_timeline": diagnostics.get("mutation_timeline"),
        }

        report_path = os.path.join(output_dir, REPORT_FILENAME)
        result["report_path"] = report_path
        try:
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(result, report_file, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not persist optimization report: {e}")
            result["report_path"] = None

        statusCallback({
            "type": "phase",
            "phase": "complete",
            "status": "Optimization complete",
            "progress": 100,
            "improvement": result.get("improvement"),
        })

        return result

    except Exception as e:
        statusCallback({"type": "error", "status": f"Error: {str(e)}", "error": True, "progress": 0})
        return {"error": str(e)}