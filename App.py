import asyncio
import os
import json
import shutil
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

import pandas as pd

from Processing.Optimization.Optimizing import startOptimization
from Types.Types import *
from Processing.Models.ModelProcessing import analyseModel
from Processing.Models.ModelDiagnostics import generateModelDiagnostics
from Processing.Models.ModelEditing import (
    loadProjectDescriptor,
    validateDescriptorPayload,
    applyModelEdit,
    saveModelDescriptor,
    visualizeModel,
    expandNodes,
    collapseNodes,
    checkEdgeCompatibility,
    undoModelEdit,
    redoModelEdit,
    exportUploadedPt2ToOnnx,
    _ProjectHistory,
    _get_draft_node_ids,
    _get_reachable_nodes,
)
from Processing.Models.node_catalog import get_catalog, TEMPLATES
from Processing.Data.DataProcessingForVisualisation import (
    analyseCSV,
    calculateDescriptiveStatistics,
    calculateCorrelationMatrix,
    analyzeDistributions,
    performDataQualityChecks,
    generateChartData,
    analyzeTargetVariable
)
from Utils.FileHandler import saveFile
from Utils.Other import make_json_serializable



# ---------------- GLOBAL STATE ----------------

CurrentSession = SessionData()
CurrentProject = ProjectData()
queue = []
optimizing = False

MIDDLEWARE_EXCEPTIONS = ["/", "/initialize", "/docs", "/openapi.json", "/optimizationStatus"]

REPORT_FILENAME = "optimization_report.json"

# ---------------- SETUP ----------------

load_dotenv()



app = FastAPI(title="MYLO AGENT", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------- HELPERS ----------------

def project_path(filename: str) -> str:
    return os.path.join("temp_project", filename)


def slot_path(filename: str) -> str:
    return project_path(filename)


def slot_exists(filename: str) -> bool:
    return bool(filename) and os.path.exists(slot_path(filename))


def _slot_entry(key: str, section: str, label: str, filename: str, accept: str, can_upload: bool, can_optimize: bool, can_download: bool):
    return {
        "key": key,
        "section": section,
        "label": label,
        "accept": accept,
        "path": filename or "",
        "name": filename or "",
        "exists": slot_exists(filename),
        "canUpload": can_upload,
        "canVisualize": bool(filename),
        "canOptimize": can_optimize,
        "canDownload": can_download,
    }


def project_slots():
    return [
        _slot_entry(
            "uploaded_pt2",
            "uploaded",
            "Uploaded PyTorch",
            CurrentProject.uploadedPt2Filepath,
            ".pt2,.ptpkg,.torchpackage,.pt,.pth",
            True,
            True,
            True,
        ),
        _slot_entry("uploaded_onnx", "uploaded", "Uploaded ONNX", CurrentProject.uploadedOnnxFilepath, ".onnx", True, False, True),
        _slot_entry("optimized_pt2", "optimized", "Optimized PT2", CurrentProject.optimizedPt2Filepath, ".pt2", False, False, True),
        _slot_entry("optimized_onnx", "optimized", "Optimized ONNX", CurrentProject.optimizedOnnxFilepath, ".onnx", False, False, True),
    ]


def clear_slot(filename: str):
    full_path = slot_path(filename)
    if os.path.exists(full_path):
        os.remove(full_path)


def cleanup_existing_source_models(extension: str):
    if not os.path.exists("temp_project"):
        return

    extension_groups = {
        ".pt": {".pt", ".pth", ".ptpkg", ".torchpackage"},
        ".pth": {".pt", ".pth", ".ptpkg", ".torchpackage"},
        ".ptpkg": {".pt", ".pth", ".ptpkg", ".torchpackage"},
        ".torchpackage": {".pt", ".pth", ".ptpkg", ".torchpackage"},
        ".pt2": {".pt2"},
        ".onnx": {".onnx"},
    }

    target_extensions = extension_groups.get(extension.lower(), {extension.lower()})

    for filename in os.listdir("temp_project"):
        full_path = project_path(filename)
        if not os.path.isfile(full_path):
            continue

        lower_name = filename.lower()
        _, existing_extension = os.path.splitext(lower_name)
        if existing_extension in target_extensions and "optimized" not in lower_name:
            os.remove(full_path)


async def save_uploaded_to_slot(file: UploadFile, previous_filename: str = ""):
    temp_path = await saveFile(file, path="temp_project")
    target_filename = os.path.basename(file.filename or "")
    return target_filename


def loadProjectData():
    global CurrentProject
    if not CurrentProject.id and os.path.exists(project_path("projectInfo.json")):
        with open(project_path("projectInfo.json")) as f:
            data = json.load(f)
            CurrentProject.name = data.get("name", "")
            CurrentProject.id = data.get("id", "")
            CurrentProject.csvFilepath = data.get("csvFilepath", "")
            CurrentProject.modelFilepath = data.get("modelFilepath", "")
            CurrentProject.uploadedPt2Filepath = data.get("uploadedPt2Filepath", CurrentProject.modelFilepath or "")
            CurrentProject.uploadedOnnxFilepath = data.get("uploadedOnnxFilepath", "")
            CurrentProject.optimizedPt2Filepath = data.get("optimizedPt2Filepath", "")
            CurrentProject.optimizedOnnxFilepath = data.get("optimizedOnnxFilepath", "")
            if "weightsFilepath" in data:
                CurrentProject.weightsFilepath = data.get("weightsFilepath")


def load_optimization_report():
    report_file = project_path(REPORT_FILENAME)
    if not os.path.exists(report_file):
        return None
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def optimization_report_summary():
    """Slim subset of the report safe to embed in /getProject responses."""
    report = load_optimization_report()
    if not report:
        return None
    return {
        "summary": report.get("summary"),
        "improvement": report.get("improvement"),
        "baseline": report.get("baseline"),
        "champion": report.get("champion"),
        "architecture_diff_counts": (report.get("architecture_diff") or {}).get("counts"),
        "onnx_exported": report.get("onnx_exported"),
    }


loadProjectData()


# ---------------- MIDDLEWARE ----------------

@app.middleware("http")
async def checkRequest(request: Request, call_next):
    global CurrentSession

    if request.method == "OPTIONS":
        return await call_next(request)

    if request.url.path not in MIDDLEWARE_EXCEPTIONS and not CurrentSession.initialized:
        return JSONResponse({"error": "Session not initialized"}, 400)

    if request.url.path not in MIDDLEWARE_EXCEPTIONS:
        apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
        if apiKey != CurrentSession.apiKey:
            return JSONResponse({"error": "Invalid API key"}, 401)

    return await call_next(request)


# ---------------- SESSION ----------------

@app.get("/")
async def root():
    return {"message": "MYLO AGENT is running!"}


@app.post("/")
def echo():
    return JSONResponse({"Status": CurrentSession.initialized})


@app.post("/initialize")
async def initialize(request: Request):
    global CurrentSession

    if CurrentSession.initialized:
        return JSONResponse({"error": "Already initialized"}, 400)

    data = await request.json()
    apiKey = data.get("apiKey")

    if not apiKey:
        return JSONResponse({"error": "API key required"}, 400)

    CurrentSession = SessionData()
    CurrentSession.apiKey = apiKey
    CurrentSession.initialized = True

    return JSONResponse({"message": "Initialized"})


@app.post("/disconnect")
def disconnect():
    global CurrentSession
    CurrentSession = SessionData()
    return JSONResponse({"message": "Disconnected"})


# ---------------- MODEL ----------------

@app.post("/loadModel")
async def loadModel(file: UploadFile = File(...), weightsFile: Optional[UploadFile] = File(None)):
    try:
        global CurrentProject

        previous_state = {
            "modelFilepath": CurrentProject.modelFilepath,
            "uploadedPt2Filepath": CurrentProject.uploadedPt2Filepath,
            "uploadedOnnxFilepath": CurrentProject.uploadedOnnxFilepath,
            "weightsFilepath": getattr(CurrentProject, "weightsFilepath", ""),
        }
        saved_paths = []
        previous_slot_filename = ""
        replaced_backup = ""
        weights_replaced_backup = ""
        metadata_backups = {}

        def rollback_upload():
            for saved_path in saved_paths:
                if os.path.exists(saved_path):
                    os.remove(saved_path)
            if replaced_backup and os.path.exists(replaced_backup):
                shutil.move(replaced_backup, project_path(previous_slot_filename))
            if weights_replaced_backup and os.path.exists(weights_replaced_backup):
                shutil.move(weights_replaced_backup, project_path(previous_state["weightsFilepath"]))
            CurrentProject.modelFilepath = previous_state["modelFilepath"]
            CurrentProject.uploadedPt2Filepath = previous_state["uploadedPt2Filepath"]
            CurrentProject.uploadedOnnxFilepath = previous_state["uploadedOnnxFilepath"]
            CurrentProject.weightsFilepath = previous_state["weightsFilepath"]
            for metadata_name, backup_path in metadata_backups.items():
                current_path = project_path(metadata_name)
                if os.path.exists(current_path):
                    os.remove(current_path)
                if backup_path and os.path.exists(backup_path):
                    shutil.move(backup_path, current_path)
            CurrentProject.dumpInTemp()

        uploaded_extension = os.path.splitext(file.filename or "")[1].lower()
        if uploaded_extension in (".pt2", ".ptpkg", ".torchpackage", ".pt", ".pth"):
            previous_slot_filename = CurrentProject.uploadedPt2Filepath
            target_name = os.path.basename(file.filename or "")
            if previous_slot_filename == target_name and slot_exists(previous_slot_filename):
                replaced_backup = project_path(f".{target_name}.upload-backup")
                shutil.copy2(project_path(previous_slot_filename), replaced_backup)
            model_name = await save_uploaded_to_slot(file, CurrentProject.uploadedPt2Filepath)
            saved_paths.append(project_path(model_name))
            CurrentProject.uploadedPt2Filepath = model_name
            CurrentProject.modelFilepath = model_name
        elif uploaded_extension == ".onnx":
            previous_slot_filename = CurrentProject.uploadedOnnxFilepath
            target_name = os.path.basename(file.filename or "")
            if previous_slot_filename == target_name and slot_exists(previous_slot_filename):
                replaced_backup = project_path(f".{target_name}.upload-backup")
                shutil.copy2(project_path(previous_slot_filename), replaced_backup)
            model_name = await save_uploaded_to_slot(file, CurrentProject.uploadedOnnxFilepath)
            saved_paths.append(project_path(model_name))
            CurrentProject.uploadedOnnxFilepath = model_name
        else:
            return JSONResponse({"error": f"Unsupported model extension: {uploaded_extension}"}, 400)

        if weightsFile:
            weights_target_name = os.path.basename(weightsFile.filename or "")
            if (
                weights_target_name
                and weights_target_name == previous_state["weightsFilepath"]
                and slot_exists(weights_target_name)
            ):
                weights_replaced_backup = project_path(f".{weights_target_name}.upload-backup")
                shutil.copy2(project_path(weights_target_name), weights_replaced_backup)
            weights_path = await saveFile(weightsFile, path="temp_project")
            CurrentProject.weightsFilepath = os.path.basename(weights_path)
            saved_paths.append(weights_path)

        full_model_path = project_path(model_name)
        weights_full_path = ""
        if getattr(CurrentProject, "weightsFilepath", ""):
            weights_full_path = project_path(CurrentProject.weightsFilepath)

        for metadata_name in ("descriptor.json", "modelInfo.json"):
            metadata_path = project_path(metadata_name)
            backup_path = project_path(f".{metadata_name}.upload-backup")
            if os.path.exists(metadata_path):
                shutil.copy2(metadata_path, backup_path)
                metadata_backups[metadata_name] = backup_path
            else:
                metadata_backups[metadata_name] = ""

        result = analyseModel(
            full_model_path,
            originalFilename=file.filename,
            weightsPath=weights_full_path,
        )

        if (
            uploaded_extension in (".pt", ".pth")
            and isinstance(result, dict)
            and (result.get("summary") or {}).get("editable") is False
        ):
            result = {
                "error": (
                    "This .pt/.pth file contains weights or a raw torch.save model but no editable architecture. "
                    "Upload a torch.package archive or a MYLO checkpoint containing model_config."
                ),
                "format": (result.get("summary") or {}).get("format"),
                "editable": False,
            }

        if isinstance(result, dict) and "error" in result:
            rollback_upload()
            return JSONResponse(result, 400)

        CurrentProject.dumpInTemp()

        with open(project_path("modelInfo.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)

        if previous_slot_filename and previous_slot_filename != model_name:
            clear_slot(previous_slot_filename)
        if replaced_backup and os.path.exists(replaced_backup):
            os.remove(replaced_backup)
        if weights_replaced_backup and os.path.exists(weights_replaced_backup):
            os.remove(weights_replaced_backup)
        for backup_path in metadata_backups.values():
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)

        return JSONResponse(result)

    except Exception as e:
        if "rollback_upload" in locals():
            try:
                rollback_upload()
            except Exception as rollback_error:
                print(f"Error rolling back model upload: {rollback_error}")
        print(f"Error in /loadModel: {e}")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/generateOnnxFromModel")
@app.post("/generateOnnxFromPt2")
async def generateOnnxFromPt2():
    try:
        global CurrentProject
        loadProjectData()
        if not CurrentProject.uploadedPt2Filepath:
            return JSONResponse({"error": "No uploaded editable PyTorch model found"}, 400)

        result = exportUploadedPt2ToOnnx(
            CurrentProject.uploadedPt2Filepath,
            CurrentProject.uploadedOnnxFilepath,
        )
        if "error" in result:
            return JSONResponse(result, 400)

        CurrentProject.uploadedOnnxFilepath = result["onnxFile"]
        CurrentProject.dumpInTemp()
        return JSONResponse({
            **result,
            "files": project_slots(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/download-optimized")
async def downloadOptimized(request: Request):
    data = await request.json()
    slot = (data.get("slot") or "").lower()
    file_name = os.path.basename(data.get("fileName") or "")

    if file_name:
        full_path = slot_path(file_name)
        if not os.path.exists(full_path):
            return JSONResponse({"error": f"Optimized file not found: {file_name}"}, 404)
        return FileResponse(full_path, filename=file_name)

    if slot == "pt2":
        filename = CurrentProject.optimizedPt2Filepath
    elif slot == "onnx":
        filename = CurrentProject.optimizedOnnxFilepath
    else:
        return JSONResponse({"error": "slot must be 'pt2' or 'onnx'"}, 400)

    if not filename:
        return JSONResponse({"error": f"Optimized file not found for slot: {slot}"}, 404)

    full_path = slot_path(filename)
    if not os.path.exists(full_path):
        return JSONResponse({"error": f"Optimized file not found: {filename}"}, 404)

    return FileResponse(full_path, filename=filename)


@app.post("/visualizeProjectFile")
async def visualizeProjectFile(request: Request):
    try:
        data = await request.json()
        file_name = os.path.basename(data.get("filePath") or data.get("fileName") or "")

        if not file_name:
            return JSONResponse({"error": "fileName is required"}, 400)

        full_path = project_path(file_name)
        if not os.path.exists(full_path):
            return JSONResponse({"error": f"File not found: {file_name}"}, 404)

        result = analyseModel(full_path, originalFilename=file_name)
        if isinstance(result, dict) and "error" in result:
            return JSONResponse(result, 400)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/modelDiagnostics")
async def modelDiagnostics(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        descriptor = loadProjectDescriptor(CurrentProject.modelFilepath)
        result = generateModelDiagnostics(
            descriptor,
            weights_path=(
                getattr(CurrentProject, "weightsFilepath", "")
                or CurrentProject.modelFilepath
            ),
            export_name=body.get("exportName", "model_diagnostics.onnx"),
            report_name=body.get("reportName", "model_diagnostics_report.json"),
            mode=body.get("mode", "dummy"),
            csv_path=project_path(CurrentProject.csvFilepath) if CurrentProject.csvFilepath else "",
            input_features=body.get("inputFeatures", []),
            output_features=body.get("outputFeatures", body.get("targetFeature", [])),
            epochs=body.get("epochs", 10),
            problem_type=body.get("problemType", "regression"),
            batch_size=body.get("batchSize", 32),
            validation_split=body.get("validationSplit", 0.2),
            learning_rate=body.get("learningRate", 0.001),
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)



# ---------------- MODEL EDITOR ----------------

@app.post("/getModelDescriptor")
async def getModelDescriptor():
    try:
        descriptor = loadProjectDescriptor()
        return JSONResponse({"descriptor": descriptor.to_dict()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)


@app.post("/validateModelDescriptor")
async def validateModelDescriptor(request: Request):
    """Validate a descriptor payload.

    By default uses lenient validation (allows draft nodes).
    Pass strict=true to enforce full validation (save/compile-time check).
    """
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        strict = bool(body.get("strict", False))
        if not descriptor_dict:
            return JSONResponse({"error": "descriptor is required"}, 400)
        return JSONResponse(validateDescriptorPayload(descriptor_dict, strict=strict))
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


def _handle_edit_request(body: dict, persist: bool, dry_run: bool) -> JSONResponse:
    """Shared handler for editModel and previewEdit endpoints.

    Editing (persist=False): lenient validation, draft nodes allowed.
    Saving (persist=True): strict validation, all nodes must be valid.
    """
    operation = body.get("operation")
    payload = body.get("payload", {})
    if not operation:
        return JSONResponse({"error": "operation is required"}, 400)

    result = applyModelEdit(
        descriptor_dict=body.get("descriptor"),
        operation=operation,
        payload=payload,
        persist=persist,
        view_mode=body.get("viewMode", "summary"),
        expanded_nodes=body.get("expandedNodes", []),
        dry_run=dry_run,
    )
    status = 200 if result.get("success") or result.get("valid") else 400
    return JSONResponse(result, status)


@app.post("/editModel")
async def editModel(request: Request):
    """Apply an edit operation. Uses lenient validation by default.

    Draft/disconnected nodes are allowed. Only the active connected subgraph
    is validated. Set persist=true to enforce strict validation on save.
    """
    try:
        body = await request.json()
        return _handle_edit_request(body, persist=bool(body.get("persist", False)), dry_run=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/previewEdit")
async def previewEdit(request: Request):
    """Preview an edit without persisting. Always uses lenient validation."""
    try:
        body = await request.json()
        return _handle_edit_request(body, persist=False, dry_run=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/checkEdge")
async def checkEdge(request: Request):
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        source = body.get("source") or body.get("from")
        target = body.get("target") or body.get("to")
        source_port = body.get("sourcePort", body.get("source_port", "output"))
        target_port = body.get("targetPort", body.get("target_port", "input"))

        if not descriptor_dict or not source or not target:
            return JSONResponse({"error": "descriptor, source, and target are required"}, 400)

        result = checkEdgeCompatibility(
            descriptor_dict=descriptor_dict,
            source=source,
            target=target,
            source_port=source_port,
            target_port=target_port,
        )
        status = 200 if result.get("compatible") else 400
        return JSONResponse(result, status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/saveModelDescriptor")
async def saveModelDescriptorEndpoint(request: Request):
    """Save a model descriptor. Uses STRICT validation — no draft nodes allowed."""
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        if not descriptor_dict:
            return JSONResponse({"error": "descriptor is required"}, 400)

        result = saveModelDescriptor(
            descriptor_dict=descriptor_dict,
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        status = 200 if result.get("success") or result.get("valid") else 400
        return JSONResponse(result, status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/compileModelDescriptor")
async def compileModelDescriptor(request: Request):
    """Validate descriptor strictly for compilation/export."""
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        if not descriptor_dict:
            return JSONResponse({"error": "descriptor is required"}, 400)

        from Core.ArchitectureDescriptor import ArchitectureDescriptor
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.normalize_inplace()

        draft_ids = _get_draft_node_ids(descriptor)

        if draft_ids:
            return JSONResponse({
                "valid": False,
                "strict": True,
                "code": "DRAFT_NODES_EXIST",
                "message": f"Cannot compile: {len(draft_ids)} disconnected node(s) found",
                "draftNodes": sorted(draft_ids),
                "error": "All nodes must be connected to an Input before compiling. "
                         "Connect draft nodes or remove them.",
            }, 400)

        # Draft-free graphs use strict validation.
        try:
            descriptor.validate()
            shapes = descriptor._propagate_shapes(mutate=False)
            return JSONResponse({
                "valid": True,
                "strict": True,
                "descriptor": descriptor.to_dict(),
                "shapes": {k: v for k, v in shapes.items()},
                "message": "Descriptor is valid for compilation",
            })
        except Exception as exc:
            return JSONResponse({
                "valid": False,
                "strict": True,
                "code": "VALIDATION_ERROR",
                "message": str(exc),
                "error": str(exc),
            }, 400)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/visualizeModel")
async def visualizeModelEndpoint(request: Request):
    try:
        body = await request.json()
        result = visualizeModel(
            descriptor_dict=body.get("descriptor"),
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        if "error" in result and not result.get("valid", True):
            return JSONResponse(result, 400)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/expandModelNodes")
async def expandModelNodes(request: Request):
    """Expand individual descriptor nodes into ONNX op-level sub-graphs."""
    try:
        body = await request.json()
        node_ids = body.get("nodeIds") or body.get("node_ids") or []
        if not node_ids:
            return JSONResponse({"error": "nodeIds is required"}, 400)

        result = expandNodes(
            descriptor_dict=body.get("descriptor"),
            node_ids=node_ids,
            current_expanded=body.get("expandedNodes", []),
        )
        if "error" in result:
            return JSONResponse(result, 400)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/collapseModelNodes")
async def collapseModelNodes(request: Request):
    """Collapse expanded ONNX sub-graphs back to high-level descriptor nodes."""
    try:
        body = await request.json()
        node_ids = body.get("nodeIds") or body.get("node_ids") or []
        if not node_ids:
            return JSONResponse({"error": "nodeIds is required"}, 400)

        result = collapseNodes(
            descriptor_dict=body.get("descriptor"),
            node_ids=node_ids,
            current_expanded=body.get("expandedNodes", []),
        )
        if "error" in result:
            return JSONResponse(result, 400)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/editor/catalog")
async def editor_catalog():
    """Return the node catalog for the model editor UI."""
    return JSONResponse(get_catalog())


@app.get("/editor/templates")
async def editor_templates():
    """Return common model template layouts."""
    return JSONResponse({"templates": TEMPLATES})


@app.post("/editor/undo")
async def editor_undo(request: Request):
    """Revert the last persisted model edit."""
    try:
        body = await request.json()
        result = undoModelEdit(
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/editor/redo")
async def editor_redo(request: Request):
    """Re-apply the last undone model edit."""
    try:
        body = await request.json()
        result = redoModelEdit(
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------- CSV ----------------

@app.post("/loadCSV")
async def loadCsv(file: UploadFile = File(...)):
    try:
        global CurrentProject

        if CurrentProject.csvFilepath:
            path = project_path(CurrentProject.csvFilepath)
            if os.path.exists(path):
                os.remove(path)

        csv_path = await saveFile(file, path="temp_project")
        csv_name = os.path.basename(csv_path)

        CurrentProject.csvFilepath = csv_name
        CurrentProject.dumpInTemp()

        result = analyseCSV(project_path(csv_name))

        with open(project_path("csvInfo.json"), "w") as f:
            json.dump(result, f)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/advancedAnalysis")
async def getAdvancedAnalysis():
    try:
        loadProjectData()

        if not CurrentProject.csvFilepath:
            return JSONResponse({"error": "No CSV file loaded"}, 400)

        csv_path = project_path(CurrentProject.csvFilepath)
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "CSV file not found"}, 404)

        df = pd.read_csv(csv_path)

        descriptive_stats = calculateDescriptiveStatistics(df)
        correlation_matrix = calculateCorrelationMatrix(df)
        distributions = analyzeDistributions(df)
        data_quality = performDataQualityChecks(df)

        result = {
            "descriptive_statistics": descriptive_stats,
            "correlation_analysis": correlation_matrix,
            "distribution_analysis": distributions,
            "data_quality_checks": data_quality
        }

        result = make_json_serializable(result)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/correlationAnalysis")
async def getCorrelationAnalysis():
    try:
        loadProjectData()

        if not CurrentProject.csvFilepath:
            return JSONResponse({"error": "No CSV file loaded"}, 400)

        csv_path = project_path(CurrentProject.csvFilepath)
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "CSV file not found"}, 404)

        df = pd.read_csv(csv_path)
        result = calculateCorrelationMatrix(df)

        result = make_json_serializable(result)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/chartData")
async def getChartData():
    try:
        loadProjectData()

        if not CurrentProject.csvFilepath:
            return JSONResponse({"error": "No CSV file loaded"}, 400)

        csv_path = project_path(CurrentProject.csvFilepath)
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "CSV file not found"}, 404)

        df = pd.read_csv(csv_path)
        result = generateChartData(df)

        result = make_json_serializable(result)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/targetAnalysis")
async def getTargetAnalysis(request: Request):
    try:
        loadProjectData()

        if not CurrentProject.csvFilepath:
            return JSONResponse({"error": "No CSV file loaded"}, 400)

        body = await request.json()
        target_col = body.get("targetColumn")

        if not target_col:
            return JSONResponse({"error": "targetColumn is required"}, 400)

        csv_path = project_path(CurrentProject.csvFilepath)
        if not os.path.exists(csv_path):
            return JSONResponse({"error": "CSV file not found"}, 404)

        df = pd.read_csv(csv_path)
        result = analyzeTargetVariable(df, target_col)

        result = make_json_serializable(result)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------- OPTIMIZATION ----------------

@app.post("/optimizeModel")
async def optimizeModel(request: Request):
    global CurrentProject, queue, optimizing

    if optimizing:
        return JSONResponse({"error": "Optimization already in progress"}, 400)

    optimizing = True
    loadProjectData()

    queue = []
    body = await request.json()
    requestedModelFile = os.path.basename(body.get("modelFilepath", "") or "")
    inputFeatures = body.get("inputFeatures", [])
    targetFeature = body.get("targetFeature", [])
    epochs = body.get("epochs", 10)
    encoding = body.get("encoding", "none")
    strategy = body.get("strategy", "brute-force")
    generations = body.get("generations", 5)
    problemType = body.get("problemType", "regression")

    if requestedModelFile:
        requested_model_path = project_path(requestedModelFile)
        if not os.path.exists(requested_model_path):
            optimizing = False
            return JSONResponse({"error": f"Model file not found: {requestedModelFile}"}, 404)
        CurrentProject.modelFilepath = requestedModelFile
        CurrentProject.dumpInTemp()

    if CurrentProject.modelFilepath.endswith(".onnx"):
        optimizing = False
        return JSONResponse({"error": "ONNX optimization disabled"}, 400)

    def callback(status):
        queue.append(status)

    try:
        result = await startOptimization(
            CurrentProject,
            OptimizationRequest(
                encoding=encoding,
                strategy=strategy,
                inputFeatures=inputFeatures,
                targetFeature=targetFeature,
                epochs=epochs,
                generations=generations,
                problemType=problemType
            ),
            callback
        )
    except Exception as e:
        result = {"error": str(e)}
        queue.append({"type": "error", "status": f"Error: {e}", "error": True, "progress": 0})
    finally:
        # Always release the global lock, even if the optimizer blew up.
        optimizing = False

    if isinstance(result, dict) and result.get("status") == "success":
        optimized_pt2_path = result.get("model_path")
        optimized_onnx_path = result.get("model_onnx_path")
        if optimized_pt2_path and os.path.exists(optimized_pt2_path):
            CurrentProject.optimizedPt2Filepath = os.path.basename(optimized_pt2_path)
        if optimized_onnx_path and os.path.exists(optimized_onnx_path):
            CurrentProject.optimizedOnnxFilepath = os.path.basename(optimized_onnx_path)
        CurrentProject.dumpInTemp()

        queue.append({
            "type": "complete",
            "status": "Optimization complete",
            "progress": 100,
            "improvement": result.get("improvement"),
            "summary": result.get("summary"),
        })
    elif isinstance(result, dict) and "error" in result:
        queue.append({"type": "error", "status": f"Optimization failed: {result.get('error')}", "error": True, "progress": 0})

    return JSONResponse({"status_updates": queue, "result": result})


@app.get("/optimizationStatus")
async def optimizationStatus(request: Request):
    global queue, optimizing
    if not optimizing:
        return JSONResponse({"status": "No optimization in progress"})

    async def statusGenerator():
        while True:
            if await request.is_disconnected():
                break
            if queue:
                message = queue.pop(0)
                yield f"data: {json.dumps(message)}\n\n"
            elif not optimizing:
                yield f"data: {json.dumps({'type': 'done', 'status': 'Optimization finished', 'progress': 100})}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(statusGenerator(), media_type="text/event-stream")


@app.get("/optimizationReport")
async def optimizationReport():
    """Return the full persisted optimization report (metrics, mutation
    timeline, lineage, training curves, architecture diff)."""
    report = load_optimization_report()
    if report is None:
        return JSONResponse({"error": "No optimization report available"}, 404)
    return JSONResponse(report)


# ---------------- PROJECT ----------------

@app.post("/newProject")
async def newProject(request: Request):
    shutil.rmtree("temp_project", ignore_errors=True)
    os.makedirs("temp_project", exist_ok=True)

    shutil.rmtree("temp_export", ignore_errors=True)
    os.makedirs("temp_export", exist_ok=True)

    _ProjectHistory.clear_project()

    global CurrentProject
    CurrentProject = ProjectData()

    data = await request.json()
    CurrentProject.name = data.get("name", "New Project")
    CurrentProject.id = data.get("id", "")
    CurrentProject.modelFilepath = ""
    CurrentProject.uploadedPt2Filepath = ""
    CurrentProject.uploadedOnnxFilepath = ""
    CurrentProject.optimizedPt2Filepath = ""
    CurrentProject.optimizedOnnxFilepath = ""

    CurrentProject.dumpInTemp()

    return JSONResponse({
        "message": "Success",
        "data": {
            "name": CurrentProject.name,
            "csvFilepath": "",
            "modelFilepath": "",
            "id": CurrentProject.id
        }
    })


@app.post("/getProject")
async def getProject():
    global CurrentProject

    loadProjectData()

    modelData = {}
    csvData = {}

    if os.path.exists(project_path("modelInfo.json")):
        with open(project_path("modelInfo.json")) as f:
            modelData = json.load(f)

    if os.path.exists(project_path("csvInfo.json")):
        with open(project_path("csvInfo.json")) as f:
            csvData = json.load(f)

    report_summary = optimization_report_summary()

    return JSONResponse({
        "projectData": {
            "name": CurrentProject.name,
            "csvFile": CurrentProject.csvFilepath or "",
            "modelFile": CurrentProject.modelFilepath or "",
            "uploadedPt2File": CurrentProject.uploadedPt2Filepath or "",
            "uploadedOnnxFile": CurrentProject.uploadedOnnxFilepath or "",
            "optimizedPt2File": CurrentProject.optimizedPt2Filepath or "",
            "optimizedOnnxFile": CurrentProject.optimizedOnnxFilepath or "",
            "id": CurrentProject.id
        },
        "projectFiles": project_slots(),
        "modelData": modelData,
        "csvData": csvData,
        "hasOptimizationReport": report_summary is not None,
        "optimizationSummary": report_summary
    })


@app.post("/projectFiles")
async def getProjectFiles():
    loadProjectData()
    return JSONResponse({
        "projectData": {
            "name": CurrentProject.name,
            "csvFile": CurrentProject.csvFilepath or "",
            "modelFile": CurrentProject.modelFilepath or "",
            "uploadedPt2File": CurrentProject.uploadedPt2Filepath or "",
            "uploadedOnnxFile": CurrentProject.uploadedOnnxFilepath or "",
            "optimizedPt2File": CurrentProject.optimizedPt2Filepath or "",
            "optimizedOnnxFile": CurrentProject.optimizedOnnxFilepath or "",
            "id": CurrentProject.id
        },
        "files": project_slots(),
    })


@app.post("/exportProject")
async def exportProject():
    try:
        shutil.rmtree("temp_export", ignore_errors=True)
        os.makedirs("temp_export", exist_ok=True)

        global CurrentProject
        CurrentProject.dumpInTemp()

        zip_path = os.path.join("temp_export", f"{CurrentProject.name}.mylo")

        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk("temp_project"):
                for f in files:
                    path = os.path.join(root, f)
                    zipf.write(path, os.path.relpath(path, "temp_project"))

        return FileResponse(zip_path, media_type='application/zip', filename=f"{CurrentProject.name}.mylo")

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/loadProject")
async def loadProject(file: UploadFile = File(...)):
    try:
        shutil.rmtree("temp_project", ignore_errors=True)
        os.makedirs("temp_project", exist_ok=True)

        shutil.rmtree("temp_export", ignore_errors=True)
        os.makedirs("temp_export", exist_ok=True)

        _ProjectHistory.clear_project()

        temp_path = await saveFile(file, path="temp_export")

        with ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall("temp_project")

        os.remove(temp_path)

        global CurrentProject
        CurrentProject = ProjectData()

        info_path = project_path("projectInfo.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                data = json.load(f)
                CurrentProject.modelFilepath = data.get("modelFilepath", "")
                CurrentProject.csvFilepath = data.get("csvFilepath", "")
                CurrentProject.id = data.get("id", "")
                CurrentProject.name = data.get("name", "")
                CurrentProject.uploadedPt2Filepath = data.get("uploadedPt2Filepath", CurrentProject.modelFilepath or "")
                CurrentProject.uploadedOnnxFilepath = data.get("uploadedOnnxFilepath", "")
                CurrentProject.optimizedPt2Filepath = data.get("optimizedPt2Filepath", "")
                CurrentProject.optimizedOnnxFilepath = data.get("optimizedOnnxFilepath", "")
                if "weightsFilepath" in data:
                    CurrentProject.weightsFilepath = data.get("weightsFilepath")

        if CurrentProject.csvFilepath and not os.path.exists(project_path(CurrentProject.csvFilepath)):
            return JSONResponse({"error": "CSV file missing after load"}, 500)

        for filepath in [
            CurrentProject.uploadedPt2Filepath,
            CurrentProject.uploadedOnnxFilepath,
            CurrentProject.optimizedPt2Filepath,
            CurrentProject.optimizedOnnxFilepath,
        ]:
            if filepath and not os.path.exists(project_path(filepath)):
                return JSONResponse({"error": f"Model file missing after load: {filepath}"}, 500)

        if not CurrentProject.modelFilepath:
            CurrentProject.modelFilepath = CurrentProject.uploadedPt2Filepath or ""

        CurrentProject.dumpInTemp()

        return JSONResponse({"message": "Loaded"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ---------------- RUN ----------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("SERVER_IP", "127.0.0.1"),
        port=int(os.getenv("SERVER_PORT", 8000)),
        log_level="debug"
    )
