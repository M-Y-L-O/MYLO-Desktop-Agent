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
    log_blocked,
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

# EventSource cannot send Authorization headers, so SSE must be exempt.
# /disconnect is exempt so a browser with a stale/missing key can reset the session.
MIDDLEWARE_EXCEPTIONS = ["/", "/initialize", "/optimizationStatus", "/disconnect"]

REPORT_FILENAME = "optimization_report.json"

# ---------------- SETUP ----------------

load_dotenv()

_enable_docs = os.getenv("ENABLE_AGENT_DOCS", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(
    title="MYLO AGENT",
    version="0.2.2",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)

def _cors_origins() -> list[str]:
    """Browser origins allowed to call this loopback agent.

    LOCAL mode talks from the page origin directly to 127.0.0.1, so a deployed
    web UI (e.g. Vercel) must be listed here or CORS preflight fails.
    Override with CORS_ORIGINS=https://app.example.com,http://localhost:3000
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://mylo-web-deployument.vercel.app",
    ]


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    """Chrome Private Network Access: public HTTPS pages → loopback agent."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response



# ---------------- HELPERS ----------------

def project_path(filename: str) -> str:
    return os.path.join("temp_project", filename)


def _client_is_loopback(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in ("127.0.0.1", "::1", "localhost", "testclient")


def safe_extract_zip(zip_ref: ZipFile, dest_dir: str) -> None:
    """Extract zip members only if every path stays under dest_dir (Zip Slip safe)."""
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    for member in zip_ref.infolist():
        name = member.filename
        if not name or name.endswith("/"):
            continue
        # Reject absolute / drive-letter / backslash traversal variants
        if os.path.isabs(name) or name.startswith(("/", "\\")) or ":" in name.split("/")[0]:
            raise ValueError(f"Illegal zip entry path: {name}")
        target = os.path.abspath(os.path.join(dest_dir, name))
        try:
            common = os.path.commonpath([dest_dir, target])
        except ValueError as exc:
            raise ValueError(f"Illegal zip entry path: {name}") from exc
        if common != dest_dir:
            raise ValueError(f"Illegal zip entry path: {name}")
    zip_ref.extractall(dest_dir)


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
        _slot_entry("uploaded_pt2", "uploaded", "Uploaded PT2", CurrentProject.uploadedPt2Filepath, ".pt2", True, True, True),
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
        ".pt": {".pt", ".pth"},
        ".pth": {".pt", ".pth"},
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
    if previous_filename and previous_filename != target_filename:
        clear_slot(previous_filename)
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


# Register CORS last so it is outermost: early auth responses still get ACAO headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    bootstrap = os.getenv("AGENT_BOOTSTRAP_TOKEN", "").strip()
    if bootstrap:
        provided = request.headers.get("X-Bootstrap-Token") or data.get("bootstrapToken")
        if provided != bootstrap:
            return JSONResponse({"error": "Invalid bootstrap token"}, 401)
    elif not _client_is_loopback(request):
        return JSONResponse({"error": "Initialize only allowed from localhost"}, 403)

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

        uploaded_extension = os.path.splitext(file.filename or "")[1].lower()
        if uploaded_extension == ".pt2":
            model_name = await save_uploaded_to_slot(file, CurrentProject.uploadedPt2Filepath)
            CurrentProject.uploadedPt2Filepath = model_name
            CurrentProject.modelFilepath = model_name
        elif uploaded_extension == ".onnx":
            model_name = await save_uploaded_to_slot(file, CurrentProject.uploadedOnnxFilepath)
            CurrentProject.uploadedOnnxFilepath = model_name
        else:
            return JSONResponse({"error": f"Unsupported model extension: {uploaded_extension}"}, 400)

        if weightsFile:
            weights_path = await saveFile(weightsFile, path="temp_project")
            CurrentProject.weightsFilepath = os.path.basename(weights_path)

        CurrentProject.dumpInTemp()

        full_model_path = project_path(model_name)
        weights_full_path = ""
        if getattr(CurrentProject, "weightsFilepath", ""):
            weights_full_path = project_path(CurrentProject.weightsFilepath)

        result = analyseModel(
            full_model_path,
            originalFilename=file.filename,
            weightsPath=weights_full_path,
        )

        if isinstance(result, dict) and "error" in result:
            raise ValueError(result["error"])

        # New model upload starts a fresh edit history
        _ProjectHistory.clear_project()

        with open(project_path("modelInfo.json"), "w") as f:
            json.dump(result, f)

        return JSONResponse(result)

    except Exception as e:
        print(f"Error in /loadModel: {e}")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/generateOnnxFromPt2")
async def generateOnnxFromPt2():
    """Convert the uploaded PT2 into the uploaded ONNX slot."""
    try:
        global CurrentProject
        loadProjectData()
        if not CurrentProject.uploadedPt2Filepath:
            return JSONResponse({"error": "No uploaded PT2 model found"}, 400)

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
            weights_path=getattr(CurrentProject, "weightsFilepath", ""),
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
        log_blocked("getModelDescriptor", str(e), code="LOAD_FAILED")
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
            log_blocked("validate", "descriptor is required", code="MISSING_ARGUMENT")
            return JSONResponse({"error": "descriptor is required"}, 400)
        return JSONResponse(validateDescriptorPayload(descriptor_dict, strict=strict))
    except Exception as e:
        log_blocked("validate", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


def _handle_edit_request(body: dict, persist: bool, dry_run: bool) -> JSONResponse:
    """Shared handler for editModel and previewEdit endpoints.

    Editing (persist=False): lenient validation, draft nodes allowed.
    Saving (persist=True): strict validation, all nodes must be valid.
    """
    operation = body.get("operation")
    payload = body.get("payload", {})
    if not operation:
        log_blocked(
            "previewEdit" if dry_run else "editModel",
            "operation is required",
            code="MISSING_ARGUMENT",
            persist=persist,
            dry_run=dry_run,
        )
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
        log_blocked("editModel", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/previewEdit")
async def previewEdit(request: Request):
    """Preview an edit without persisting. Always uses lenient validation."""
    try:
        body = await request.json()
        return _handle_edit_request(body, persist=False, dry_run=True)
    except Exception as e:
        log_blocked("previewEdit", str(e), code="UNKNOWN_ERROR")
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
            log_blocked(
                "checkEdge",
                "descriptor, source, and target are required",
                code="MISSING_ARGUMENT",
                hasDescriptor=bool(descriptor_dict),
                source=source,
                target=target,
            )
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
        log_blocked("checkEdge", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/saveModelDescriptor")
async def saveModelDescriptorEndpoint(request: Request):
    """Save a model descriptor. Uses STRICT validation — no draft nodes allowed."""
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        if not descriptor_dict:
            log_blocked("save", "descriptor is required", code="MISSING_ARGUMENT")
            return JSONResponse({"error": "descriptor is required"}, 400)

        result = saveModelDescriptor(
            descriptor_dict=descriptor_dict,
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        status = 200 if result.get("success") or result.get("valid") else 400
        return JSONResponse(result, status)
    except Exception as e:
        log_blocked("save", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/compileModelDescriptor")
async def compileModelDescriptor(request: Request):
    """Validate descriptor strictly for compilation/export."""
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        if not descriptor_dict:
            log_blocked("compile", "descriptor is required", code="MISSING_ARGUMENT")
            return JSONResponse({"error": "descriptor is required"}, 400)

        from Core.ArchitectureDescriptor import ArchitectureDescriptor
        descriptor = ArchitectureDescriptor.from_dict(descriptor_dict)
        descriptor.normalize_inplace()

        draft_ids = _get_draft_node_ids(descriptor)

        if draft_ids:
            reason = f"Cannot compile: {len(draft_ids)} disconnected node(s) found"
            log_blocked("compile", reason, code="DRAFT_NODES_EXIST", draftNodes=sorted(draft_ids))
            return JSONResponse({
                "valid": False,
                "strict": True,
                "code": "DRAFT_NODES_EXIST",
                "message": reason,
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
            log_blocked("compile", str(exc), code="VALIDATION_ERROR")
            return JSONResponse({
                "valid": False,
                "strict": True,
                "code": "VALIDATION_ERROR",
                "message": str(exc),
                "error": str(exc),
            }, 400)

    except Exception as e:
        log_blocked("compile", str(e), code="UNKNOWN_ERROR")
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
        log_blocked("visualize", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/expandModelNodes")
async def expandModelNodes(request: Request):
    """Expand individual descriptor nodes into ONNX op-level sub-graphs."""
    try:
        body = await request.json()
        node_ids = body.get("nodeIds") or body.get("node_ids") or []
        if not node_ids:
            log_blocked("expandNodes", "nodeIds is required", code="MISSING_ARGUMENT")
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
        log_blocked("expandNodes", str(e), code="UNKNOWN_ERROR")
        return JSONResponse({"error": str(e)}, 500)


@app.post("/collapseModelNodes")
async def collapseModelNodes(request: Request):
    """Collapse expanded ONNX sub-graphs back to high-level descriptor nodes."""
    try:
        body = await request.json()
        node_ids = body.get("nodeIds") or body.get("node_ids") or []
        if not node_ids:
            log_blocked("collapseNodes", "nodeIds is required", code="MISSING_ARGUMENT")
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
        log_blocked("collapseNodes", str(e), code="UNKNOWN_ERROR")
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
        log_blocked("undo", str(e), code="UNKNOWN_ERROR")
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
        log_blocked("redo", str(e), code="UNKNOWN_ERROR")
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

    return JSONResponse({
        "status_updates": queue,
        "result": make_json_serializable(result),
    })


@app.get("/optimizationStatus")
async def optimizationStatus(request: Request):
    """SSE progress feed.

    The UI opens EventSource *before* POST /optimizeModel, so this must always
    return text/event-stream and briefly wait for optimizing=True.
    """
    global queue, optimizing

    async def statusGenerator():
        waited = 0.0
        while not optimizing and waited < 15.0:
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.1)
            waited += 0.1

        if not optimizing:
            yield f"data: {json.dumps({'type': 'idle', 'status': 'No optimization in progress', 'progress': 0})}\n\n"
            return

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
            safe_extract_zip(zip_ref, "temp_project")

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
