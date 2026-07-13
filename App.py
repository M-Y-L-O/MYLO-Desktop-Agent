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

import json
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

from Processing.Optimization.Optimizing import startOptimization
from Types.Types import *
from Processing.Models.ModelProcessing import analyseModel
from Processing.Models.ModelEditing import (
    loadProjectDescriptor,
    validateDescriptorPayload,
    applyModelEdit,
    saveModelDescriptor,
    visualizeModel,
    expandNodes,
    collapseNodes,
    getEditorCatalog,
)
from Processing.Data.DataProcessingForVisualisation import analyseCSV
from Processing.Optimization.Optimizing import startOptimization
from Utils.FileHandler import saveFile


# ---------------- GLOBAL STATE ----------------

CurrentSession = SessionData()
CurrentProject = ProjectData()
queue = []
optimizing = False

MIDDLEWARE_EXCEPTIONS = ["/", "/initialize", "/docs", "/openapi.json", "/optimizationStatus"]

load_dotenv()


# ---------------- SETUP ----------------

app = FastAPI(title="MYLO AGENT", version="0.2.1")

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
        _slot_entry("uploaded_pt2", "uploaded", "Uploaded PT2", CurrentProject.uploadedPt2Filepath, ".pt2", True, True, False),
        _slot_entry("uploaded_onnx", "uploaded", "Uploaded ONNX", CurrentProject.uploadedOnnxFilepath, ".onnx", True, False, False),
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

        with open(project_path("modelInfo.json"), "w") as f:
            json.dump(result, f)

        return JSONResponse(result)

    except Exception as e:
        print(f"Error in /loadModel: {e}")
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
    try:
        body = await request.json()
        descriptor_dict = body.get("descriptor")
        if not descriptor_dict:
            return JSONResponse({"error": "descriptor is required"}, 400)
        return JSONResponse(validateDescriptorPayload(descriptor_dict))
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/editModel")
async def editModel(request: Request):
    try:
        body = await request.json()
        operation = body.get("operation")
        payload = body.get("payload", {})
        if not operation:
            return JSONResponse({"error": "operation is required"}, 400)

        result = applyModelEdit(
            descriptor_dict=body.get("descriptor"),
            operation=operation,
            payload=payload,
            persist=bool(body.get("persist", False)),
            view_mode=body.get("viewMode", "summary"),
            expanded_nodes=body.get("expandedNodes", []),
        )
        status = 200 if result.get("success") or result.get("valid") else 400
        return JSONResponse(result, status)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/saveModelDescriptor")
async def saveModelDescriptorEndpoint(request: Request):
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


@app.post("/getModelEditorCatalog")
async def getModelEditorCatalog():
    return JSONResponse(getEditorCatalog())


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

    result = await startOptimization(CurrentProject, OptimizationRequest(encoding=encoding, strategy=strategy, inputFeatures=inputFeatures, targetFeature=targetFeature, epochs=epochs, generations=generations, problemType=problemType), callback)
    if isinstance(result, dict) and result.get("status") == "success":
        optimized_pt2_path = result.get("model_path")
        optimized_onnx_path = result.get("model_onnx_path")
        if optimized_pt2_path and os.path.exists(optimized_pt2_path):
            CurrentProject.optimizedPt2Filepath = os.path.basename(optimized_pt2_path)
        if optimized_onnx_path and os.path.exists(optimized_onnx_path):
            CurrentProject.optimizedOnnxFilepath = os.path.basename(optimized_onnx_path)
        CurrentProject.dumpInTemp()
    optimizing = False
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
            await asyncio.sleep(0.1)

    return StreamingResponse(statusGenerator(), media_type="text/event-stream")

# ---------------- PROJECT ----------------

@app.post("/newProject")
async def newProject(request: Request):
    shutil.rmtree("temp_project", ignore_errors=True)
    os.makedirs("temp_project", exist_ok=True)

    shutil.rmtree("temp_export", ignore_errors=True)
    os.makedirs("temp_export", exist_ok=True)

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
        "csvData": csvData
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