import os
import json
import shutil
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

import json
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

from Processing.Optimization.Optimizing import startOptimization
from Types.Types import *
from Processing.Models.ONNXProcessing import analyseOnnx as load_onnx_model
from Processing.Models.DescriptorHandling import loadDescriptorFromBytes, descriptorToGraph
from Processing.Data.DataProcessingForVisualisation import analyseCSV
from Processing.Optimization.Optimizing import optimizeModel as runOptimizationModel
from Utils.FileHandler import saveFile


# ---------------- GLOBAL STATE ----------------

CurrentSession = SessionData()
CurrentProject = ProjectData()

MIDDLEWARE_EXCEPTIONS = ["/", "/initialize", "/docs", "/openapi.json"]

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


def try_extract_descriptor(model_bytes: bytes, is_pytorch: bool):
    try:
        return loadDescriptorFromBytes(
            model_bytes=model_bytes,
            is_pytorch=is_pytorch,
            input_dim=1,
            output_dim=1
        )
    except Exception:
        return None


def parse_model_file(filename: str, model_bytes: bytes, weights_present: bool):
    if filename.endswith(".onnx"):
        return "onnx", None

    if filename.endswith(".pt2"):
        descriptor = try_extract_descriptor(model_bytes, True)
        if not descriptor:
            raise ValueError("Invalid .pt2 descriptor")
        return "descriptor", descriptor

    if weights_present:
        descriptor = try_extract_descriptor(model_bytes, False)
        return "descriptor", descriptor

    if filename.endswith((".pt", ".pth")):
        descriptor = try_extract_descriptor(model_bytes, True)
        if descriptor:
            return "descriptor", descriptor
        return "raw_pytorch", None

    raise ValueError(f"Unsupported model format: {filename}")


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

        # cleanup
        if CurrentProject.modelFilepath:
            path = project_path(CurrentProject.modelFilepath)
            if os.path.exists(path):
                os.remove(path)

        if getattr(CurrentProject, "weightsFilepath", None):
            path = project_path(CurrentProject.weightsFilepath)
            if os.path.exists(path):
                os.remove(path)

        # save
        model_path = await saveFile(file, path="temp_project")
        model_name = os.path.basename(model_path)
        CurrentProject.modelFilepath = model_name

        if weightsFile:
            weights_path = await saveFile(weightsFile, path="temp_project")
            CurrentProject.weightsFilepath = os.path.basename(weights_path)

        CurrentProject.dumpInTemp()

        full_model_path = project_path(model_name)

        with open(full_model_path, "rb") as f:
            model_bytes = f.read()

        mode, descriptor = parse_model_file(file.filename.lower(), model_bytes, weightsFile is not None)

        if mode == "onnx":
            result = load_onnx_model(full_model_path)

        elif mode == "descriptor":
            if not descriptor:
                raise ValueError("Descriptor extraction failed")
            result = descriptorToGraph(descriptor)

        else:
            result = {"type": "raw_pytorch", "warning": "Limited visualization"}

        with open(project_path("modelInfo.json"), "w") as f:
            json.dump(result, f)

        return JSONResponse(result)

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


# ---------------- OPTIMIZATION ----------------

@app.post("/optimizeModel")
async def optimizeModel(request: Request):
    queue = []
    inputFeatures = request.get("inputFeatures", [])
    targetFeature = request.get("targetFeature", "")
    epochs = request.get("epochs", 10)
    encoding = request.get("encoding", "none")
    strategy = request.get("strategy", "brute-force")
    generations = request.get("generations", 5)

    if CurrentProject.modelFilepath.endswith(".onnx"):
        return JSONResponse({"error": "ONNX optimization disabled"}, 400)

    body = await request.json()
    queue = []

    def callback(status):
        queue.append(status)

    startOptimization(CurrentProject, OptimizationRequest(encoding=encoding, strategy=strategy, inputFeatures=inputFeatures, targetFeature=targetFeature, epochs=epochs, generations=generations), statusCallback)

    return JSONResponse({"status_updates": queue, "result": result})


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

    if not CurrentProject.id and os.path.exists(project_path("projectInfo.json")):
        with open(project_path("projectInfo.json")) as f:
            data = json.load(f)
            CurrentProject.name = data.get("name", "")
            CurrentProject.id = data.get("id", "")
            CurrentProject.csvFilepath = data.get("csvFilepath", "")
            CurrentProject.modelFilepath = data.get("modelFilepath", "")
            if "weightsFilepath" in data:
                CurrentProject.weightsFilepath = data.get("weightsFilepath")

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
            "id": CurrentProject.id
        },
        "modelData": modelData,
        "csvData": csvData
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
                if "weightsFilepath" in data:
                    CurrentProject.weightsFilepath = data.get("weightsFilepath")

        # validation 
        if CurrentProject.modelFilepath and not os.path.exists(project_path(CurrentProject.modelFilepath)):
            return JSONResponse({"error": "Model file missing after load"}, 500)

        if CurrentProject.csvFilepath and not os.path.exists(project_path(CurrentProject.csvFilepath)):
            return JSONResponse({"error": "CSV file missing after load"}, 500)

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