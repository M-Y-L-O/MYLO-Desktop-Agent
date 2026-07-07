import os
from urllib import request

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

import json
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

from Types.Types import *
from Processing.ModelProcessing import load_onnx_model
from Processing.DataProcessingForVisualisation import analyseCSV
from Utils.FileHandler import saveFile, loadData


from tkinter import Tk

root = Tk()
root.withdraw()

# APP SETUP

CurrentSession = SessionData()
CurrentProject = ProjectData()

MIDDLEWARE_EXCEPTIONS = ["/initialize", "/docs", "/openapi.json"]

load_dotenv()

app = FastAPI(
    title="MYLO AGENT",
    version="0.1.0",
    
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = []

# MIDDLEWARE

@app.middleware("http")
async def checkRequest(request: Request, call_next):
    global CurrentSession

    if request.method == "OPTIONS":
        return await call_next(request)

    if(request.url.path not in MIDDLEWARE_EXCEPTIONS and not CurrentSession.initialized):
        return JSONResponse(content={"error": "Session not initialized"}, status_code=400)
    
    if(request.url.path not in MIDDLEWARE_EXCEPTIONS):
        apiKey = request.headers.get("Authorization", "").replace("Bearer ", "")
        if apiKey != CurrentSession.apiKey:
            return JSONResponse(content={"error": "Invalid API key"}, status_code=401)
        
    response = await call_next(request)
    return response

# ROUTES

@app.get("/")
async def root():
    return {"message": "MYLO AGENT is running!"}

@app.post("/")
def echo():
    return JSONResponse(content={"Status": CurrentSession.initialized}, status_code=200)

#Session handling routes

@app.post("/initialize")
async def initialize(request: Request):
    try:
        global CurrentSession
        if(CurrentSession.initialized):
            return JSONResponse(content={"error": "Client already initialized"}, status_code=400)
        
        CurrentSession = SessionData()

        data = await request.json()
        apiKey = data.get("apiKey", "")
        if not apiKey:
            return JSONResponse(content={"error": "API key is required"}, status_code=400)
        CurrentSession.apiKey = apiKey
        CurrentSession.initialized = True
        print("Session initialized with API key:", CurrentSession.apiKey)

        return JSONResponse(content={"message": "Session initialized successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/disconnect")
def disconnect():
    try:
        global CurrentSession
        CurrentSession.apiKey = ""
        CurrentSession.initialized = False
        return JSONResponse(content={"message": "Session disconnected successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#Data handling routes

@app.post("/loadModel")
async def loadModel(file: UploadFile = File(...)):
    try:
        global CurrentProject
        
        tempPath = await saveFile(file)

        if(CurrentProject.modelFilepath != ""):
            loadedModel = os.path.join("temp_project", os.path.basename(CurrentProject.modelFilepath))
            if(os.path.exists(loadedModel)):
                os.remove(loadedModel)

        CurrentProject.modelFilepath = tempPath
        CurrentProject.dumpInTemp()
        
        result = load_onnx_model(tempPath)

        with open("temp_project/modelInfo.json", "w") as f:
            json.dump(result, f)

        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/loadCSV")
async def loadCsv(file: UploadFile = File(...)):
    try:
        global CurrentProject

        tempPath = await saveFile(file)
       
        if(CurrentProject.csvFilepath != ""):
            loadedCSV = os.path.join("temp_project", os.path.basename(CurrentProject.csvFilepath))
            if(os.path.exists(loadedCSV)):
                os.remove(loadedCSV)

        CurrentProject.csvFilepath = tempPath
        CurrentProject.dumpInTemp()

        result = analyseCSV(tempPath)

        with open("temp_project/csvInfo.json", "w") as f:
            json.dump(result, f)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Tools

@app.post("/optimizeModel")
async def optimizeModel(request: Request):
    queue = []
    inputFeatures = request.get("inputFeatures", [])
    targetFeature = request.get("targetFeature", "")
    epochs = request.get("epochs", 10)
    encoding = request.get("encoding", "none")
    strategy = request.get("strategy", "brute-force")

    def statusCallback(status):
        queue.append(status)

    optimizeModel(CurrentProject, OptimizationRequest(encoding=encoding, strategy=strategy, inputFeatures=inputFeatures, targetFeature=targetFeature, epochs=epochs), statusCallback)

#Project handling routes

@app.post("/newProject")
async def newProject(Request: Request):
    try:
        if(os.path.exists("temp_project")):
            shutil.rmtree("temp_project")
        
        os.makedirs("temp_project")

        if(os.path.exists("temp_export")):
            shutil.rmtree("temp_export")
        
        os.makedirs("temp_export")
        global CurrentProject
        CurrentProject = ProjectData()
        
        bodyData = await Request.json()
        print("Received new project data:", bodyData)
        CurrentProject.name = bodyData.get("name", "New Project")
        CurrentProject.csvFilepath = ""
        CurrentProject.modelFilepath = ""
        CurrentProject.id = bodyData.get("id", "")
        CurrentProject.dumpInTemp()

        return JSONResponse(content={"message": "Success", "data": {"name": CurrentProject.name, "csvFilepath": CurrentProject.csvFilepath, "modelFilepath": CurrentProject.modelFilepath, "id": CurrentProject.id}})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/getProject")
async def getProject(Request: Request):
    try:
        global CurrentProject
        if(CurrentProject.id == ""):
            if(os.path.exists("temp_project/projectInfo.json")):
                with open("temp_project/projectInfo.json", "r") as f:
                    projectData = json.load(f)
                    CurrentProject.name = projectData.get("name", "")
                    CurrentProject.csvFilepath = projectData.get("csvFilepath", "")
                    CurrentProject.modelFilepath = projectData.get("modelFilepath", "")
                    CurrentProject.id = projectData.get("id", "")

        projectData = {
            "name": CurrentProject.name,
            "csvFile": os.path.basename(CurrentProject.csvFilepath) if CurrentProject.csvFilepath else "",
            "modelFile": os.path.basename(CurrentProject.modelFilepath) if CurrentProject.modelFilepath else "",
            "id":CurrentProject.id
        }

        modelData={}
        if(os.path.exists("temp_project/modelInfo.json")):
            with open("temp_project/modelInfo.json", "r") as f:
                modelData = json.load(f)
        csvData={}
        if(os.path.exists("temp_project/csvInfo.json")):
            with open("temp_project/csvInfo.json", "r") as f:
                csvData = json.load(f)

        result = {
            "projectData": projectData,
            "modelData": modelData,
            "csvData": csvData
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/exportProject")
async def exportProject(Request: Request):
    try:
        if(os.path.exists("temp_export")):
            shutil.rmtree("temp_export")
        
        os.makedirs("temp_export")
        global CurrentProject
        auxProject = ProjectData()
        auxProject.modelFilepath = CurrentProject.modelFilepath
        auxProject.csvFilepath = CurrentProject.csvFilepath
        auxProject.name = CurrentProject.name
        auxProject.id = CurrentProject.id
        if(CurrentProject.csvFilepath != ""):
            loadedCSV = os.path.join("temp_project", os.path.basename(CurrentProject.csvFilepath))
            if(not os.path.exists(loadedCSV)):
                shutil.copy(CurrentProject.csvFilepath, loadedCSV)
                auxProject.csvFilepath = loadedCSV
        
        if(CurrentProject.modelFilepath != ""):
            loadedModel = os.path.join("temp_project", os.path.basename(CurrentProject.modelFilepath))
            if(not os.path.exists(loadedModel)):
                shutil.copy(CurrentProject.modelFilepath, loadedModel)
                auxProject.modelFilepath = loadedModel
        

        auxProject.dumpInTemp()

        savePath = os.path.join("temp_export", f"{auxProject.name}.mylo")
        with ZipFile(savePath, 'w', compression=ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk("temp_project"):
                for file in files:
                    zip_ref.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "temp_project"))
        return FileResponse(savePath, media_type='application/zip', filename=f"{auxProject.name}.mylo")
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/loadProject")
async def loadProject(file: UploadFile = File(...)):
    try:
        if(os.path.exists("temp_project")):
            shutil.rmtree("temp_project")
        
        os.makedirs("temp_project")

        if(os.path.exists("temp_export")):
            shutil.rmtree("temp_export")
        
        os.makedirs("temp_export")

        tempPath = await saveFile(file, path="temp_export")
        with ZipFile(tempPath, 'r') as zip_ref:
            zip_ref.extractall("temp_project")
            zip_ref.close()
        global CurrentProject

        with open("temp_project/projectInfo.json", "r") as f:
            projectData = json.load(f)
            CurrentProject.modelFilepath = projectData.get("modelFilepath", "")
            CurrentProject.csvFilepath = projectData.get("csvFilepath", "")
            CurrentProject.id = projectData.get("id", "")
            CurrentProject.name = projectData.get("name", "")
            CurrentProject.dumpInTemp()

        os.remove(tempPath)
        
        return JSONResponse(content={"message": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")), log_level="debug")