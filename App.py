import os
from urllib import request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

import json
from zipfile import ZipFile

from Types.Types import *
from Processing.ModelProcessing import load_onnx_model
from Processing.DataProcessing import analyseCSV
from Utils.FileHandler import pickModelFile,pickCSVFile, pickProjectFile

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
        print(f"API Key from request: {apiKey}, CurrentSession.apiKey: {CurrentSession.apiKey}")
        if apiKey != CurrentSession.apiKey:
            return JSONResponse(content={"error": "Invalid API key"}, status_code=401)
        
    response = await call_next(request)
    return response

# ROUTES

@app.get("/")
async def root():
    return {"message": "MYLO AGENT is running!"}

#Session handling routes

@app.post("/initialize")
async def initialize(request: Request):
    try:
        global CurrentSession
        CurrentSession = SessionData()

        data = await request.json()
        apiKey = data.get("apiKey", "")
        if not apiKey:
            return JSONResponse(content={"error": "API key is required"}, status_code=400)
        CurrentSession.apiKey = apiKey
        CurrentSession.initialized = True
        return JSONResponse(content={"message": "Session initialized successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/disconnect")
async def disconnect():
    try:
        global CurrentSession
        CurrentSession.apiKey = ""
        CurrentSession.initialized = False
        return JSONResponse(content={"message": "Session disconnected successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#Data handling routes

@app.post("/loadModel")
async def loadModel():
    try:
        global CurrentProject
        path = pickModelFile()

        if(CurrentProject.modelFilepath != ""):
            loadedModel = os.path.join("temp_project", os.path.basename(CurrentProject.modelFilepath))
            if(os.path.exists(loadedModel)):
                os.remove(loadedModel)

        CurrentProject.modelFilepath = path
        
        result = load_onnx_model(path)

        with open("temp_project/modelInfo.json", "w") as f:
            json.dump(result, f)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/loadCSV")
async def loadCsv():
    try:
        global CurrentProject

        path = pickCSVFile()
        if(CurrentProject.csvFilepath != ""):
            loadedCSV = os.path.join("temp_project", os.path.basename(CurrentProject.csvFilepath))
            if(os.path.exists(loadedCSV)):
                os.remove(loadedCSV)

        CurrentProject.csvFilepath = path
        result = analyseCSV(path)

        with open("temp_project/csvInfo.json", "w") as f:
            json.dump(result, f)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#Project handling routes

@app.post("/newProject")
async def newProject():
    try:
        if(os.path.exists("temp_project")):
            import shutil
            shutil.rmtree("temp_project")
        
        global glCurrentProject
        glCurrentProject = ProjectData()
        
        return JSONResponse(content={"message": "New project created successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/exportProject")
async def exportProject():
    try:
        # Implementation for exporting project
        return JSONResponse(content={"message": "Project exported successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/loadProject")
async def loadProject():
    try:
        path = pickProjectFile()
        with ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall("temp_project")
            zip_ref.close()
        return JSONResponse(content={"message": "Project loaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")))