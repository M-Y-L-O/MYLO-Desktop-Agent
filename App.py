import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

import json

from Types.RequestTypes import *
from Processing.ModelProcessing import load_onnx_model
from Processing.DataProcessing import analyseCSV
from Utils.FileHandler import pickModelFile,pickCSVFile

# APP SETUP

load_dotenv()

app = FastAPI(
    title="MYLO AGENT",
    version="0.0.1",
    
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

# ROUTES
@app.get("/")
async def root():
    return {"message": "MYLO AGENT is running!"}

@app.post("/loadModel")
async def loadModel():
    try:
        path = pickModelFile()
        result = load_onnx_model(path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/loadCSV")
async def loadCsv():
    try:
        path = pickCSVFile()
        result = analyseCSV(path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("SERVER_PORT")))