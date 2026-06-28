from Types.Types import *
from Utils.FileHandler import loadData, readBinary
from Utils.Other import getDevice
import asyncio
import concurrent.futures
from Processing.Data.DataProcessingForTraining import encodeData, mapOriginalToEncodedColumns
import torch

async def optimizeModel(project:ProjectData, requestInfo:OptimizationRequest, statusCalback):
    statusCalback({"status":"Processing request...", "progress":0})
    data = loadData(project.csvFilepath)
    statusCalback({"status":"Encoding data...", "progress":3})
    result = encodeData(data, requestInfo.encoding)
    encodedDf = result[0]
    encodedMetadata = result[1]

    mappedInputFeatures = mapOriginalToEncodedColumns(requestInfo.inputFeatures, encodedMetadata, encodedDf)
    mappedTargetFeature = mapOriginalToEncodedColumns([requestInfo.targetFeature], encodedMetadata, encodedDf)[0]

    statusCalback({"status":"Loading model...", "progress":5})
    model = readBinary(project.modelFilepath)

    isPytorchModel = project.modelFilepath.endswith(".pt") or project.modelFilepath.endswith(".pth") or project.modelFilepath.endswith(".pt2")

    statusCalback({"status":"Optimizing model...", "progress":10})
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: findOptimalArchitecture(model, encodedDf, requestInfo, isPytorchModel, statusCallback=statusCalback)
        )
    
    if "modelPath" not in result:
        return {"error": "Optimization failed", "details": result}

def findOptimalArchitecture(modelBytes, df, requestInfo:OptimizationRequest, isPytorchModel, statusCallback=None):
    try:

        device = getDevice()

        torch.set_num_threads(4)

        if df is None or df.empty:
            return {"error": "Encoded DataFrame is empty"}
        
        if modelBytes is None or len(modelBytes) == 0:
            return {"error": "Model bytes are empty"}
        
        statusCallback({"status":"Analasying model...", "progress":15})
        
        
