import os
from tkinter import filedialog
import pandas as pd

from fastapi.params import File
import tempfile

def loadModel(path:str):
    if os.path.exists(path):
        with open(path, 'r') as file:
            binary_data = file.read()
            return binary_data
    else:
        raise FileNotFoundError("Model file not found")
    
async def saveFile(file:File, path:str = "temp_project"):
    tempPath = os.path.join(path, file.filename)

    with open(tempPath, "wb") as f:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk:
                break
            f.write(chunk)
            
    return tempPath

def loadData(path:str):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError("Data file not found")
    
def readBinary(path:str):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            binary_data = file.read()
            return binary_data
        
def createTempFile(fileBytes, extension):
  with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
    temp_file.write(fileBytes)
    temp_file_path = temp_file.name
  
  return temp_file_path