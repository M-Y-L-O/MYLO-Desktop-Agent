import os
from tkinter import filedialog

from fastapi.params import File

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