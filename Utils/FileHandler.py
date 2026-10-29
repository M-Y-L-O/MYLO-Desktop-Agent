import os
import pandas as pd
from typing import Optional

from fastapi.params import File
import tempfile

def loadModel(path:str):
    if os.path.exists(path):
        with open(path, 'r') as file:
            binary_data = file.read()
            return binary_data
    else:
        raise FileNotFoundError("Model file not found")

def _safe_filename(name: Optional[str]) -> str:
    base = os.path.basename(name or "upload.bin")
    base = base.replace("\x00", "").strip()
    if not base or base in (".", ".."):
        return "upload.bin"
    return base

async def saveFile(file:File, path:str = "temp_project"):
    dest_dir = os.path.abspath(path)
    os.makedirs(dest_dir, exist_ok=True)
    filename = _safe_filename(getattr(file, "filename", None))
    tempPath = os.path.abspath(os.path.join(dest_dir, filename))
    if os.path.commonpath([dest_dir, tempPath]) != dest_dir:
        raise ValueError("Invalid upload filename")

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
