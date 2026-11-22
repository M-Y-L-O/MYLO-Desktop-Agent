import os
import tempfile

import pandas as pd


async def saveFile(file, path: str = "temp_project"):
    filename = os.path.basename(str(file.filename or "").replace("\\", "/"))
    if not filename:
        raise ValueError("Uploaded file must have a filename")

    os.makedirs(path, exist_ok=True)
    tempPath = os.path.join(path, filename)

    with open(tempPath, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return tempPath


def loadData(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    raise FileNotFoundError("Data file not found")


def readBinary(path: str):
    if os.path.exists(path):
        with open(path, "rb") as file:
            return file.read()


def createTempFile(fileBytes, extension):
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(fileBytes)
        return temp_file.name
